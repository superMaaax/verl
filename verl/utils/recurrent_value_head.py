# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helpers for stateful critic value heads."""

from __future__ import annotations

import inspect
from types import MethodType

import torch
from torch import nn

from verl.trainer.ppo.value_categorical import ValueHeadArchitectureSpec


class StatefulValueHead(nn.Module):
    """Base class for token value heads that maintain state across the sequence."""

    architecture: str = "stateful"

    def __init__(self, input_size: int, state_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size

        self._runtime_attention_mask = None
        self._runtime_position_ids = None
        self._last_debug_metrics: dict[str, float] = {}

    def set_runtime_context(self, *, attention_mask=None, position_ids=None) -> None:
        self._runtime_attention_mask = attention_mask
        self._runtime_position_ids = position_ids

    def clear_runtime_context(self) -> None:
        self._runtime_attention_mask = None
        self._runtime_position_ids = None

    def clear_debug_metrics(self) -> None:
        self._last_debug_metrics = {}

    def get_debug_metrics(self) -> dict[str, float]:
        return dict(self._last_debug_metrics)

    def _update_debug_metrics(self, metrics: dict[str, torch.Tensor | float]) -> None:
        debug_metrics: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().item()
            debug_metrics[key] = float(value)
        self._last_debug_metrics = debug_metrics

    def _normalize_attention_mask(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        attention_mask = self._runtime_attention_mask
        if attention_mask is None:
            return None

        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        elif attention_mask.dim() != 2:
            raise ValueError(f"attention_mask must have 1 or 2 dims, got shape {tuple(attention_mask.shape)}")

        expected_shape = hidden_states.shape[:2]
        if tuple(attention_mask.shape) != tuple(expected_shape):
            raise ValueError(
                "attention_mask shape must match the hidden-state sequence shape. "
                f"expected {tuple(expected_shape)}, got {tuple(attention_mask.shape)}"
            )
        return attention_mask.to(device=hidden_states.device)

    def _normalize_position_ids(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        position_ids = self._runtime_position_ids
        if position_ids is None:
            return None

        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        elif position_ids.dim() == 2:
            pass
        elif position_ids.dim() == 3:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not currently support 3D position_ids "
                f"(for example, multimodal mRoPE). Got shape {tuple(position_ids.shape)}."
            )
        else:
            raise ValueError(f"position_ids must have 1, 2, or 3 dims, got shape {tuple(position_ids.shape)}")

        expected_shape = hidden_states.shape[:2]
        if tuple(position_ids.shape) != tuple(expected_shape):
            raise ValueError(
                "position_ids shape must match the hidden-state sequence shape. "
                f"expected {tuple(expected_shape)}, got {tuple(position_ids.shape)}"
            )
        return position_ids.to(device=hidden_states.device)

    def _prepare_runtime_inputs(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, bool, torch.Tensor | None, torch.Tensor | None]:
        if hidden_states.dim() not in {2, 3}:
            raise ValueError(
                f"{self.__class__.__name__} expects hidden states of shape [T, H] or [B, T, H], "
                f"got {hidden_states.shape}"
            )

        squeeze_batch = hidden_states.dim() == 2
        if squeeze_batch:
            hidden_states = hidden_states.unsqueeze(0)

        attention_mask = self._normalize_attention_mask(hidden_states)
        position_ids = None
        if self._runtime_position_ids is not None:
            try:
                position_ids = self._normalize_position_ids(hidden_states)
            except NotImplementedError:
                if attention_mask is None:
                    raise
                # Keep padding-mask support for multimodal models that use 3D position_ids.
                position_ids = None
        return hidden_states, squeeze_batch, attention_mask, position_ids

    def _sequence_reset_mask(self, position_ids: torch.Tensor | None) -> torch.Tensor | None:
        if position_ids is None:
            return None

        reset_mask = torch.ones_like(position_ids, dtype=torch.bool)
        if position_ids.shape[1] > 1:
            reset_mask[:, 1:] = position_ids[:, 1:] <= position_ids[:, :-1]
        return reset_mask

    def _masked_readout(
        self, *, next_state: torch.Tensor, prev_state: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = torch.where(valid_mask.unsqueeze(-1), next_state, prev_state)
        step_logits = self.readout(state)
        step_logits = torch.where(valid_mask.unsqueeze(-1), step_logits, torch.zeros_like(step_logits))
        return state, step_logits

    def _masked_feature_mean(self, values: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        values = values.detach()
        if valid_mask is None:
            return values.mean()

        weights = valid_mask.to(dtype=values.dtype).unsqueeze(-1)
        denom = (weights.sum() * values.shape[-1]).clamp_min(1.0)
        return (values * weights).sum() / denom

    def _masked_feature_std(self, values: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        values = values.detach()
        mean = self._masked_feature_mean(values, valid_mask)
        if valid_mask is None:
            variance = (values - mean).pow(2).mean()
        else:
            weights = valid_mask.to(dtype=values.dtype).unsqueeze(-1)
            denom = (weights.sum() * values.shape[-1]).clamp_min(1.0)
            variance = ((values - mean).pow(2) * weights).sum() / denom
        return variance.clamp_min(0.0).sqrt()

    def value_head_parameters(self) -> tuple[torch.nn.Parameter | None, ...]:
        raise NotImplementedError

    def forward_sequence(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, squeeze_batch, attention_mask, position_ids = self._prepare_runtime_inputs(hidden_states)
        outputs = self.forward_sequence(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        if squeeze_batch:
            outputs = outputs.squeeze(0)
        return outputs


class RecurrentValueHead(StatefulValueHead):
    """GRU-based token value head with optional runtime sequence-boundary context."""

    architecture = "gru"

    def __init__(self, input_size: int, hidden_size: int, output_size: int, *, bias: bool = True):
        super().__init__(input_size=input_size, state_size=hidden_size, output_size=output_size)
        self.hidden_size = hidden_size

        self.gru = nn.GRUCell(input_size=input_size, hidden_size=hidden_size, bias=bias)
        self.readout = nn.Linear(hidden_size, output_size, bias=bias)

    def value_head_parameters(self) -> tuple[torch.nn.Parameter | None, ...]:
        return (
            self.gru.weight_ih,
            self.gru.weight_hh,
            self.gru.bias_ih,
            self.gru.bias_hh,
            self.readout.weight,
            self.readout.bias,
        )

    def forward_sequence(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        state = hidden_states.new_zeros(batch_size, self.hidden_size)
        zero_state = hidden_states.new_zeros(batch_size, self.hidden_size)
        valid_mask = attention_mask.to(dtype=torch.bool) if attention_mask is not None else None
        reset_mask = self._sequence_reset_mask(position_ids)
        logits = []
        state_norm_sum = hidden_states.new_tensor(0.0, dtype=torch.float32)
        state_norm_count = hidden_states.new_tensor(0.0, dtype=torch.float32)

        for step in range(seq_len):
            step_hidden = hidden_states[:, step, :]
            prev_state = state

            if reset_mask is not None:
                prev_state = torch.where(reset_mask[:, step].unsqueeze(-1), zero_state, prev_state)

            next_state = self.gru(step_hidden, prev_state)

            if valid_mask is not None:
                state, step_logits = self._masked_readout(
                    next_state=next_state,
                    prev_state=prev_state,
                    valid_mask=valid_mask[:, step],
                )
            else:
                state = next_state
                step_logits = self.readout(state)

            step_state_norm = state.detach().float().norm(dim=-1)
            if valid_mask is not None:
                step_weight = valid_mask[:, step].to(dtype=step_state_norm.dtype)
                state_norm_sum += (step_state_norm * step_weight).sum()
                state_norm_count += step_weight.sum()
            else:
                state_norm_sum += step_state_norm.sum()
                state_norm_count += step_state_norm.new_tensor(step_state_norm.numel())

            logits.append(step_logits)

        self._update_debug_metrics(
            {
                "critic/value_head/state_norm": state_norm_sum / state_norm_count.clamp_min(1.0),
            }
        )
        return torch.stack(logits, dim=1)


class GatedCarryValueHead(StatefulValueHead):
    """Running-belief value head with a gated carry update."""

    architecture = "gated_carry"

    def __init__(self, input_size: int, hidden_size: int, output_size: int, *, bias: bool = True):
        super().__init__(input_size=input_size, state_size=hidden_size, output_size=output_size)
        self.hidden_size = hidden_size

        self.candidate_proj = nn.Linear(input_size, hidden_size, bias=bias)
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=bias)
        self.readout = nn.Linear(hidden_size, output_size, bias=bias)

    def value_head_parameters(self) -> tuple[torch.nn.Parameter | None, ...]:
        return (
            self.candidate_proj.weight,
            self.candidate_proj.bias,
            self.gate_proj.weight,
            self.gate_proj.bias,
            self.readout.weight,
            self.readout.bias,
        )

    def forward_sequence(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        candidate_states = torch.tanh(self.candidate_proj(hidden_states))
        carry_gates = torch.sigmoid(self.gate_proj(hidden_states))

        state = hidden_states.new_zeros(batch_size, self.hidden_size)
        zero_state = hidden_states.new_zeros(batch_size, self.hidden_size)
        valid_mask = attention_mask.to(dtype=torch.bool) if attention_mask is not None else None
        reset_mask = self._sequence_reset_mask(position_ids)
        logits = []
        state_norm_sum = hidden_states.new_tensor(0.0, dtype=torch.float32)
        state_norm_count = hidden_states.new_tensor(0.0, dtype=torch.float32)

        for step in range(seq_len):
            prev_state = state
            if reset_mask is not None:
                prev_state = torch.where(reset_mask[:, step].unsqueeze(-1), zero_state, prev_state)

            step_gate = carry_gates[:, step, :]
            step_candidate = candidate_states[:, step, :]
            next_state = step_gate * prev_state + (1.0 - step_gate) * step_candidate

            if valid_mask is not None:
                state, step_logits = self._masked_readout(
                    next_state=next_state,
                    prev_state=prev_state,
                    valid_mask=valid_mask[:, step],
                )
            else:
                state = next_state
                step_logits = self.readout(state)

            step_state_norm = state.detach().float().norm(dim=-1)
            if valid_mask is not None:
                step_weight = valid_mask[:, step].to(dtype=step_state_norm.dtype)
                state_norm_sum += (step_state_norm * step_weight).sum()
                state_norm_count += step_weight.sum()
            else:
                state_norm_sum += step_state_norm.sum()
                state_norm_count += step_state_norm.new_tensor(step_state_norm.numel())

            logits.append(step_logits)

        self._update_debug_metrics(
            {
                "critic/value_head/state_norm": state_norm_sum / state_norm_count.clamp_min(1.0),
                "critic/value_head/gate_mean": self._masked_feature_mean(carry_gates, valid_mask),
                "critic/value_head/gate_std": self._masked_feature_std(carry_gates, valid_mask),
                "critic/value_head/candidate_abs_mean": self._masked_feature_mean(candidate_states.abs(), valid_mask),
            }
        )
        return torch.stack(logits, dim=1)


GatedCarryCritic = GatedCarryValueHead


def _get_supported_value_head_attr(model: nn.Module) -> tuple[str, nn.Module] | None:
    for attr_name in ("score", "classifier"):
        module = getattr(model, attr_name, None)
        if isinstance(module, (nn.Linear, StatefulValueHead)):
            return attr_name, module
    return None


def _build_stateful_value_head(head_module: nn.Linear, head_spec: ValueHeadArchitectureSpec) -> StatefulValueHead:
    common_kwargs = {
        "input_size": head_module.in_features,
        "hidden_size": head_spec.recurrent_state_size,
        "output_size": head_module.out_features,
        "bias": head_module.bias is not None,
    }
    if head_spec.architecture == "gru":
        return RecurrentValueHead(**common_kwargs)
    if head_spec.architecture == "gated_carry":
        return GatedCarryValueHead(**common_kwargs)

    raise ValueError(f"Unsupported stateful value-head architecture: {head_spec.architecture}")


def patch_recurrent_value_head(model: nn.Module, head_spec: ValueHeadArchitectureSpec) -> nn.Module:
    """Replace the scalar token head with a stateful head when requested."""
    head_spec.validate()
    if not head_spec.is_recurrent():
        return model

    target = _get_supported_value_head_attr(model)
    if target is None:
        raise ValueError(
            "Stateful critic heads require a HuggingFace token-classification model with a linear "
            "`score` or `classifier` head."
        )

    attr_name, head_module = target
    if isinstance(head_module, StatefulValueHead):
        if (
            head_module.architecture == head_spec.architecture
            and head_module.state_size == head_spec.recurrent_state_size
        ):
            _patch_forward_runtime_context(model, attr_name)
            return model
        raise ValueError(
            f"Model already has a stateful value head with architecture={head_module.architecture}, "
            f"but the requested architecture is {head_spec.architecture}."
        )

    stateful_head = _build_stateful_value_head(head_module, head_spec)
    setattr(model, attr_name, stateful_head)
    _patch_forward_runtime_context(model, attr_name)
    return model


def _patch_forward_runtime_context(model: nn.Module, attr_name: str) -> None:
    if getattr(model, "_recurrent_value_head_forward_patched", False):
        setattr(model, "_recurrent_value_head_attr_name", attr_name)
        return

    original_forward = model.forward
    signature = inspect.signature(original_forward)

    def wrapped_forward(self, *args, **kwargs):
        head_name = getattr(self, "_recurrent_value_head_attr_name", attr_name)
        head = getattr(self, head_name, None)
        if not isinstance(head, StatefulValueHead):
            return original_forward(*args, **kwargs)

        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids")
        if attention_mask is None or position_ids is None:
            try:
                bound_args = signature.bind_partial(*args, **kwargs)
            except TypeError:
                bound_args = None
            if bound_args is not None:
                if attention_mask is None:
                    attention_mask = bound_args.arguments.get("attention_mask")
                if position_ids is None:
                    position_ids = bound_args.arguments.get("position_ids")

        head.set_runtime_context(attention_mask=attention_mask, position_ids=position_ids)
        try:
            return original_forward(*args, **kwargs)
        finally:
            head.clear_runtime_context()

    model.forward = MethodType(wrapped_forward, model)
    setattr(model, "_recurrent_value_head_forward_patched", True)
    setattr(model, "_recurrent_value_head_attr_name", attr_name)
