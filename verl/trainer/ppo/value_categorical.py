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
"""Utilities for categorical value heads and target projection."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

SUPPORTED_VALUE_HEAD_TYPES = {"scalar", "categorical"}
SUPPORTED_VALUE_HEAD_ARCHITECTURES = {"linear", "gru", "gated_carry"}
SUPPORTED_VALUE_TARGET_TYPES = {"two_hot", "hl_gauss", "one_hot"}
SUPPORTED_VALUE_TARGET_SCALING = {"identity", "affine"}
SUPPORTED_VALUE_RANGE_BEHAVIOR = {"error", "warn", "clip"}


def _get_config_value(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)

    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)

    return getattr(config, key, default)


@dataclass(frozen=True)
class ValueHeadSpec:
    head_type: str = "scalar"
    num_bins: int = 11
    value_min: float = 0.0
    value_max: float = 1.0
    target_type: str = "two_hot"
    hl_gauss_sigma: float | None = None
    hl_gauss_sigma_ratio: float = 0.75
    target_scaling: str = "identity"
    target_scale_min: float = 0.0
    target_scale_max: float = 1.0
    target_out_of_range: str = "error"

    def validate(self) -> None:
        if self.head_type not in SUPPORTED_VALUE_HEAD_TYPES:
            raise ValueError(
                f"value_head_type must be one of {sorted(SUPPORTED_VALUE_HEAD_TYPES)}, got {self.head_type}"
            )

        if self.value_max <= self.value_min:
            raise ValueError(f"value_max must be greater than value_min, got [{self.value_min}, {self.value_max}]")

        if self.head_type == "categorical":
            if self.num_bins < 2:
                raise ValueError(f"value_num_bins must be >= 2 for categorical value head, got {self.num_bins}")
            if self.target_type not in SUPPORTED_VALUE_TARGET_TYPES:
                raise ValueError(
                    f"value_target_type must be one of {sorted(SUPPORTED_VALUE_TARGET_TYPES)}, got {self.target_type}"
                )
            if self.target_scaling not in SUPPORTED_VALUE_TARGET_SCALING:
                raise ValueError(
                    f"value_target_scaling must be one of {sorted(SUPPORTED_VALUE_TARGET_SCALING)}, "
                    f"got {self.target_scaling}"
                )
            if self.target_out_of_range not in SUPPORTED_VALUE_RANGE_BEHAVIOR:
                raise ValueError(
                    f"value_target_out_of_range must be one of {sorted(SUPPORTED_VALUE_RANGE_BEHAVIOR)}, "
                    f"got {self.target_out_of_range}"
                )
            if self.hl_gauss_sigma is not None and self.hl_gauss_sigma <= 0:
                raise ValueError(f"value_hl_gauss_sigma must be > 0 when set, got {self.hl_gauss_sigma}")
            if self.hl_gauss_sigma_ratio <= 0:
                raise ValueError(f"value_hl_gauss_sigma_ratio must be > 0, got {self.hl_gauss_sigma_ratio}")
            if self.target_scaling == "affine" and self.target_scale_max <= self.target_scale_min:
                raise ValueError(
                    "value_target_scale_max must be greater than value_target_scale_min when using affine scaling, "
                    f"got [{self.target_scale_min}, {self.target_scale_max}]"
                )

    def is_categorical(self) -> bool:
        return self.head_type == "categorical"

    def bin_step(self) -> float:
        return (self.value_max - self.value_min) / (self.num_bins - 1)

    def effective_hl_gauss_sigma(self) -> float:
        if self.hl_gauss_sigma is not None:
            return self.hl_gauss_sigma
        return self.hl_gauss_sigma_ratio * self.bin_step()

    def support(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.linspace(self.value_min, self.value_max, self.num_bins, device=device, dtype=dtype)

    def bin_edges(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        support = self.support(device=device, dtype=dtype)
        edges = torch.empty(self.num_bins + 1, device=support.device, dtype=support.dtype)
        edges[1:-1] = (support[:-1] + support[1:]) * 0.5
        half_step = self.bin_step() * 0.5
        edges[0] = support[0] - half_step
        edges[-1] = support[-1] + half_step
        return edges


@dataclass(frozen=True)
class ValueHeadArchitectureSpec:
    architecture: str = "linear"
    gru_hidden_size: int = 256
    state_size: int | None = None

    @property
    def recurrent_state_size(self) -> int:
        return self.gru_hidden_size if self.state_size is None else self.state_size

    def validate(self) -> None:
        if self.architecture not in SUPPORTED_VALUE_HEAD_ARCHITECTURES:
            raise ValueError(
                "value_head_architecture must be one of "
                f"{sorted(SUPPORTED_VALUE_HEAD_ARCHITECTURES)}, got {self.architecture}"
            )

        if self.is_recurrent() and self.recurrent_state_size <= 0:
            raise ValueError(
                "value_head_state_size/value_head_gru_hidden_size must be > 0, "
                f"got {self.recurrent_state_size}"
            )

    def is_recurrent(self) -> bool:
        return self.architecture in {"gru", "gated_carry"}


def extract_value_head_spec(config: Any) -> ValueHeadSpec:
    hl_gauss_sigma = _get_config_value(config, "value_hl_gauss_sigma", None)
    if hl_gauss_sigma is not None:
        hl_gauss_sigma = float(hl_gauss_sigma)

    spec = ValueHeadSpec(
        head_type=_get_config_value(config, "value_head_type", "scalar"),
        num_bins=int(_get_config_value(config, "value_num_bins", 11)),
        value_min=float(_get_config_value(config, "value_min", 0.0)),
        value_max=float(_get_config_value(config, "value_max", 1.0)),
        target_type=_get_config_value(config, "value_target_type", "two_hot"),
        hl_gauss_sigma=hl_gauss_sigma,
        hl_gauss_sigma_ratio=float(_get_config_value(config, "value_hl_gauss_sigma_ratio", 0.75)),
        target_scaling=_get_config_value(config, "value_target_scaling", "identity"),
        target_scale_min=float(_get_config_value(config, "value_target_scale_min", 0.0)),
        target_scale_max=float(_get_config_value(config, "value_target_scale_max", 1.0)),
        target_out_of_range=_get_config_value(config, "value_target_out_of_range", "error"),
    )
    spec.validate()
    return spec


def extract_value_head_architecture_spec(config: Any) -> ValueHeadArchitectureSpec:
    state_size = _get_config_value(config, "value_head_state_size", None)
    if state_size is not None:
        state_size = int(state_size)

    spec = ValueHeadArchitectureSpec(
        architecture=_get_config_value(config, "value_head_architecture", "linear"),
        gru_hidden_size=int(_get_config_value(config, "value_head_gru_hidden_size", 256)),
        state_size=state_size,
    )
    spec.validate()
    return spec


def apply_value_head_spec_to_hf_config(hf_config: Any, spec: ValueHeadSpec) -> None:
    # Persist value-head settings on HF config so model-engine workers can access them.
    setattr(hf_config, "value_head_type", spec.head_type)
    setattr(hf_config, "value_num_bins", spec.num_bins)
    setattr(hf_config, "value_min", spec.value_min)
    setattr(hf_config, "value_max", spec.value_max)
    setattr(hf_config, "value_target_type", spec.target_type)
    setattr(hf_config, "value_hl_gauss_sigma", spec.hl_gauss_sigma)
    setattr(hf_config, "value_hl_gauss_sigma_ratio", spec.hl_gauss_sigma_ratio)
    setattr(hf_config, "value_target_scaling", spec.target_scaling)
    setattr(hf_config, "value_target_scale_min", spec.target_scale_min)
    setattr(hf_config, "value_target_scale_max", spec.target_scale_max)
    setattr(hf_config, "value_target_out_of_range", spec.target_out_of_range)
    setattr(hf_config, "num_labels", spec.num_bins if spec.is_categorical() else 1)


def apply_value_head_architecture_spec_to_hf_config(hf_config: Any, spec: ValueHeadArchitectureSpec) -> None:
    recurrent_state_size = spec.recurrent_state_size if spec.is_recurrent() else spec.gru_hidden_size
    setattr(hf_config, "value_head_architecture", spec.architecture)
    setattr(hf_config, "value_head_gru_hidden_size", recurrent_state_size)
    setattr(hf_config, "value_head_state_size", spec.recurrent_state_size if spec.is_recurrent() else spec.state_size)


def value_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)


def value_probs_to_scaled_scalar(probs: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    return (probs * support).sum(dim=-1)


def scale_scalar_values(values: torch.Tensor, spec: ValueHeadSpec) -> torch.Tensor:
    if spec.target_scaling == "identity":
        return values
    # Affine scaling from raw value range to categorical support range.
    raw_span = spec.target_scale_max - spec.target_scale_min
    support_span = spec.value_max - spec.value_min
    return ((values - spec.target_scale_min) / raw_span) * support_span + spec.value_min


def unscale_scalar_values(values: torch.Tensor, spec: ValueHeadSpec) -> torch.Tensor:
    if spec.target_scaling == "identity":
        return values
    raw_span = spec.target_scale_max - spec.target_scale_min
    support_span = spec.value_max - spec.value_min
    return ((values - spec.value_min) / support_span) * raw_span + spec.target_scale_min


def clamp_scaled_targets(
    scaled_targets: torch.Tensor,
    spec: ValueHeadSpec,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if valid_mask is None:
        valid_mask = torch.ones_like(scaled_targets, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(torch.bool)

    out_of_range = ((scaled_targets < spec.value_min) | (scaled_targets > spec.value_max)) & valid_mask
    denom = valid_mask.float().sum().clamp_min(1.0)
    out_of_range_fraction = out_of_range.float().sum() / denom
    if out_of_range.any():
        msg = (
            f"Categorical critic target is outside [{spec.value_min}, {spec.value_max}] after scaling. "
            f"out_of_range_fraction={out_of_range_fraction.item():.6f}. "
            "Adjust value_target_scaling/value_target_scale_{min,max} or support range."
        )
        if spec.target_out_of_range == "error":
            raise ValueError(msg)
        if spec.target_out_of_range == "warn":
            warnings.warn(msg, stacklevel=2)

    return scaled_targets.clamp(min=spec.value_min, max=spec.value_max), out_of_range_fraction


def project_two_hot(scaled_targets: torch.Tensor, spec: ValueHeadSpec, support: torch.Tensor) -> torch.Tensor:
    step = spec.bin_step()
    lower_pos = (scaled_targets - support[0]) / step
    lower_idx = torch.floor(lower_pos).long().clamp(min=0, max=spec.num_bins - 1)
    upper_idx = (lower_idx + 1).clamp(max=spec.num_bins - 1)

    lower_vals = support.gather(0, lower_idx.view(-1)).view_as(scaled_targets)
    upper_weight = ((scaled_targets - lower_vals) / step).clamp(min=0.0, max=1.0)
    lower_weight = 1.0 - upper_weight

    probs = torch.zeros(
        (*scaled_targets.shape, spec.num_bins), device=scaled_targets.device, dtype=scaled_targets.dtype
    )
    probs.scatter_add_(dim=-1, index=lower_idx.unsqueeze(-1), src=lower_weight.unsqueeze(-1))
    probs.scatter_add_(dim=-1, index=upper_idx.unsqueeze(-1), src=upper_weight.unsqueeze(-1))
    return probs


def project_one_hot(scaled_targets: torch.Tensor, spec: ValueHeadSpec, support: torch.Tensor) -> torch.Tensor:
    step = spec.bin_step()
    idx = torch.round((scaled_targets - support[0]) / step).long().clamp(min=0, max=spec.num_bins - 1)
    return F.one_hot(idx, num_classes=spec.num_bins).to(dtype=scaled_targets.dtype)


def project_hl_gauss(scaled_targets: torch.Tensor, spec: ValueHeadSpec, edges: torch.Tensor) -> torch.Tensor:
    sigma = spec.effective_hl_gauss_sigma()
    denom = math.sqrt(2.0) * sigma
    cdf_evals = torch.erf((edges - scaled_targets.unsqueeze(-1)) / denom)
    z = (cdf_evals[..., -1] - cdf_evals[..., 0]).unsqueeze(-1).clamp_min(1e-12)
    probs = (cdf_evals[..., 1:] - cdf_evals[..., :-1]) / z
    probs = probs.clamp_min(0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def project_scalar_targets_to_probs(
    scaled_targets: torch.Tensor,
    spec: ValueHeadSpec,
    support: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    if spec.target_type == "two_hot":
        return project_two_hot(scaled_targets, spec, support)
    if spec.target_type == "one_hot":
        return project_one_hot(scaled_targets, spec, support)
    if spec.target_type == "hl_gauss":
        return project_hl_gauss(scaled_targets, spec, edges)
    raise ValueError(f"Unsupported value_target_type: {spec.target_type}")


def value_logits_to_scalar_expectation(
    value_logits: torch.Tensor, spec: ValueHeadSpec
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if value_logits.shape[-1] != spec.num_bins:
        raise ValueError(
            f"value_logits has trailing dim {value_logits.shape[-1]} but value_num_bins={spec.num_bins}"
        )
    logits_fp32 = value_logits.float()
    support = spec.support(device=logits_fp32.device, dtype=logits_fp32.dtype)
    probs = value_logits_to_probs(logits_fp32)
    scaled_values = value_probs_to_scaled_scalar(probs, support)
    scalar_values = unscale_scalar_values(scaled_values, spec)
    return scalar_values, probs, scaled_values


def soft_target_cross_entropy(value_logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(value_logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1)


def categorical_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)


def expected_bin_index(probs: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
    return (probs * idx).sum(dim=-1)
