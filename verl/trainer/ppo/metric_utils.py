# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.import_utils import deprecated


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def _safe_mean_max_min(values: torch.Tensor) -> tuple[float, float, float]:
    if values.numel() == 0:
        return 0.0, 0.0, 0.0
    return (
        torch.mean(values).detach().item(),
        torch.max(values).detach().item(),
        torch.min(values).detach().item(),
    )


def _mean_or_zero(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return torch.mean(values).detach().item()


def _variance_or_zero(values: torch.Tensor) -> float:
    if values.numel() <= 1:
        return 0.0
    return torch.var(values.float(), unbiased=False).detach().item()


def _safe_pearson_corr(values_a: torch.Tensor, values_b: torch.Tensor) -> float:
    if values_a.numel() <= 1 or values_b.numel() <= 1:
        return 0.0
    values_a = values_a.float().reshape(-1)
    values_b = values_b.float().reshape(-1)
    values_a = values_a - values_a.mean()
    values_b = values_b - values_b.mean()
    denom = torch.sqrt(values_a.pow(2).mean() * values_b.pow(2).mean()).clamp_min(1e-8)
    return ((values_a * values_b).mean() / denom).detach().item()


def _safe_var(values: torch.Tensor) -> torch.Tensor:
    values = values.float().reshape(-1)
    if values.numel() <= 1:
        return values.new_tensor(0.0)
    return torch.var(values, unbiased=False)


def _safe_variance_ratio(
    numerator_values: torch.Tensor,
    denominator_values: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[float, float, float]:
    numerator_var = _safe_var(numerator_values)
    denominator_var = _safe_var(denominator_values)
    if denominator_var <= eps:
        return 1.0, numerator_var.detach().item(), denominator_var.detach().item()
    ratio = numerator_var / denominator_var
    return ratio.detach().item(), numerator_var.detach().item(), denominator_var.detach().item()


def _select_masked_position_values(
    values: torch.Tensor,
    mask: torch.Tensor,
    row_mask: torch.Tensor | None = None,
    position: str = "last",
) -> torch.Tensor:
    """Select one scalar per row from the masked response-aligned sequence values."""
    mask = mask.to(dtype=torch.bool)
    valid_rows = mask.any(dim=-1)
    if row_mask is not None:
        valid_rows = valid_rows & row_mask.to(dtype=torch.bool)
    if not valid_rows.any():
        return values.new_empty((0,), dtype=values.dtype)

    masked_values = values[valid_rows]
    masked_positions = mask[valid_rows]
    seq_positions = torch.arange(masked_positions.shape[-1], device=values.device).unsqueeze(0).expand_as(masked_positions)

    if position == "first":
        gather_idx = masked_positions.float().argmax(dim=-1)
    elif position == "last":
        gather_idx = torch.where(
            masked_positions,
            seq_positions + 1,
            torch.zeros_like(seq_positions),
        ).max(dim=-1).values - 1
    else:
        raise ValueError(f"Unsupported masked position selector: {position}.")

    return masked_values.gather(1, gather_idx.long().unsqueeze(-1)).squeeze(-1)


def _get_non_tensor_batch(batch: DataProto) -> dict[str, Any]:
    non_tensor_batch = getattr(batch, "non_tensor_batch", None)
    return non_tensor_batch if isinstance(non_tensor_batch, dict) else {}


def _build_prompt_groups(prompt_ids: np.ndarray | list[Any] | None) -> list[list[int]]:
    if prompt_ids is None:
        return []
    groups: dict[Any, list[int]] = defaultdict(list)
    for idx, prompt_id in enumerate(np.asarray(prompt_ids, dtype=object).tolist()):
        groups[prompt_id].append(idx)
    return list(groups.values())


def _compute_within_prompt_variance_ratio(
    returns_tokenwise: torch.Tensor,
    baseline_tokenwise: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_ids: np.ndarray | list[Any] | None,
) -> dict[str, float]:
    groups = _build_prompt_groups(prompt_ids)
    if not groups:
        return {}

    raw_advantages = returns_tokenwise - baseline_tokenwise
    per_prompt_ratios: list[float] = []
    pooled_centered_returns: list[torch.Tensor] = []
    pooled_centered_advantages: list[torch.Tensor] = []

    for row_indices in groups:
        group_returns = torch.masked_select(returns_tokenwise[row_indices], response_mask[row_indices])
        if group_returns.numel() <= 1:
            continue
        group_advantages = torch.masked_select(raw_advantages[row_indices], response_mask[row_indices])
        group_ratio, _, _ = _safe_variance_ratio(group_advantages, group_returns)
        per_prompt_ratios.append(group_ratio)
        pooled_centered_returns.append(group_returns.float() - group_returns.float().mean())
        pooled_centered_advantages.append(group_advantages.float() - group_advantages.float().mean())

    if not per_prompt_ratios:
        return {
            "var_ratio_within_prompt_mean": 1.0,
            "var_ratio_within_prompt_median": 1.0,
            "var_ratio_within_prompt_pooled": 1.0,
        }

    pooled_ratio, _, _ = _safe_variance_ratio(
        torch.cat(pooled_centered_advantages, dim=0),
        torch.cat(pooled_centered_returns, dim=0),
    )
    ratios_tensor = torch.tensor(per_prompt_ratios, dtype=torch.float32)
    return {
        "var_ratio_within_prompt_mean": ratios_tensor.mean().item(),
        "var_ratio_within_prompt_median": ratios_tensor.median().item(),
        "var_ratio_within_prompt_pooled": pooled_ratio,
    }


def _compute_position_variance_ratios(
    rollout_returns: torch.Tensor,
    baseline_tokenwise: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    position_specs: tuple[tuple[str, float | None], ...] = (
        ("25", 0.25),
        ("50", 0.50),
        ("75", 0.75),
        ("90", 0.90),
        ("final", None),
    )
    valid_rows = response_mask.any(dim=-1)
    if valid_rows.sum().item() <= 1:
        return {}

    valid_lengths = response_mask.sum(dim=-1).to(dtype=torch.long)[valid_rows]
    valid_returns = rollout_returns[valid_rows].float()
    valid_baseline = baseline_tokenwise[valid_rows].float()
    metrics = {}

    for label, fraction in position_specs:
        if fraction is None:
            position_idx = valid_lengths - 1
        else:
            position_idx = torch.ceil(valid_lengths.float() * fraction).to(dtype=torch.long) - 1
        position_idx = position_idx.clamp(min=0)
        row_idx = torch.arange(valid_baseline.shape[0], device=valid_baseline.device)
        selected_baseline = valid_baseline[row_idx, position_idx]
        ratio, _, _ = _safe_variance_ratio(valid_returns - selected_baseline, valid_returns)
        metrics[f"var_ratio_global_pos_{label}"] = ratio
        metrics[f"variance_reduction_gain_global_pos_{label}"] = 1.0 - ratio

    return metrics


def _compute_variance_ratio_metrics(
    prefix: str,
    rollout_returns: torch.Tensor,
    baseline_tokenwise: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_ids: np.ndarray | list[Any] | None,
    include_position_metrics: bool = False,
) -> tuple[dict[str, float], float, float]:
    returns_tokenwise = rollout_returns.unsqueeze(-1).expand_as(baseline_tokenwise).float()
    baseline_tokenwise = baseline_tokenwise.float()
    flat_returns = torch.masked_select(returns_tokenwise, response_mask)
    flat_raw_advantages = torch.masked_select(returns_tokenwise - baseline_tokenwise, response_mask)
    global_ratio, raw_advantage_var, return_var = _safe_variance_ratio(flat_raw_advantages, flat_returns)

    metrics = {
        f"{prefix}/var_ratio_global": global_ratio,
        f"{prefix}/variance_reduction_gain_global": 1.0 - global_ratio,
    }

    within_prompt_metrics = _compute_within_prompt_variance_ratio(
        returns_tokenwise=returns_tokenwise,
        baseline_tokenwise=baseline_tokenwise,
        response_mask=response_mask,
        prompt_ids=prompt_ids,
    )
    if "var_ratio_within_prompt_mean" in within_prompt_metrics:
        within_prompt_metrics["var_ratio_within_prompt"] = within_prompt_metrics["var_ratio_within_prompt_mean"]
    for name, value in within_prompt_metrics.items():
        metrics[f"{prefix}/{name}"] = value
        metrics[f"{prefix}/{name.replace('var_ratio', 'variance_reduction_gain')}"] = 1.0 - value

    if include_position_metrics:
        position_metrics = _compute_position_variance_ratios(
            rollout_returns=rollout_returns,
            baseline_tokenwise=baseline_tokenwise,
            response_mask=response_mask,
        )
        for name, value in position_metrics.items():
            metrics[f"{prefix}/{name}"] = value

    return metrics, raw_advantage_var, return_var


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/prompt_end_value/mean: Mean value at the end of prompt. For prompt-residual critics,
              this uses the prompt-prior head alone; otherwise it uses the token-0 value prediction
              from the main critic output. (if use_critic=True)
            - critic/trajectory_end_value/mean: Mean of the last response-aligned value on each trajectory
              (if use_critic=True)
            - critic/vf_rho: Var(returns - values) / Var(returns) on valid response tokens (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - critic/rollout_return/* and prompt/end-vs-return diagnostics when rollout_returns are available
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()
    raw_response_mask = batch.batch.get("raw_response_mask", batch.batch["response_mask"]).bool()

    max_prompt_length = prompt_mask.size(-1)

    prompt_length = prompt_mask.sum(-1).float()
    response_length = raw_response_mask.sum(-1).float()

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask
    valid_training_seq_mask = response_mask.any(dim=-1)

    valid_sequence_score = sequence_score[valid_training_seq_mask]
    valid_sequence_reward = sequence_reward[valid_training_seq_mask]

    score_mean, score_max, score_min = _safe_mean_max_min(valid_sequence_score)

    reward_mean, reward_max, reward_min = _safe_mean_max_min(valid_sequence_reward)

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)
    advantages_mean, advantages_max, advantages_min = _safe_mean_max_min(valid_adv)
    returns_mean, returns_max, returns_min = _safe_mean_max_min(valid_returns)
    non_tensor_batch = _get_non_tensor_batch(batch)

    if use_critic:
        values = batch.batch["values"]
        prompt_prior_values = batch.batch.get("prompt_prior_values")
        valid_values = torch.masked_select(values, response_mask)
        values_mean, values_max, values_min = _safe_mean_max_min(valid_values)
        if valid_returns.numel() > 1 and valid_values.numel() > 1:
            return_diff_var = torch.var(valid_returns - valid_values)
            return_var = torch.var(valid_returns)
            vf_rho = (return_diff_var / (return_var + 1e-5)).detach().item()
            vf_explained_var = 1.0 - vf_rho
        else:
            vf_rho = 0.0
            vf_explained_var = 0.0

        # For prompt-residual critics, report the prompt-prior head alone at prompt end.
        # Otherwise, values are aligned with action positions, so index 0 corresponds
        # to the value at the last prompt token (before generating token 0).
        # Use valid_training_seq_mask so training diagnostics reflect effective samples only.
        if prompt_prior_values is not None:
            prompt_end_values = prompt_prior_values[valid_training_seq_mask]
        else:
            prompt_end_values = values[valid_training_seq_mask, :1].squeeze(-1)
        prompt_end_value_mean = _mean_or_zero(prompt_end_values)
        # Use raw_response_mask to recover the final response position even if later
        # training-time masking trims intermediate or terminal tokens.
        trajectory_end_values = _select_masked_position_values(
            values=values,
            mask=raw_response_mask,
            row_mask=valid_training_seq_mask,
            position="last",
        )
        trajectory_end_value_mean = _mean_or_zero(trajectory_end_values)

        prompt_residual_metrics = {}
        if "rollout_returns" in batch.batch:
            rollout_returns = batch.batch["rollout_returns"]
            valid_rollout_returns = rollout_returns[valid_training_seq_mask]
            rollout_return_mean, rollout_return_max, rollout_return_min = _safe_mean_max_min(valid_rollout_returns)
            prompt_residual_metrics.update(
                {
                    "critic/rollout_return/mean": rollout_return_mean,
                    "critic/rollout_return/max": rollout_return_max,
                    "critic/rollout_return/min": rollout_return_min,
                    "critic/prompt_end_return_corr": _safe_pearson_corr(prompt_end_values, valid_rollout_returns),
                    "critic/trajectory_end_return_corr": _safe_pearson_corr(
                        trajectory_end_values,
                        valid_rollout_returns,
                    ),
                    "critic/prompt_end_vs_return_gap": _mean_or_zero(
                        torch.abs(prompt_end_values - valid_rollout_returns)
                    ),
                    "critic/trajectory_end_vs_return_gap": _mean_or_zero(
                        torch.abs(trajectory_end_values - valid_rollout_returns)
                    ),
                    "critic/prompt_to_trajectory_value_delta_mean": _mean_or_zero(
                        trajectory_end_values - prompt_end_values
                    ),
                }
            )

        if prompt_prior_values is not None and "rollout_returns" in batch.batch:
            repeated_rollout_returns = rollout_returns.unsqueeze(-1).expand_as(values)
            prompt_ids = non_tensor_batch.get("uid")

            valid_prompt_prior = prompt_prior_values[valid_training_seq_mask]
            valid_prompt_residual = valid_rollout_returns - valid_prompt_prior
            valid_combined_residual = (repeated_rollout_returns - values)[response_mask]

            prompt_prior_baseline = prompt_prior_values.unsqueeze(-1).expand_as(values) * response_mask.to(
                dtype=values.dtype
            )
            prompt_prior_var_metrics, actor_raw_adv_var, actor_raw_return_var = _compute_variance_ratio_metrics(
                prefix="prompt_prior",
                rollout_returns=rollout_returns,
                baseline_tokenwise=prompt_prior_baseline,
                response_mask=response_mask,
                prompt_ids=prompt_ids,
                include_position_metrics=False,
            )
            prompt_residual_metrics.update(prompt_prior_var_metrics)
            actor_baseline_metrics = {
                f"baseline/{key.split('/', 1)[1]}": value for key, value in prompt_prior_var_metrics.items()
            }

            residual_values = batch.batch.get("residual_values")
            if residual_values is not None:
                valid_residual_values = residual_values[response_mask]
                residual_var_metrics, _, _ = _compute_variance_ratio_metrics(
                    prefix="residual",
                    rollout_returns=rollout_returns,
                    baseline_tokenwise=residual_values,
                    response_mask=response_mask,
                    prompt_ids=prompt_ids,
                    include_position_metrics=True,
                )
                prompt_residual_metrics.update(residual_var_metrics)

                combined_var_metrics, actor_raw_adv_var, actor_raw_return_var = _compute_variance_ratio_metrics(
                    prefix="combined",
                    rollout_returns=rollout_returns,
                    baseline_tokenwise=values,
                    response_mask=response_mask,
                    prompt_ids=prompt_ids,
                    include_position_metrics=True,
                )
                prompt_residual_metrics.update(combined_var_metrics)
                actor_baseline_metrics = {
                    f"baseline/{key.split('/', 1)[1]}": value for key, value in combined_var_metrics.items()
                }
                residual_mean = _mean_or_zero(valid_residual_values)
                residual_return_corr = _safe_pearson_corr(
                    valid_residual_values,
                    repeated_rollout_returns[response_mask],
                )
            else:
                residual_mean = 0.0
                residual_return_corr = 0.0

            prompt_residual_metrics.update(actor_baseline_metrics)
            prompt_residual_metrics.update(
                {
                    "critic/prompt_prior_mean": _mean_or_zero(valid_prompt_prior),
                    "critic/residual_mean": residual_mean,
                    "critic/combined_value_mean": values_mean,
                    "critic/prompt_prior_return_corr": _safe_pearson_corr(valid_prompt_prior, valid_rollout_returns),
                    "critic/residual_return_corr": residual_return_corr,
                    "critic/prompt_prior_vs_return_gap": _mean_or_zero(
                        torch.abs(valid_prompt_prior - valid_rollout_returns)
                    ),
                    "critic/combined_vs_return_gap": _mean_or_zero(torch.abs(valid_combined_residual)),
                    "critic/prompt_prior_residual_variance": _variance_or_zero(valid_prompt_residual),
                    "critic/combined_residual_variance": _variance_or_zero(valid_combined_residual),
                    "actor/raw_advantage_var": actor_raw_adv_var,
                    "actor/raw_return_var": actor_raw_return_var,
                    "actor/raw_advantage_var_ratio_vs_returns": (
                        actor_raw_adv_var / max(actor_raw_return_var, 1e-8) if actor_raw_return_var > 0 else 1.0
                    ),
                    "actor/raw_advantage_var_ratio_vs_return_tokens": (
                        actor_raw_adv_var / max(actor_raw_return_var, 1e-8) if actor_raw_return_var > 0 else 1.0
                    ),
                }
            )

    # Aborted samples and non-aborted response length statistics
    # response_length_non_aborted/*: statistics computed on non-aborted samples only
    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()

    non_aborted_response_length = response_length[non_aborted_mask]
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = torch.mean(non_aborted_response_length).detach().item()
        non_aborted_response_length_max = torch.max(non_aborted_response_length).detach().item()
        non_aborted_response_length_min = torch.min(non_aborted_response_length).detach().item()
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_response_length).float()).detach().item()
        )
    else:
        non_aborted_response_length_mean = 0.0
        non_aborted_response_length_max = 0.0
        non_aborted_response_length_min = 0.0
        non_aborted_response_length_clip_ratio = 0.0

    metrics = {
        # score
        "critic/score/mean": score_mean,
        "critic/score/max": score_max,
        "critic/score/min": score_min,
        # reward
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
        # adv
        "critic/advantages/mean": advantages_mean,
        "critic/advantages/max": advantages_max,
        "critic/advantages/min": advantages_min,
        # returns
        "critic/returns/mean": returns_mean,
        "critic/returns/max": returns_max,
        "critic/returns/min": returns_min,
        **(
            {
                # values
                "critic/values/mean": values_mean,
                "critic/values/max": values_max,
                "critic/values/min": values_min,
                # prompt end value (state value before generating first response token)
                "critic/prompt_end_value/mean": prompt_end_value_mean,
                # final response-aligned value available on the trajectory
                "critic/trajectory_end_value/mean": trajectory_end_value_mean,
                # rho = Var(R - V_hat) / Var(R), using the same valid-token mask as vf_explained_var.
                "critic/vf_rho": vf_rho,
                # vf explained var
                "critic/vf_explained_var": vf_explained_var,
                **prompt_residual_metrics,
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # response length (non-aborted only)
        # These statistics exclude aborted samples to avoid skew from zeros
        "response_length_non_aborted/mean": non_aborted_response_length_mean,
        "response_length_non_aborted/max": non_aborted_response_length_max,
        "response_length_non_aborted/min": non_aborted_response_length_min,
        "response_length_non_aborted/clip_ratio": non_aborted_response_length_clip_ratio,
        # aborted ratio
        # Fraction of samples whose response length is zero
        "response/aborted_ratio": aborted_ratio,
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # multi-turn conversation
    if "__num_turns__" in non_tensor_batch:
        num_turns = non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    if "tool_call_counts" in non_tensor_batch:
        tool_call_counts = non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()

    return metrics


def compute_overlong_filtering_metrics(batch: DataProto) -> dict[str, float]:
    if "is_overlong" not in batch.batch or "raw_response_mask" not in batch.batch:
        return {}

    is_overlong = batch.batch["is_overlong"].to(bool)
    raw_response_mask = batch.batch["raw_response_mask"].to(torch.float32)
    overlong_filtered_response_mask = batch.batch.get("overlong_filtered_response_mask", batch.batch["response_mask"])
    overlong_filtered_response_mask = overlong_filtered_response_mask.to(torch.float32)
    raw_valid_seq_mask = raw_response_mask.sum(dim=-1) > 0
    valid_mask = overlong_filtered_response_mask.sum(dim=-1) > 0

    raw_response_lengths = raw_response_mask.sum(dim=-1)
    raw_response_tokens = raw_response_mask.sum()
    valid_response_tokens = overlong_filtered_response_mask.sum()
    masked_response_tokens = raw_response_tokens - valid_response_tokens

    def _mean_by_group(values: torch.Tensor, group_mask: torch.Tensor) -> float:
        selected = values[group_mask]
        return _mean_or_zero(selected)

    metrics = {
        "overlong_filtering/sample_fraction": is_overlong.to(torch.float32).mean().detach().item(),
        "overlong_filtering/masked_token_fraction": (
            masked_response_tokens / (raw_response_tokens + 1e-8)
        ).detach().item(),
        "overlong_filtering/effective_batch_size_before": float(raw_valid_seq_mask.sum().item()),
        "overlong_filtering/effective_batch_size_after": float(valid_mask.sum().item()),
        "overlong_filtering/effective_response_tokens_before": raw_response_tokens.detach().item(),
        "overlong_filtering/effective_response_tokens_after": valid_response_tokens.detach().item(),
        "overlong_filtering/mean_response_length_valid": _mean_by_group(raw_response_lengths, valid_mask),
        "overlong_filtering/mean_response_length_overlong": _mean_by_group(raw_response_lengths, is_overlong),
    }

    if "token_level_scores" in batch.batch:
        sequence_score = batch.batch["token_level_scores"].sum(dim=-1)
        metrics["overlong_filtering/score_mean_valid"] = _mean_by_group(sequence_score, valid_mask)
        metrics["overlong_filtering/score_mean_overlong"] = _mean_by_group(sequence_score, is_overlong)

    if "token_level_rewards" in batch.batch:
        sequence_reward = batch.batch["token_level_rewards"].sum(dim=-1)
        metrics["overlong_filtering/reward_mean_valid"] = _mean_by_group(sequence_reward, valid_mask)
        metrics["overlong_filtering/reward_mean_overlong"] = _mean_by_group(sequence_reward, is_overlong)

    if "acc" in batch.non_tensor_batch:
        acc = torch.as_tensor(batch.non_tensor_batch["acc"], dtype=torch.float32)
        metrics["overlong_filtering/success_mean_valid"] = _mean_by_group(acc, valid_mask.cpu())
        metrics["overlong_filtering/success_mean_overlong"] = _mean_by_group(acc, is_overlong.cpu())
    elif "acc" in batch.batch:
        acc = batch.batch["acc"].to(torch.float32)
        metrics["overlong_filtering/success_mean_valid"] = _mean_by_group(acc, valid_mask)
        metrics["overlong_filtering/success_mean_overlong"] = _mean_by_group(acc, is_overlong)

    return metrics


def compute_grpo_baseline_metrics(batch: DataProto) -> dict[str, float]:
    """Compute GRPO baseline variance reduction metrics.

    This logs the GRPO analogue of critic rho:

        rho = Var(R - b_group) / Var(R)

    where R is the sequence-level reward and b_group is the prompt-group mean
    baseline. To match the current GRPO advantage implementation, singleton
    groups use a zero baseline instead of subtracting their own reward.

    In addition to the sequence-level metric, log tokenwise companions that
    broadcast the sequence reward and baseline over valid response tokens. The
    tokenwise version uses the same masked-token weighting style as PPO's
    ``critic/vf_rho`` and is therefore the better overlay target when comparing
    GRPO against PPO on the same chart. For cross-run dashboard compatibility,
    also expose the tokenwise GRPO metric through the ``critic/vf_*`` aliases
    when no learned critic values are present on the batch.
    """
    non_tensor_batch = _get_non_tensor_batch(batch)
    if "token_level_rewards" not in batch.batch or "uid" not in non_tensor_batch:
        return {}

    response_mask = batch.batch.get("response_mask")
    if response_mask is None:
        response_mask = torch.ones_like(batch.batch["token_level_rewards"], dtype=torch.bool)
    else:
        response_mask = response_mask.bool()
    valid_seq_mask = response_mask.sum(dim=-1) > 0
    if valid_seq_mask.sum() <= 1:
        return {}

    sequence_rewards = batch.batch["token_level_rewards"].sum(dim=-1)[valid_seq_mask]
    group_ids = non_tensor_batch["uid"][valid_seq_mask.cpu().numpy()]
    id2rewards = defaultdict(list)
    for i, group_id in enumerate(group_ids):
        id2rewards[group_id].append(sequence_rewards[i])

    baselines = torch.zeros_like(sequence_rewards)
    for i, group_id in enumerate(group_ids):
        rewards = id2rewards[group_id]
        if len(rewards) > 1:
            baselines[i] = torch.mean(torch.stack(rewards))

    residual_var = torch.var(sequence_rewards - baselines)
    reward_var = torch.var(sequence_rewards)
    baseline_rho = residual_var / (reward_var + 1e-5)

    valid_response_mask = response_mask[valid_seq_mask]
    repeated_sequence_rewards = sequence_rewards.unsqueeze(-1).expand_as(valid_response_mask)
    repeated_baselines = baselines.unsqueeze(-1).expand_as(valid_response_mask)
    tokenwise_sequence_rewards = repeated_sequence_rewards[valid_response_mask]
    tokenwise_residuals = (repeated_sequence_rewards - repeated_baselines)[valid_response_mask]
    tokenwise_residual_var = torch.var(tokenwise_residuals)
    tokenwise_reward_var = torch.var(tokenwise_sequence_rewards)
    baseline_rho_tokenwise = tokenwise_residual_var / (tokenwise_reward_var + 1e-5)

    metrics = {
        "grpo/reward_var": reward_var.detach().item(),
        "grpo/residual_var": residual_var.detach().item(),
        "grpo/baseline_rho": baseline_rho.detach().item(),
        "grpo/baseline_explained_var": (1.0 - baseline_rho).detach().item(),
        "grpo/reward_var_tokenwise": tokenwise_reward_var.detach().item(),
        "grpo/residual_var_tokenwise": tokenwise_residual_var.detach().item(),
        "grpo/baseline_rho_tokenwise": baseline_rho_tokenwise.detach().item(),
        "grpo/baseline_explained_var_tokenwise": (1.0 - baseline_rho_tokenwise).detach().item(),
    }
    if "values" not in batch.batch:
        metrics["critic/vf_rho"] = baseline_rho_tokenwise.detach().item()
        metrics["critic/vf_explained_var"] = (1.0 - baseline_rho_tokenwise).detach().item()
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def compute_zero_critic_metrics() -> dict[str, float]:
    """Metrics emitted when using a fixed zero value function instead of a critic.

    By convention, zero critic corresponds to a baseline that does not reduce
    return variance, so vf_rho=1.0 and vf_explained_var=0.0.
    """
    return {
        "critic/vf_loss": 0.0,
        "critic/vf_clipfrac": 0.0,
        "critic/vf_rho": 1.0,
        "critic/vf_explained_var": 0.0,
        "critic/vpred_mean": 0.0,
        "critic/values/mean": 0.0,
        "critic/values/max": 0.0,
        "critic/values/min": 0.0,
        "critic/prompt_end_value/mean": 0.0,
        "critic/trajectory_end_value/mean": 0.0,
        "critic/grad_norm": 0.0,
    }


def compute_variance_proxy_metrics(batch: DataProto, gradient_norm: float = None) -> dict[str, float]:
    """
    Compute variance proxy metrics using the simplified expected squared norm approach.

    This metric provides a computationally efficient way to monitor gradient variance
    during training. It works for any advantage estimator as long as sum_pi_squared
    is available from the actor.

    Theory:
    - Full variance: Var(g̃) = E[||g̃||²] - ||g_true||²
    - Simplified proxy (when ||g_true||² ≈ 0): Var(g̃) ≈ E[||g̃||²]
    - Using W-score approximation: E[||g̃||²] ≈ E[A² × W(τ)]

    Where W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²] is the score-norm proxy.
    """
    metrics = {}

    # Check if we have the necessary data (sum_pi_squared is required for W-score)
    if "sum_pi_squared" not in batch.batch or "old_log_probs" not in batch.batch or "advantages" not in batch.batch:
        return metrics

    # Compute W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²]
    pi_t = torch.exp(batch.batch["old_log_probs"])
    w_per_timestep = 1 - 2 * pi_t + batch.batch["sum_pi_squared"]

    # Get response mask to only consider valid tokens
    response_mask = batch.batch["response_mask"]
    valid_seq_mask = response_mask.sum(dim=-1) > 0
    if not valid_seq_mask.any():
        return {
            "variance_proxy/proxy1_signal_strength": gradient_norm**2 if gradient_norm is not None else 0.0,
            "variance_proxy/proxy2_total_power": 0.0,
            "variance_proxy/proxy3_pure_noise": 0.0,
            "variance_proxy/expected_a_squared": 0.0,
            "variance_proxy/expected_w": 0.0,
        }

    # Use pre-computed rollout IS weights from batch (for variance proxy consistency with training loss)
    # IS weights are computed centrally in ray_trainer.py to avoid duplication
    rollout_is_weights = None
    if "rollout_is_weights" in batch.batch:
        # Extract pre-computed IS weights from batch (already computed in trainer)
        rollout_is_weights = batch.batch["rollout_is_weights"]

        # Scale W by (rollout IS weight)² for optimal baseline under biased estimation
        w_per_timestep = w_per_timestep * (rollout_is_weights**2).detach()

        # Note: IS weight statistics and mismatch metrics are logged in ray_trainer.py

    # Get scalar advantages (mean over timesteps)
    advantages = batch.batch["advantages"]
    # Compute mean advantage per trajectory using masked_mean
    advantages_scalar = verl_F.masked_mean(advantages, response_mask, axis=-1)[valid_seq_mask]

    # Compute W values (sum over timesteps)
    w_values = verl_F.masked_sum(w_per_timestep, response_mask, axis=-1)[valid_seq_mask]

    # ====== COMPUTE VARIANCE PROXIES ======
    # Variance proxy should match the actual gradient computation:
    # - If IS weights were computed/applied: use them in variance proxy calculation
    # - Otherwise: compute on-policy variance proxy

    # ====== PROXY 1: Signal Strength ||ḡ||² ======
    # The squared norm of the mean gradient (provided from training loop)
    proxy1_signal_strength = gradient_norm**2 if gradient_norm is not None else None

    # ====== PROXY 2: Total Power E[||ĝ_τ||²] ======
    # Measures the average of squared gradient norms (Signal + Noise)
    if rollout_is_weights is not None:
        # Off-policy with IS correction applied: use clamped weights consistently with actual gradient computation
        rollout_is_weights_scalar = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)[valid_seq_mask]
        # Recover original W (before IS correction was applied in line 657)
        # Clamp to avoid division by zero when IS weights are zero
        w_original = verl_F.masked_sum(
            w_per_timestep / torch.clamp((rollout_is_weights**2).detach(), min=1e-10), response_mask, axis=-1
        )[valid_seq_mask]
        # Clamp W to avoid negative values (which would cause NaN in sqrt)
        w_original = torch.clamp(w_original, min=0.0)
        # Proxy 2 for off-policy: E[ρ̄² × A² × W]
        proxy2_total_power = ((rollout_is_weights_scalar**2) * (advantages_scalar**2) * w_original).mean()

    else:
        # On-policy Proxy 2: E[A² × W]
        # Clamp W to avoid negative values (which would cause NaN in sqrt)
        w_values_clamped = torch.clamp(w_values, min=0.0)
        proxy2_total_power = (advantages_scalar**2 * w_values_clamped).mean()

    # ====== PROXY 3: Pure Noise - Variance of Mean Vector ======
    # Requires ||ḡ||² from actual batch gradient
    # Formula: (1/(N-1)) × (Proxy2 - Proxy1)
    proxy3_pure_noise = None
    if proxy1_signal_strength is not None:
        batch_size = advantages_scalar.shape[0]
        if batch_size > 1:
            proxy3_pure_noise = (1.0 / (batch_size - 1)) * (proxy2_total_power - proxy1_signal_strength)
            # Ensure non-negative (can be negative due to numerical errors)
            proxy3_pure_noise = max(
                0.0, proxy3_pure_noise.item() if torch.is_tensor(proxy3_pure_noise) else proxy3_pure_noise
            )

    # Decompose into components for analysis
    expected_a_squared = (advantages_scalar**2).mean()
    expected_w = w_values.mean()

    metrics.update(
        {
            # Proxy 1: Signal Strength ||ḡ||²
            "variance_proxy/proxy1_signal_strength": (
                proxy1_signal_strength if proxy1_signal_strength is not None else 0.0
            ),
            # Proxy 2: Total Power E[||ĝ_τ||²]
            "variance_proxy/proxy2_total_power": proxy2_total_power.detach().item(),
            # Proxy 3: Pure Noise - Variance of Mean Vector
            "variance_proxy/proxy3_pure_noise": proxy3_pure_noise if proxy3_pure_noise is not None else 0.0,
            # Component metrics for debugging
            "variance_proxy/expected_a_squared": expected_a_squared.detach().item(),
            "variance_proxy/expected_w": expected_w.detach().item(),
        }
    )

    return metrics


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)
    data_np = np.array(data, dtype=object)
    n_data = len(data_np)

    # generate bootstrap indices, shape: (n_bootstrap, subset_size)
    bootstrap_idxs = np.random.choice(n_data, size=(n_bootstrap, subset_size), replace=True)

    # pre-allocate result array, shape: (n_fns, n_bootstrap)
    n_fns = len(reduce_fns)
    metric_results = np.empty((n_fns, n_bootstrap), dtype=np.float64)

    # compute metric results for each bootstrap sample
    for fn_idx, reduce_fn in enumerate(reduce_fns):
        # bootstrap sample and compute metric
        for boot_idx in range(n_bootstrap):
            sample = data_np[bootstrap_idxs[boot_idx]]
            metric_results[fn_idx, boot_idx] = reduce_fn(sample)

    # compute mean and std for each metric function
    result = [
        (float(np.mean(metric_results[fn_idx])), float(np.std(metric_results[fn_idx]))) for fn_idx in range(n_fns)
    ]
    return result


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    np_mean = np.mean
    np_std = np.std
    reduce_fns_best_worst = [np.max, np.min]
    n_bootstrap = 1000

    # 2. cache ns list
    def gen_ns(n_resps: int) -> list[int]:
        if n_resps <= 1:
            return []
        ns = []
        n = 2
        while n < n_resps:
            ns.append(n)
            n *= 2
        ns.append(n_resps)
        return ns

    ns_cache = {}

    # 3. cache metric results
    data_src2uid2var2metric = {}

    # 4. flatten loop
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        # create uid dict
        uid_dict = data_src2uid2var2metric.setdefault(data_source, {})

        for uid, var2vals in uid2var2vals.items():
            pred_vals = var2vals.get("pred")
            has_pred = pred_vals is not None
            var_dict = uid_dict.setdefault(uid, {})

            for var_name, var_vals in var2vals.items():
                # skip empty or string values
                if not var_vals or isinstance(var_vals[0], str):
                    continue

                # compute mean and std
                n_resps = len(var_vals)
                metric = {f"mean@{n_resps}": float(np_mean(var_vals))}

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = float(np_std(var_vals))

                    # cache ns list
                    if n_resps not in ns_cache:
                        ns_cache[n_resps] = gen_ns(n_resps)
                    ns = ns_cache[n_resps]

                    # compute best/worst metrics
                    for n in ns:
                        # compute best/worst metrics
                        (bon_mean, bon_std), (won_mean, won_std) = bootstrap_metric(
                            data=var_vals,
                            subset_size=n,
                            reduce_fns=reduce_fns_best_worst,
                            n_bootstrap=n_bootstrap,
                            seed=seed,
                        )
                        metric[f"best@{n}/mean"] = bon_mean
                        metric[f"best@{n}/std"] = bon_std
                        metric[f"worst@{n}/mean"] = won_mean
                        metric[f"worst@{n}/std"] = won_std

                        # compute maj metrics
                        if has_pred:
                            # create vote_data
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, pred_vals, strict=True)
                            ]
                            # compute maj metrics
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                n_bootstrap=n_bootstrap,
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"] = maj_n_mean
                            metric[f"maj@{n}/std"] = maj_n_std

                var_dict[var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)
    return data_src2var2metric2val
