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

import pytest
import torch

from verl.trainer.ppo.core_algos import compute_categorical_value_loss, compute_value_loss
from verl.trainer.ppo.value_categorical import (
    ValueHeadSpec,
    clamp_scaled_targets,
    project_scalar_targets_to_probs,
    scale_scalar_values,
    value_logits_to_scalar_expectation,
)


def _make_spec(
    target_type: str = "two_hot",
    *,
    target_scaling: str = "identity",
    target_scale_min: float = 0.0,
    target_scale_max: float = 1.0,
    target_out_of_range: str = "error",
    sigma: float | None = None,
) -> ValueHeadSpec:
    spec = ValueHeadSpec(
        head_type="categorical",
        num_bins=11,
        value_min=0.0,
        value_max=1.0,
        target_type=target_type,
        hl_gauss_sigma=sigma,
        target_scaling=target_scaling,
        target_scale_min=target_scale_min,
        target_scale_max=target_scale_max,
        target_out_of_range=target_out_of_range,
    )
    spec.validate()
    return spec


def test_two_hot_projection_is_normalized_and_matches_scalar_target():
    spec = _make_spec(target_type="two_hot")
    targets = torch.tensor([0.0, 0.03, 0.2, 0.37, 0.6, 0.99, 1.0], dtype=torch.float32)
    support = spec.support(dtype=torch.float32)
    edges = spec.bin_edges(dtype=torch.float32)
    probs = project_scalar_targets_to_probs(targets, spec, support, edges)

    assert probs.shape == (targets.shape[0], spec.num_bins)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(targets), atol=1e-6)
    reconstructed = (probs * support).sum(dim=-1)
    assert torch.allclose(reconstructed, targets, atol=1e-6)


def test_hl_gauss_projection_is_normalized():
    spec = _make_spec(target_type="hl_gauss", sigma=0.05)
    targets = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
    support = spec.support(dtype=torch.float32)
    edges = spec.bin_edges(dtype=torch.float32)
    probs = project_scalar_targets_to_probs(targets, spec, support, edges)

    assert probs.shape == (targets.shape[0], spec.num_bins)
    assert torch.all(probs >= 0.0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(targets), atol=1e-6)


def test_expected_value_from_probs():
    spec = _make_spec()
    logits = torch.zeros(4, spec.num_bins, dtype=torch.float32)
    values, probs, scaled_values = value_logits_to_scalar_expectation(logits, spec)

    assert probs.shape == logits.shape
    assert values.shape == (4,)
    assert scaled_values.shape == (4,)
    assert torch.allclose(values, torch.full_like(values, 0.5), atol=1e-6)


def test_categorical_value_loss_shapes_and_metrics():
    torch.manual_seed(0)
    spec = _make_spec(target_type="hl_gauss", sigma=0.05)
    logits = torch.randn(2, 3, spec.num_bins, dtype=torch.float32)
    returns = torch.rand(2, 3, dtype=torch.float32)
    values = torch.rand(2, 3, dtype=torch.float32)
    response_mask = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.bool)

    vf_loss, vf_clipfrac, vpreds, metrics = compute_categorical_value_loss(
        value_logits=logits,
        returns=returns,
        values=values,
        response_mask=response_mask,
        cliprange_value=0.2,
        value_spec=spec,
    )

    assert vf_loss.ndim == 0
    assert vf_clipfrac.ndim == 0
    assert vpreds.shape == returns.shape
    assert "critic/value_entropy" in metrics
    assert "critic/target_entropy" in metrics
    assert "critic/value_bin_idx_mean" in metrics
    assert "critic/value_target_out_of_range_fraction" in metrics


def test_identity_scaling_out_of_range_is_not_silent():
    spec = _make_spec(target_out_of_range="error")
    targets = torch.tensor([0.2, 1.1], dtype=torch.float32)
    with pytest.raises(ValueError, match="outside"):
        clamp_scaled_targets(targets, spec)


def test_affine_scaling_maps_raw_targets_to_support():
    spec = _make_spec(
        target_scaling="affine",
        target_scale_min=-2.0,
        target_scale_max=2.0,
        target_out_of_range="error",
    )
    raw_targets = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float32)
    scaled_targets = scale_scalar_values(raw_targets, spec)
    assert torch.allclose(scaled_targets, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32), atol=1e-6)


def test_scalar_value_loss_path_still_works():
    vpreds = torch.tensor([[0.1, 0.4, 0.8]], dtype=torch.float32)
    values = torch.tensor([[0.1, 0.3, 0.7]], dtype=torch.float32)
    returns = torch.tensor([[0.0, 0.6, 1.0]], dtype=torch.float32)
    response_mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=0.2,
    )

    assert vf_loss.ndim == 0
    assert vf_clipfrac.ndim == 0
