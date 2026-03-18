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
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, GPT2Config

from verl.trainer.main_ppo import _validate_recurrent_critic_runtime_support
from verl.trainer.ppo.value_categorical import (
    ValueHeadArchitectureSpec,
    ValueHeadSpec,
    apply_value_head_architecture_spec_to_hf_config,
    apply_value_head_spec_to_hf_config,
    extract_value_head_architecture_spec,
)
from verl.utils.model import load_valuehead_model
from verl.utils.recurrent_value_head import GatedCarryValueHead, RecurrentValueHead, patch_recurrent_value_head
from verl.workers.config import FSDPCriticConfig, HFModelConfig


STATEFUL_HEAD_CASES = (
    ("gru", RecurrentValueHead),
    ("gated_carry", GatedCarryValueHead),
)


def _make_tiny_gpt2_config() -> GPT2Config:
    return GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
        num_labels=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _make_recurrent_spec(architecture: str = "gru", hidden_size: int = 8) -> ValueHeadArchitectureSpec:
    spec = ValueHeadArchitectureSpec(architecture=architecture, state_size=hidden_size)
    spec.validate()
    return spec


def _get_value_head_module(model):
    for attr_name in ("score", "classifier"):
        module = getattr(model, attr_name, None)
        if module is not None:
            return module
    return None


@pytest.mark.parametrize(("architecture", "head_cls"), STATEFUL_HEAD_CASES)
def test_stateful_value_head_resets_on_position_id_boundaries(architecture, head_cls):
    torch.manual_seed(0)
    head = head_cls(input_size=4, hidden_size=6, output_size=2)

    seq_a = torch.randn(3, 4)
    seq_b = torch.randn(2, 4)
    packed = torch.cat([seq_a, seq_b], dim=0)

    expected = torch.cat([head(seq_a), head(seq_b)], dim=0)

    head.set_runtime_context(position_ids=torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long))
    actual = head(packed.unsqueeze(0)).squeeze(0)
    head.clear_runtime_context()

    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize(("architecture", "head_cls"), STATEFUL_HEAD_CASES)
def test_stateful_value_head_resets_on_position_id_boundaries_with_attention_mask(architecture, head_cls):
    torch.manual_seed(0)
    head = head_cls(input_size=4, hidden_size=6, output_size=1)

    seq_a = torch.randn(3, 4)
    seq_b = torch.randn(2, 4)
    packed = torch.cat([seq_a, seq_b], dim=0).unsqueeze(0)

    expected = torch.cat([head(seq_a), head(seq_b)], dim=0)

    head.set_runtime_context(
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long),
    )
    actual = head(packed).squeeze(0)
    head.clear_runtime_context()

    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize(("architecture", "head_cls"), STATEFUL_HEAD_CASES)
def test_stateful_value_head_respects_attention_mask(architecture, head_cls):
    torch.manual_seed(0)
    head = head_cls(input_size=4, hidden_size=5, output_size=1)

    seq_a = torch.randn(4, 4)
    seq_b = torch.randn(2, 4)

    padded = torch.zeros(2, 4, 4)
    padded[0] = seq_a
    padded[1, :2] = seq_b
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.long)

    head.set_runtime_context(attention_mask=attention_mask)
    actual = head(padded)
    head.clear_runtime_context()

    expected_a = head(seq_a)
    expected_b = head(seq_b)

    assert torch.allclose(actual[0], expected_a, atol=1e-6)
    assert torch.allclose(actual[1, :2], expected_b, atol=1e-6)
    assert torch.count_nonzero(actual[1, 2:]) == 0


@pytest.mark.parametrize(("architecture", "head_cls"), STATEFUL_HEAD_CASES)
def test_stateful_value_head_rejects_3d_position_ids(architecture, head_cls):
    head = head_cls(input_size=4, hidden_size=5, output_size=1)
    head.set_runtime_context(position_ids=torch.zeros(3, 1, 4, dtype=torch.long))

    with pytest.raises(NotImplementedError, match="3D position_ids"):
        head(torch.randn(1, 4, 4))

    head.clear_runtime_context()


def test_gated_carry_value_head_exposes_debug_metrics():
    torch.manual_seed(0)
    head = GatedCarryValueHead(input_size=4, hidden_size=6, output_size=1)
    hidden_states = torch.randn(2, 5, 4)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.long)

    head.set_runtime_context(attention_mask=attention_mask)
    outputs = head(hidden_states)
    head.clear_runtime_context()

    assert outputs.shape == (2, 5, 1)
    metrics = head.get_debug_metrics()
    assert set(metrics) == {
        "critic/value_head/state_norm",
        "critic/value_head/gate_mean",
        "critic/value_head/gate_std",
        "critic/value_head/candidate_abs_mean",
    }
    assert 0.0 <= metrics["critic/value_head/gate_mean"] <= 1.0
    assert metrics["critic/value_head/gate_std"] >= 0.0

    head.clear_debug_metrics()
    assert head.get_debug_metrics() == {}


@pytest.mark.parametrize(("architecture", "head_cls"), STATEFUL_HEAD_CASES)
def test_load_valuehead_model_round_trip_for_stateful_hf_checkpoint(tmp_path, architecture, head_cls):
    torch.manual_seed(0)
    config = _make_tiny_gpt2_config()
    value_spec = ValueHeadSpec()
    recurrent_spec = _make_recurrent_spec(architecture=architecture, hidden_size=7)
    apply_value_head_spec_to_hf_config(config, value_spec)
    apply_value_head_architecture_spec_to_hf_config(config, recurrent_spec)

    model = AutoModelForTokenClassification.from_config(config)
    patch_recurrent_value_head(model, recurrent_spec)

    for _, param in model.named_parameters():
        torch.nn.init.uniform_(param, a=-0.2, b=0.2)

    model.save_pretrained(tmp_path)

    reloaded_config = AutoConfig.from_pretrained(tmp_path)
    loaded = load_valuehead_model(tmp_path, torch.float32, reloaded_config, trust_remote_code=False)

    assert isinstance(_get_value_head_module(loaded), head_cls)
    for key, expected_tensor in model.state_dict().items():
        assert key in loaded.state_dict()
        assert torch.allclose(loaded.state_dict()[key], expected_tensor)


@pytest.mark.parametrize(("architecture", "head_cls"), STATEFUL_HEAD_CASES)
def test_load_valuehead_model_can_bootstrap_stateful_head_from_base_lm_checkpoint(tmp_path, architecture, head_cls):
    torch.manual_seed(0)
    base_config = _make_tiny_gpt2_config()
    lm = AutoModelForCausalLM.from_config(base_config)
    lm.save_pretrained(tmp_path)

    critic_config = AutoConfig.from_pretrained(tmp_path)
    apply_value_head_spec_to_hf_config(critic_config, ValueHeadSpec())
    apply_value_head_architecture_spec_to_hf_config(
        critic_config,
        _make_recurrent_spec(architecture=architecture, hidden_size=9),
    )

    critic_model = load_valuehead_model(
        tmp_path,
        torch.float32,
        critic_config,
        trust_remote_code=False,
        value_head_init_mean=0.0,
        value_head_init_std=0.0,
    )

    critic_head = _get_value_head_module(critic_model)
    assert isinstance(critic_head, head_cls)
    for param in critic_head.value_head_parameters():
        if param is not None:
            assert torch.count_nonzero(param) == 0

    outputs = critic_model(input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
    assert outputs.logits.shape == (1, 4, 1)


def test_extract_value_head_architecture_spec_prefers_state_size_alias():
    spec = extract_value_head_architecture_spec(
        {
            "value_head_architecture": "gated_carry",
            "value_head_state_size": 13,
            "value_head_gru_hidden_size": 7,
        }
    )

    assert spec.architecture == "gated_carry"
    assert spec.recurrent_state_size == 13


@pytest.mark.parametrize("architecture", ("gru", "gated_carry"))
def test_fsdp_stateful_critic_rejects_ulysses_sequence_parallel(tmp_path, architecture):
    AutoModelForCausalLM.from_config(_make_tiny_gpt2_config()).save_pretrained(tmp_path)

    with pytest.raises(ValueError, match="ulysses_sequence_parallel_size > 1"):
        FSDPCriticConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=4,
            ppo_micro_batch_size_per_gpu=2,
            forward_micro_batch_size_per_gpu=2,
            use_dynamic_bsz=False,
            rollout_n=1,
            ulysses_sequence_parallel_size=2,
            value_head_architecture=architecture,
            model_config=HFModelConfig(
                path=str(tmp_path),
                tokenizer_path=str(tmp_path),
                load_tokenizer=False,
            ),
        )


@pytest.mark.parametrize("architecture", ("gru", "gated_carry"))
def test_stateful_critic_rejects_new_engine_worker_path(architecture):
    config = OmegaConf.create(
        {
            "trainer": {"use_legacy_worker_impl": "disable"},
            "critic": {"enable": True, "value_head_architecture": architecture},
        }
    )

    with pytest.raises(ValueError, match="legacy FSDP critic worker"):
        _validate_recurrent_critic_runtime_support(config)
