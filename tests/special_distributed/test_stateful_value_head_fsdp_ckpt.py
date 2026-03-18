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

import json
import os
import shutil
import tempfile

import torch
import torch.distributed
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoModelForTokenClassification, GPT2Config

from verl.trainer.ppo.value_categorical import (
    ValueHeadArchitectureSpec,
    ValueHeadSpec,
    apply_value_head_architecture_spec_to_hf_config,
    apply_value_head_spec_to_hf_config,
)
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import MixedPrecisionPolicy, apply_fsdp2
from verl.utils.recurrent_value_head import GatedCarryValueHead, RecurrentValueHead, patch_recurrent_value_head


def _get_value_head_module(model):
    for attr_name in ("score", "classifier"):
        module = getattr(model, attr_name, None)
        if module is not None:
            return module
    raise AttributeError("Expected token-classification model to expose `score` or `classifier`.")


def _build_config(architecture: str, state_size: int) -> GPT2Config:
    config = GPT2Config(
        vocab_size=97,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        classifier_dropout=0.0,
        num_labels=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    apply_value_head_spec_to_hf_config(config, ValueHeadSpec())
    apply_value_head_architecture_spec_to_hf_config(
        config,
        ValueHeadArchitectureSpec(architecture=architecture, state_size=state_size),
    )
    return config


def _make_inputs(batch_size: int, seq_len: int, vocab_size: int, *, offset: int) -> tuple[torch.Tensor, ...]:
    device = get_device_name()
    input_ids = (torch.arange(batch_size * seq_len, device=device).reshape(batch_size, seq_len) + offset) % vocab_size
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
    if seq_len >= 8:
        position_ids[:, seq_len // 2 :] -= seq_len // 2

    return input_ids, attention_mask, position_ids


def test_stateful_value_head_fsdp_checkpoint(architecture: str = "gated_carry", strategy: str = "fsdp") -> None:
    assert get_torch_device().device_count() >= 2, "need at least 2 gpus for test"

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    _, _, world_size = initialize_global_process_group()
    device_mesh = init_device_mesh(get_device_name(), mesh_shape=(world_size,), mesh_dim_names=("dp",))

    config = _build_config(architecture=architecture, state_size=12)

    with torch.device(get_device_name()):
        model = AutoModelForTokenClassification.from_config(config, torch_dtype=torch.bfloat16)
        patch_recurrent_value_head(model, ValueHeadArchitectureSpec(architecture=architecture, state_size=12))
        expected_head_cls = GatedCarryValueHead if architecture == "gated_carry" else RecurrentValueHead
        assert isinstance(_get_value_head_module(model), expected_head_cls)
        model = model.to(device=get_device_name(), dtype=torch.bfloat16)

    if strategy == "fsdp":
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        model = FSDP(
            model,
            use_orig_params=False,
            device_id=get_torch_device().current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=device_mesh,
        )
    else:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )
        apply_fsdp2(model, {"mesh": device_mesh, "mp_policy": mp_policy}, {})

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    checkpoint_manager = FSDPCheckpointManager(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)

    batch_size = 4
    seq_len = 8
    input_ids1, attention_mask1, position_ids1 = _make_inputs(batch_size, seq_len, config.vocab_size, offset=0)
    input_ids2, attention_mask2, position_ids2 = _make_inputs(batch_size, seq_len, config.vocab_size, offset=17)

    outputs1 = model(input_ids=input_ids1, attention_mask=attention_mask1, position_ids=position_ids1)
    loss1 = outputs1.logits.mean()
    loss1.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    temp_dir = tempfile.mkdtemp() if torch.distributed.get_rank() == 0 else None
    temp_dir_holder = [temp_dir]
    torch.distributed.broadcast_object_list(temp_dir_holder, src=0)
    temp_dir = temp_dir_holder[0]
    checkpoint_path = os.path.join(temp_dir, "checkpoint")
    checkpoint_manager.save_checkpoint(local_path=checkpoint_path, hdfs_path=None, global_step=0)
    saved_state_dict = model.state_dict()

    if torch.distributed.get_rank() == 0:
        with open(os.path.join(checkpoint_path, "huggingface", "config.json"), encoding="utf-8") as f:
            saved_config = json.load(f)
        assert saved_config["value_head_architecture"] == architecture
        assert saved_config["value_head_state_size"] == 12
        assert saved_config["value_head_gru_hidden_size"] == 12

    torch.distributed.barrier()

    outputs2 = model(input_ids=input_ids2, attention_mask=attention_mask2, position_ids=position_ids2)
    loss2 = outputs2.logits.mean()
    loss2.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    with torch.no_grad():
        logits_before_load = model(input_ids=input_ids2, attention_mask=attention_mask2, position_ids=position_ids2).logits

    checkpoint_manager.load_checkpoint(checkpoint_path)
    loaded_state_dict = model.state_dict()
    for key in loaded_state_dict:
        assert key in saved_state_dict, f"Key {key} not found in saved state dict"
        torch.testing.assert_close(loaded_state_dict[key], saved_state_dict[key], atol=0.0, rtol=0.0)

    outputs3 = model(input_ids=input_ids2, attention_mask=attention_mask2, position_ids=position_ids2)
    loss3 = outputs3.logits.mean()
    loss3.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    with torch.no_grad():
        logits_after_load = model(input_ids=input_ids2, attention_mask=attention_mask2, position_ids=position_ids2).logits

    torch.testing.assert_close(logits_before_load, logits_after_load, atol=0.0, rtol=0.0)

    if torch.distributed.get_rank() == 0:
        shutil.rmtree(temp_dir)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    test_stateful_value_head_fsdp_checkpoint(
        architecture=os.environ.get("VALUE_HEAD_ARCHITECTURE", "gated_carry"),
        strategy=os.environ.get("STRATEGY", "fsdp"),
    )
