#!/usr/bin/env python3
"""
Run policy + critic over a dataset and plot average value curves
for correct vs wrong responses.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import random
import re
import subprocess
import sys
from types import MethodType
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from verl.utils.reward_score import default_compute_score
import datetime


ABLATION_MODES = ("none", "zero_weight_hh", "reset_each_token", "no_direct_carry")


def _has_hf_weights(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    for name in ["model.safetensors", "pytorch_model.bin"]:
        if (model_dir / name).exists():
            return True
    for pattern in ["model-*.safetensors", "pytorch_model-*.bin"]:
        if list(model_dir.glob(pattern)):
            return True
    return False


def _merge_fsdp_checkpoint(local_dir: Path, target_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(local_dir),
        "--target_dir",
        str(target_dir),
    ]
    print(f"[merge] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _get_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


def _prepare_tokenizer(model_dir: Path, trust_remote_code: bool, fix_mistral_regex: bool):
    kwargs = {"trust_remote_code": trust_remote_code}
    if fix_mistral_regex:
        kwargs["fix_mistral_regex"] = True
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir), **kwargs)
    except TypeError:
        # Older transformers may not accept fix_mistral_regex
        kwargs.pop("fix_mistral_regex", None)
        tok = AutoTokenizer.from_pretrained(str(model_dir), **kwargs)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _get_generation_config(model_dir: Path) -> GenerationConfig | None:
    try:
        return GenerationConfig.from_pretrained(str(model_dir))
    except Exception:
        return None


def _load_policy(model_dir: Path, dtype: torch.dtype, device: torch.device, trust_remote_code: bool):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model


def _load_critic(model_dir: Path, dtype: torch.dtype, device: torch.device, trust_remote_code: bool):
    from transformers import AutoConfig
    from verl.trainer.ppo.value_categorical import extract_value_head_spec

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    value_spec = None
    try:
        value_spec = extract_value_head_spec(config)
    except Exception:
        value_spec = None

    is_categorical = value_spec is not None and value_spec.is_categorical()
    if is_categorical:
        config.num_labels = int(value_spec.num_bins)
    else:
        config.num_labels = 1

    try:
        from verl.utils.model import load_valuehead_model

        model = load_valuehead_model(
            str(model_dir),
            torch_dtype=dtype,
            model_config=config,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        if is_categorical:
            raise RuntimeError(
                "Failed to load categorical critic with token-classification head. "
                "Please check checkpoint/config consistency."
            ) from exc
        try:
            from trl import AutoModelForCausalLMWithValueHead
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("trl is required to load value-head critic models") from exc

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            str(model_dir),
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    # Recurrent token-classification critics are built from config and can retain
    # float32 value-head weights after checkpoint load, so cast explicitly.
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, value_spec


def _find_recurrent_value_head(critic):
    from verl.utils.recurrent_value_head import RecurrentValueHead

    for attr_name in ("score", "classifier"):
        module = getattr(critic, attr_name, None)
        if isinstance(module, RecurrentValueHead):
            return attr_name, module
    return None, None


def _forward_reset_each_token(head, hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.dim() not in {2, 3}:
        raise ValueError(
            f"RecurrentValueHead expects hidden states of shape [T, H] or [B, T, H], got {hidden_states.shape}"
        )

    squeeze_batch = hidden_states.dim() == 2
    if squeeze_batch:
        hidden_states = hidden_states.unsqueeze(0)

    attention_mask = head._normalize_attention_mask(hidden_states)
    batch_size, seq_len, _ = hidden_states.shape
    zero_state = hidden_states.new_zeros(batch_size, head.hidden_size)
    logits = []

    for step in range(seq_len):
        step_hidden = hidden_states[:, step, :]
        next_state = head.gru(step_hidden, zero_state)

        if attention_mask is not None:
            valid_mask = attention_mask[:, step].to(dtype=torch.bool)
            step_state = torch.where(valid_mask.unsqueeze(-1), next_state, zero_state)
            step_logits = head.readout(step_state)
            step_logits = torch.where(valid_mask.unsqueeze(-1), step_logits, torch.zeros_like(step_logits))
        else:
            step_logits = head.readout(next_state)

        logits.append(step_logits)

    outputs = torch.stack(logits, dim=1)
    if squeeze_batch:
        outputs = outputs.squeeze(0)
    return outputs


def _gru_cell_without_direct_carry(gru, step_hidden: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
    gate_x = F.linear(step_hidden, gru.weight_ih, gru.bias_ih)
    gate_h = F.linear(prev_state, gru.weight_hh, gru.bias_hh)

    i_r, i_z, i_n = gate_x.chunk(3, dim=1)
    h_r, h_z, h_n = gate_h.chunk(3, dim=1)

    reset_gate = torch.sigmoid(i_r + h_r)
    update_gate = torch.sigmoid(i_z + h_z)
    new_gate = torch.tanh(i_n + reset_gate * h_n)

    # Standard GRU: new_gate + update_gate * (prev_state - new_gate)
    # = (1 - update_gate) * new_gate + update_gate * prev_state.
    # This ablation removes only the direct carry term update_gate * prev_state.
    return new_gate - update_gate * new_gate


def _forward_no_direct_carry(head, hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.dim() not in {2, 3}:
        raise ValueError(
            f"RecurrentValueHead expects hidden states of shape [T, H] or [B, T, H], got {hidden_states.shape}"
        )

    squeeze_batch = hidden_states.dim() == 2
    if squeeze_batch:
        hidden_states = hidden_states.unsqueeze(0)

    attention_mask = head._normalize_attention_mask(hidden_states)
    position_ids = None if attention_mask is not None else head._normalize_position_ids(hidden_states)

    batch_size, seq_len, _ = hidden_states.shape
    state = hidden_states.new_zeros(batch_size, head.hidden_size)
    zero_state = hidden_states.new_zeros(batch_size, head.hidden_size)
    logits = []
    prev_positions = None

    for step in range(seq_len):
        step_hidden = hidden_states[:, step, :]
        prev_state = state

        if position_ids is not None:
            cur_positions = position_ids[:, step]
            if prev_positions is None:
                reset_mask = torch.ones(batch_size, dtype=torch.bool, device=hidden_states.device)
            else:
                reset_mask = cur_positions <= prev_positions
            prev_state = torch.where(reset_mask.unsqueeze(-1), zero_state, prev_state)
            prev_positions = cur_positions

        next_state = _gru_cell_without_direct_carry(head.gru, step_hidden, prev_state)

        if attention_mask is not None:
            valid_mask = attention_mask[:, step].to(dtype=torch.bool)
            state = torch.where(valid_mask.unsqueeze(-1), next_state, prev_state)
            step_logits = head.readout(state)
            step_logits = torch.where(valid_mask.unsqueeze(-1), step_logits, torch.zeros_like(step_logits))
        else:
            state = next_state
            step_logits = head.readout(state)

        logits.append(step_logits)

    outputs = torch.stack(logits, dim=1)
    if squeeze_batch:
        outputs = outputs.squeeze(0)
    return outputs


@contextmanager
def _critic_ablation_context(critic, mode: str):
    if mode == "none":
        yield
        return

    _, recurrent_head = _find_recurrent_value_head(critic)
    if recurrent_head is None:
        raise ValueError(f"Ablation mode '{mode}' requires a recurrent GRU value head on the critic.")

    if mode == "zero_weight_hh":
        with torch.no_grad():
            original_weight_hh = recurrent_head.gru.weight_hh.detach().clone()
            recurrent_head.gru.weight_hh.zero_()
        try:
            yield
        finally:
            with torch.no_grad():
                recurrent_head.gru.weight_hh.copy_(original_weight_hh)
        return

    if mode == "reset_each_token":
        original_forward = recurrent_head.forward
        recurrent_head.forward = MethodType(_forward_reset_each_token, recurrent_head)
        try:
            yield
        finally:
            recurrent_head.forward = original_forward
        return

    if mode == "no_direct_carry":
        original_forward = recurrent_head.forward
        recurrent_head.forward = MethodType(_forward_no_direct_carry, recurrent_head)
        try:
            yield
        finally:
            recurrent_head.forward = original_forward
        return

    raise ValueError(f"Unknown ablation mode: {mode}")


def _stringify_chat_messages(messages) -> str:
    parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            parts.append(str(msg))
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _normalize_prompt(prompt, tokenizer) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        if "messages" in prompt:
            messages = prompt["messages"]
            if hasattr(tokenizer, "apply_chat_template"):
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return _stringify_chat_messages(messages)
        for key in ("prompt", "text", "content"):
            if key in prompt:
                return str(prompt[key])
        return json.dumps(prompt, ensure_ascii=True)
    if isinstance(prompt, list):
        if len(prompt) == 0:
            return ""
        if all(isinstance(x, dict) for x in prompt):
            if hasattr(tokenizer, "apply_chat_template"):
                return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            return _stringify_chat_messages(prompt)
        if all(isinstance(x, str) for x in prompt):
            return "\n".join(prompt)
        return "\n".join(str(x) for x in prompt)
    return str(prompt)


def _normalize_reference(ref) -> str | None:
    if ref is None:
        return None
    if isinstance(ref, str):
        return ref
    if isinstance(ref, list):
        if all(isinstance(x, str) for x in ref):
            return "\n".join(ref)
        return "\n".join(str(x) for x in ref)
    if isinstance(ref, dict):
        for key in ("response", "answer", "text", "content"):
            if key in ref:
                return str(ref[key])
        return json.dumps(ref, ensure_ascii=True)
    return str(ref)


def _normalize_text_for_match(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_filter_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return _normalize_text_for_match(json.dumps(value, ensure_ascii=True, sort_keys=True))
    return _normalize_text_for_match(str(value))


def _build_level_filtered_indices(ds, level_key: str, levels: list[str] | None) -> list[int]:
    if not levels:
        return list(range(len(ds)))

    if level_key not in ds.column_names:
        cols = ", ".join(ds.column_names)
        raise KeyError(f"Level key '{level_key}' not found in dataset columns: {cols}")

    target_levels = set()
    for level in levels:
        normalized = _normalize_filter_value(level)
        if normalized:
            target_levels.add(normalized)
    if not target_levels:
        raise ValueError("No valid level value was provided in --levels.")

    level_values = ds[level_key]
    selected = []
    for idx, value in enumerate(level_values):
        if _normalize_filter_value(value) in target_levels:
            selected.append(idx)
    return selected


def _is_correct(
    response: str,
    reference: str | None,
    mode: str,
    data_source: str | None = None,
    score_threshold: float = 1.0,
) -> bool:
    if reference is None:
        return False
    if mode == "exact":
        return _normalize_text_for_match(response) == _normalize_text_for_match(reference)
    if mode == "contains":
        return _normalize_text_for_match(reference) in _normalize_text_for_match(response)
    if mode == "regex":
        try:
            return re.search(reference, response, flags=re.DOTALL) is not None
        except re.error:
            return False
    if mode == "verl":
        if data_source is None:
            raise ValueError("correct_match=verl requires data_source in the dataset row")
        score = default_compute_score(data_source, response, reference)
        return score >= score_threshold
    raise ValueError(f"Unknown correct_match mode: {mode}")


def _extract_reference_from_row(row, response_key: str):
    if response_key in row and row[response_key] is not None:
        return row[response_key]
    reward_model = row.get("reward_model")
    if reward_model is None:
        return None
    if isinstance(reward_model, dict):
        if "ground_truth" in reward_model:
            return reward_model["ground_truth"]
        return None
    if isinstance(reward_model, str):
        try:
            parsed = json.loads(reward_model)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed.get("ground_truth")
    return None


def _dist_setup():
    """Return (enabled, rank, world_size, local_rank)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        init_kwargs = {"backend": "nccl"}
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(
            # timeout=datetime.timedelta(hours=2),
            **init_kwargs,
        )
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def _dist_barrier(enabled: bool):
    if enabled:
        dist.barrier()


def _dist_cleanup(enabled: bool):
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def _accumulate_bins(values_list, num_bins, sum_arr, cnt_arr):
    n = len(values_list)
    if n == 0:
        return
    for i, v in enumerate(values_list):
        b = int((i / n) * num_bins)
        if b >= num_bins:
            b = num_bins - 1
        sum_arr[b] += float(v)
        cnt_arr[b] += 1


def _finalize_curve(sum_arr, cnt_arr):
    curve = np.full_like(sum_arr, np.nan, dtype=np.float64)
    mask = cnt_arr > 0
    curve[mask] = sum_arr[mask] / cnt_arr[mask]
    return curve.tolist()


def _allreduce_numpy(sum_arr, cnt_arr, enabled: bool):
    if not enabled:
        return sum_arr, cnt_arr
    sum_t = torch.tensor(sum_arr, device="cuda", dtype=torch.float64)
    cnt_t = torch.tensor(cnt_arr, device="cuda", dtype=torch.int64)
    dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(cnt_t, op=dist.ReduceOp.SUM)
    return sum_t.cpu().numpy(), cnt_t.cpu().numpy()


def _allgather_float_list(local_values, enabled: bool):
    if not enabled:
        return [float(v) for v in local_values]

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, [float(v) for v in local_values])
    merged = []
    for part in gathered:
        if part:
            merged.extend(float(v) for v in part)
    return merged


def _allreduce_delta_totals(sum_delta: float, abs_sum_delta: float, token_count: int, enabled: bool):
    if not enabled:
        return float(sum_delta), float(abs_sum_delta), int(token_count)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    totals = torch.tensor(
        [float(sum_delta), float(abs_sum_delta), float(token_count)],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return float(totals[0].item()), float(totals[1].item()), int(round(float(totals[2].item())))


def _run_generation(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    gen_config: GenerationConfig | None,
    max_prompt_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_len,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    if gen_config is not None:
        gen_kwargs["generation_config"] = gen_config

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    response_ids = output_ids[:, prompt_len:]
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return output_ids, response_ids, response_text, prompt_len


def _compute_critic_values(
    critic,
    input_ids: torch.Tensor,
    prompt_len: int,
    response_len: int,
    value_spec=None,
    ablation_mode: str = "none",
) -> torch.Tensor:
    from verl.trainer.ppo.value_categorical import value_logits_to_scalar_expectation

    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        with _critic_ablation_context(critic, ablation_mode):
            outputs = critic(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    if hasattr(critic, "v_head"):
        values = outputs[2]
    else:
        values = outputs.logits

    is_categorical = value_spec is not None and value_spec.is_categorical()
    if is_categorical:
        values, _, _ = value_logits_to_scalar_expectation(values, value_spec)
    elif values.dim() == 3:
        values = values.squeeze(-1)

    if response_len <= 0:
        return values[:, :0]

    seq_len = values.shape[1]
    start = min(max(int(prompt_len), 0), seq_len)
    end = min(start + int(response_len), seq_len)
    response_values = values[:, start:end]

    # Fallback for unexpected sequence/value alignment differences:
    # keep only the post-prompt tail to avoid accidentally including prompt tokens.
    if response_values.shape[1] != response_len:
        response_values = values[:, start:]

    return response_values


def normalize_and_bin_sequences(sequences, num_bins=100):
    if not sequences:
        return [np.nan] * num_bins

    bins = [[] for _ in range(num_bins)]

    for sequence in sequences:
        if not sequence or len(sequence) == 0:
            continue
        for i, value in enumerate(sequence):
            if value is not None:
                bin_index = int((i / len(sequence)) * num_bins)
                bin_index = min(bin_index, num_bins - 1)
                bins[bin_index].append(value)

    averages = []
    for bin_values in bins:
        if bin_values:
            averages.append(float(np.mean(bin_values)))
        else:
            averages.append(float("nan"))

    return averages


def _save_curves(out_dir: Path, correct_values, wrong_values, num_bins: int):
    correct_curve = normalize_and_bin_sequences(correct_values, num_bins=num_bins)
    wrong_curve = normalize_and_bin_sequences(wrong_values, num_bins=num_bins)

    curves = {
        "num_bins": num_bins,
        "correct_curve": correct_curve,
        "wrong_curve": wrong_curve,
        "num_correct": len(correct_values),
        "num_wrong": len(wrong_values),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    curves_path = out_dir / "curves.json"
    with curves_path.open("w", encoding="utf-8") as f:
        json.dump(curves, f, ensure_ascii=True, indent=2)

    try:
        import matplotlib.pyplot as plt  # noqa: WPS433

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(correct_curve, label=f"correct (n={len(correct_values)})", linewidth=1.5)
        ax.plot(wrong_curve, label=f"wrong (n={len(wrong_values)})", linewidth=1.5)
        ax.set_title("Average Critic Values Over Normalized Response Position")
        ax.set_xlabel("Normalized token position (bins)")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "curves.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Failed to write curves plot: {exc}")

    print(f"[saved] {curves_path}")
    print(f"[saved] {out_dir / 'curves.png'}")


def _summarize_distribution(values):
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p25": None,
            "median": None,
            "p75": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def _build_histogram(values, bins):
    if len(values) == 0:
        return [0.0 for _ in range(len(bins) - 1)]
    hist, _ = np.histogram(values, bins=bins, density=True)
    return hist.astype(np.float64).tolist()


def _save_final_value_distribution(
    out_dir: Path,
    correct_final_values,
    wrong_final_values,
    num_bins: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    all_values = list(correct_final_values) + list(wrong_final_values)
    bins = None
    distribution = {
        "num_bins": int(num_bins),
        "correct_final_value_stats": _summarize_distribution(correct_final_values),
        "wrong_final_value_stats": _summarize_distribution(wrong_final_values),
        "note": "Final value means critic value at the last generated response token.",
    }

    if all_values:
        all_arr = np.asarray(all_values, dtype=np.float64)
        vmin = float(all_arr.min())
        vmax = float(all_arr.max())
        if np.isclose(vmin, vmax):
            span = max(abs(vmin) * 1e-3, 1e-6)
            bins = np.linspace(vmin - span, vmax + span, num_bins + 1)
        else:
            bins = np.linspace(vmin, vmax, num_bins + 1)
        distribution["bin_edges"] = bins.tolist()
        distribution["correct_density"] = _build_histogram(correct_final_values, bins)
        distribution["wrong_density"] = _build_histogram(wrong_final_values, bins)

    dist_path = out_dir / "final_value_distribution.json"
    with dist_path.open("w", encoding="utf-8") as df:
        json.dump(distribution, df, ensure_ascii=True, indent=2)

    try:
        import matplotlib.pyplot as plt  # noqa: WPS433

        fig = plt.figure(figsize=(9, 4))
        ax = fig.add_subplot(1, 1, 1)

        has_any = False
        if len(correct_final_values) > 0:
            ax.hist(
                correct_final_values,
                bins=bins if bins is not None else num_bins,
                density=True,
                alpha=0.45,
                label=f"correct (n={len(correct_final_values)})",
            )
            has_any = True
        if len(wrong_final_values) > 0:
            ax.hist(
                wrong_final_values,
                bins=bins if bins is not None else num_bins,
                density=True,
                alpha=0.45,
                label=f"wrong (n={len(wrong_final_values)})",
            )
            has_any = True
        if has_any:
            ax.legend()

        ax.set_title("Distribution of Final Response-Token Critic Value")
        ax.set_xlabel("Final response-token value")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / "final_value_distribution.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Failed to write final value distribution plot: {exc}")

    print(f"[saved] {dist_path}")
    print(f"[saved] {out_dir / 'final_value_distribution.png'}")


def _summarize_delta_totals(sum_delta: float, abs_sum_delta: float, token_count: int):
    if token_count <= 0:
        return {
            "token_count": 0,
            "mean_delta": None,
            "mean_abs_delta": None,
        }

    return {
        "token_count": int(token_count),
        "mean_delta": float(sum_delta / token_count),
        "mean_abs_delta": float(abs_sum_delta / token_count),
    }


def _save_ablation_report(
    out_dir: Path,
    ablation_mode: str,
    num_bins: int,
    correct_curve,
    wrong_curve,
    correct_curve_ablated,
    wrong_curve_ablated,
    correct_delta_curve,
    wrong_delta_curve,
    overall_delta_stats,
    correct_delta_stats,
    wrong_delta_stats,
    correct_final_deltas,
    wrong_final_deltas,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "ablation_mode": ablation_mode,
        "delta_definition": "ablated_value - original_value",
        "num_bins": int(num_bins),
        "correct_curve": correct_curve,
        "wrong_curve": wrong_curve,
        "correct_curve_ablated": correct_curve_ablated,
        "wrong_curve_ablated": wrong_curve_ablated,
        "correct_delta_curve": correct_delta_curve,
        "wrong_delta_curve": wrong_delta_curve,
        "overall_token_delta_stats": overall_delta_stats,
        "correct_token_delta_stats": correct_delta_stats,
        "wrong_token_delta_stats": wrong_delta_stats,
        "correct_final_value_delta_stats": _summarize_distribution(correct_final_deltas),
        "wrong_final_value_delta_stats": _summarize_distribution(wrong_final_deltas),
    }

    report_path = out_dir / "ablation_report.json"
    with report_path.open("w", encoding="utf-8") as rf:
        json.dump(payload, rf, ensure_ascii=True, indent=2)

    try:
        import matplotlib.pyplot as plt  # noqa: WPS433

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(correct_curve, label="correct", linewidth=1.8, color="#1f77b4")
        axes[0].plot(
            correct_curve_ablated,
            label=f"correct ({ablation_mode})",
            linewidth=1.8,
            linestyle="--",
            color="#1f77b4",
        )
        axes[0].plot(wrong_curve, label="wrong", linewidth=1.8, color="#ff7f0e")
        axes[0].plot(
            wrong_curve_ablated,
            label=f"wrong ({ablation_mode})",
            linewidth=1.8,
            linestyle="--",
            color="#ff7f0e",
        )
        axes[0].set_title(f"Critic Value Curves With Ablation: {ablation_mode}")
        axes[0].set_ylabel("Value")
        axes[0].grid(True, linestyle="--", alpha=0.4)
        axes[0].legend()

        axes[1].plot(correct_delta_curve, label="correct delta", linewidth=1.8, color="#2ca02c")
        axes[1].plot(wrong_delta_curve, label="wrong delta", linewidth=1.8, color="#d62728")
        axes[1].axhline(0.0, color="#444444", linewidth=1.0, linestyle=":")
        axes[1].set_title("Ablated - Original Value Curve")
        axes[1].set_xlabel("Normalized token position (bins)")
        axes[1].set_ylabel("Delta")
        axes[1].grid(True, linestyle="--", alpha=0.4)
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(out_dir / "ablation_curves.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Failed to write ablation plot: {exc}")

    print(f"[saved] {report_path}")
    print(f"[saved] {out_dir / 'ablation_curves.png'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug PPO critic values over an entire dataset.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/nfs/shuozhe/verl/train_log/max_response_length=2048/global_step_200",
        help="Path to the PPO checkpoint directory (contains actor/critic).",
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to training parquet file.")
    parser.add_argument(
        "--level_key",
        type=str,
        default="level",
        help="Dataset column key used for level filtering.",
    )
    parser.add_argument(
        "--levels",
        type=str,
        nargs="+",
        default=None,
        help='Optional level values to keep. Example: --levels "Level 5" or --levels "Level 1" "Level 5".',
    )
    parser.add_argument("--prompt_key", type=str, default="prompt", help="Prompt column key in dataset.")
    parser.add_argument(
        "--response_key",
        type=str,
        default="ground_truth",
        help="Optional response column key for reference output.",
    )
    parser.add_argument("--max_prompt_len", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu. Default: auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--fix_mistral_regex", action="store_true")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/nfs/shuozhe/verl/outputs/critic_debug_all",
        help="Output directory for plots and dumps.",
    )
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--num_bins", type=int, default=100)
    parser.add_argument(
        "--dist_bins",
        type=int,
        default=80,
        help="Histogram bins for final-value distribution plot.",
    )
    parser.add_argument(
        "--correct_match",
        type=str,
        default="verl",
        choices=["exact", "contains", "regex", "verl"],
    )
    parser.add_argument(
        "--data_source_key",
        type=str,
        default="data_source",
        help="Dataset column key for verl reward_score routing when correct_match=verl.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=1.0,
        help="Score threshold for marking correct when correct_match=verl.",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=ABLATION_MODES,
        help=(
            "Optional recurrent-memory ablation for the critic. "
            "'reset_each_token' is the cleaner behavioral ablation; "
            "'zero_weight_hh' only removes the hidden-to-hidden affine term; "
            "'no_direct_carry' removes the direct z_t * h_{t-1} carry path while keeping recurrent gates."
        ),
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    dist_enabled, rank, world_size, local_rank = _dist_setup()

    ckpt_dir = Path(args.checkpoint_dir)
    actor_ckpt = ckpt_dir / "actor"
    critic_ckpt = ckpt_dir / "critic"

    merged_root = ckpt_dir / "merged_hf"
    actor_hf = merged_root / "actor"
    critic_hf = merged_root / "critic"

    if not args.skip_merge:
        if (not dist_enabled) or rank == 0:
            if not _has_hf_weights(actor_hf):
                _merge_fsdp_checkpoint(actor_ckpt, actor_hf)
            if not _has_hf_weights(critic_hf):
                _merge_fsdp_checkpoint(critic_ckpt, critic_hf)
        _dist_barrier(dist_enabled)

    if not _has_hf_weights(actor_hf):
        raise FileNotFoundError(f"Actor HF weights not found in {actor_hf}")
    if not _has_hf_weights(critic_hf):
        raise FileNotFoundError(f"Critic HF weights not found in {critic_hf}")

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}" if dist_enabled else "cuda")
        else:
            device = torch.device("cpu")
    dtype = _get_dtype(args.dtype)

    tokenizer = _prepare_tokenizer(actor_hf, trust_remote_code=args.trust_remote_code, fix_mistral_regex=args.fix_mistral_regex)
    actor = _load_policy(actor_hf, dtype=dtype, device=device, trust_remote_code=args.trust_remote_code)
    critic, critic_value_spec = _load_critic(
        critic_hf,
        dtype=dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
    )
    if args.ablation != "none":
        _, recurrent_head = _find_recurrent_value_head(critic)
        if recurrent_head is None:
            raise ValueError(
                f"--ablation {args.ablation} was requested, but the loaded critic does not expose a recurrent GRU head."
            )
    gen_config = _get_generation_config(actor_hf)

    ds = load_dataset("parquet", data_files=args.dataset_path, split="train")
    dataset_total = len(ds)
    filtered_indices = _build_level_filtered_indices(ds, args.level_key, args.levels)
    filtered_total = len(filtered_indices)

    if (not dist_enabled) or rank == 0:
        if args.levels:
            levels_str = ", ".join(args.levels)
            print(
                f"[filter] matched {filtered_total}/{dataset_total} rows "
                f"where {args.level_key} in {{{levels_str}}}"
            )
        else:
            print(f"[filter] no level filter; using all {dataset_total} rows")

    start = max(0, args.start_index)
    end = args.end_index if args.end_index is not None else filtered_total
    end = min(end, filtered_total)

    if args.max_examples is not None:
        end = min(end, start + args.max_examples)

    selected_indices = filtered_indices[start:end]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / (f"responses_rank{rank}.jsonl" if dist_enabled else "responses.jsonl")

    correct_sum = np.zeros(args.num_bins, dtype=np.float64)
    correct_cnt = np.zeros(args.num_bins, dtype=np.int64)
    wrong_sum = np.zeros(args.num_bins, dtype=np.float64)
    wrong_cnt = np.zeros(args.num_bins, dtype=np.int64)
    num_correct_local = 0
    num_wrong_local = 0
    correct_final_values_local = []
    wrong_final_values_local = []
    ablation_enabled = args.ablation != "none"
    if ablation_enabled:
        correct_sum_ablated = np.zeros(args.num_bins, dtype=np.float64)
        correct_cnt_ablated = np.zeros(args.num_bins, dtype=np.int64)
        wrong_sum_ablated = np.zeros(args.num_bins, dtype=np.float64)
        wrong_cnt_ablated = np.zeros(args.num_bins, dtype=np.int64)
        correct_delta_sum = np.zeros(args.num_bins, dtype=np.float64)
        correct_delta_cnt = np.zeros(args.num_bins, dtype=np.int64)
        wrong_delta_sum = np.zeros(args.num_bins, dtype=np.float64)
        wrong_delta_cnt = np.zeros(args.num_bins, dtype=np.int64)
        correct_final_values_ablated_local = []
        wrong_final_values_ablated_local = []
        correct_final_deltas_local = []
        wrong_final_deltas_local = []
        correct_delta_total_local = 0.0
        correct_abs_delta_total_local = 0.0
        correct_delta_tokens_local = 0
        wrong_delta_total_local = 0.0
        wrong_abs_delta_total_local = 0.0
        wrong_delta_tokens_local = 0

    with jsonl_path.open("w", encoding="utf-8") as f:
        indices = selected_indices[rank::world_size] if dist_enabled else selected_indices
        total_local = len(indices)
        processed = 0
        for idx in indices:
            row = ds[int(idx)]
            prompt_raw = row.get(args.prompt_key)
            ref_raw = _extract_reference_from_row(row, args.response_key)
            data_source = row.get(args.data_source_key) if args.data_source_key in row else None

            prompt = _normalize_prompt(prompt_raw, tokenizer)
            reference = _normalize_reference(ref_raw)

            output_ids, response_ids, response_text, prompt_len = _run_generation(
                model=actor,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                gen_config=gen_config,
                max_prompt_len=args.max_prompt_len,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
            )

            response_len = response_ids.shape[1]
            values = _compute_critic_values(
                critic,
                output_ids,
                prompt_len=prompt_len,
                response_len=response_len,
                value_spec=critic_value_spec,
            )
            values_list = values[0].detach().cpu().tolist()
            response_ids_list = response_ids[0].detach().cpu().tolist()
            ablated_values_list = None
            delta_values_list = None
            if ablation_enabled:
                values_ablated = _compute_critic_values(
                    critic,
                    output_ids,
                    prompt_len=prompt_len,
                    response_len=response_len,
                    value_spec=critic_value_spec,
                    ablation_mode=args.ablation,
                )
                ablated_values_list = values_ablated[0].detach().cpu().tolist()

            aligned_len = len(response_ids_list)
            if ablated_values_list is None:
                aligned_len = min(aligned_len, len(values_list))
            else:
                aligned_len = min(aligned_len, len(values_list), len(ablated_values_list))

            if len(values_list) != aligned_len or len(response_ids_list) != aligned_len:
                values_list = values_list[:aligned_len]
                response_ids_list = response_ids_list[:aligned_len]
            if ablated_values_list is not None and len(ablated_values_list) != aligned_len:
                ablated_values_list = ablated_values_list[:aligned_len]
            if ablated_values_list is not None:
                delta_values_arr = np.asarray(ablated_values_list, dtype=np.float64) - np.asarray(values_list, dtype=np.float64)
                delta_values_list = delta_values_arr.tolist()

            final_response_value = float(values_list[-1]) if values_list else None
            final_response_value_ablated = float(ablated_values_list[-1]) if ablated_values_list else None
            final_response_value_delta = float(delta_values_list[-1]) if delta_values_list else None

            correct = _is_correct(
                response_text,
                reference,
                mode=args.correct_match,
                data_source=data_source,
                score_threshold=args.score_threshold,
            )
            if correct:
                num_correct_local += 1
                _accumulate_bins(values_list, args.num_bins, correct_sum, correct_cnt)
                if final_response_value is not None:
                    correct_final_values_local.append(final_response_value)
                if ablation_enabled:
                    _accumulate_bins(ablated_values_list, args.num_bins, correct_sum_ablated, correct_cnt_ablated)
                    _accumulate_bins(delta_values_list, args.num_bins, correct_delta_sum, correct_delta_cnt)
                    if final_response_value_ablated is not None:
                        correct_final_values_ablated_local.append(final_response_value_ablated)
                    if final_response_value_delta is not None:
                        correct_final_deltas_local.append(final_response_value_delta)
                    correct_delta_total_local += float(delta_values_arr.sum())
                    correct_abs_delta_total_local += float(np.abs(delta_values_arr).sum())
                    correct_delta_tokens_local += int(delta_values_arr.size)
            else:
                num_wrong_local += 1
                _accumulate_bins(values_list, args.num_bins, wrong_sum, wrong_cnt)
                if final_response_value is not None:
                    wrong_final_values_local.append(final_response_value)
                if ablation_enabled:
                    _accumulate_bins(ablated_values_list, args.num_bins, wrong_sum_ablated, wrong_cnt_ablated)
                    _accumulate_bins(delta_values_list, args.num_bins, wrong_delta_sum, wrong_delta_cnt)
                    if final_response_value_ablated is not None:
                        wrong_final_values_ablated_local.append(final_response_value_ablated)
                    if final_response_value_delta is not None:
                        wrong_final_deltas_local.append(final_response_value_delta)
                    wrong_delta_total_local += float(delta_values_arr.sum())
                    wrong_abs_delta_total_local += float(np.abs(delta_values_arr).sum())
                    wrong_delta_tokens_local += int(delta_values_arr.size)

            record = {
                "index": idx,
                "prompt": prompt,
                "response": response_text,
                "reference": None if reference is None else str(reference),
                "correct": bool(correct),
                "response_ids": response_ids_list,
                "values": values_list,
                "final_response_value": final_response_value,
                "rank": rank,
                "data_source": data_source,
            }
            if ablation_enabled:
                record.update(
                    {
                        "ablation_mode": args.ablation,
                        "values_ablated": ablated_values_list,
                        "value_deltas": delta_values_list,
                        "final_response_value_ablated": final_response_value_ablated,
                        "final_response_value_delta": final_response_value_delta,
                    }
                )
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

            processed += 1
            if processed % args.save_every == 0:
                if total_local:
                    pct = 100.0 * processed / total_local
                    print(f"[rank {rank}] processed {processed}/{total_local} ({pct:.1f}%)")
                else:
                    print(f"[rank {rank}] processed {processed}")

    correct_sum, correct_cnt = _allreduce_numpy(correct_sum, correct_cnt, dist_enabled)
    wrong_sum, wrong_cnt = _allreduce_numpy(wrong_sum, wrong_cnt, dist_enabled)
    correct_final_values = _allgather_float_list(correct_final_values_local, dist_enabled)
    wrong_final_values = _allgather_float_list(wrong_final_values_local, dist_enabled)
    if ablation_enabled:
        correct_sum_ablated, correct_cnt_ablated = _allreduce_numpy(correct_sum_ablated, correct_cnt_ablated, dist_enabled)
        wrong_sum_ablated, wrong_cnt_ablated = _allreduce_numpy(wrong_sum_ablated, wrong_cnt_ablated, dist_enabled)
        correct_delta_sum, correct_delta_cnt = _allreduce_numpy(correct_delta_sum, correct_delta_cnt, dist_enabled)
        wrong_delta_sum, wrong_delta_cnt = _allreduce_numpy(wrong_delta_sum, wrong_delta_cnt, dist_enabled)
        correct_final_values_ablated = _allgather_float_list(correct_final_values_ablated_local, dist_enabled)
        wrong_final_values_ablated = _allgather_float_list(wrong_final_values_ablated_local, dist_enabled)
        correct_final_deltas = _allgather_float_list(correct_final_deltas_local, dist_enabled)
        wrong_final_deltas = _allgather_float_list(wrong_final_deltas_local, dist_enabled)
        (
            correct_delta_total,
            correct_abs_delta_total,
            correct_delta_tokens,
        ) = _allreduce_delta_totals(
            correct_delta_total_local,
            correct_abs_delta_total_local,
            correct_delta_tokens_local,
            dist_enabled,
        )
        (
            wrong_delta_total,
            wrong_abs_delta_total,
            wrong_delta_tokens,
        ) = _allreduce_delta_totals(
            wrong_delta_total_local,
            wrong_abs_delta_total_local,
            wrong_delta_tokens_local,
            dist_enabled,
        )

    if dist_enabled:
        tot = torch.tensor([num_correct_local, num_wrong_local], device="cuda", dtype=torch.long)
        dist.all_reduce(tot, op=dist.ReduceOp.SUM)
        num_correct = int(tot[0].item())
        num_wrong = int(tot[1].item())
    else:
        num_correct, num_wrong = num_correct_local, num_wrong_local

    if (not dist_enabled) or rank == 0:
        correct_curve = _finalize_curve(correct_sum, correct_cnt)
        wrong_curve = _finalize_curve(wrong_sum, wrong_cnt)

        curves = {
            "num_bins": args.num_bins,
            "correct_curve": correct_curve,
            "wrong_curve": wrong_curve,
            "num_correct": num_correct,
            "num_wrong": num_wrong,
        }

        curves_path = out_dir / "curves.json"
        with curves_path.open("w", encoding="utf-8") as cf:
            json.dump(curves, cf, ensure_ascii=True, indent=2)

        try:
            import matplotlib.pyplot as plt  # noqa: WPS433

            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(correct_curve, label=f"correct (n={num_correct})", linewidth=1.5)
            ax.plot(wrong_curve, label=f"wrong (n={num_wrong})", linewidth=1.5)
            ax.set_title("Average Critic Values Over Normalized Response Position")
            ax.set_xlabel("Normalized token position (bins)")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "curves.png", dpi=150)
            plt.close(fig)
        except Exception as exc:
            print(f"[warn] Failed to write curves plot: {exc}")

        if ablation_enabled:
            correct_curve_ablated = _finalize_curve(correct_sum_ablated, correct_cnt_ablated)
            wrong_curve_ablated = _finalize_curve(wrong_sum_ablated, wrong_cnt_ablated)
            correct_delta_curve = _finalize_curve(correct_delta_sum, correct_delta_cnt)
            wrong_delta_curve = _finalize_curve(wrong_delta_sum, wrong_delta_cnt)

            correct_delta_stats = _summarize_delta_totals(
                correct_delta_total,
                correct_abs_delta_total,
                correct_delta_tokens,
            )
            wrong_delta_stats = _summarize_delta_totals(
                wrong_delta_total,
                wrong_abs_delta_total,
                wrong_delta_tokens,
            )
            overall_delta_stats = _summarize_delta_totals(
                correct_delta_total + wrong_delta_total,
                correct_abs_delta_total + wrong_abs_delta_total,
                correct_delta_tokens + wrong_delta_tokens,
            )

            _save_ablation_report(
                out_dir=out_dir,
                ablation_mode=args.ablation,
                num_bins=args.num_bins,
                correct_curve=correct_curve,
                wrong_curve=wrong_curve,
                correct_curve_ablated=correct_curve_ablated,
                wrong_curve_ablated=wrong_curve_ablated,
                correct_delta_curve=correct_delta_curve,
                wrong_delta_curve=wrong_delta_curve,
                overall_delta_stats=overall_delta_stats,
                correct_delta_stats=correct_delta_stats,
                wrong_delta_stats=wrong_delta_stats,
                correct_final_deltas=correct_final_deltas,
                wrong_final_deltas=wrong_final_deltas,
            )

        _save_final_value_distribution(
            out_dir=out_dir,
            correct_final_values=correct_final_values,
            wrong_final_values=wrong_final_values,
            num_bins=args.dist_bins,
        )

        meta = {
            "checkpoint_dir": str(ckpt_dir),
            "dataset_path": args.dataset_path,
            "level_key": args.level_key,
            "levels": args.levels,
            "prompt_key": args.prompt_key,
            "response_key": args.response_key,
            "data_source_key": args.data_source_key,
            "start_index": start,
            "end_index": end,
            "dataset_total_rows": dataset_total,
            "level_filtered_rows": filtered_total,
            "selected_rows": len(selected_indices),
            "max_examples": args.max_examples,
            "correct_match": args.correct_match,
            "score_threshold": args.score_threshold,
            "num_correct": num_correct,
            "num_wrong": num_wrong,
            "num_correct_with_final_value": len(correct_final_values),
            "num_wrong_with_final_value": len(wrong_final_values),
            "world_size": world_size,
            "ablation_mode": args.ablation,
        }
        if ablation_enabled:
            meta.update(
                {
                    "num_correct_with_final_value_ablated": len(correct_final_values_ablated),
                    "num_wrong_with_final_value_ablated": len(wrong_final_values_ablated),
                    "num_correct_with_final_value_delta": len(correct_final_deltas),
                    "num_wrong_with_final_value_delta": len(wrong_final_deltas),
                }
            )
        with (out_dir / "metadata.json").open("w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=True, indent=2)

        print(f"[saved] {curves_path}")
        print(f"[saved] {out_dir / 'curves.png'}")
        print(f"[saved] {out_dir / 'metadata.json'}")
        if ablation_enabled:
            print(f"[saved] {out_dir / 'ablation_report.json'}")
            print(f"[saved] {out_dir / 'ablation_curves.png'}")
        print(f"[saved] per-rank jsonl: {out_dir}/responses_rank*.jsonl")

    _dist_cleanup(dist_enabled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
