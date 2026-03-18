#!/usr/bin/env python3
"""Rescore saved generated responses with a different critic ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from debug_critic_values_all import (
    ABLATION_MODES,
    _accumulate_bins,
    _compute_critic_values,
    _finalize_curve,
    _get_dtype,
    _load_critic,
    _prepare_tokenizer,
    _save_ablation_report,
    _save_final_value_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescore saved responses with a different critic ablation.")
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Checkpoint directory containing merged_hf/{actor,critic} or raw actor/critic shards.",
    )
    parser.add_argument(
        "--source_run_dir",
        required=True,
        help="Directory containing responses_rank*.jsonl from a previous debug_critic_values_all run.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for the rescored ablation run.",
    )
    parser.add_argument(
        "--ablation",
        required=True,
        choices=[mode for mode in ABLATION_MODES if mode != "none"],
        help="Critic ablation mode to apply during rescoring.",
    )
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default=None, help="cuda or cpu. Default: auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--fix_mistral_regex", action="store_true")
    parser.add_argument("--max_prompt_len", type=int, default=2048)
    return parser.parse_args()


def _has_hf_weights(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (model_dir / name).exists():
            return True
    for pattern in ("model-*.safetensors", "pytorch_model-*.bin"):
        if list(model_dir.glob(pattern)):
            return True
    return False


def _resolve_hf_dirs(checkpoint_dir: Path) -> tuple[Path, Path]:
    merged_root = checkpoint_dir / "merged_hf"
    actor_hf = merged_root / "actor"
    critic_hf = merged_root / "critic"

    if _has_hf_weights(actor_hf) and _has_hf_weights(critic_hf):
        return actor_hf, critic_hf

    raise FileNotFoundError(
        f"Merged HF weights were not found under {merged_root}. "
        "Run debug_critic_values_all.py once without --skip_merge for this checkpoint first."
    )


def _all_response_paths(source_run_dir: Path) -> list[Path]:
    paths = sorted(source_run_dir.glob("responses_rank*.jsonl"))
    if paths:
        return paths
    single = source_run_dir / "responses.jsonl"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"No responses_rank*.jsonl or responses.jsonl found in {source_run_dir}")


def _save_curves_json(out_dir: Path, correct_curve, wrong_curve, num_correct: int, num_wrong: int, num_bins: int) -> None:
    payload = {
        "num_bins": int(num_bins),
        "correct_curve": correct_curve,
        "wrong_curve": wrong_curve,
        "num_correct": int(num_correct),
        "num_wrong": int(num_wrong),
    }
    with (out_dir / "curves.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _save_curves_plot(out_dir: Path, correct_curve, wrong_curve, num_correct: int, num_wrong: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - runtime dependency
        print(f"[warn] Failed to write curves plot: {exc}")
        return

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


def main() -> int:
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    source_run_dir = Path(args.source_run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    actor_hf, critic_hf = _resolve_hf_dirs(checkpoint_dir)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _get_dtype(args.dtype)

    tokenizer = _prepare_tokenizer(
        actor_hf,
        trust_remote_code=args.trust_remote_code,
        fix_mistral_regex=args.fix_mistral_regex,
    )
    critic, critic_value_spec = _load_critic(
        critic_hf,
        dtype=dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
    )

    source_metadata_path = source_run_dir / "metadata.json"
    source_metadata = None
    if source_metadata_path.exists():
        with source_metadata_path.open("r", encoding="utf-8") as f:
            source_metadata = json.load(f)

    num_bins = int(source_metadata["num_bins"]) if source_metadata and "num_bins" in source_metadata else 100
    dist_bins = int(source_metadata["dist_bins"]) if source_metadata and "dist_bins" in source_metadata else 80

    correct_sum = np.zeros(num_bins, dtype=np.float64)
    correct_cnt = np.zeros(num_bins, dtype=np.int64)
    wrong_sum = np.zeros(num_bins, dtype=np.float64)
    wrong_cnt = np.zeros(num_bins, dtype=np.int64)
    correct_sum_ablated = np.zeros(num_bins, dtype=np.float64)
    correct_cnt_ablated = np.zeros(num_bins, dtype=np.int64)
    wrong_sum_ablated = np.zeros(num_bins, dtype=np.float64)
    wrong_cnt_ablated = np.zeros(num_bins, dtype=np.int64)
    correct_delta_sum = np.zeros(num_bins, dtype=np.float64)
    correct_delta_cnt = np.zeros(num_bins, dtype=np.int64)
    wrong_delta_sum = np.zeros(num_bins, dtype=np.float64)
    wrong_delta_cnt = np.zeros(num_bins, dtype=np.int64)

    num_correct = 0
    num_wrong = 0
    correct_final_values = []
    wrong_final_values = []
    correct_final_values_ablated = []
    wrong_final_values_ablated = []
    correct_final_deltas = []
    wrong_final_deltas = []
    correct_delta_total = 0.0
    correct_abs_delta_total = 0.0
    correct_delta_tokens = 0
    wrong_delta_total = 0.0
    wrong_abs_delta_total = 0.0
    wrong_delta_tokens = 0

    output_path = out_dir / "responses_rank0.jsonl"

    with output_path.open("w", encoding="utf-8") as out_f:
        for src_path in _all_response_paths(source_run_dir):
            with src_path.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    record = json.loads(line)

                    prompt = record["prompt"]
                    response_ids_list = list(record.get("response_ids", []))
                    baseline_values = list(record.get("values", []))

                    prompt_inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=args.max_prompt_len,
                    )
                    prompt_ids = prompt_inputs["input_ids"].to(device)
                    response_ids = torch.tensor(response_ids_list, device=device, dtype=prompt_ids.dtype).unsqueeze(0)
                    full_input_ids = torch.cat([prompt_ids, response_ids], dim=1)

                    ablated_values = _compute_critic_values(
                        critic,
                        full_input_ids,
                        prompt_len=prompt_ids.shape[1],
                        response_len=response_ids.shape[1],
                        value_spec=critic_value_spec,
                        ablation_mode=args.ablation,
                    )
                    ablated_values_list = ablated_values[0].detach().cpu().tolist()

                    aligned_len = min(len(response_ids_list), len(baseline_values), len(ablated_values_list))
                    response_ids_list = response_ids_list[:aligned_len]
                    baseline_values = baseline_values[:aligned_len]
                    ablated_values_list = ablated_values_list[:aligned_len]

                    delta_values_arr = np.asarray(ablated_values_list, dtype=np.float64) - np.asarray(
                        baseline_values, dtype=np.float64
                    )
                    delta_values_list = delta_values_arr.tolist()

                    final_response_value = float(baseline_values[-1]) if baseline_values else None
                    final_response_value_ablated = float(ablated_values_list[-1]) if ablated_values_list else None
                    final_response_value_delta = float(delta_values_list[-1]) if delta_values_list else None

                    correct = bool(record.get("correct"))
                    if correct:
                        num_correct += 1
                        _accumulate_bins(baseline_values, num_bins, correct_sum, correct_cnt)
                        _accumulate_bins(ablated_values_list, num_bins, correct_sum_ablated, correct_cnt_ablated)
                        _accumulate_bins(delta_values_list, num_bins, correct_delta_sum, correct_delta_cnt)
                        if final_response_value is not None:
                            correct_final_values.append(final_response_value)
                        if final_response_value_ablated is not None:
                            correct_final_values_ablated.append(final_response_value_ablated)
                        if final_response_value_delta is not None:
                            correct_final_deltas.append(final_response_value_delta)
                        correct_delta_total += float(delta_values_arr.sum())
                        correct_abs_delta_total += float(np.abs(delta_values_arr).sum())
                        correct_delta_tokens += int(delta_values_arr.size)
                    else:
                        num_wrong += 1
                        _accumulate_bins(baseline_values, num_bins, wrong_sum, wrong_cnt)
                        _accumulate_bins(ablated_values_list, num_bins, wrong_sum_ablated, wrong_cnt_ablated)
                        _accumulate_bins(delta_values_list, num_bins, wrong_delta_sum, wrong_delta_cnt)
                        if final_response_value is not None:
                            wrong_final_values.append(final_response_value)
                        if final_response_value_ablated is not None:
                            wrong_final_values_ablated.append(final_response_value_ablated)
                        if final_response_value_delta is not None:
                            wrong_final_deltas.append(final_response_value_delta)
                        wrong_delta_total += float(delta_values_arr.sum())
                        wrong_abs_delta_total += float(np.abs(delta_values_arr).sum())
                        wrong_delta_tokens += int(delta_values_arr.size)

                    updated = dict(record)
                    updated["ablation_mode"] = args.ablation
                    updated["values_ablated"] = ablated_values_list
                    updated["value_deltas"] = delta_values_list
                    updated["final_response_value_ablated"] = final_response_value_ablated
                    updated["final_response_value_delta"] = final_response_value_delta
                    out_f.write(json.dumps(updated, ensure_ascii=True) + "\n")

    correct_curve = _finalize_curve(correct_sum, correct_cnt)
    wrong_curve = _finalize_curve(wrong_sum, wrong_cnt)
    correct_curve_ablated = _finalize_curve(correct_sum_ablated, correct_cnt_ablated)
    wrong_curve_ablated = _finalize_curve(wrong_sum_ablated, wrong_cnt_ablated)
    correct_delta_curve = _finalize_curve(correct_delta_sum, correct_delta_cnt)
    wrong_delta_curve = _finalize_curve(wrong_delta_sum, wrong_delta_cnt)

    _save_curves_json(out_dir, correct_curve, wrong_curve, num_correct, num_wrong, num_bins)
    _save_curves_plot(out_dir, correct_curve, wrong_curve, num_correct, num_wrong)
    _save_ablation_report(
        out_dir=out_dir,
        ablation_mode=args.ablation,
        num_bins=num_bins,
        correct_curve=correct_curve,
        wrong_curve=wrong_curve,
        correct_curve_ablated=correct_curve_ablated,
        wrong_curve_ablated=wrong_curve_ablated,
        correct_delta_curve=correct_delta_curve,
        wrong_delta_curve=wrong_delta_curve,
        overall_delta_stats={
            "token_count": int(correct_delta_tokens + wrong_delta_tokens),
            "mean_delta": float((correct_delta_total + wrong_delta_total) / max(correct_delta_tokens + wrong_delta_tokens, 1)),
            "mean_abs_delta": float(
                (correct_abs_delta_total + wrong_abs_delta_total) / max(correct_delta_tokens + wrong_delta_tokens, 1)
            ),
        },
        correct_delta_stats={
            "token_count": int(correct_delta_tokens),
            "mean_delta": float(correct_delta_total / max(correct_delta_tokens, 1)),
            "mean_abs_delta": float(correct_abs_delta_total / max(correct_delta_tokens, 1)),
        },
        wrong_delta_stats={
            "token_count": int(wrong_delta_tokens),
            "mean_delta": float(wrong_delta_total / max(wrong_delta_tokens, 1)),
            "mean_abs_delta": float(wrong_abs_delta_total / max(wrong_delta_tokens, 1)),
        },
        correct_final_deltas=correct_final_deltas,
        wrong_final_deltas=wrong_final_deltas,
    )
    _save_final_value_distribution(out_dir, correct_final_values, wrong_final_values, dist_bins)

    metadata = dict(source_metadata) if source_metadata is not None else {}
    metadata.update(
        {
            "checkpoint_dir": str(checkpoint_dir),
            "source_run_dir": str(source_run_dir),
            "rescored_from_saved_responses": True,
            "ablation_mode": args.ablation,
            "num_correct": num_correct,
            "num_wrong": num_wrong,
            "num_correct_with_final_value": len(correct_final_values),
            "num_wrong_with_final_value": len(wrong_final_values),
            "num_correct_with_final_value_ablated": len(correct_final_values_ablated),
            "num_wrong_with_final_value_ablated": len(wrong_final_values_ablated),
            "num_correct_with_final_value_delta": len(correct_final_deltas),
            "num_wrong_with_final_value_delta": len(wrong_final_deltas),
            "num_bins": num_bins,
            "dist_bins": dist_bins,
        }
    )
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)

    print(f"[saved] {out_dir / 'responses_rank0.jsonl'}")
    print(f"[saved] {out_dir / 'curves.json'}")
    print(f"[saved] {out_dir / 'curves.png'}")
    print(f"[saved] {out_dir / 'ablation_report.json'}")
    print(f"[saved] {out_dir / 'ablation_curves.png'}")
    print(f"[saved] {out_dir / 'final_value_distribution.json'}")
    print(f"[saved] {out_dir / 'final_value_distribution.png'}")
    print(f"[saved] {out_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
