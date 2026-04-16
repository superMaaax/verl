#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


MODEL_SHARD_RE = re.compile(r"model_world_size_(?P<world_size>\d+)_rank_(?P<rank>\d+)\.pt$")
GRU_HH_SUFFIX = ".gru.weight_hh"
GRU_IH_SUFFIX = ".gru.weight_ih"
GRU_BIAS_IH_SUFFIX = ".gru.bias_ih"
GRU_BIAS_HH_SUFFIX = ".gru.bias_hh"
READOUT_W_SUFFIX = ".readout.weight"
READOUT_B_SUFFIX = ".readout.bias"
GATE_NAMES = ("reset", "update", "new")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a recurrent GRU critic/value head saved in a sharded FSDP checkpoint. "
            "You can pass either a global_step directory or its critic subdirectory."
        )
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_gru/global_step_450",
        help="Path to a checkpoint directory such as global_step_450 or global_step_450/critic.",
    )
    parser.add_argument(
        "--component",
        choices=("critic", "actor"),
        default="critic",
        help="Checkpoint component to inspect when the input path is a global_step directory.",
    )
    parser.add_argument(
        "--threshold",
        dest="thresholds",
        action="append",
        type=float,
        default=None,
        help=(
            "Near-zero threshold for |weight|. Repeat this flag to report multiple thresholds. "
            "Default: 1e-6, 1e-5, 1e-4."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the text report.",
    )
    return parser.parse_args()


def resolve_component_dir(checkpoint: str | Path, component: str) -> Path:
    checkpoint = Path(checkpoint).expanduser().resolve()
    if find_model_shards(checkpoint, allow_missing=True):
        return checkpoint

    component_dir = checkpoint / component
    if find_model_shards(component_dir, allow_missing=True):
        return component_dir

    raise FileNotFoundError(
        f"Could not find model shard files under {checkpoint} or {component_dir}. "
        "Expected files named like model_world_size_<N>_rank_<R>.pt."
    )


def find_model_shards(component_dir: Path, *, allow_missing: bool = False) -> list[Path]:
    shards: list[tuple[int, Path]] = []
    if component_dir.exists():
        for path in component_dir.iterdir():
            match = MODEL_SHARD_RE.fullmatch(path.name)
            if match:
                shards.append((int(match.group("rank")), path))

    shards.sort(key=lambda item: item[0])
    ordered = [path for _, path in shards]
    if ordered or allow_missing:
        return ordered

    raise FileNotFoundError(f"No model shard files found in {component_dir}")


def load_state_dicts(component_dir: Path) -> list[dict[str, Any]]:
    shard_paths = find_model_shards(component_dir)
    return [torch.load(path, map_location="cpu", weights_only=False) for path in shard_paths]


def get_gru_prefixes(state_dict: dict[str, Any]) -> list[str]:
    prefixes = []
    for key in state_dict:
        if key.endswith(GRU_HH_SUFFIX):
            prefixes.append(key[: -len(GRU_HH_SUFFIX)])
    return sorted(prefixes)


def reconstruct_param(state_dicts: list[dict[str, Any]], key: str) -> torch.Tensor:
    values = [state_dict[key] for state_dict in state_dicts]
    first = values[0]

    if hasattr(first, "to_local"):
        local_tensors = [value.to_local().cpu() for value in values]
        placements = getattr(first, "placements", ())
        shard_dim = None
        if placements:
            shard_dim = getattr(placements[0], "dim", None)
        if shard_dim is not None:
            return torch.cat(local_tensors, dim=shard_dim)
        return local_tensors[0]

    if torch.is_tensor(first):
        return first.cpu()

    raise TypeError(f"Unsupported parameter type for {key}: {type(first)}")


def threshold_key(threshold: float) -> str:
    return f"fraction_abs_le_{threshold:.0e}"


def summarize_tensor(tensor: torch.Tensor, thresholds: list[float]) -> dict[str, Any]:
    tensor = tensor.detach().float()
    total = tensor.numel()
    summary: dict[str, Any] = {
        "shape": list(tensor.shape),
        "mean_abs": float(tensor.abs().mean().item()),
        "max_abs": float(tensor.abs().max().item()),
        "l2_norm": float(tensor.norm().item()),
        "fraction_exact_zero": float((tensor == 0).float().mean().item()),
    }
    for threshold in thresholds:
        summary[threshold_key(threshold)] = float((tensor.abs() <= threshold).float().mean().item())
    if total > 0:
        summary["numel"] = int(total)
    return summary


def split_gru_gates(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    if weight.ndim != 2 or weight.shape[0] % 3 != 0:
        raise ValueError(f"Expected a GRU weight matrix with shape [3H, *], got {tuple(weight.shape)}")
    hidden_size = weight.shape[0] // 3
    return {name: chunk for name, chunk in zip(GATE_NAMES, weight.split(hidden_size, dim=0))}


def inspect_recurrent_value_head(
    checkpoint: str | Path,
    *,
    component: str = "critic",
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = [1e-6, 1e-5, 1e-4]

    component_dir = resolve_component_dir(checkpoint, component)
    state_dicts = load_state_dicts(component_dir)
    prefixes = get_gru_prefixes(state_dicts[0])
    if not prefixes:
        raise ValueError(f"No GRU value-head parameters found in {component_dir}")

    report: dict[str, Any] = {
        "component_dir": str(component_dir),
        "num_shards": len(state_dicts),
        "thresholds": thresholds,
        "modules": {},
    }

    for prefix in prefixes:
        weight_ih = reconstruct_param(state_dicts, prefix + GRU_IH_SUFFIX)
        weight_hh = reconstruct_param(state_dicts, prefix + GRU_HH_SUFFIX)
        bias_ih = reconstruct_param(state_dicts, prefix + GRU_BIAS_IH_SUFFIX)
        bias_hh = reconstruct_param(state_dicts, prefix + GRU_BIAS_HH_SUFFIX)
        readout_weight = reconstruct_param(state_dicts, prefix + READOUT_W_SUFFIX)
        readout_bias = reconstruct_param(state_dicts, prefix + READOUT_B_SUFFIX)

        weight_ih_gates = split_gru_gates(weight_ih)
        weight_hh_gates = split_gru_gates(weight_hh)

        gate_report: dict[str, Any] = {}
        for gate_name in GATE_NAMES:
            ih_stats = summarize_tensor(weight_ih_gates[gate_name], thresholds)
            hh_stats = summarize_tensor(weight_hh_gates[gate_name], thresholds)
            gate_report[gate_name] = {
                "weight_ih": ih_stats,
                "weight_hh": hh_stats,
                "recurrent_to_input_l2_ratio": float(
                    hh_stats["l2_norm"] / max(ih_stats["l2_norm"], torch.finfo(torch.float32).eps)
                ),
            }

        module_report = {
            "hidden_size": int(weight_hh.shape[1]),
            "weight_ih": summarize_tensor(weight_ih, thresholds),
            "weight_hh": summarize_tensor(weight_hh, thresholds),
            "bias_ih": summarize_tensor(bias_ih, thresholds),
            "bias_hh": summarize_tensor(bias_hh, thresholds),
            "readout_weight": summarize_tensor(readout_weight, thresholds),
            "readout_bias": summarize_tensor(readout_bias, thresholds),
            "recurrent_to_input_l2_ratio": float(
                weight_hh.float().norm().item() / max(weight_ih.float().norm().item(), torch.finfo(torch.float32).eps)
            ),
            "gate_stats": gate_report,
        }
        report["modules"][prefix] = module_report

    return report


def format_summary_line(name: str, stats: dict[str, Any], thresholds: list[float]) -> str:
    pieces = [
        f"{name}: shape={tuple(stats['shape'])}",
        f"l2={stats['l2_norm']:.6g}",
        f"mean|w|={stats['mean_abs']:.6g}",
        f"max|w|={stats['max_abs']:.6g}",
        f"exact_zero={stats['fraction_exact_zero']:.2%}",
    ]
    for threshold in thresholds:
        pieces.append(f"|w|<={threshold:.0e}:{stats[threshold_key(threshold)]:.2%}")
    return "  " + " ".join(pieces)


def render_text_report(report: dict[str, Any]) -> str:
    lines = [
        f"Checkpoint: {report['component_dir']}",
        f"Model shard files: {report['num_shards']}",
        "Interpretation note: smaller weight_hh means weaker direct dependence on h_{t-1}, "
        "but it is not a definitive proof that memory is unused.",
        "",
    ]

    thresholds = report["thresholds"]
    for module_name, module_report in report["modules"].items():
        lines.append(f"Module: {module_name} (GRU hidden_size={module_report['hidden_size']})")
        lines.append(format_summary_line("weight_ih", module_report["weight_ih"], thresholds))
        lines.append(format_summary_line("weight_hh", module_report["weight_hh"], thresholds))
        lines.append(f"  recurrent/input l2 ratio: {module_report['recurrent_to_input_l2_ratio']:.6f}")
        lines.append(format_summary_line("bias_ih", module_report["bias_ih"], thresholds))
        lines.append(format_summary_line("bias_hh", module_report["bias_hh"], thresholds))
        lines.append(format_summary_line("readout.weight", module_report["readout_weight"], thresholds))
        lines.append(format_summary_line("readout.bias", module_report["readout_bias"], thresholds))
        lines.append("  Gate stats (PyTorch GRU order: reset, update, new):")
        for gate_name in GATE_NAMES:
            gate_report = module_report["gate_stats"][gate_name]
            lines.append(
                "    "
                f"{gate_name}: hh/ih l2 ratio={gate_report['recurrent_to_input_l2_ratio']:.6f} "
                f"ih_mean|w|={gate_report['weight_ih']['mean_abs']:.6g} "
                f"hh_mean|w|={gate_report['weight_hh']['mean_abs']:.6g} "
                f"hh_|w|<={thresholds[-1]:.0e}:{gate_report['weight_hh'][threshold_key(thresholds[-1])]:.2%}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    report = inspect_recurrent_value_head(
        args.checkpoint,
        component=args.component,
        thresholds=args.thresholds or [1e-6, 1e-5, 1e-4],
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text_report(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
