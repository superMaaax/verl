#!/usr/bin/env python3
"""Compare multiple GRU ablation runs and write advisor-friendly summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RUN_SPECS = [
    (
        "reset_each_token",
        "/data/shuozhe/verl/critic_debug/05b_vh_init_e5_gru_step_450_reset_each_token_l1",
    ),
    (
        "zero_weight_hh",
        "/data/shuozhe/verl/critic_debug/05b_vh_init_e5_gru_step_450_zero_weight_hh_l1",
    ),
    (
        "no_direct_carry",
        "/data/shuozhe/verl/critic_debug/05b_vh_init_e5_gru_step_450_no_direct_carry_l1",
    ),
]

DEFAULT_OUT_DIR = "/data/shuozhe/verl/critic_debug/05b_vh_init_e5_gru_step_450_ablation_comparison_l1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple GRU ablation reports from debug_critic_values_all.py.")
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help=(
            "Run specification in the form label=/path/to/run_dir_or_ablation_report.json. "
            "If omitted, use the three known step-450 Level-1 runs."
        ),
    )
    parser.add_argument(
        "--reference",
        default="reset_each_token",
        help="Reference ablation label used when computing effect ratios.",
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help="Directory where comparison artifacts will be written.",
    )
    return parser.parse_args()


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --run spec {spec!r}; expected label=/path/to/run_dir_or_ablation_report.json")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip()).expanduser().resolve()
    if not label:
        raise ValueError(f"Invalid --run spec {spec!r}; label is empty")
    return label, path


def resolve_report_path(path: Path) -> Path:
    if path.is_file():
        return path
    report_path = path / "ablation_report.json"
    if report_path.exists():
        return report_path
    raise FileNotFoundError(f"Could not find ablation_report.json under {path}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def nanmean(values: list[float]) -> float:
    return float(np.nanmean(np.asarray(values, dtype=np.float64)))


def compute_gap_mean(correct_curve: list[float], wrong_curve: list[float]) -> float:
    correct = np.asarray(correct_curve, dtype=np.float64)
    wrong = np.asarray(wrong_curve, dtype=np.float64)
    return float(np.nanmean(correct - wrong))


def compute_sign_stats(run_dir: Path) -> dict[str, Any]:
    totals = {
        "all": {"count": 0, "negative": 0, "positive": 0, "zero": 0},
        "correct": {"count": 0, "negative": 0, "positive": 0, "zero": 0},
        "wrong": {"count": 0, "negative": 0, "positive": 0, "zero": 0},
    }

    response_paths = sorted(run_dir.glob("responses_rank*.jsonl"))
    if not response_paths:
        single_path = run_dir / "responses.jsonl"
        if single_path.exists():
            response_paths = [single_path]

    for path in response_paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                delta = record.get("final_response_value_delta")
                if delta is None:
                    continue

                is_correct = bool(record.get("correct"))
                bucket_names = ["all", "correct" if is_correct else "wrong"]
                for bucket_name in bucket_names:
                    bucket = totals[bucket_name]
                    bucket["count"] += 1
                    if delta > 0:
                        bucket["positive"] += 1
                    elif delta < 0:
                        bucket["negative"] += 1
                    else:
                        bucket["zero"] += 1

    for bucket in totals.values():
        count = bucket["count"]
        if count == 0:
            bucket["negative_fraction"] = None
            bucket["positive_fraction"] = None
            bucket["zero_fraction"] = None
        else:
            bucket["negative_fraction"] = bucket["negative"] / count
            bucket["positive_fraction"] = bucket["positive"] / count
            bucket["zero_fraction"] = bucket["zero"] / count
    return totals


def load_run(label: str, path: Path) -> dict[str, Any]:
    report_path = resolve_report_path(path)
    run_dir = report_path.parent
    report = load_json(report_path)
    metadata = maybe_load_json(run_dir / "metadata.json")
    sign_stats = compute_sign_stats(run_dir)

    summary = {
        "label": label,
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "ablation_mode": report.get("ablation_mode", label),
        "metadata": metadata,
        "sign_stats": sign_stats,
        "correct_curve": report["correct_curve"],
        "wrong_curve": report["wrong_curve"],
        "correct_curve_ablated": report["correct_curve_ablated"],
        "wrong_curve_ablated": report["wrong_curve_ablated"],
        "correct_delta_curve": report["correct_delta_curve"],
        "wrong_delta_curve": report["wrong_delta_curve"],
        "overall_token_delta_stats": report["overall_token_delta_stats"],
        "correct_token_delta_stats": report["correct_token_delta_stats"],
        "wrong_token_delta_stats": report["wrong_token_delta_stats"],
        "correct_final_value_delta_stats": report["correct_final_value_delta_stats"],
        "wrong_final_value_delta_stats": report["wrong_final_value_delta_stats"],
    }

    summary["correct_curve_mean_before"] = nanmean(summary["correct_curve"])
    summary["wrong_curve_mean_before"] = nanmean(summary["wrong_curve"])
    summary["correct_curve_mean_after"] = nanmean(summary["correct_curve_ablated"])
    summary["wrong_curve_mean_after"] = nanmean(summary["wrong_curve_ablated"])
    summary["baseline_gap_mean"] = compute_gap_mean(summary["correct_curve"], summary["wrong_curve"])
    summary["ablated_gap_mean"] = compute_gap_mean(summary["correct_curve_ablated"], summary["wrong_curve_ablated"])
    summary["gap_drop"] = summary["baseline_gap_mean"] - summary["ablated_gap_mean"]
    summary["gap_retained_fraction"] = (
        summary["ablated_gap_mean"] / summary["baseline_gap_mean"] if summary["baseline_gap_mean"] != 0 else None
    )
    return summary


def validate_shared_baseline(runs: OrderedDict[str, dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    run_items = list(runs.items())
    if not run_items:
        return warnings
    ref_label, ref_run = run_items[0]
    ref_correct = np.asarray(ref_run["correct_curve"], dtype=np.float64)
    ref_wrong = np.asarray(ref_run["wrong_curve"], dtype=np.float64)

    for label, run in run_items[1:]:
        cur_correct = np.asarray(run["correct_curve"], dtype=np.float64)
        cur_wrong = np.asarray(run["wrong_curve"], dtype=np.float64)
        if not np.allclose(ref_correct, cur_correct, atol=1e-8, equal_nan=True):
            warnings.append(f"correct_curve baseline differs between {ref_label} and {label}")
        if not np.allclose(ref_wrong, cur_wrong, atol=1e-8, equal_nan=True):
            warnings.append(f"wrong_curve baseline differs between {ref_label} and {label}")
    return warnings


def compute_relative_metrics(runs: OrderedDict[str, dict[str, Any]], reference_label: str) -> None:
    if reference_label not in runs:
        raise KeyError(f"Reference label {reference_label!r} was not found in the loaded runs: {list(runs.keys())}")

    ref = runs[reference_label]
    ref_overall = abs(ref["overall_token_delta_stats"]["mean_delta"])
    ref_correct_final = abs(ref["correct_final_value_delta_stats"]["mean"])
    ref_wrong_final = abs(ref["wrong_final_value_delta_stats"]["mean"])

    for run in runs.values():
        run["overall_effect_vs_reference"] = (
            abs(run["overall_token_delta_stats"]["mean_delta"]) / ref_overall if ref_overall else None
        )
        run["correct_final_effect_vs_reference"] = (
            abs(run["correct_final_value_delta_stats"]["mean"]) / ref_correct_final if ref_correct_final else None
        )
        run["wrong_final_effect_vs_reference"] = (
            abs(run["wrong_final_value_delta_stats"]["mean"]) / ref_wrong_final if ref_wrong_final else None
        )


def ordered_summary_rows(runs: OrderedDict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs.values():
        rows.append(
            {
                "label": run["label"],
                "ablation_mode": run["ablation_mode"],
                "overall_mean_delta": run["overall_token_delta_stats"]["mean_delta"],
                "overall_mean_abs_delta": run["overall_token_delta_stats"]["mean_abs_delta"],
                "correct_mean_delta": run["correct_token_delta_stats"]["mean_delta"],
                "wrong_mean_delta": run["wrong_token_delta_stats"]["mean_delta"],
                "correct_final_mean_delta": run["correct_final_value_delta_stats"]["mean"],
                "wrong_final_mean_delta": run["wrong_final_value_delta_stats"]["mean"],
                "baseline_gap_mean": run["baseline_gap_mean"],
                "ablated_gap_mean": run["ablated_gap_mean"],
                "gap_drop": run["gap_drop"],
                "gap_retained_fraction": run["gap_retained_fraction"],
                "overall_effect_vs_reference": run["overall_effect_vs_reference"],
                "correct_final_effect_vs_reference": run["correct_final_effect_vs_reference"],
                "wrong_final_effect_vs_reference": run["wrong_final_effect_vs_reference"],
                "all_negative_fraction": run["sign_stats"]["all"]["negative_fraction"],
                "correct_negative_fraction": run["sign_stats"]["correct"]["negative_fraction"],
                "wrong_positive_fraction": run["sign_stats"]["wrong"]["positive_fraction"],
            }
        )
    return rows


def fmt_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.{digits}f}%"


def write_summary_csv(out_path: Path, rows: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_json(out_path: Path, payload: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def write_summary_markdown(
    out_path: Path,
    rows: list[dict[str, Any]],
    runs: OrderedDict[str, dict[str, Any]],
    reference_label: str,
    warnings: list[str],
) -> None:
    if reference_label not in runs:
        raise KeyError(reference_label)

    ref = runs[reference_label]
    sorted_by_effect = sorted(rows, key=lambda row: abs(row["overall_mean_delta"]), reverse=True)
    strongest = sorted_by_effect[0]
    weakest = sorted_by_effect[-1]
    no_direct = runs.get("no_direct_carry")
    zero_hh = runs.get("zero_weight_hh")

    lines = [
        "# GRU Ablation Comparison",
        "",
        f"Reference run: `{reference_label}`",
        "",
        "## Key Findings",
    ]

    if no_direct is not None:
        lines.append(
            "- `no_direct_carry` is almost identical to `reset_each_token`: "
            f"{fmt_pct(no_direct['overall_effect_vs_reference'])} of the reference overall token effect, "
            f"{fmt_pct(no_direct['correct_final_effect_vs_reference'])} of the correct-final effect."
        )
    if zero_hh is not None:
        lines.append(
            "- `zero_weight_hh` is much weaker: "
            f"{fmt_pct(zero_hh['overall_effect_vs_reference'])} of the reference overall token effect, "
            f"{fmt_pct(zero_hh['correct_final_effect_vs_reference'])} of the correct-final effect."
        )
    lines.append(
        "- The baseline correct-wrong gap is "
        f"{fmt_float(ref['baseline_gap_mean'], digits=4)}. After ablation it becomes "
        + ", ".join(f"`{row['label']}`: {fmt_float(row['ablated_gap_mean'], digits=4)}" for row in rows)
        + "."
    )
    lines.append(
        "- Negative final-value deltas dominate in all strong ablations, especially on correct samples: "
        + ", ".join(
            f"`{row['label']}` correct-negative={fmt_pct(row['correct_negative_fraction'])}"
            for row in rows
        )
        + "."
    )
    lines.append(
        f"- Strongest overall effect: `{strongest['label']}` ({fmt_float(strongest['overall_mean_delta'], digits=4)}). "
        f"Weakest: `{weakest['label']}` ({fmt_float(weakest['overall_mean_delta'], digits=4)})."
    )
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Ablation | Overall token Δ | Correct token Δ | Wrong token Δ | Correct final Δ | Wrong final Δ | Gap after | Gap retained | Effect vs ref |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + f"{row['label']} | "
            + f"{fmt_float(row['overall_mean_delta'], digits=4)} | "
            + f"{fmt_float(row['correct_mean_delta'], digits=4)} | "
            + f"{fmt_float(row['wrong_mean_delta'], digits=4)} | "
            + f"{fmt_float(row['correct_final_mean_delta'], digits=4)} | "
            + f"{fmt_float(row['wrong_final_mean_delta'], digits=4)} | "
            + f"{fmt_float(row['ablated_gap_mean'], digits=4)} | "
            + f"{fmt_pct(row['gap_retained_fraction'])} | "
            + f"{fmt_pct(row['overall_effect_vs_reference'])} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_comparison_plot(out_path: Path, runs: OrderedDict[str, dict[str, Any]], reference_label: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - runtime dependency
        print(f"[warn] matplotlib unavailable, skipping plot export: {exc}")
        return

    labels = list(runs.keys())
    colors = {
        "reset_each_token": "#d62728",
        "no_direct_carry": "#2ca02c",
        "zero_weight_hh": "#1f77b4",
    }

    baseline = runs[reference_label]
    x = np.arange(len(baseline["correct_curve"]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(x, baseline["correct_curve"], color="#222222", linewidth=2.0, label="baseline")
    for label in labels:
        ax.plot(
            x,
            runs[label]["correct_curve_ablated"],
            linewidth=1.8,
            color=colors.get(label),
            label=label,
        )
    ax.set_title("Correct Curves")
    ax.set_xlabel("Normalized token position (bins)")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(x, baseline["wrong_curve"], color="#222222", linewidth=2.0, label="baseline")
    for label in labels:
        ax.plot(
            x,
            runs[label]["wrong_curve_ablated"],
            linewidth=1.8,
            color=colors.get(label),
            label=label,
        )
    ax.set_title("Wrong Curves")
    ax.set_xlabel("Normalized token position (bins)")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    ax = axes[1, 0]
    for label in labels:
        ax.plot(
            x,
            runs[label]["correct_delta_curve"],
            linewidth=1.8,
            color=colors.get(label),
            label=label,
        )
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle=":")
    ax.set_title("Correct Delta Curves")
    ax.set_xlabel("Normalized token position (bins)")
    ax.set_ylabel("Ablated - baseline")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    ax = axes[1, 1]
    metric_names = ["overall Δ", "correct final Δ", "wrong final Δ"]
    base_x = np.arange(len(metric_names))
    width = 0.22
    offsets = np.linspace(-width, width, num=len(labels))
    metric_getters = [
        lambda run: run["overall_token_delta_stats"]["mean_delta"],
        lambda run: run["correct_final_value_delta_stats"]["mean"],
        lambda run: run["wrong_final_value_delta_stats"]["mean"],
    ]
    for offset, label in zip(offsets, labels):
        values = [getter(runs[label]) for getter in metric_getters]
        ax.bar(base_x + offset, values, width=width, color=colors.get(label), label=label)
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle=":")
    ax.set_xticks(base_x)
    ax.set_xticklabels(metric_names)
    ax.set_title("Key Delta Metrics")
    ax.set_ylabel("Ablated - baseline")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    fig.suptitle("GRU Ablation Comparison", fontsize=16)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_svg_table(out_path: Path, rows: list[dict[str, Any]]) -> None:
    width = 1120
    row_height = 36
    header_height = 44
    margin = 24
    columns = [
        ("Ablation", 200),
        ("Overall Δ", 120),
        ("Correct Δ", 120),
        ("Wrong Δ", 120),
        ("Correct final Δ", 140),
        ("Wrong final Δ", 140),
        ("Gap after", 120),
        ("Gap kept", 120),
    ]
    height = margin * 2 + header_height + row_height * len(rows)

    x_positions = []
    cursor = margin
    for _, col_width in columns:
        x_positions.append(cursor)
        cursor += col_width

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{margin}" y="{margin}" font-size="22" fill="#222" dominant-baseline="hanging">GRU Ablation Metrics</text>',
        f'<rect x="{margin}" y="{margin + 30}" width="{sum(w for _, w in columns)}" height="{header_height}" fill="#f4f6f8" stroke="#d0d7de"/>',
    ]

    for (title, col_width), x in zip(columns, x_positions):
        parts.append(
            f'<text x="{x + 10}" y="{margin + 30 + header_height / 2}" font-size="14" fill="#222" dominant-baseline="middle">{title}</text>'
        )

    y = margin + 30 + header_height
    for row in rows:
        parts.append(
            f'<rect x="{margin}" y="{y}" width="{sum(w for _, w in columns)}" height="{row_height}" fill="white" stroke="#e5e7eb"/>'
        )
        values = [
            row["label"],
            fmt_float(row["overall_mean_delta"], digits=4),
            fmt_float(row["correct_mean_delta"], digits=4),
            fmt_float(row["wrong_mean_delta"], digits=4),
            fmt_float(row["correct_final_mean_delta"], digits=4),
            fmt_float(row["wrong_final_mean_delta"], digits=4),
            fmt_float(row["ablated_gap_mean"], digits=4),
            fmt_pct(row["gap_retained_fraction"]),
        ]
        for value, x in zip(values, x_positions):
            parts.append(
                f'<text x="{x + 10}" y="{y + row_height / 2}" font-size="13" fill="#222" dominant-baseline="middle">{value}</text>'
            )
        y += row_height

    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    run_specs = args.run if args.run is not None else [f"{label}={path}" for label, path in DEFAULT_RUN_SPECS]
    runs: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for spec in run_specs:
        label, path = parse_run_spec(spec)
        runs[label] = load_run(label, path)

    warnings = validate_shared_baseline(runs)
    compute_relative_metrics(runs, args.reference)
    rows = ordered_summary_rows(runs)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "reference_label": args.reference,
        "warnings": warnings,
        "rows": rows,
        "runs": runs,
    }
    write_summary_json(out_dir / "comparison_summary.json", payload)
    write_summary_csv(out_dir / "comparison_summary.csv", rows)
    write_summary_markdown(out_dir / "comparison_summary.md", rows, runs, args.reference, warnings)
    save_comparison_plot(out_dir / "comparison_plot.png", runs, args.reference)
    save_svg_table(out_dir / "comparison_table.svg", rows)

    print(f"[saved] {out_dir / 'comparison_summary.json'}")
    print(f"[saved] {out_dir / 'comparison_summary.csv'}")
    print(f"[saved] {out_dir / 'comparison_summary.md'}")
    print(f"[saved] {out_dir / 'comparison_plot.png'}")
    print(f"[saved] {out_dir / 'comparison_table.svg'}")
    if warnings:
        for warning in warnings:
            print(f"[warn] {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
