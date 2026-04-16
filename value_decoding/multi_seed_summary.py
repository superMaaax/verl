from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate multiple per-seed value_decoding summary_metrics.json files into a single "
            "top-level summary_metrics.json manifest."
        )
    )
    parser.add_argument("--output_path", type=str, required=True, help="Combined summary_metrics.json output path.")
    parser.add_argument("--source_script", type=str, required=True, help="Wrapper script that launched the runs.")
    parser.add_argument("--seed_values", nargs="+", required=True, help="Ordered list of seed values.")
    parser.add_argument(
        "--summary_paths",
        nargs="+",
        required=True,
        help="Ordered list of per-seed summary_metrics.json paths matching --seed_values.",
    )
    parser.add_argument(
        "--seed_output_dirs",
        nargs="+",
        required=True,
        help="Ordered list of per-seed output directories matching --seed_values.",
    )
    return parser.parse_args()


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload).__name__}.")
    return payload


def _coerce_seed(seed: str) -> int | str:
    try:
        return int(seed)
    except ValueError:
        return seed


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed_values = [_coerce_seed(str(seed)) for seed in args.seed_values]
    summary_paths = [Path(path).resolve() for path in args.summary_paths]
    seed_output_dirs = [str(Path(path).resolve()) for path in args.seed_output_dirs]

    if len(seed_values) != len(summary_paths) or len(seed_values) != len(seed_output_dirs):
        raise ValueError(
            "seed_values, summary_paths, and seed_output_dirs must have the same length, got "
            f"{len(seed_values)}, {len(summary_paths)}, and {len(seed_output_dirs)}."
        )
    if not seed_values:
        raise ValueError("At least one seed is required.")
    if len(set(str(seed) for seed in seed_values)) != len(seed_values):
        raise ValueError(f"Duplicate seed values are not allowed: {seed_values}")

    seed_runs: list[dict[str, Any]] = []
    for seed_value, summary_path, seed_output_dir in zip(seed_values, summary_paths, seed_output_dirs, strict=True):
        if not summary_path.exists():
            raise FileNotFoundError(f"Per-seed summary file does not exist: {summary_path}")
        summary_payload = _json_load(summary_path)
        recorded_seed = summary_payload.get("run_args", {}).get("seed")
        if recorded_seed is not None and str(recorded_seed) != str(seed_value):
            raise ValueError(
                f"Seed mismatch for {summary_path}: expected {seed_value!r}, found {recorded_seed!r} in run_args.seed."
            )
        seed_runs.append(
            {
                "seed": seed_value,
                "output_dir": seed_output_dir,
                "summary_metrics_path": str(summary_path),
                "summary": summary_payload,
            }
        )

    combined_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "aggregation_type": "multi_seed",
        "source_script": str(Path(args.source_script).resolve()),
        "output_path": str(output_path),
        "num_seeds": len(seed_values),
        "seed_values": seed_values,
        "seed_output_dirs": seed_output_dirs,
        "summary_paths": [str(path) for path in summary_paths],
        "seed_runs": seed_runs,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(combined_payload, handle, ensure_ascii=True, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
