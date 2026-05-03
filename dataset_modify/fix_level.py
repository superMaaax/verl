#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


COT_SUFFIX_RE = re.compile(
    r"\s*Let's think step by step and output the final answer within \\boxed\{\}\.?\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill valid numeric level values for MetaMathQA-math-500 using MATH-500 test.jsonl."
        )
    )
    parser.add_argument(
        "--math500_jsonl",
        type=Path,
        default=Path("/data/shuozhe/saved_dataset/MATH-500/test.jsonl"),
    )
    parser.add_argument(
        "--target_dir",
        type=Path,
        default=Path("/data/shuozhe/saved_dataset/MetaMathQA-math-500"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train.parquet", "test.parquet"],
        help="Parquet files under target_dir to process.",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Disable writing .bak timestamp copies before overwrite.",
    )
    return parser.parse_args()


def normalize_problem_text(text: str) -> str:
    text = COT_SUFFIX_RE.sub("", text)
    text = " ".join(text.strip().split())
    return text


def extract_prompt_text(prompt_value: Any) -> str:
    if prompt_value is None:
        return ""

    # prompt is typically a list of {'role': ..., 'content': ...}
    messages: list[Any]
    if isinstance(prompt_value, list):
        messages = prompt_value
    elif hasattr(prompt_value, "tolist"):
        maybe_list = prompt_value.tolist()
        if isinstance(maybe_list, list):
            messages = maybe_list
        else:
            messages = [maybe_list]
    else:
        messages = [prompt_value]

    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user" and "content" in msg:
            return str(msg["content"])
    for msg in messages:
        if isinstance(msg, dict) and "content" in msg:
            return str(msg["content"])

    return str(messages[0]) if messages else ""


def parse_existing_level(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value) if value.is_integer() else None

    s = str(value).strip()
    if not s:
        return None

    if s.isdigit():
        return int(s)

    match = re.fullmatch(r"(?i)level\s*(\d+)", s)
    if match:
        return int(match.group(1))

    return None


def build_math500_level_map(math500_jsonl: Path) -> dict[str, int]:
    level_map: dict[str, int] = {}

    with math500_jsonl.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            obj = json.loads(line)
            problem = obj.get("problem")
            level = obj.get("level")
            if problem is None or level is None:
                raise ValueError(
                    f"Invalid MATH-500 row at line {line_number}: missing problem/level."
                )

            key = normalize_problem_text(str(problem))
            level_int = int(level)
            if key in level_map and level_map[key] != level_int:
                raise ValueError(
                    "Conflicting levels for the same normalized problem text: "
                    f"{key!r} -> {level_map[key]} vs {level_int}"
                )
            level_map[key] = level_int

    if not level_map:
        raise ValueError(f"No rows loaded from {math500_jsonl}")

    return level_map


def process_parquet(parquet_path: Path, level_map: dict[str, int], create_backup: bool) -> dict[str, int]:
    table = pq.read_table(parquet_path)
    names = table.schema.names
    if "prompt" not in names:
        raise ValueError(f"Missing 'prompt' column in {parquet_path}")

    prompts = table["prompt"].to_pylist()
    existing_levels = table["level"].to_pylist() if "level" in names else [None] * len(prompts)

    if len(existing_levels) != len(prompts):
        raise ValueError(
            f"Column length mismatch in {parquet_path}: prompt={len(prompts)}, level={len(existing_levels)}"
        )

    new_levels: list[int | None] = []
    matched_from_math500 = 0
    kept_existing = 0
    missing_after_fix = 0

    for prompt_value, existing in zip(prompts, existing_levels):
        prompt_text = extract_prompt_text(prompt_value)
        normalized = normalize_problem_text(prompt_text)

        mapped_level = level_map.get(normalized)
        if mapped_level is not None:
            new_levels.append(mapped_level)
            matched_from_math500 += 1
            continue

        existing_level = parse_existing_level(existing)
        if existing_level is not None:
            new_levels.append(existing_level)
            kept_existing += 1
        else:
            new_levels.append(None)
            missing_after_fix += 1

    level_array = pa.array(new_levels, type=pa.int64())
    if "level" in names:
        level_idx = names.index("level")
        updated = table.set_column(level_idx, "level", level_array)
    else:
        updated = table.append_column("level", level_array)

    if create_backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = parquet_path.with_suffix(parquet_path.suffix + f".bak_{ts}")
        parquet_path.replace(backup_path)
        pq.write_table(updated, parquet_path)
    else:
        tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
        pq.write_table(updated, tmp_path)
        tmp_path.replace(parquet_path)

    return {
        "rows": len(prompts),
        "matched_from_math500": matched_from_math500,
        "kept_existing": kept_existing,
        "missing_after_fix": missing_after_fix,
    }


def main() -> None:
    args = parse_args()

    level_map = build_math500_level_map(args.math500_jsonl)
    print(f"Loaded {len(level_map)} unique problem->level mappings from {args.math500_jsonl}")

    all_stats: dict[str, dict[str, int]] = {}

    for split in args.splits:
        parquet_path = args.target_dir / split
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing file: {parquet_path}")
        stats = process_parquet(parquet_path, level_map, create_backup=not args.no_backup)
        all_stats[split] = stats
        print(
            f"[{split}] rows={stats['rows']} "
            f"matched_from_math500={stats['matched_from_math500']} "
            f"kept_existing={stats['kept_existing']} "
            f"missing_after_fix={stats['missing_after_fix']}"
        )

    test_stats = all_stats.get("test.parquet")
    if test_stats is not None and test_stats["matched_from_math500"] != test_stats["rows"]:
        raise RuntimeError(
            "Validation failed: test.parquet should be fully matched to MATH-500, "
            f"but only {test_stats['matched_from_math500']}/{test_stats['rows']} matched."
        )

    print("Done.")


if __name__ == "__main__":
    main()
