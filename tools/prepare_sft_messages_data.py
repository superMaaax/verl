#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import pandas as pd


def to_builtin(value):
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if hasattr(value, "tolist"):
        return to_builtin(value.tolist())
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Convert prompt/response data into VERL MultiTurnSFTDataset messages parquet.")
    parser.add_argument("--input", required=True, help="Input .parquet or .jsonl file.")
    parser.add_argument("--output", required=True, help="Output SFT messages .parquet file.")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--dedup-by-prompt", action="store_true")
    parser.add_argument("--no-filter-correct", action="store_true", help="Do not filter is_correct==true rows when that column exists.")
    parser.add_argument("--enable-thinking", choices=["true", "false"], default=None, help="Optional per-row enable_thinking column value.")
    return parser.parse_args()


def read_input(path: Path, max_jsonl_records: int = -1) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
                    if max_jsonl_records >= 0 and len(records) >= max_jsonl_records:
                        break
        return pd.DataFrame(records)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input extension {path.suffix!r}; expected .parquet or .jsonl")


IM_START_RE = re.compile(r"<\|im_start\|>(system|user|assistant)\n")
IM_END = "<|im_end|>"


def parse_prompt(prompt: str) -> list[dict[str, str]]:
    prompt = str(prompt)
    matches = list(IM_START_RE.finditer(prompt))
    if not matches:
        return [{"role": "user", "content": prompt.strip()}]
    messages = []
    for index, match in enumerate(matches):
        role = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        content = prompt[start:end].strip()
        if content.endswith(IM_END):
            content = content[: -len(IM_END)].strip()
        if role == "assistant" and content == "":
            continue
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError(f"Parsed no messages from prompt prefix: {prompt[:200]!r}")
    return messages


def is_missing(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def convert(df: pd.DataFrame, *, filter_correct: bool, dedup_by_prompt: bool, enable_thinking: str | None) -> pd.DataFrame:
    if "messages" in df.columns:
        out = df.copy()
        out["messages"] = out["messages"].apply(to_builtin)
        if enable_thinking is not None:
            out["enable_thinking"] = enable_thinking == "true"
    else:
        missing = sorted({"prompt", "response"} - set(df.columns))
        if missing:
            raise ValueError(
                f"Cannot convert to SFT messages format. Missing columns: {missing}. "
                "Expected either a 'messages' column or 'prompt' and 'response' columns."
            )
        if filter_correct and "is_correct" in df.columns:
            df = df[df["is_correct"].astype(bool)].copy()
        if dedup_by_prompt:
            sort_cols = [col for col in ["example_id", "response_index"] if col in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols)
            df = df.drop_duplicates(subset=["prompt"], keep="first").copy()

        rows = []
        metadata_cols = [
            "example_id",
            "response_index",
            "num_responses_per_prompt",
            "task_score",
            "is_correct",
            "dataset_source",
            "level",
            "type",
            "answer",
            "id",
        ]
        for _, row in df.iterrows():
            messages = parse_prompt(row["prompt"])
            messages.append({"role": "assistant", "content": str(row["response"]).strip()})
            out_row = {"messages": messages}
            for col in metadata_cols:
                if col in row.index and not is_missing(row[col]):
                    out_row[col] = row[col]
            if enable_thinking is not None:
                out_row["enable_thinking"] = enable_thinking == "true"
            rows.append(out_row)
        out = pd.DataFrame(rows)

    return out


def validate(out: pd.DataFrame, source: Path) -> None:
    if len(out) == 0:
        raise ValueError(f"Prepared SFT dataset is empty after filtering/conversion: {source}")
    if "messages" not in out.columns:
        raise ValueError("Prepared SFT dataset is missing required 'messages' column.")
    for row_index, sample_messages in enumerate(out["messages"]):
        if not isinstance(sample_messages, (list, tuple)) or len(sample_messages) < 2:
            raise ValueError(
                f"Prepared SFT row {row_index} has invalid messages; expected at least user and assistant messages."
            )
        for message_index, message in enumerate(sample_messages):
            if not isinstance(message, dict):
                raise ValueError(f"Prepared SFT row {row_index} message {message_index} is not a dict: {type(message)!r}")
            if message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValueError(
                    f"Prepared SFT row {row_index} message {message_index} has invalid role: {message.get('role')!r}"
                )
            if "content" not in message:
                raise ValueError(f"Prepared SFT row {row_index} message {message_index} is missing content.")
        if sample_messages[-1].get("role") != "assistant":
            raise ValueError(f"Prepared SFT row {row_index} must end with an assistant message for supervised loss.")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    df = read_input(input_path, max_jsonl_records=args.max_samples)
    out = convert(
        df,
        filter_correct=not args.no_filter_correct,
        dedup_by_prompt=args.dedup_by_prompt,
        enable_thinking=args.enable_thinking,
    )
    if args.max_samples >= 0:
        out = out.head(args.max_samples).copy()
    validate(out, input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    sample_messages = out.iloc[0]["messages"]
    print(f"prepared_sft_rows={len(out)}")
    print(f"prepared_sft_columns={list(out.columns)}")
    print(f"prepared_sft_output={output_path}")
    print(f"sample_roles={[message.get('role') for message in sample_messages]}")
    print(f"sample_user_chars={len(str(sample_messages[0].get('content', '')))}")
    print(f"sample_assistant_chars={len(str(sample_messages[-1].get('content', '')))}")


if __name__ == "__main__":
    main()
