from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

HF_EXACT_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")
HF_INDEX_WEIGHT_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
HF_SHARD_PATTERNS = ("model-*.safetensors", "pytorch_model-*.bin")
MATH_DATA_SOURCES = {
    "lighteval/MATH",
    "DigitalLearningGmbH/MATH-lighteval",
    "HuggingFaceH4/MATH-500",
    "math_500",
}


def _read_hf_index_shard_names(index_path: Path) -> list[str]:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        return []

    shard_names: list[str] = []
    seen: set[str] = set()
    for candidate in weight_map.values():
        if not isinstance(candidate, str) or not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        shard_names.append(candidate)
    return shard_names


def _find_missing_hf_weight_files(model_dir: Path) -> list[Path]:
    if not model_dir.exists():
        return []

    for name in HF_EXACT_WEIGHT_NAMES:
        if (model_dir / name).is_file():
            return []

    for name in HF_INDEX_WEIGHT_NAMES:
        index_path = model_dir / name
        if not index_path.is_file():
            continue
        shard_names = _read_hf_index_shard_names(index_path)
        if not shard_names:
            return [index_path]
        return [model_dir / shard_name for shard_name in shard_names if not (model_dir / shard_name).is_file()]

    for pattern in HF_SHARD_PATTERNS:
        if any(model_dir.glob(pattern)):
            return []

    return []


def _has_hf_weights(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    if _find_missing_hf_weight_files(model_dir):
        return False
    for name in HF_EXACT_WEIGHT_NAMES:
        if (model_dir / name).is_file():
            return True
    for name in HF_INDEX_WEIGHT_NAMES:
        if (model_dir / name).is_file():
            return True
    for pattern in HF_SHARD_PATTERNS:
        if any(model_dir.glob(pattern)):
            return True
    return False


def _has_hf_config(model_dir: Path) -> bool:
    return model_dir.exists() and (model_dir / "config.json").is_file()


def has_complete_hf_checkpoint(model_dir: Path) -> bool:
    return _has_hf_weights(model_dir) and _has_hf_config(model_dir)


def _has_fsdp_checkpoint_shards(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    if (model_dir / "fsdp_config.json").is_file():
        return True
    return any(model_dir.glob("model_world_size_*_rank_*.pt"))


def _resolve_actor_local_dir(checkpoint_dir: Path) -> Path:
    actor_dir = checkpoint_dir / "actor"
    for candidate in (actor_dir, checkpoint_dir):
        if has_complete_hf_checkpoint(candidate) or _has_fsdp_checkpoint_shards(candidate) or _has_hf_config(candidate):
            return candidate
    return actor_dir


def _resolve_actor_hf_source_dir(checkpoint_dir: Path) -> Path:
    actor_local_dir = _resolve_actor_local_dir(checkpoint_dir)
    candidates = [
        checkpoint_dir / "merged_hf" / "actor",
        actor_local_dir / "huggingface",
        actor_local_dir,
        checkpoint_dir / "huggingface" / "actor",
        checkpoint_dir / "huggingface",
        checkpoint_dir,
    ]
    for candidate in candidates:
        if _has_hf_config(candidate):
            return candidate

    tried = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Unable to locate actor Hugging Face config metadata under checkpoint "
        f"{checkpoint_dir}.\nTried:\n{tried}"
    )


def _merge_fsdp_actor_checkpoint(local_dir: Path, target_dir: Path, *, hf_model_config_path: Path) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
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
        "--hf_model_config_path",
        str(hf_model_config_path),
    ]
    subprocess.run(cmd, check=True)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolve_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(model_dir: Path, *, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"Tokenizer at {model_dir} has no pad_token and no eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_actor_model(
    model_dir: Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool = False,
):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model


def resolve_eos_token_ids(model_dir: Path, tokenizer) -> tuple[int, ...]:
    eos_token_ids: list[int] = []
    try:
        generation_config = GenerationConfig.from_pretrained(str(model_dir))
    except Exception:
        generation_config = None

    for candidate in (getattr(generation_config, "eos_token_id", None), tokenizer.eos_token_id):
        if candidate is None:
            continue
        if isinstance(candidate, int):
            eos_token_ids.append(int(candidate))
            continue
        eos_token_ids.extend(int(token_id) for token_id in candidate if token_id is not None)

    return tuple(sorted(set(eos_token_ids)))


@dataclass(frozen=True)
class ExampleRecord:
    example_id: int
    prompt_text: str
    data_source: str
    ground_truth: Any
    prompt_token_ids: tuple[int, ...] | None = None


def _is_missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
    except Exception:
        return False

    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def _stringify_chat_messages(messages: list[Any]) -> str:
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            parts.append(str(message))
            continue
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _coerce_prompt(prompt: Any) -> Any:
    if isinstance(prompt, np.ndarray):
        return prompt.tolist()
    return prompt


def normalize_prompt(prompt: Any, tokenizer) -> str:
    prompt = _coerce_prompt(prompt)

    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, dict):
        if "messages" in prompt:
            return normalize_prompt(prompt["messages"], tokenizer)
        for key in ("prompt", "text", "content"):
            if key in prompt:
                return str(prompt[key])
        return json.dumps(prompt, ensure_ascii=True)

    if isinstance(prompt, list):
        if not prompt:
            return ""
        if all(isinstance(item, dict) for item in prompt):
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                except Exception:
                    pass
            return _stringify_chat_messages(prompt)
        if all(isinstance(item, str) for item in prompt):
            return "\n".join(prompt)
        return "\n".join(str(item) for item in prompt)

    return str(prompt)


def extract_ground_truth(row: pd.Series, response_key: str | None) -> Any:
    if response_key and response_key in row and not _is_missing(row[response_key]):
        return row[response_key]

    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth")

    return None


def load_examples(
    dataset_path: str | Path,
    *,
    tokenizer,
    prompt_key: str = "prompt",
    response_key: str | None = None,
    start_index: int = 0,
    max_examples: int | None = None,
    shuffle_examples: bool = False,
    seed: int = 0,
    pretokenize_max_length: int | None = None,
) -> list[ExampleRecord]:
    dataframe = pd.read_parquet(Path(dataset_path))
    indices = list(range(len(dataframe)))

    if start_index:
        indices = indices[start_index:]

    if shuffle_examples:
        random.Random(seed).shuffle(indices)

    if max_examples is not None:
        indices = indices[:max_examples]

    examples: list[ExampleRecord] = []
    for example_id in indices:
        row = dataframe.iloc[example_id]
        prompt_text = normalize_prompt(row[prompt_key], tokenizer)
        prompt_token_ids = None
        if pretokenize_max_length is not None:
            tokenized = tokenizer(
                prompt_text,
                truncation=True,
                max_length=pretokenize_max_length,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            prompt_token_ids = tuple(int(token_id) for token_id in tokenized["input_ids"])
        data_source = row.get("data_source", "")
        data_source = "" if _is_missing(data_source) else str(data_source)
        ground_truth = extract_ground_truth(row, response_key=response_key)
        examples.append(
            ExampleRecord(
                example_id=int(example_id),
                prompt_text=prompt_text,
                data_source=data_source,
                ground_truth=ground_truth,
                prompt_token_ids=prompt_token_ids,
            )
        )
    return examples


def _math_last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def _math_remove_boxed(string: str) -> str:
    if "\\boxed " in string:
        left = "\\boxed "
        if not string.startswith(left):
            raise ValueError(f"Unexpected boxed string: {string}")
        return string[len(left) :]

    left = "\\boxed{"
    if not string.startswith(left) or not string.endswith("}"):
        raise ValueError(f"Unexpected boxed string: {string}")
    return string[len(left) : -1]


def _math_fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _math_fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        if string != f"{a}/{b}":
            return string
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _math_remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _math_fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_substr = "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _math_strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _math_remove_right_units(string)
    string = string.replace("\\\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _math_fix_sqrt(string)
    string = string.replace(" ", "")
    string = _math_fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _math_fix_a_slash_b(string)
    return string


def _math_is_equiv(string_a: str | None, string_b: str | None) -> bool:
    if string_a is None and string_b is None:
        return True
    if string_a is None or string_b is None:
        return False
    try:
        return _math_strip_string(string_a) == _math_strip_string(string_b)
    except Exception:
        return string_a == string_b


def _math_compute_score(solution_str: str, ground_truth: Any) -> float:
    if ground_truth is None:
        raise ValueError("Ground-truth answer is required for correctness scoring.")
    try:
        boxed_answer = _math_last_boxed_only_string(solution_str)
        if boxed_answer is None:
            return 0.0
        return 1.0 if _math_is_equiv(_math_remove_boxed(boxed_answer), str(ground_truth)) else 0.0
    except Exception:
        return 0.0


def score_response(example: ExampleRecord, response_text: str) -> float:
    if example.data_source in MATH_DATA_SOURCES:
        return _math_compute_score(response_text, example.ground_truth)
    raise NotImplementedError(
        "This plotting script currently supports MATH-style scoring only. "
        f"Received data_source={example.data_source!r}."
    )


class ActorSamplingMode(str, Enum):
    GREEDY = "greedy"
    SAMPLE = "sample"


def set_decode_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _filter_logits(logits: torch.Tensor, *, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    filtered = logits.clone()

    if top_k > 0:
        top_k = min(top_k, filtered.shape[-1])
        kth_values = torch.topk(filtered, k=top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < kth_values, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(filtered, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        filtered = filtered.masked_fill(indices_to_remove, float("-inf"))

    return filtered


def sample_token_from_actor(
    logits: torch.Tensor,
    *,
    sampling_mode: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    if sampling_mode == ActorSamplingMode.GREEDY.value or temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())

    scaled_logits = logits.float() / temperature
    filtered_logits = _filter_logits(scaled_logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(filtered_logits, dim=-1)
    if not torch.isfinite(probs).all() or torch.sum(probs) <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    return int(torch.multinomial(probs, num_samples=1).item())


class ActorStepper:
    def __init__(self, model, prompt_ids: torch.Tensor, *, use_cache: bool = True):
        self.model = model
        self.sequence_ids = prompt_ids
        self.attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)
        self.request_cache = bool(use_cache)
        self.use_cache = False
        self.past_key_values = None
        self.current_logits = self._initialize()

    @torch.inference_mode()
    def _initialize(self) -> torch.Tensor:
        outputs = self.model(
            input_ids=self.sequence_ids,
            attention_mask=self.attention_mask,
            use_cache=self.request_cache,
        )
        past_key_values = getattr(outputs, "past_key_values", None)
        self.use_cache = bool(self.request_cache and past_key_values is not None)
        self.past_key_values = past_key_values if self.use_cache else None
        return outputs.logits[:, -1, :]

    @torch.inference_mode()
    def append(self, token_id: int) -> None:
        token_tensor = torch.tensor([[token_id]], device=self.sequence_ids.device, dtype=self.sequence_ids.dtype)
        self.sequence_ids = torch.cat([self.sequence_ids, token_tensor], dim=1)
        self.attention_mask = torch.cat(
            [self.attention_mask, torch.ones_like(token_tensor, device=self.attention_mask.device)],
            dim=1,
        )

        if self.use_cache and self.past_key_values is not None:
            try:
                outputs = self.model(
                    input_ids=token_tensor,
                    attention_mask=self.attention_mask,
                    past_key_values=self.past_key_values,
                    use_cache=True,
                )
            except Exception:
                self.use_cache = False
                self.past_key_values = None
                outputs = self.model(
                    input_ids=self.sequence_ids,
                    attention_mask=self.attention_mask,
                    use_cache=False,
                )
        else:
            outputs = self.model(
                input_ids=self.sequence_ids,
                attention_mask=self.attention_mask,
                use_cache=False,
            )

        if self.use_cache:
            self.past_key_values = getattr(outputs, "past_key_values", None)
            self.use_cache = self.past_key_values is not None
        self.current_logits = outputs.logits[:, -1, :]


@dataclass(frozen=True)
class EntropyTrace:
    example_result: dict[str, Any]
    token_ids: tuple[int, ...]
    token_texts: tuple[str, ...]
    entropies: tuple[float, ...]
    token_logprobs: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the first correct actor generation in a dataset slice and plot "
            "next-token entropy over generated token positions."
        )
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/shuozhe/verl/step_boundary_detect/output_plot",
    )
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--response_key", type=str, default=None)
    parser.add_argument(
        "--example_id",
        type=int,
        default=None,
        help="Optional exact dataset row to try. If unset, scan forward from --start_index.",
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_scan_examples", type=int, default=128)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--actor_sampling_mode",
        type=str,
        default=ActorSamplingMode.GREEDY.value,
        choices=[mode.value for mode in ActorSamplingMode],
    )
    parser.add_argument("--actor_temperature", type=float, default=1.0)
    parser.add_argument("--actor_top_p", type=float, default=1.0)
    parser.add_argument("--actor_top_k", type=int, default=0)
    parser.add_argument(
        "--moving_average_window",
        type=int,
        default=32,
        help="Optional smoothing window for an overlaid moving-average curve; use 0 or 1 to disable.",
    )
    parser.add_argument("--plot_width", type=float, default=14.0)
    parser.add_argument("--plot_height", type=float, default=6.0)
    parser.add_argument("--plot_dpi", type=int, default=180)
    parser.add_argument("--plot_title", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="correct_generation_entropy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--disable_actor_cache", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.example_id is not None and args.example_id < 0:
        raise ValueError("--example_id must be non-negative when provided.")
    if args.start_index < 0:
        raise ValueError("--start_index must be non-negative.")
    if args.max_scan_examples <= 0:
        raise ValueError("--max_scan_examples must be positive.")
    if args.max_prompt_length <= 0:
        raise ValueError("--max_prompt_length must be positive.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive.")
    if args.actor_temperature < 0.0:
        raise ValueError("--actor_temperature must be non-negative.")
    if not (0.0 < args.actor_top_p <= 1.0):
        raise ValueError("--actor_top_p must be in (0, 1].")
    if args.actor_top_k < 0:
        raise ValueError("--actor_top_k must be non-negative.")
    if args.moving_average_window < 0:
        raise ValueError("--moving_average_window must be non-negative.")
    if args.plot_width <= 0 or args.plot_height <= 0:
        raise ValueError("--plot_width and --plot_height must be positive.")
    if args.plot_dpi <= 0:
        raise ValueError("--plot_dpi must be positive.")


def _resolve_actor_hf_dir(checkpoint_dir: Path, *, skip_merge: bool) -> Path:
    if has_complete_hf_checkpoint(checkpoint_dir):
        return checkpoint_dir

    merged_actor_dir = checkpoint_dir / "merged_hf" / "actor"
    if has_complete_hf_checkpoint(merged_actor_dir):
        return merged_actor_dir

    actor_local_dir = _resolve_actor_local_dir(checkpoint_dir)
    if has_complete_hf_checkpoint(actor_local_dir):
        return actor_local_dir

    if skip_merge:
        raise FileNotFoundError(
            f"Actor HF checkpoint not found in {merged_actor_dir}. "
            "Disable --skip_merge or point --checkpoint_dir at a complete merged actor checkpoint."
        )

    if not _has_fsdp_checkpoint_shards(actor_local_dir):
        raise FileNotFoundError(
            f"Actor checkpoint at {actor_local_dir} is neither a complete HF checkpoint "
            "nor an FSDP checkpoint with model shards."
        )

    hf_source_dir = _resolve_actor_hf_source_dir(checkpoint_dir)
    _merge_fsdp_actor_checkpoint(
        actor_local_dir,
        merged_actor_dir,
        hf_model_config_path=hf_source_dir,
    )

    if not has_complete_hf_checkpoint(merged_actor_dir):
        raise FileNotFoundError(
            f"Actor HF checkpoint is incomplete in {merged_actor_dir} after merge."
        )
    return merged_actor_dir


def _entropy_from_logits(logits: torch.Tensor) -> float:
    logits_fp32 = logits.float()
    log_probs = torch.log_softmax(logits_fp32, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return float(entropy.item())


def _example_seed(base_seed: int, *, example_id: int) -> int:
    return int(base_seed + (example_id + 1) * 1_000_003)


def _token_texts(tokenizer, token_ids: Sequence[int]) -> tuple[str, ...]:
    tokens = tokenizer.convert_ids_to_tokens(list(token_ids))
    if tokens is None:
        return tuple(str(token_id) for token_id in token_ids)
    return tuple(str(token) for token in tokens)


def _load_scan_examples(args: argparse.Namespace, tokenizer) -> list[ExampleRecord]:
    start_index = args.example_id if args.example_id is not None else args.start_index
    max_examples = 1 if args.example_id is not None else args.max_scan_examples
    return load_examples(
        args.dataset_path,
        tokenizer=tokenizer,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        start_index=start_index,
        max_examples=max_examples,
        shuffle_examples=False,
        seed=args.seed,
        pretokenize_max_length=args.max_prompt_length,
    )


def _decode_example_entropy_trace(
    *,
    actor,
    tokenizer,
    example: ExampleRecord,
    actor_device: torch.device,
    max_prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    sampling_mode: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    use_actor_cache: bool,
) -> EntropyTrace:
    set_decode_seed(seed)

    if example.prompt_token_ids is not None:
        prompt_ids = torch.tensor([list(example.prompt_token_ids)], device=actor_device, dtype=torch.long)
    else:
        tokenized = tokenizer(
            example.prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        )
        prompt_ids = tokenized["input_ids"].to(actor_device)

    actor_state = ActorStepper(actor, prompt_ids, use_cache=use_actor_cache)
    generated_token_ids: list[int] = []
    entropies: list[float] = []
    token_logprobs: list[float] = []
    eos_emitted = False

    start_time = time.perf_counter()
    for _ in range(max_new_tokens):
        logits = actor_state.current_logits
        actor_log_probs = torch.log_softmax(logits.float(), dim=-1)
        entropy = _entropy_from_logits(logits)
        token_id = sample_token_from_actor(
            logits.squeeze(0),
            sampling_mode=sampling_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        token_logprob = float(actor_log_probs[0, token_id].item())

        generated_token_ids.append(int(token_id))
        entropies.append(entropy)
        token_logprobs.append(token_logprob)
        actor_state.append(token_id)

        if token_id in eos_token_ids:
            eos_emitted = True
            break

    latency_sec = time.perf_counter() - start_time
    response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    task_score = float(score_response(example, response_text))
    token_texts = _token_texts(tokenizer, generated_token_ids)
    response_length = len(generated_token_ids)

    example_result = {
        "example_id": int(example.example_id),
        "data_source": example.data_source,
        "prompt": example.prompt_text,
        "ground_truth": None if example.ground_truth is None else str(example.ground_truth),
        "generated_response": response_text,
        "response_length": int(response_length),
        "task_score": task_score,
        "eos_emitted": bool(eos_emitted),
        "max_length_hit": bool(max_new_tokens > 0 and not eos_emitted and response_length >= max_new_tokens),
        "latency_sec": latency_sec,
        "tokens_per_second": (response_length / latency_sec) if latency_sec > 0 else None,
        "actor_sampling_mode": sampling_mode,
        "actor_temperature": float(temperature),
        "actor_top_p": float(top_p),
        "actor_top_k": int(top_k),
    }
    return EntropyTrace(
        example_result=example_result,
        token_ids=tuple(generated_token_ids),
        token_texts=token_texts,
        entropies=tuple(float(value) for value in entropies),
        token_logprobs=tuple(float(value) for value in token_logprobs),
    )


def _scan_for_correct_trace(
    *,
    actor,
    tokenizer,
    examples: Sequence[ExampleRecord],
    actor_device: torch.device,
    args: argparse.Namespace,
    eos_token_ids: tuple[int, ...],
) -> EntropyTrace:
    use_actor_cache = not args.disable_actor_cache
    for scan_index, example in enumerate(examples, start=1):
        trace = _decode_example_entropy_trace(
            actor=actor,
            tokenizer=tokenizer,
            example=example,
            actor_device=actor_device,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            sampling_mode=args.actor_sampling_mode,
            temperature=args.actor_temperature,
            top_p=args.actor_top_p,
            top_k=args.actor_top_k,
            seed=_example_seed(args.seed, example_id=example.example_id),
            use_actor_cache=use_actor_cache,
        )
        example_result = trace.example_result
        is_correct = float(example_result["task_score"]) >= 1.0
        print(
            "example_id={example_id} score={score:.1f} len={length} correct={correct} ({index}/{total})".format(
                example_id=example_result["example_id"],
                score=float(example_result["task_score"]),
                length=example_result["response_length"],
                correct=is_correct,
                index=scan_index,
                total=len(examples),
            ),
            flush=True,
        )
        if is_correct:
            return trace

    start_label = args.example_id if args.example_id is not None else args.start_index
    raise SystemExit(
        f"Did not find a correct generation after scanning {len(examples)} example(s) starting at dataset index {start_label}."
    )


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="valid")


def _default_plot_title(trace: EntropyTrace) -> str:
    example_result = trace.example_result
    return (
        "Correct Generation Entropy over Token Positions\n"
        f"example_id={example_result['example_id']} score={float(example_result['task_score']):.1f} "
        f"tokens={example_result['response_length']}"
    )


def _plot_entropy_trace(
    *,
    trace: EntropyTrace,
    plot_path: Path,
    plot_title: str,
    moving_average_window: int,
    plot_width: float,
    plot_height: float,
    plot_dpi: int,
) -> None:
    entropy_values = np.asarray(trace.entropies, dtype=np.float64)
    positions = np.arange(1, len(entropy_values) + 1, dtype=np.int64)
    if positions.size == 0:
        raise ValueError("Cannot plot an empty entropy trace.")

    fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=plot_dpi)
    ax.plot(
        positions,
        entropy_values,
        color="#1f77b4",
        linewidth=1.2,
        alpha=0.9,
        label="token entropy",
    )

    if moving_average_window > 1 and len(entropy_values) >= moving_average_window:
        averaged = _moving_average(entropy_values, moving_average_window)
        averaged_positions = np.arange(moving_average_window, len(entropy_values) + 1, dtype=np.int64)
        ax.plot(
            averaged_positions,
            averaged,
            color="#d62728",
            linewidth=2.0,
            alpha=0.95,
            label=f"{moving_average_window}-token moving avg",
        )

    mean_entropy = float(entropy_values.mean())
    peak_index = int(np.argmax(entropy_values))
    peak_position = int(positions[peak_index])
    peak_entropy = float(entropy_values[peak_index])

    ax.axhline(
        mean_entropy,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.0,
        alpha=0.85,
        label=f"mean={mean_entropy:.3f}",
    )
    ax.scatter(
        [peak_position],
        [peak_entropy],
        color="#ff7f0e",
        s=28,
        zorder=4,
        label=f"peak={peak_entropy:.3f} @ {peak_position}",
    )

    if trace.example_result["eos_emitted"]:
        ax.axvline(
            positions[-1],
            color="#9467bd",
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
            label="EOS emitted",
        )

    ax.set_title(plot_title)
    ax.set_xlabel("Generated token position")
    ax.set_ylabel("Entropy")
    ax.set_xlim(1, int(positions[-1]))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    summary_lines = [
        f"example_id={trace.example_result['example_id']}",
        f"score={float(trace.example_result['task_score']):.1f}",
        f"tokens={trace.example_result['response_length']}",
        f"eos_emitted={trace.example_result['eos_emitted']}",
    ]
    ax.text(
        0.013,
        0.985,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.9,
        },
    )

    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def _entropy_stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def _write_trace_json(
    *,
    path: Path,
    trace: EntropyTrace,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    actor_hf_dir: Path,
    plot_path: Path,
) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "checkpoint_dir": str(checkpoint_dir),
        "actor_hf_dir": str(actor_hf_dir),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "plot_path": str(plot_path),
        "config": {
            "prompt_key": args.prompt_key,
            "response_key": args.response_key,
            "example_id": args.example_id,
            "start_index": args.start_index,
            "max_scan_examples": args.max_scan_examples,
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "device": args.device,
            "actor_sampling_mode": args.actor_sampling_mode,
            "actor_temperature": args.actor_temperature,
            "actor_top_p": args.actor_top_p,
            "actor_top_k": args.actor_top_k,
            "moving_average_window": args.moving_average_window,
            "seed": args.seed,
            "skip_merge": args.skip_merge,
            "disable_actor_cache": args.disable_actor_cache,
            "trust_remote_code": args.trust_remote_code,
        },
        "example_result": trace.example_result,
        "entropy_stats": _entropy_stats(trace.entropies),
        "token_ids": list(trace.token_ids),
        "token_texts": list(trace.token_texts),
        "entropies": list(trace.entropies),
        "token_logprobs": list(trace.token_logprobs),
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _validate_args(args)

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing actor from {checkpoint_dir}", flush=True)
    actor_hf_dir = _resolve_actor_hf_dir(checkpoint_dir, skip_merge=args.skip_merge)
    actor_device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    tokenizer = load_tokenizer(actor_hf_dir, trust_remote_code=args.trust_remote_code)
    eos_token_ids = resolve_eos_token_ids(actor_hf_dir, tokenizer)
    examples = _load_scan_examples(args, tokenizer)
    if not examples:
        raise SystemExit("No examples were loaded for scanning.")

    print(
        f"Loading actor model from {actor_hf_dir} on {actor_device} with dtype={dtype}",
        flush=True,
    )
    actor = load_actor_model(
        actor_hf_dir,
        dtype=dtype,
        device=actor_device,
        trust_remote_code=args.trust_remote_code,
    )

    print(
        f"Scanning {len(examples)} example(s) for a correct generation; eos_token_ids={eos_token_ids}",
        flush=True,
    )
    trace = _scan_for_correct_trace(
        actor=actor,
        tokenizer=tokenizer,
        examples=examples,
        actor_device=actor_device,
        args=args,
        eos_token_ids=eos_token_ids,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    example_id = int(trace.example_result["example_id"])
    stem = f"{args.output_prefix}_{timestamp}_example_{example_id}"
    plot_path = output_dir / f"{stem}.png"
    json_path = output_dir / f"{stem}.json"

    plot_title = args.plot_title or _default_plot_title(trace)
    _plot_entropy_trace(
        trace=trace,
        plot_path=plot_path,
        plot_title=plot_title,
        moving_average_window=args.moving_average_window,
        plot_width=args.plot_width,
        plot_height=args.plot_height,
        plot_dpi=args.plot_dpi,
    )
    _write_trace_json(
        path=json_path,
        trace=trace,
        args=args,
        checkpoint_dir=checkpoint_dir,
        actor_hf_dir=actor_hf_dir,
        plot_path=plot_path,
    )

    print(f"Found correct example_id={example_id} score={float(trace.example_result['task_score']):.1f}", flush=True)
    print(f"Wrote entropy plot: {plot_path}", flush=True)
    print(f"Wrote entropy trace JSON: {json_path}", flush=True)


if __name__ == "__main__":
    main()
