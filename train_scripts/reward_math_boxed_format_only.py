from __future__ import annotations

from typing import Any, Optional

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off", ""}:
            return False
        raise ValueError(f"Cannot interpret {value!r} as boolean.")
    return bool(value)


def extract_boxed_answer(solution_str: str, require_nonempty_box: bool = True) -> Optional[str]:
    """Extract the last boxed answer and optionally require non-empty content."""
    try:
        boxed = last_boxed_only_string(solution_str)
    except Exception:
        return None

    if boxed is None:
        return None

    try:
        answer = remove_boxed(boxed).strip()
    except Exception:
        return None

    if require_nonempty_box and not answer:
        return None

    return answer


def compute_score(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    *,
    format_reward: float = 1.0,
    missing_reward: float = 0.0,
    require_nonempty_box: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Reward only whether the response contains a boxed final answer.

    This intentionally ignores `ground_truth` correctness so PPO can be trained
    on format adherence alone.
    """
    format_reward = float(format_reward)
    missing_reward = float(missing_reward)
    require_nonempty_box = _coerce_bool(require_nonempty_box)

    boxed_answer = extract_boxed_answer(solution_str, require_nonempty_box=require_nonempty_box)
    format_ok = boxed_answer is not None
    reward = format_reward if format_ok else missing_reward

    regular_eval = default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    if isinstance(regular_eval, dict):
        regular_acc = regular_eval.get("acc", regular_eval.get("score", 0.0))
    else:
        regular_acc = regular_eval

    return {
        "score": float(reward),
        "acc": float(regular_acc),
        "format_ok": format_ok,
        "boxed_answer": boxed_answer if boxed_answer is not None else "",
    }
