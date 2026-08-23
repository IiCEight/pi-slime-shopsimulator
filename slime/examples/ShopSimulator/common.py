"""Small helpers shared by ShopSimulator rollout and evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

SUB_SCORES = ("r_type", "r_att", "r_option", "r_price")


def prompt_text(prompt: str | list[dict[str, Any]]) -> str:
    """Flatten a dataset prompt into the string passed to Pi."""
    if isinstance(prompt, str):
        return prompt
    parts: list[str] = []
    for message in prompt or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
    text = "\n".join(part for part in parts if part)
    if not text:
        raise ValueError("prompt has no user text")
    return text


def _candidate_outcome(row: dict[str, Any]) -> dict[str, Any]:
    detail = row.get("reward_detail") or {}
    scored = bool(detail)
    values = {
        name: float(detail.get(name, 1.0 if name == "r_option" else 0.0)) if scored else 0.0
        for name in SUB_SCORES
    }
    hard_reward = 1.0
    for value in values.values():
        hard_reward *= value
    purchase, goal = row.get("purchase_asin"), row.get("goal_asin")
    return {
        "task_id": row.get("task_id"),
        "done": 1 if scored else 0,
        "r_loose": float(row.get("reward", 0.0) or 0.0),
        "r_hard": hard_reward,
        "r_success": 1 if scored and all(value == 1 for value in values.values()) else 0,
        "right_product": 1 if scored and purchase is not None and purchase == goal else 0,
        "sub_scores": values,
    }


def official_metrics(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean@k-style averages and per-task pass@k statistics."""
    outcomes = [_candidate_outcome(row) for row in candidates]
    if not outcomes:
        return {"samples": 0}

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 6)

    metrics: dict[str, Any] = {
        "samples": len(outcomes),
        "done_rate": mean([outcome["done"] for outcome in outcomes]),
        "r_loose": mean([outcome["r_loose"] for outcome in outcomes]),
        "r_hard": mean([outcome["r_hard"] for outcome in outcomes]),
        "r_success": mean([outcome["r_success"] for outcome in outcomes]),
        "right_product_rate": mean([outcome["right_product"] for outcome in outcomes]),
        **{
            name: mean([outcome["sub_scores"][name] for outcome in outcomes])
            for name in SUB_SCORES
        },
    }
    if any(outcome["task_id"] is None for outcome in outcomes):
        metrics["pass_at_k"] = None
        return metrics

    by_task: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        by_task[outcome["task_id"]].append(outcome)
    counts = {len(group) for group in by_task.values()}
    metrics["pass_at_k"] = {
        "tasks": len(by_task),
        "samples_per_task": (
            min(counts) if len(counts) == 1 else {"min": min(counts), "max": max(counts)}
        ),
        "reward_variance_task_fraction": mean([
            1 if max(item["r_loose"] for item in group) != min(item["r_loose"] for item in group) else 0
            for group in by_task.values()
        ]),
        "all_zero_reward_task_fraction": mean([
            1 if max(item["r_loose"] for item in group) == 0 else 0
            for group in by_task.values()
        ]),
        "pass_success": mean([
            1 if any(item["r_success"] for item in group) else 0
            for group in by_task.values()
        ]),
        "pass_positive_reward": mean([
            1 if any(item["r_loose"] > 0 for item in group) else 0
            for group in by_task.values()
        ]),
        "pass_done": mean([
            1 if any(item["done"] for item in group) else 0
            for group in by_task.values()
        ]),
        "pass_right_product": mean([
            1 if any(item["right_product"] for item in group) else 0
            for group in by_task.values()
        ]),
    }
    return metrics
