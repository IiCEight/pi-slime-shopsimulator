"""Per-sample ShopSimulator rollout driven by the real pi harness."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import secrets
from collections import defaultdict
from typing import Any

import aiohttp

from slime.agent.adapters import OpenAIAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from .common import prompt_text
from .pi_harness import InfrastructureError, PiRunResult, run_pi

logger = logging.getLogger(__name__)


def _candidate_fragments(candidate: Sample | list[Sample]) -> list[Sample]:
    return candidate if isinstance(candidate, list) else [candidate]



def validate_complete_groups(args, groups) -> None:
    """Validate candidate boundaries and atomically mask incomplete groups."""
    expected = int(args.n_samples_per_prompt)
    for group in groups:
        if len(group) != expected:
            raise RuntimeError(f"ShopSimulator group has {len(group)} candidates, expected {expected}")
        group_index = None
        candidate_ids = set()
        group_fragments = []
        drop_reasons = []
        for candidate in group:
            fragments = _candidate_fragments(candidate)
            if not fragments:
                raise RuntimeError("ShopSimulator candidate has no training fragments")
            first = fragments[0]
            candidate_id = first.rollout_id if first.rollout_id is not None else first.index
            if candidate_id is None or candidate_id in candidate_ids:
                raise RuntimeError(f"invalid or duplicate ShopSimulator candidate id: {candidate_id}")
            candidate_ids.add(candidate_id)
            if group_index is None:
                group_index = first.group_index
            for fragment in fragments:
                fragment_id = fragment.rollout_id if fragment.rollout_id is not None else fragment.index
                if fragment.group_index != group_index or fragment_id != candidate_id:
                    raise RuntimeError("ShopSimulator fan-out changed group or candidate identity")
                if fragment.remove_sample or fragment.status == Sample.Status.ABORTED:
                    drop_reasons.append(f"candidate {candidate_id} aborted")
            group_fragments.extend(fragments)

        if drop_reasons:
            reason = "; ".join(sorted(set(drop_reasons)))
            for fragment in group_fragments:
                fragment.remove_sample = True
                fragment.metadata = dict(fragment.metadata or {})
                fragment.metadata["group_filtered"] = True
                fragment.metadata["group_filter_reason"] = reason
            logger.warning("Filter complete ShopSimulator group %s: %s", group_index, reason)
            continue

        for candidate in group:
            fragments = _candidate_fragments(candidate)
            candidate_id = fragments[0].rollout_id
            if candidate_id is None:
                candidate_id = fragments[0].index
            if sum(sum(fragment.loss_mask or []) for fragment in fragments) <= 0:
                raise RuntimeError(f"ShopSimulator candidate {candidate_id} has no trainable action tokens")


def normalize_candidate_group_rewards(args, samples: list[Sample]):
    """Normalize complete candidate groups; filtered groups receive zero advantage."""
    expected = int(args.n_samples_per_prompt)
    raw_rewards = [float(sample.get_reward_value(args)) for sample in samples]
    grouped: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for position, sample in enumerate(samples):
        if sample.group_index is None:
            raise RuntimeError("ShopSimulator sample is missing group_index")
        candidate_id = sample.rollout_id if sample.rollout_id is not None else sample.index
        if candidate_id is None:
            raise RuntimeError("ShopSimulator sample is missing candidate identity")
        grouped[int(sample.group_index)][int(candidate_id)].append(position)

    normalized = [0.0] * len(samples)
    use_std = bool(getattr(args, "grpo_std_normalization", True))
    nonzero_variance_groups = 0
    for group_index, candidates in grouped.items():
        if len(candidates) != expected:
            raise RuntimeError(
                f"ShopSimulator group {group_index} has {len(candidates)} candidates, expected {expected}"
            )
        group_positions = [position for positions in candidates.values() for position in positions]
        removed = [
            samples[position].remove_sample or samples[position].status == Sample.Status.ABORTED
            for position in group_positions
        ]
        if any(removed):
            if not all(samples[position].remove_sample for position in group_positions):
                raise RuntimeError(f"ShopSimulator group {group_index} was only partially filtered")
            continue

        candidate_rewards = []
        for candidate_id, positions in candidates.items():
            values = [raw_rewards[position] for position in positions]
            if not all(math.isfinite(value) for value in values) or any(value != values[0] for value in values[1:]):
                raise RuntimeError(f"inconsistent reward across candidate {candidate_id} fan-out fragments")
            candidate_rewards.append((candidate_id, values[0], positions))
        if max(value for _, value, _ in candidate_rewards) != min(value for _, value, _ in candidate_rewards):
            nonzero_variance_groups += 1
        mean = sum(value for _, value, _ in candidate_rewards) / expected
        centered = [value - mean for _, value, _ in candidate_rewards]
        if use_std:
            std = math.sqrt(sum(value * value for value in centered) / max(expected - 1, 1))
            centered = [value / (std + 1e-6) for value in centered]
        for (_, _, positions), advantage in zip(candidate_rewards, centered, strict=True):
            for position in positions:
                normalized[position] = advantage
    require_signal_per_rollout = os.environ.get(
        "SHOP_REQUIRE_NONZERO_VARIANCE_PER_ROLLOUT", "1"
    ) not in {"0", "false", "False"}
    if (
        require_signal_per_rollout
        and not getattr(args, "debug_rollout_only", False)
        and nonzero_variance_groups == 0
    ):
        raise RuntimeError("zero-signal ShopSimulator rollout: no complete candidate group has reward variance")
    return raw_rewards, normalized


class AdapterService(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        public_host = os.environ.get("ADAPTER_PUBLIC_HOST", "127.0.0.1")
        bind_host = os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0")
        port = int(os.environ.get("ADAPTER_PORT", "18080"))
        max_turns = int(os.environ.get("SHOP_MAX_TURNS", "40"))
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        self.max_response_len = int(getattr(args, "rollout_max_response_len", 0) or 0)
        self.max_model_turns = max_turns
        self.adapter = OpenAIAdapter(
            tokenizer=tokenizer,
            sglang_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
            tool_parser=getattr(args, "sglang_tool_call_parser", None) or None,
            reasoning_parser=getattr(args, "sglang_reasoning_parser", None) or None,
            max_turns_per_sid=max_turns,
        )
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=bind_host,
            port=port,
            thread_name="shop-openai-adapter",
            runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
        )
        self.adapter_url = f"http://{public_host}:{self.app_handle.port}"


def make_unique_session_id(sample: Sample, task_id: int) -> str:
    indices = f"{sample.index}-{sample.group_index}" if sample.index is not None else secrets.token_hex(8)
    return f"shop-{task_id}-{indices}-{secrets.token_hex(4)}"


def abort_sample(
    sample: Sample,
    reason: str,
    *,
    error_kind: str = "infrastructure_error",
    error_message: str | None = None,
) -> list[Sample]:
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    sample.remove_sample = True
    sample.status = Sample.Status.ABORTED
    sample.metadata = {
        **(sample.metadata or {}),
        "abort_reason": reason,
        "error_kind": error_kind,
        "error_message": error_message or reason,
    }
    return [sample]


async def release_env(env_url: str, env_idx: int, rollout_session_id: str) -> None:
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as client:
            async with client.post(env_url, json={
                "action": "release_one",
                "env_idx": env_idx,
                "rollout_session_id": rollout_session_id,
            }) as response:
                body = await response.json(content_type=None)
                result = body.get("result", {}) if isinstance(body, dict) else {}
                if response.status >= 400 or (isinstance(result, dict) and result.get("error")):
                    logger.warning("failed to release ShopSimulator env %s: %s", env_idx, body)
    except Exception as exc:
        logger.warning("failed to release ShopSimulator env %s: %s", env_idx, exc)


def _sample_prompt(sample: Sample) -> str:
    try:
        return prompt_text(sample.prompt)
    except ValueError as exc:
        raise InfrastructureError(f"sample prompt is unusable: {exc}") from exc


def _task_id(sample: Sample) -> int:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if "task_id" not in metadata:
        raise InfrastructureError("sample metadata is missing task_id")
    try:
        value = int(metadata["task_id"])
    except (TypeError, ValueError) as exc:
        raise InfrastructureError("sample metadata.task_id is not an integer") from exc
    if value < 0:
        raise InfrastructureError("sample metadata.task_id must be non-negative")
    return value


async def finish_candidate_session(
    state: AdapterService,
    session_id: str,
    *,
    base_sample: Sample,
    task_id: int,
    result: PiRunResult,
) -> list[Sample]:
    """Convert a completed pi process into trainable fragments.

    A turn cap is an environment-horizon truncation, not an infrastructure
    failure. The adapter owns the structured signal so a generic pi error or
    unrelated HTTP 429 can never be mistaken for a valid truncation.
    """
    termination_reason = state.adapter.session_termination_reason(session_id)
    model_turns = state.adapter.session_turn_count(session_id)
    terminal = result.done or result.over
    # A trustworthy environment result has highest priority. The adapter turn
    # cap is next; expected Pi cancellation at either boundary is non-fatal.
    turn_limited = termination_reason == "turn_limit" and not terminal
    fatal_error = (
        result.error_kind in {"infrastructure_error", "model_or_process_error"}
        or (result.error is not None and result.error_kind is None)
        or result.exit_code != 0
    )
    if fatal_error and not terminal and not turn_limited:
        raise InfrastructureError(result.error or f"pi exited {result.exit_code}", result=result)

    if result.done:
        termination_reason = "environment_done"
        outcome_kind = "environment_terminal"
    elif result.over:
        termination_reason = "environment_over"
        outcome_kind = "environment_terminal"
    elif turn_limited:
        termination_reason = "turn_limit"
        outcome_kind = "turn_limit"
    elif result.tool_errors:
        termination_reason = "agent_tool_error"
        outcome_kind = "agent_tool_error"
    else:
        termination_reason = termination_reason or "agent_stop"
        outcome_kind = "agent_stop"
    reward = float(result.reward) if result.done else 0.0
    error_message = result.error
    if error_message is None and result.tool_errors:
        error_message = result.tool_errors[-1]["message"]
    samples = await state.adapter.finish_session(
        session_id,
        base_sample=base_sample,
        reward=reward,
        extra_metadata={
            "task_id": task_id,
            "env_done": result.done,
            "env_over": result.over,
            "env_idx": result.env_idx,
            "reward_detail": result.reward_detail,
            "purchase_asin": result.purchase_asin,
            "goal_asin": result.goal_asin,
            "pi_exit_code": result.exit_code,
            "pi_error": result.error,
            "error_kind": outcome_kind,
            "error_message": error_message,
            "pi_tool_errors": result.tool_errors,
            "pi_tool_calls": result.tool_calls,
            "model_turns": model_turns,
            "max_model_turns": state.max_model_turns,
            "termination_reason": termination_reason,
            "truncated": turn_limited,
        },
    )
    if turn_limited:
        for sample in samples:
            sample.status = Sample.Status.TRUNCATED
            sample.remove_sample = False
            sample.metadata = dict(sample.metadata or {})
            sample.metadata["truncated"] = True
    return samples


async def generate(args, base_sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    del evaluation  # ShopSimulator uses the same environment reward in train/eval.
    try:
        task_id = _task_id(base_sample)
        state = AdapterService(args)
    except InfrastructureError as exc:
        return abort_sample(
            base_sample,
            f"infrastructure:{exc}",
            error_message=str(exc),
        )
    except Exception as exc:
        return abort_sample(
            base_sample,
            f"infrastructure:{type(exc).__name__}: {exc}",
            error_message=str(exc),
        )

    session_id = make_unique_session_id(base_sample, task_id)
    base_sample.session_id = session_id
    state.adapter.open_session(
        session_id,
        sampling_defaults=sampling_params,
        max_context_tokens=state.max_context_len,
    )
    result: PiRunResult | None = None
    env_url = os.environ.get("SHOP_ENV_URL", "http://127.0.0.1:5000/api/shop_agent")
    timeout_sec = float(os.environ.get("SHOP_ROLLOUT_TIMEOUT_SEC", "600"))
    try:
        result = await run_pi(
            session_id=session_id,
            task_id=task_id,
            adapter_url=state.adapter_url,
            env_url=env_url,
            pi_bin=os.environ.get("PI_BIN", "pi"),
            prompt=_sample_prompt(base_sample),
            timeout_sec=timeout_sec,
            context_window=state.max_context_len,
            max_tokens=state.max_response_len,
        )
        samples = await finish_candidate_session(
            state,
            session_id,
            base_sample=base_sample,
            task_id=task_id,
            result=result,
        )
        if not samples:
            return abort_sample(
                base_sample,
                "adapter_session_empty",
                error_message="adapter returned no trainable fragments",
            )
        return samples
    except InfrastructureError as exc:
        if exc.result is not None:
            result = exc.result
        error_kind = result.error_kind if result is not None else "infrastructure_error"
        if error_kind not in {"infrastructure_error", "model_or_process_error"}:
            error_kind = "infrastructure_error"
        return abort_sample(
            base_sample,
            f"infrastructure:{exc}",
            error_kind=error_kind,
            error_message=str(exc),
        )
    except Exception as exc:
        logger.exception("ShopSimulator rollout failed for task %s", task_id)
        return abort_sample(
            base_sample,
            f"infrastructure:{type(exc).__name__}: {exc}",
            error_message=str(exc),
        )
    finally:
        if result is not None and result.env_idx is not None and not (result.done or result.over):
            await release_env(env_url, result.env_idx, session_id)
        await state.adapter.drop_session(session_id, wait_timeout=30)
