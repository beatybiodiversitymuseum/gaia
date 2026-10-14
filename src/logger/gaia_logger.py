"""Bounded JSON execution traces and interaction feedback for Gaia."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, cast

from src.gaia_agent.runtime.contracts import (
    AgentCancelledError,
    AgentRuntimeError,
    AgentResult,
    ChatMessage,
    ContextInfo,
    ModelRequestMetrics,
    ModelTurnRuntime,
    TokenUsage,
    ToolCall,
    ToolBatchRuntime,
    ToolResult,
)


TRACE_VERSION = 5
REDACTED_VALUE = "[REDACTED]"
GaiaLoggerStatus = Literal["completed", "cancelled", "failed"]

MAX_FEEDBACK_REASONS = 8
MAX_FEEDBACK_REASON_CHARACTERS = 128
MAX_FEEDBACK_DETAILS_CHARACTERS = 2_000
# Repository-path normalization can expand each submitted character.
GAIA_FEEDBACK_TRACE_RESERVE_BYTES = 65_536
_MINIMUM_TRACE_BODY_BYTES = 1_024

_SENSITIVE_KEYS = frozenset(
    {
        "accesstoken",
        "apikey",
        "authorization",
        "clientsecret",
        "cookie",
        "password",
        "passwd",
        "privatekey",
        "refreshtoken",
        "secret",
        "setcookie",
    }
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_NAMED_SECRET_PATTERN = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|"
    r"passwd|secret|client[_-]?secret|authorization|cookie)\b\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_MISSING_TOOL_RESULT = {
    "status": "missing",
    "reason": "no structured tool result was recorded",
}

# ---------------------------------------------------------------------------
# Trace contracts
# ---------------------------------------------------------------------------


class GaiaLoggerError(RuntimeError):
    """Indicate an invalid trace record or persistence operation."""


@dataclass(frozen=True, slots=True)
class GaiaLoggerLimits:
    """Bound trace values, interaction collections, and total file size."""

    max_text_characters: int = 8_192
    max_json_characters: int = 32_768
    max_collection_items: int = 1_024
    max_total_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        for field_name in (
            "max_text_characters",
            "max_json_characters",
            "max_collection_items",
            "max_total_bytes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.max_total_bytes < 1_024:
            raise ValueError("max_total_bytes must be at least 1024")


@dataclass(frozen=True, slots=True)
class GaiaLoggerRecord:
    """Hold one completed, cancelled, or failed interaction."""

    run_id: str
    status: GaiaLoggerStatus
    started_at: str
    ended_at: str
    elapsed_seconds: float
    invocation: dict[str, object]
    transcript: tuple[ChatMessage, ...]
    tool_results: tuple[ToolResult, ...]
    registered_tool_count: int
    model_turns: int
    final_response: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    cause_type: str | None = None
    model_turn_runtime: tuple[ModelTurnRuntime, ...] = ()
    tool_batch_runtime: tuple[ToolBatchRuntime, ...] = ()
    context_info: ContextInfo | None = None

    def __post_init__(self) -> None:
        for field_name in ("run_id", "started_at", "ended_at"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.status not in {"completed", "cancelled", "failed"}:
            raise ValueError(
                "trace status must be 'completed', 'cancelled', or 'failed'"
            )
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be a finite non-negative number")
        for field_name in ("registered_tool_count", "model_turns"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

        invocation = _json_object(self.invocation, label="trace invocation")
        transcript = tuple(self.transcript)
        tool_results = tuple(self.tool_results)
        model_turn_runtime = tuple(self.model_turn_runtime)
        tool_batch_runtime = tuple(self.tool_batch_runtime)
        if any(not isinstance(item, ChatMessage) for item in transcript):
            raise TypeError("trace transcript must contain only ChatMessage values")
        if any(not isinstance(item, ToolResult) for item in tool_results):
            raise TypeError("trace tool results must contain only ToolResult values")
        if any(
            not isinstance(item, ModelTurnRuntime)
            for item in model_turn_runtime
        ):
            raise TypeError(
                "trace model runtime must contain only ModelTurnRuntime values"
            )
        if any(
            not isinstance(item, ToolBatchRuntime)
            for item in tool_batch_runtime
        ):
            raise TypeError(
                "trace tool batch runtime must contain only ToolBatchRuntime values"
            )
        object.__setattr__(self, "invocation", invocation)
        object.__setattr__(self, "transcript", transcript)
        object.__setattr__(self, "tool_results", tool_results)
        object.__setattr__(self, "model_turn_runtime", model_turn_runtime)
        object.__setattr__(self, "tool_batch_runtime", tool_batch_runtime)
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))

        if self.status == "completed":
            if (
                not isinstance(self.final_response, str)
                or not self.final_response.strip()
            ):
                raise ValueError("completed traces require a non-empty final response")
            if any(
                value is not None
                for value in (self.error_type, self.error_message, self.cause_type)
            ):
                raise ValueError("completed traces cannot contain error fields")
            return
        if self.status == "cancelled":
            if self.final_response is not None:
                raise ValueError(
                    "cancelled traces cannot contain a final response"
                )
            if any(
                value is not None
                for value in (self.error_type, self.error_message, self.cause_type)
            ):
                raise ValueError("cancelled traces cannot contain error fields")
            return
        if self.final_response is not None:
            raise ValueError("failed traces cannot contain a final response")
        if not isinstance(self.error_type, str) or not self.error_type.strip():
            raise ValueError("failed traces require a non-empty error type")
        if not isinstance(self.error_message, str):
            raise TypeError("failed traces require an error message")
        if self.cause_type is not None and not isinstance(self.cause_type, str):
            raise TypeError("trace cause type must be a string or None")


@dataclass(frozen=True, slots=True)
class GaiaLoggerConfig:
    """Describe one repository-local, bounded agent trace."""

    path: Path
    run_id: str
    repository_root: Path
    invocation: dict[str, object] = field(default_factory=dict)
    limits: GaiaLoggerLimits = field(default_factory=GaiaLoggerLimits)
    feedback_enabled: bool = False

    def __post_init__(self) -> None:
        path = Path(self.path)
        repository_root = Path(self.repository_root)
        if not path.name:
            raise ValueError("trace path must identify a file")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("trace run id must be a non-empty string")
        if not isinstance(self.limits, GaiaLoggerLimits):
            raise TypeError("trace limits must be a GaiaLoggerLimits value")
        if not isinstance(self.feedback_enabled, bool):
            raise TypeError("feedback_enabled must be a boolean")
        if self.feedback_enabled and self.limits.max_total_bytes < (
            GAIA_FEEDBACK_TRACE_RESERVE_BYTES + _MINIMUM_TRACE_BODY_BYTES
        ):
            raise ValueError(
                "Gaia trace byte limit must include feedback capacity"
            )
        invocation = _json_object(self.invocation, label="trace invocation")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "repository_root", repository_root)
        object.__setattr__(self, "invocation", invocation)

# ---------------------------------------------------------------------------
# Trace construction and persistence
# ---------------------------------------------------------------------------


def create_chat_trace_config(
    *,
    chat_directory: Path,
    interaction: int,
    repository_root: Path,
    invocation: dict[str, object],
    feedback_enabled: bool = False,
) -> GaiaLoggerConfig:
    """Build the shared trace target and metadata for one chat interaction."""
    label = f"{interaction:04d}"
    return GaiaLoggerConfig(
        path=chat_directory / f"interaction_{label}.json",
        run_id=f"{chat_directory.name}-interaction-{label}",
        repository_root=repository_root,
        invocation={**invocation, "interaction": interaction},
        feedback_enabled=feedback_enabled,
    )


def build_trace_record(
    outcome: AgentResult | BaseException,
    *,
    run_id: str,
    started_at: str,
    ended_at: str,
    elapsed_seconds: float,
    invocation: dict[str, object] | None = None,
    initial_messages: Iterable[ChatMessage] = (),
    registered_tool_count: int = 0,
    history_message_count: int = 0,
) -> GaiaLoggerRecord:
    """Capture a Gaia outcome, excluding completed conversation history."""
    if not isinstance(outcome, (AgentResult, BaseException)):
        raise TypeError("outcome must be an AgentResult or exception")
    if (
        isinstance(history_message_count, bool)
        or not isinstance(history_message_count, int)
        or history_message_count < 0
    ):
        raise ValueError("history_message_count must be a non-negative integer")
    run = outcome if isinstance(outcome, (AgentResult, AgentRuntimeError)) else None
    status: GaiaLoggerStatus
    if isinstance(outcome, AgentResult):
        status = "completed"
    elif isinstance(outcome, AgentCancelledError):
        status = "cancelled"
    else:
        status = "failed"
    transcript = run.transcript if run is not None else tuple(initial_messages)
    error = (
        outcome if isinstance(outcome, BaseException) and status == "failed" else None
    )
    return GaiaLoggerRecord(
        run_id=run_id,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        elapsed_seconds=elapsed_seconds,
        invocation=invocation or {},
        transcript=transcript[history_message_count:],
        tool_results=run.tool_results if run is not None else (),
        registered_tool_count=(
            run.registered_tool_count if run is not None else registered_tool_count
        ),
        model_turns=run.model_turns if run is not None else 0,
        final_response=(
            outcome.final_response if isinstance(outcome, AgentResult) else None
        ),
        error_type=type(error).__name__ if error is not None else None,
        error_message=str(error) if error is not None else None,
        cause_type=(
            type(error.__cause__).__name__
            if error is not None and error.__cause__ is not None else None
        ),
        model_turn_runtime=run.model_turn_runtime if run is not None else (),
        tool_batch_runtime=run.tool_batch_runtime if run is not None else (),
        context_info=outcome.context_info if isinstance(outcome, AgentResult) else None,
    )


def serialize_agent_trace(
    record: GaiaLoggerRecord,
    *,
    limits: GaiaLoggerLimits | None = None,
    repository_root: Path | None = None,
) -> str:
    """Serialize one interaction as bounded, deterministic, two-space JSON."""
    if not isinstance(record, GaiaLoggerRecord):
        raise TypeError("record must be a GaiaLoggerRecord")
    active_limits = limits if limits is not None else GaiaLoggerLimits()
    if not isinstance(active_limits, GaiaLoggerLimits):
        raise TypeError("limits must be a GaiaLoggerLimits or None")
    root = str(Path(repository_root)) if repository_root is not None else ""
    document, omitted = _interaction_document(record, active_limits, root)
    return _fit_total_bytes(document, omitted, active_limits)


def write_gaia_trace(config: GaiaLoggerConfig, record: GaiaLoggerRecord) -> None:
    """Atomically persist a Gaia interaction, reserving room for feedback."""
    if not isinstance(config, GaiaLoggerConfig):
        raise TypeError("config must be a GaiaLoggerConfig")
    limits = config.limits
    if config.feedback_enabled:
        limits = replace(
            limits,
            max_total_bytes=limits.max_total_bytes - GAIA_FEEDBACK_TRACE_RESERVE_BYTES,
        )
    text = serialize_agent_trace(
        record, limits=limits, repository_root=config.repository_root
    )
    _replace_owner_only(config.path, text.encode("utf-8"))


def write_gaia_trace_feedback(
    config: GaiaLoggerConfig,
    *,
    interaction: int,
    rating: str,
    reasons: tuple[str, ...],
    details: str,
    submitted_at: str,
) -> None:
    """Atomically add or replace feedback on one completed interaction trace."""
    if not isinstance(config, GaiaLoggerConfig):
        raise TypeError("config must be a GaiaLoggerConfig")
    if not config.feedback_enabled:
        raise ValueError("trace config does not enable feedback")
    try:
        document = json.loads(config.path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("trace root must be a JSON object")
        run = document.get("run")
        if not isinstance(run, dict) or run.get("status") != "completed":
            raise ValueError("trace must describe a completed run")
        invocation = run.get("invocation")
        linked_interaction = (
            invocation.get("interaction")
            if isinstance(invocation, dict)
            else None
        )
        if (
            isinstance(linked_interaction, bool)
            or not isinstance(linked_interaction, int)
            or linked_interaction != interaction
        ):
            raise ValueError("trace interaction does not match")
        document["user_feedback"] = _sanitize_value(
            {
                "rating": rating,
                "reasons": list(reasons),
                "details": details,
                "submitted_at": submitted_at,
            },
            str(config.repository_root),
        )
        payload = _pretty_json(document).encode("utf-8")
        if len(payload) > config.limits.max_total_bytes:
            raise ValueError("feedback exceeds the trace byte limit")
        _replace_owner_only(config.path, payload)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise GaiaLoggerError(
            "Gaia interaction feedback could not be recorded"
        ) from error


def _replace_owner_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

# ---------------------------------------------------------------------------
# Interaction document
# ---------------------------------------------------------------------------


def _interaction_document(
    record: GaiaLoggerRecord,
    limits: GaiaLoggerLimits,
    repository_root: str,
) -> tuple[dict[str, object], dict[str, int]]:
    user_messages = [
        message.content
        for message in record.transcript
        if message.role == "user"
    ]
    omitted = {
        "user_prompts": max(0, len(user_messages) - 1),
        "reasoning_traces": 0,
        "tools_requested": 0,
        "model_turn_runtime": 0,
        "tool_batches": 0,
    }
    user_prompt = (
        _trace_value(
            user_messages[-1],
            limits.max_text_characters,
            repository_root,
        )
        if user_messages
        else None
    )

    reasoning_traces, tool_requests = _interaction_steps(
        record,
        limits,
        repository_root,
    )
    reasoning_total = len(reasoning_traces)
    reasoning_traces, reasoning_omitted = _limit_collection(
        reasoning_traces,
        limits.max_collection_items,
    )
    tool_requests, tools_omitted = _limit_collection(
        tool_requests,
        limits.max_collection_items,
    )
    omitted["reasoning_traces"] += reasoning_omitted
    omitted["tools_requested"] += tools_omitted

    model_runtime = [
        _model_runtime_entry(item, limits)
        for item in record.model_turn_runtime
    ]
    tool_batches = [
        {
            "turn": item.turn,
            "tool_calls": item.tool_calls,
            "wall_elapsed_seconds": item.wall_elapsed_seconds,
        }
        for item in record.tool_batch_runtime
    ]
    model_runtime, model_runtime_omitted = _limit_collection(
        model_runtime,
        limits.max_collection_items,
    )
    tool_batches, tool_batches_omitted = _limit_collection(
        tool_batches,
        limits.max_collection_items,
    )
    omitted["model_turn_runtime"] += model_runtime_omitted
    omitted["tool_batches"] += tool_batches_omitted

    run = _run_document(
        record, limits, repository_root, len(user_messages), reasoning_total
    )
    returned_result = _returned_result_document(record, limits, repository_root)

    document: dict[str, object] = {
        "run": run,
        "runtime": {
            "token_summary": _token_summary_document(
                record.model_turns,
                record.model_turn_runtime,
            ),
            "model_turns": model_runtime,
            "tool_batches": tool_batches,
        },
        "user_prompt": user_prompt,
        "reasoning_traces": reasoning_traces,
        "tools_requested": tool_requests,
        "returned_result": returned_result,
    }
    if record.context_info is not None:
        runtime_document = cast(dict[str, object], document["runtime"])
        runtime_document["context"] = {
            "prompt_tokens": record.context_info.prompt_tokens,
            "limit_tokens": record.context_info.context_limit_tokens,
        }
    _sync_omissions(run, omitted)
    return document, omitted


def _run_document(
    record: GaiaLoggerRecord,
    limits: GaiaLoggerLimits,
    repository_root: str,
    user_message_count: int,
    reasoning_total: int,
) -> dict[str, object]:
    invocation = _trace_value(
        record.invocation,
        limits.max_json_characters,
        repository_root,
    )
    run: dict[str, object] = {
        "trace_version": TRACE_VERSION,
        "run_id": _bounded_metadata_text(record.run_id, limits),
        "status": record.status,
        "started_at": _bounded_metadata_text(record.started_at, limits),
        "ended_at": _bounded_metadata_text(record.ended_at, limits),
        "elapsed_seconds": record.elapsed_seconds,
        "counts": {
            "model_turns": record.model_turns,
            "messages": len(record.transcript),
            "user_messages": user_message_count,
            "reasoning_traces": reasoning_total,
            "tool_requests": sum(
                len(message.tool_calls)
                for message in record.transcript
                if message.role == "assistant"
            ),
            "tool_results": len(record.tool_results),
            "registered_tools": record.registered_tool_count,
            "model_runtime_records": len(record.model_turn_runtime),
            "tool_batches": len(record.tool_batch_runtime),
        },
        "invocation": invocation,
    }

    return run


def _returned_result_document(
    record: GaiaLoggerRecord,
    limits: GaiaLoggerLimits,
    repository_root: str,
) -> dict[str, object]:
    returned_result: dict[str, object]
    if record.status == "completed":
        returned_result = {
            "status": "completed",
            "content": _trace_value(
                record.final_response,
                limits.max_text_characters,
                repository_root,
            ),
        }
    elif record.status == "failed":
        error: dict[str, object] = {
            "type": _bounded_metadata_text(record.error_type or "", limits),
            "message": _trace_value(
                record.error_message,
                limits.max_text_characters,
                repository_root,
            ),
        }
        if record.cause_type is not None:
            error["cause_type"] = _bounded_metadata_text(
                record.cause_type,
                limits,
            )
        returned_result = {"status": "failed", "error": error}
    else:
        returned_result = {"status": "cancelled"}

    return returned_result


def _model_runtime_entry(
    runtime: ModelTurnRuntime,
    limits: GaiaLoggerLimits,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "turn": runtime.turn,
        "status": runtime.status,
        "client_round_trip_seconds": runtime.client_round_trip_seconds,
    }
    if runtime.purpose == "compaction":
        entry["purpose"] = runtime.purpose
    if runtime.response_id is not None:
        entry["response_id"] = _bounded_metadata_text(
            runtime.response_id,
            limits,
        )
    if runtime.usage is not None:
        entry["token_usage"] = _token_usage_document(runtime.usage)
    if runtime.provider_metrics is not None:
        entry["provider_metrics"] = _provider_metrics_document(
            runtime.provider_metrics
        )
    if runtime.error_type is not None:
        entry["error_type"] = _bounded_metadata_text(
            runtime.error_type,
            limits,
        )
    return entry


def _token_usage_document(usage: TokenUsage) -> dict[str, int]:
    document = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    if usage.cached_prompt_tokens is not None:
        document["cached_prompt_tokens"] = usage.cached_prompt_tokens
    if usage.reasoning_tokens is not None:
        document["reasoning_tokens"] = usage.reasoning_tokens
    return document


def _token_summary_document(
    model_turns: int,
    model_runtime: tuple[ModelTurnRuntime, ...],
) -> dict[str, object]:
    usages = tuple(
        runtime.usage
        for runtime in model_runtime
        if runtime.usage is not None
    )
    reported_turns = len(usages)
    summary: dict[str, object] = {
        "coverage": _token_coverage(model_turns, reported_turns),
        "model_turns_total": model_turns,
        "model_turns_with_usage": reported_turns,
        "model_turns_without_usage": max(0, model_turns - reported_turns),
    }
    if not usages:
        return summary

    summary["reported_totals"] = {
        "prompt_tokens": sum(usage.prompt_tokens for usage in usages),
        "generated_tokens": sum(usage.completion_tokens for usage in usages),
        "combined_tokens": sum(usage.total_tokens for usage in usages),
    }
    for field_name in ("cached_prompt_tokens", "reasoning_tokens"):
        values = tuple(
            value
            for usage in usages
            if (value := getattr(usage, field_name)) is not None
        )
        if values:
            summary[field_name] = _optional_token_summary(
                values,
                model_turns=model_turns,
            )
    return summary


def _optional_token_summary(
    values: tuple[int, ...],
    *,
    model_turns: int,
) -> dict[str, object]:
    reported_turns = len(values)
    return {
        "coverage": _token_coverage(model_turns, reported_turns),
        "model_turns_with_value": reported_turns,
        "model_turns_without_value": max(0, model_turns - reported_turns),
        "reported_tokens": sum(values),
    }


def _token_coverage(model_turns: int, reported_turns: int) -> str:
    if reported_turns == 0:
        return "unavailable"
    if reported_turns == model_turns:
        return "complete"
    return "partial"


def _provider_metrics_document(
    metrics: ModelRequestMetrics,
) -> dict[str, object]:
    document: dict[str, object] = {}
    if metrics.provider is not None:
        document["provider"] = metrics.provider
    for field_name in (
        "time_to_first_token_seconds",
        "generation_seconds",
        "queue_seconds",
        "mean_inter_token_latency_seconds",
        "tokens_per_second",
    ):
        value = getattr(metrics, field_name)
        if value is not None:
            document[field_name] = value
    return document


def _interaction_steps(
    record: GaiaLoggerRecord,
    limits: GaiaLoggerLimits,
    repository_root: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    results_by_call_id = {
        result.call.id: result for result in record.tool_results
    }

    reasoning_traces: list[dict[str, object]] = []
    tool_requests: list[dict[str, object]] = []
    response_turns = iter(
        runtime.turn for runtime in record.model_turn_runtime
        if runtime.purpose == "response" and runtime.status == "completed"
    )
    turn = 0
    for message in record.transcript:
        if message.role != "assistant":
            continue
        turn = next(response_turns, turn + 1)
        if message.reasoning_content is not None:
            reasoning_traces.append(
                {
                    "turn": turn,
                    "content": _trace_value(
                        message.reasoning_content,
                        limits.max_text_characters,
                        repository_root,
                    ),
                }
            )
        for call in message.tool_calls:
            result = results_by_call_id.pop(call.id, None)
            tool_requests.append(
                _tool_request(
                    call,
                    turn=turn,
                    result=result,
                    limits=limits,
                    repository_root=repository_root,
                )
            )

    return reasoning_traces, tool_requests


def _tool_request(
    call: ToolCall,
    *,
    turn: int,
    result: ToolResult | None,
    limits: GaiaLoggerLimits,
    repository_root: str,
) -> dict[str, object]:
    tool_name = call.name
    arguments = _trace_value(
        call.arguments,
        limits.max_json_characters,
        repository_root,
    )
    if result is None:
        returned_result: object = dict(_MISSING_TOOL_RESULT)
    else:
        returned_result = _trace_value(
            result.payload(),
            limits.max_json_characters,
            repository_root,
        )
    request = {
        "turn": turn,
        "call_id": _bounded_metadata_text(call.id, limits),
        "name": _bounded_metadata_text(tool_name, limits),
        "arguments": arguments,
        "returned_result": returned_result,
    }
    if result is not None and result.elapsed_seconds is not None:
        request["elapsed_seconds"] = result.elapsed_seconds
    return request


def _limit_collection(
    values: list[dict[str, object]],
    limit: int,
) -> tuple[list[dict[str, object]], int]:
    if len(values) <= limit:
        return values, 0
    return values[:limit], len(values) - limit


def _fit_total_bytes(
    document: dict[str, object],
    omitted: dict[str, int],
    limits: GaiaLoggerLimits,
) -> str:
    text = _pretty_json(document)
    run = cast(dict[str, object], document["run"])
    while len(text.encode("utf-8")) > limits.max_total_bytes:
        collection_name, remove_index = _largest_optional_item(document)
        if collection_name is None or remove_index is None:
            break
        collection = _trace_collection(document, collection_name)
        collection.pop(remove_index)
        omitted[collection_name] += 1
        _sync_omissions(run, omitted)
        text = _pretty_json(document)

    if len(text.encode("utf-8")) <= limits.max_total_bytes:
        return text

    compactors = (
        lambda: _compact_mapping_field(run, "invocation"),
        lambda: _compact_mapping_field(document, "user_prompt"),
        lambda: _compact_returned_result(document),
        lambda: _compact_run_identity(run),
    )
    for compact in compactors:
        compact()
        text = _pretty_json(document)
        if len(text.encode("utf-8")) <= limits.max_total_bytes:
            return text
    raise GaiaLoggerError("trace bounds cannot retain required interaction fields")


def _largest_optional_item(
    document: dict[str, object],
) -> tuple[str | None, int | None]:
    candidates: list[tuple[int, int, str, int]] = []
    for collection_priority, name in enumerate(
        (
            "reasoning_traces",
            "tools_requested",
            "model_turn_runtime",
            "tool_batches",
        )
    ):
        values = _trace_collection(document, name)
        for index, value in enumerate(values):
            size = len(_pretty_json(value).encode("utf-8"))
            candidates.append((size, -collection_priority, name, index))
    if not candidates:
        return None, None
    _, _, name, index = max(candidates)
    return name, index


def _trace_collection(
    document: dict[str, object],
    name: str,
) -> list[object]:
    if name in {"reasoning_traces", "tools_requested"}:
        return cast(list[object], document[name])
    runtime = cast(dict[str, object], document["runtime"])
    runtime_name = "model_turns" if name == "model_turn_runtime" else name
    return cast(list[object], runtime[runtime_name])


def _sync_omissions(run: dict[str, object], omitted: dict[str, int]) -> None:
    visible = {key: value for key, value in omitted.items() if value}
    if visible:
        run["omitted"] = visible
    else:
        run.pop("omitted", None)


def _compact_mapping_field(
    mapping: dict[str, object],
    field_name: str,
) -> None:
    if field_name not in mapping:
        return
    value = mapping[field_name]
    mapping[field_name] = _total_limit_representation(value)


def _compact_returned_result(
    document: dict[str, object],
) -> None:
    returned = cast(dict[str, object], document["returned_result"])
    if returned.get("status") == "completed":
        _compact_mapping_field(returned, "content")
        return
    if "error" in returned:
        error = cast(dict[str, object], returned["error"])
        _compact_mapping_field(error, "message")


def _compact_run_identity(run: dict[str, object]) -> None:
    for field_name in ("run_id", "started_at", "ended_at"):
        _compact_mapping_field(run, field_name)


def _total_limit_representation(value: object) -> dict[str, object]:
    encoded = _compact_json(value)
    return {
        "truncated": True,
        "reason": "max_total_bytes",
        "sha256": _sha256(encoded),
    }

# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


def _bounded_metadata_text(value: str, limits: GaiaLoggerLimits) -> object:
    return _trace_value(value, min(limits.max_text_characters, 256))


def _trace_value(value: object, limit: int, repository_root: str = "") -> object:
    sanitized = _sanitize_value(value, repository_root)
    encoded = sanitized if isinstance(sanitized, str) else _compact_json(sanitized)
    if len(encoded) <= limit:
        return sanitized
    return {
        "truncated": True,
        "preview": encoded[:limit],
        "original_characters": len(encoded),
        "sha256": _sha256(encoded),
    }


def _sanitize_value(value: object, repository_root: str = "") -> object:
    """Normalize paths and redact secrets in already validated trace values."""
    if isinstance(value, dict):
        return {
            key: (
                REDACTED_VALUE
                if "".join(
                    character for character in key.lower() if character.isalnum()
                )
                in _SENSITIVE_KEYS
                else _sanitize_value(item, repository_root)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item, repository_root) for item in value]
    if isinstance(value, str):
        if repository_root:
            value = value.replace(repository_root, "<repository>")
        value = _BEARER_PATTERN.sub("Bearer [REDACTED]", value)
        return _NAMED_SECRET_PATTERN.sub(
            lambda match: f"{match.group(1)}{REDACTED_VALUE}", value
        )
    return value


def _json_object(value: object, *, label: str) -> dict[str, object]:
    normalized = _json_value(value, label=label)
    if not isinstance(normalized, dict):
        raise TypeError(f"{label} must be a JSON object")
    return normalized


def _json_value(value: object, *, label: str) -> object:
    try:
        return json.loads(
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise GaiaLoggerError(f"{label} must contain only JSON values") from error


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _pretty_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ) + "\n"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
