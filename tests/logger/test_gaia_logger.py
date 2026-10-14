"""Tests for concise, bounded, redacted agent interaction traces."""

from __future__ import annotations

import asyncio
import json
import stat
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.gaia_agent.runtime.contracts import (
    AgentCancelledError,
    AgentResult,
    ChatMessage,
    ContextInfo,
    ModelClientExecutionError,
    ModelRequestMetrics,
    ModelTurn,
    ModelTurnRuntime,
    RegisteredTool,
    ToolCall,
    ToolBatchRuntime,
    ToolDefinition,
    ToolResult,
    TokenUsage,
)
from src.gaia_agent.runtime.loop import AgentRuntime
from src.logger.gaia_logger import (
    GaiaLoggerError,
    GaiaLoggerRecord,
    GAIA_FEEDBACK_TRACE_RESERVE_BYTES,
    GaiaLoggerConfig,
    MAX_FEEDBACK_DETAILS_CHARACTERS,
    MAX_FEEDBACK_REASON_CHARACTERS,
    MAX_FEEDBACK_REASONS,
    GaiaLoggerLimits,
    build_trace_record,
    serialize_agent_trace,
    write_gaia_trace,
    write_gaia_trace_feedback,
)


def _tool_definition(name: str = "hard_filter_occurrences") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Search trusted records. This schema must not enter the trace.",
        parameters={
            "type": "object",
            "properties": {"filters": {"type": "object"}},
            "required": ["filters"],
        },
        strict=True,
    )


def test_compaction_usage_keeps_response_trace_turns_and_active_context() -> None:
    result = replace(
        _completed_result(), model_turns=3,
        context_info=ContextInfo(128, 131_072),
        model_turn_runtime=(
            ModelTurnRuntime(1, "completed", 0.1, purpose="compaction"),
            ModelTurnRuntime(2, "completed", 0.2),
            ModelTurnRuntime(3, "completed", 0.3),
        ),
    )
    document = _document(serialize_agent_trace(_trace_record(result)))
    assert document["runtime"]["model_turns"][0]["purpose"] == "compaction"
    assert document["tools_requested"][0]["turn"] == 2
    assert document["reasoning_traces"][0]["turn"] == 2
    assert document["runtime"]["context"] == {
        "prompt_tokens": 128, "limit_tokens": 131_072,
    }


def _completed_result() -> AgentResult:
    definition = _tool_definition()
    call = ToolCall(
        "call-1",
        definition.name,
        {"filters": {"country": "Canada"}, "api_key": "do-not-persist"},
    )
    result = ToolResult(
        call=call,
        ok=True,
        result={
            "match_count": 1,
            "matches": [
                {
                    "dataset_key": "ctc_birds",
                    "record": {"scientificName": "Corvus corax"},
                }
            ],
        },
    )
    return AgentResult(
        context=(),
        final_response="One matching raven record.",
        transcript=(
            ChatMessage(role="system", content="Large private system prompt."),
            ChatMessage(role="user", content="Find birds; api_key=do-not-persist"),
            ChatMessage(
                role="assistant",
                content=None,
                reasoning_content="I should use the explicit filter tool.",
                tool_calls=(call,),
            ),
            ChatMessage(
                role="tool",
                content=result.to_json(),
                tool_call_id=call.id,
                name=call.name,
            ),
            ChatMessage(
                role="assistant",
                content="One matching raven record.",
                reasoning_content="The bounded result contains one match.",
            ),
        ),
        tool_results=(result,),
        model_turns=2,
        registered_tool_count=1,
    )


def _trace_record(result: AgentResult | None = None):
    return build_trace_record(
        result or _completed_result(),
        run_id="run-123",
        started_at="2026-08-22T12:00:00Z",
        ended_at="2026-08-22T12:00:01Z",
        elapsed_seconds=1.25,
        invocation={
            "operation": "query",
            "model": "local-model",
            "authorization": "Bearer do-not-persist",
        },
    )


def _document(serialized: str) -> dict[str, object]:
    document = json.loads(serialized)
    assert list(document) == [
        "run",
        "runtime",
        "user_prompt",
        "reasoning_traces",
        "tools_requested",
        "returned_result",
    ]
    assert serialized == (
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    return document


def test_serializes_one_concise_interaction_and_redacts_secrets() -> None:
    record = _trace_record()

    first = serialize_agent_trace(record)
    second = serialize_agent_trace(record)
    document = _document(first)

    assert first == second
    assert document["run"] == {
        "trace_version": 5,
        "run_id": "run-123",
        "status": "completed",
        "started_at": "2026-08-22T12:00:00Z",
        "ended_at": "2026-08-22T12:00:01Z",
        "elapsed_seconds": 1.25,
        "counts": {
            "model_turns": 2,
            "messages": 5,
            "user_messages": 1,
            "reasoning_traces": 2,
            "tool_requests": 1,
            "tool_results": 1,
            "registered_tools": 1,
            "model_runtime_records": 0,
            "tool_batches": 0,
        },
        "invocation": {
            "authorization": "[REDACTED]",
            "model": "local-model",
            "operation": "query",
        },
    }
    assert document["runtime"] == {
        "token_summary": {
            "coverage": "unavailable",
            "model_turns_total": 2,
            "model_turns_with_usage": 0,
            "model_turns_without_usage": 2,
        },
        "model_turns": [],
        "tool_batches": [],
    }
    assert document["user_prompt"] == "Find birds; api_key=[REDACTED]"
    assert document["reasoning_traces"] == [
        {"turn": 1, "content": "I should use the explicit filter tool."},
        {"turn": 2, "content": "The bounded result contains one match."},
    ]
    assert document["tools_requested"] == [
        {
            "turn": 1,
            "call_id": "call-1",
            "name": "hard_filter_occurrences",
            "arguments": {
                "api_key": "[REDACTED]",
                "filters": {"country": "Canada"},
            },
            "returned_result": record.tool_results[0].payload(),
        }
    ]
    assert document["returned_result"] == {
        "status": "completed",
        "content": "One matching raven record.",
    }
    assert "Large private system prompt" not in first
    assert "This schema must not enter the trace" not in first
    assert '"parameters"' not in first
    assert first.count("Corvus corax") == 1
    assert first.count("One matching raven record.") == 1
    assert "do-not-persist" not in first
    assert "omitted" not in document["run"]


def test_serializes_partial_usage_provider_and_concurrent_tool_runtime() -> None:
    result = _completed_result()
    timed_result = replace(
        result,
        tool_results=(replace(result.tool_results[0], elapsed_seconds=0.4),),
        model_turn_runtime=(
            ModelTurnRuntime(
                turn=1,
                status="completed",
                client_round_trip_seconds=2.0,
                response_id="chatcmpl-1",
                usage=TokenUsage(
                    prompt_tokens=100,
                    completion_tokens=20,
                    total_tokens=120,
                    cached_prompt_tokens=60,
                    reasoning_tokens=8,
                ),
                provider_metrics=ModelRequestMetrics(
                    provider="vllm",
                    time_to_first_token_seconds=0.25,
                    generation_seconds=1.5,
                    queue_seconds=0.01,
                    mean_inter_token_latency_seconds=0.05,
                    tokens_per_second=20,
                ),
            ),
            ModelTurnRuntime(
                turn=2,
                status="completed",
                client_round_trip_seconds=1.0,
                response_id="chatcmpl-2",
            ),
        ),
        tool_batch_runtime=(
            ToolBatchRuntime(
                turn=1,
                tool_calls=1,
                wall_elapsed_seconds=0.6,
            ),
        ),
    )

    document = _document(serialize_agent_trace(_trace_record(timed_result)))

    assert document["run"]["counts"]["model_runtime_records"] == 2
    assert document["run"]["counts"]["tool_batches"] == 1
    assert document["runtime"] == {
        "token_summary": {
            "coverage": "partial",
            "model_turns_total": 2,
            "model_turns_with_usage": 1,
            "model_turns_without_usage": 1,
            "reported_totals": {
                "prompt_tokens": 100,
                "generated_tokens": 20,
                "combined_tokens": 120,
            },
            "cached_prompt_tokens": {
                "coverage": "partial",
                "model_turns_with_value": 1,
                "model_turns_without_value": 1,
                "reported_tokens": 60,
            },
            "reasoning_tokens": {
                "coverage": "partial",
                "model_turns_with_value": 1,
                "model_turns_without_value": 1,
                "reported_tokens": 8,
            },
        },
        "model_turns": [
            {
                "turn": 1,
                "status": "completed",
                "client_round_trip_seconds": 2.0,
                "response_id": "chatcmpl-1",
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "cached_prompt_tokens": 60,
                    "reasoning_tokens": 8,
                },
                "provider_metrics": {
                    "provider": "vllm",
                    "time_to_first_token_seconds": 0.25,
                    "generation_seconds": 1.5,
                    "queue_seconds": 0.01,
                    "mean_inter_token_latency_seconds": 0.05,
                    "tokens_per_second": 20.0,
                },
            },
            {
                "turn": 2,
                "status": "completed",
                "client_round_trip_seconds": 1.0,
                "response_id": "chatcmpl-2",
            },
        ],
        "tool_batches": [
            {
                "turn": 1,
                "tool_calls": 1,
                "wall_elapsed_seconds": 0.6,
            }
        ],
    }
    assert document["tools_requested"][0]["elapsed_seconds"] == 0.4
    assert "arguments" not in json.dumps(document["runtime"])


def test_token_summary_sums_complete_provider_usage() -> None:
    result = replace(
        _completed_result(),
        model_turn_runtime=(
            ModelTurnRuntime(
                turn=1,
                status="completed",
                client_round_trip_seconds=2.0,
                usage=TokenUsage(
                    prompt_tokens=100,
                    completion_tokens=20,
                    total_tokens=120,
                    cached_prompt_tokens=60,
                    reasoning_tokens=8,
                ),
            ),
            ModelTurnRuntime(
                turn=2,
                status="completed",
                client_round_trip_seconds=1.0,
                usage=TokenUsage(
                    prompt_tokens=150,
                    completion_tokens=30,
                    total_tokens=180,
                    cached_prompt_tokens=100,
                    reasoning_tokens=5,
                ),
            ),
        ),
    )

    summary = _document(serialize_agent_trace(_trace_record(result)))["runtime"][
        "token_summary"
    ]

    assert summary == {
        "coverage": "complete",
        "model_turns_total": 2,
        "model_turns_with_usage": 2,
        "model_turns_without_usage": 0,
        "reported_totals": {
            "prompt_tokens": 250,
            "generated_tokens": 50,
            "combined_tokens": 300,
        },
        "cached_prompt_tokens": {
            "coverage": "complete",
            "model_turns_with_value": 2,
            "model_turns_without_value": 0,
            "reported_tokens": 160,
        },
        "reasoning_tokens": {
            "coverage": "complete",
            "model_turns_with_value": 2,
            "model_turns_without_value": 0,
            "reported_tokens": 13,
        },
    }


def test_token_summary_marks_unavailable_provider_usage() -> None:
    result = replace(
        _completed_result(),
        model_turn_runtime=(
            ModelTurnRuntime(
                turn=1,
                status="completed",
                client_round_trip_seconds=2.0,
            ),
            ModelTurnRuntime(
                turn=2,
                status="completed",
                client_round_trip_seconds=1.0,
            ),
        ),
    )

    summary = _document(serialize_agent_trace(_trace_record(result)))["runtime"][
        "token_summary"
    ]

    assert summary == {
        "coverage": "unavailable",
        "model_turns_total": 2,
        "model_turns_with_usage": 0,
        "model_turns_without_usage": 2,
    }


def test_parallel_calls_keep_request_order_and_correlate_reversed_results() -> None:
    first_call = ToolCall("call-z", "lookup", {
        "filename": "source.xlsx",
        "output_filename": "review.xlsx",
        "operations": [{"operation": "set_cells", "sheet": "Summary", "cells": []}],
    })
    second_call = ToolCall("call-a", "lookup", {
        "filename": "other.xlsx", "sheet": "birds", "range": "A1:D3",
    })
    first_result = ToolResult(first_call, ok=True, result={"answer": "first"})
    second_result = ToolResult(second_call, ok=True, result={"answer": "second"})
    result = AgentResult(
        context=(),
        final_response="done",
        transcript=(
            ChatMessage(role="user", content="Run both."),
            ChatMessage(
                role="assistant",
                content=None,
                reasoning_content="Both requests can run in this turn.",
                tool_calls=(first_call, second_call),
            ),
            ChatMessage(role="assistant", content="done"),
        ),
        tool_results=(second_result, first_result),
        model_turns=2,
    )

    document = _document(serialize_agent_trace(_trace_record(result)))

    assert document["reasoning_traces"] == [
        {"turn": 1, "content": "Both requests can run in this turn."}
    ]
    tools = document["tools_requested"]
    assert [tool["call_id"] for tool in tools] == ["call-z", "call-a"]
    assert list(tools[0]["arguments"]) == ["filename", "output_filename", "operations"]
    assert list(tools[0]["arguments"]["operations"][0]) == ["operation", "sheet", "cells"]
    assert list(tools[1]["arguments"]) == ["filename", "sheet", "range"]
    assert [tool["returned_result"]["result"]["answer"] for tool in tools] == [
        "first",
        "second",
    ]


def test_missing_and_failed_tool_results_are_explicit() -> None:
    missing_call = ToolCall("call-missing", "lookup", {"value": 1})
    failed_call = ToolCall("call-failed", "lookup", {"value": 2})
    failed_result = ToolResult(
        failed_call,
        ok=False,
        error_code="not_found",
        error_message="No match.",
    )
    result = AgentResult(
        context=(),
        final_response="recovered",
        transcript=(
            ChatMessage(role="user", content="Run both."),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=(missing_call, failed_call),
            ),
            ChatMessage(role="assistant", content="recovered"),
        ),
        tool_results=(failed_result,),
        model_turns=2,
    )

    tools = _document(serialize_agent_trace(_trace_record(result)))[
        "tools_requested"
    ]

    assert tools[0]["returned_result"] == {
        "status": "missing",
        "reason": "no structured tool result was recorded",
    }
    assert tools[1]["returned_result"] == failed_result.payload()


def test_collection_bounds_report_omissions_and_keep_order() -> None:
    calls = tuple(
        ToolCall(f"call-{index}", "lookup", {"value": index})
        for index in range(5)
    )
    transcript: list[ChatMessage] = [
        ChatMessage(role="user", content="Older context."),
        ChatMessage(role="user", content="Current request."),
    ]
    for index, call in enumerate(calls):
        transcript.append(
            ChatMessage(
                role="assistant",
                content=None,
                reasoning_content=f"reason-{index}",
                tool_calls=(call,),
            )
        )
    result = AgentResult(
        context=(),
        final_response="done",
        transcript=tuple(transcript),
        tool_results=(),
        model_turns=5,
    )

    serialized = serialize_agent_trace(
        _trace_record(result),
        limits=GaiaLoggerLimits(max_collection_items=2),
    )
    document = _document(serialized)

    assert document["user_prompt"] == "Current request."
    assert [item["turn"] for item in document["reasoning_traces"]] == [1, 2]
    assert [item["call_id"] for item in document["tools_requested"]] == [
        "call-0",
        "call-1",
    ]
    assert document["run"]["omitted"] == {
        "user_prompts": 1,
        "reasoning_traces": 3,
        "tools_requested": 3,
    }


def test_total_byte_bound_can_omit_nested_runtime_collections() -> None:
    result = AgentResult(
        context=(),
        final_response="done",
        transcript=(
            ChatMessage(role="user", content="Run it."),
            ChatMessage(role="assistant", content="done"),
        ),
        tool_results=(),
        model_turns=20,
        model_turn_runtime=tuple(
            ModelTurnRuntime(
                turn=turn,
                status="completed",
                client_round_trip_seconds=float(turn),
                response_id=f"chatcmpl-{turn}-" + ("x" * 120),
                usage=TokenUsage(
                    prompt_tokens=turn,
                    completion_tokens=1,
                    total_tokens=turn + 1,
                ),
            )
            for turn in range(1, 21)
        ),
        tool_batch_runtime=tuple(
            ToolBatchRuntime(
                turn=turn,
                tool_calls=2,
                wall_elapsed_seconds=float(turn) / 10,
            )
            for turn in range(1, 21)
        ),
    )
    limits = GaiaLoggerLimits(
        max_collection_items=100,
        max_total_bytes=1_300,
    )

    serialized = serialize_agent_trace(_trace_record(result), limits=limits)
    document = _document(serialized)

    assert len(serialized.encode("utf-8")) <= limits.max_total_bytes
    assert document["run"]["omitted"]["model_turn_runtime"] > 0
    assert document["run"]["omitted"]["tool_batches"] > 0
    assert len(document["runtime"]["model_turns"]) < 20
    assert len(document["runtime"]["tool_batches"]) < 20
    assert document["runtime"]["token_summary"] == {
        "coverage": "complete",
        "model_turns_total": 20,
        "model_turns_with_usage": 20,
        "model_turns_without_usage": 0,
        "reported_totals": {
            "prompt_tokens": 210,
            "generated_tokens": 20,
            "combined_tokens": 230,
        },
    }


def test_value_and_total_bounds_use_explicit_representations() -> None:
    calls = tuple(
        ToolCall(f"call-{index}", "lookup", {"value": "é" * 500})
        for index in range(8)
    )
    result = AgentResult(
        context=(),
        final_response="final-" + ("z" * 500),
        transcript=(
            ChatMessage(role="user", content="prompt-" + ("x" * 500)),
            *(
                ChatMessage(
                    role="assistant",
                    content=None,
                    reasoning_content="reason-" + ("y" * 500),
                    tool_calls=(call,),
                )
                for call in calls
            ),
        ),
        tool_results=(),
        model_turns=8,
    )
    limits = GaiaLoggerLimits(
        max_text_characters=100,
        max_json_characters=120,
        max_collection_items=8,
        max_total_bytes=2_200,
    )

    serialized = serialize_agent_trace(_trace_record(result), limits=limits)
    document = _document(serialized)

    assert len(serialized.encode("utf-8")) <= limits.max_total_bytes
    assert document["user_prompt"]["truncated"] is True
    assert document["returned_result"]["content"]["truncated"] is True
    assert document["run"]["omitted"]["reasoning_traces"] > 0
    assert document["run"]["omitted"]["tools_requested"] > 0


def test_atomically_writes_owner_only_trace_to_caller_path(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "agent_trace.json"
    config = GaiaLoggerConfig(
        path=target,
        run_id="run-123",
        repository_root=tmp_path,
    )

    write_gaia_trace(config, _trace_record())

    _document(target.read_text(encoding="utf-8"))
    assert target.stat().st_size <= config.limits.max_total_bytes
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not list(target.parent.glob("*.tmp"))


def test_failed_atomic_replace_preserves_existing_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "agent_trace.json"
    original = b'{"existing": "trace"}\n'
    target.write_bytes(original)
    config = GaiaLoggerConfig(
        path=target,
        run_id="run-123",
        repository_root=tmp_path,
    )

    reject_replace = Mock(side_effect=OSError("replacement failed"))

    monkeypatch.setattr("src.logger.gaia_logger.os.replace", reject_replace)

    with pytest.raises(OSError, match="replacement failed"):
        write_gaia_trace(config, _trace_record())

    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


class _FailingClient:
    calls = 0

    async def context_info(self, *, messages, tools, enable_thinking=None):
        return ContextInfo(100, 131_072)

    async def count_text_tokens(self, text):
        return len(text)

    async def complete(self, *, messages, tools, max_completion_tokens=None, enable_thinking=None):
        self.calls += 1
        if self.calls == 1:
            return ModelTurn(
                reasoning_content="I will call echo.",
                tool_calls=(ToolCall("call-1", "echo", {"filters": {}}),),
                finish_reason="tool_calls",
            )
        raise RuntimeError("transport detail")


async def _echo(arguments):
    return {"echo": arguments}


def test_model_client_failure_preserves_prior_interactions() -> None:
    definition = _tool_definition("echo")

    runtime = AgentRuntime(
        _FailingClient(),
        (RegisteredTool(definition, _echo),),
    )
    with pytest.raises(ModelClientExecutionError) as captured:
        asyncio.run(runtime.run((ChatMessage(role="user", content="Echo."),)))

    error = captured.value
    assert isinstance(error.__cause__, RuntimeError)
    record = build_trace_record(
        error,
        run_id="failed-run",
        started_at="2026-08-22T12:00:00Z",
        ended_at="2026-08-22T12:00:02Z",
        elapsed_seconds=2,
    )
    document = _document(serialize_agent_trace(record))

    assert document["user_prompt"] == "Echo."
    assert document["reasoning_traces"] == [
        {"turn": 1, "content": "I will call echo."}
    ]
    assert document["tools_requested"][0]["returned_result"]["ok"] is True
    assert [item["status"] for item in document["runtime"]["model_turns"]] == [
        "completed",
        "failed",
    ]
    assert document["runtime"]["model_turns"][1]["error_type"] == "RuntimeError"
    assert document["runtime"]["tool_batches"][0]["turn"] == 1
    assert document["tools_requested"][0]["elapsed_seconds"] >= 0
    assert document["returned_result"] == {
        "status": "failed",
        "error": {
            "type": "ModelClientExecutionError",
            "message": "model client failed: RuntimeError",
            "cause_type": "RuntimeError",
        },
    }


def test_generic_failure_omits_system_prompt_and_classifies_error() -> None:
    record = build_trace_record(
        OSError("endpoint unavailable"),
        run_id="setup-failure",
        started_at="2026-08-22T12:00:00Z",
        ended_at="2026-08-22T12:00:00Z",
        elapsed_seconds=0.1,
        invocation={"operation": "query"},
        initial_messages=(
            ChatMessage(role="system", content="Never persist this prompt."),
            ChatMessage(role="user", content="Find a bird."),
        ),
        registered_tool_count=1,
    )

    serialized = serialize_agent_trace(record)
    document = _document(serialized)

    assert document["user_prompt"] == "Find a bird."
    assert "Never persist this prompt" not in serialized
    assert document["run"]["counts"]["registered_tools"] == 1
    assert document["returned_result"] == {
        "status": "failed",
        "error": {"type": "OSError", "message": "endpoint unavailable"},
    }


def test_cancelled_run_has_a_first_class_trace_outcome() -> None:
    error = AgentCancelledError(
        "Stopped thinking",
        transcript=(ChatMessage(role="user", content="Find birds."),),
        tool_results=(),
        model_turns=1,
    )
    record = build_trace_record(
        error,
        run_id="cancelled-run",
        started_at="2026-08-22T12:00:00Z",
        ended_at="2026-08-22T12:00:01Z",
        elapsed_seconds=1,
    )

    document = _document(serialize_agent_trace(record))

    assert document["run"]["status"] == "cancelled"
    assert document["returned_result"] == {"status": "cancelled"}
    assert document["user_prompt"] == "Find birds."


def test_feedback_annotation_is_redacted_atomic_and_replaceable(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    trace_path = tmp_path / "interaction_0001.json"
    config = GaiaLoggerConfig(
        path=trace_path,
        run_id="feedback-run",
        repository_root=repository_root,
        invocation={"interaction": 1},
        feedback_enabled=True,
    )
    original = {
        "run": {
            "status": "completed",
            "invocation": {"interaction": 1},
        },
        "returned_result": {"status": "completed", "content": "雪"},
    }
    trace_path.write_text(
        json.dumps(original, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    trace_path.chmod(0o644)

    write_gaia_trace_feedback(
        config,
        interaction=1,
        rating="bad",
        reasons=(f"Source path: {repository_root}",),
        details=f"Wrong locality — 雪; api_key=private at {repository_root}",
        submitted_at="2026-09-02T12:00:00.000Z",
    )

    annotated = json.loads(trace_path.read_text(encoding="utf-8"))
    assert annotated["returned_result"] == original["returned_result"]
    assert annotated["user_feedback"] == {
        "rating": "bad",
        "reasons": ["Source path: <repository>"],
        "details": "Wrong locality — 雪; api_key=[REDACTED] at <repository>",
        "submitted_at": "2026-09-02T12:00:00.000Z",
    }
    assert stat.S_IMODE(trace_path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp"))

    write_gaia_trace_feedback(
        config,
        interaction=1,
        rating="good",
        reasons=(),
        details="Looks right — 正確",
        submitted_at="2026-09-02T12:01:00.000Z",
    )

    replaced = json.loads(trace_path.read_text(encoding="utf-8"))
    assert replaced["user_feedback"] == {
        "rating": "good",
        "reasons": [],
        "details": "Looks right — 正確",
        "submitted_at": "2026-09-02T12:01:00.000Z",
    }
    assert stat.S_IMODE(trace_path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp"))

    trace_path.write_text("not JSON", encoding="utf-8")

    with pytest.raises(GaiaLoggerError, match="could not be recorded"):
        write_gaia_trace_feedback(
            config,
            interaction=1,
            rating="good",
            reasons=(),
            details="",
            submitted_at="2026-09-02T12:00:00.000Z",
        )


def test_trace_reserves_capacity_only_when_feedback_is_enabled(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "interaction_0001.json"
    trace_body_limit = 16_384
    limits = GaiaLoggerLimits(
        max_total_bytes=(
            GAIA_FEEDBACK_TRACE_RESERVE_BYTES + trace_body_limit
        ),
    )
    config = GaiaLoggerConfig(
        path=trace_path,
        run_id="feedback-capacity-run",
        repository_root=Path("."),
        invocation={"interaction": 1},
        limits=limits,
        feedback_enabled=True,
    )
    record = GaiaLoggerRecord(
        run_id=config.run_id,
        status="completed",
        started_at="2026-09-02T12:00:00.000Z",
        ended_at="2026-09-02T12:00:01.000Z",
        elapsed_seconds=1,
        invocation={"interaction": 1},
        transcript=(
            ChatMessage(role="user", content="Question"),
            *(
                ChatMessage(
                    role="assistant",
                    content=None,
                    reasoning_content="x" * 1_000,
                )
                for _index in range(100)
            ),
            ChatMessage(role="assistant", content="Answer"),
        ),
        tool_results=(),
        registered_tool_count=0,
        model_turns=101,
        final_response="Answer",
    )

    write_gaia_trace(config, record)

    assert trace_path.stat().st_size <= trace_body_limit
    assert json.loads(trace_path.read_text(encoding="utf-8"))["run"][
        "invocation"
    ]["interaction"] == 1

    write_gaia_trace_feedback(
        config,
        interaction=1,
        rating="bad",
        reasons=tuple(
            "." * (MAX_FEEDBACK_REASON_CHARACTERS - 1) + str(index)
            for index in range(MAX_FEEDBACK_REASONS)
        ),
        details="." * MAX_FEEDBACK_DETAILS_CHARACTERS,
        submitted_at="2026-09-02T12:01:00.000Z",
    )

    assert trace_path.stat().st_size <= limits.max_total_bytes

    plain_trace_path = tmp_path / "plain_trace.json"
    plain_config = GaiaLoggerConfig(
        path=plain_trace_path,
        run_id="plain-run",
        repository_root=Path("."),
        limits=GaiaLoggerLimits(max_total_bytes=trace_body_limit),
    )

    write_gaia_trace(plain_config, record)

    assert plain_trace_path.stat().st_size <= trace_body_limit


@pytest.mark.parametrize(
    ("status", "linked_interaction"),
    [("failed", 1), ("cancelled", 1), ("completed", 2), ("completed", True)],
)
def test_feedback_rejects_ineligible_trace_without_modifying_it(
    tmp_path: Path,
    status: str,
    linked_interaction: int | bool,
) -> None:
    target = tmp_path / "interaction_0001.json"
    original = json.dumps(
        {
            "run": {
                "status": status,
                "invocation": {"interaction": linked_interaction},
            },
        }
    ).encode("utf-8")
    target.write_bytes(original)
    config = GaiaLoggerConfig(
        path=target,
        run_id="feedback-run",
        repository_root=tmp_path,
        invocation={"interaction": 1},
        feedback_enabled=True,
    )

    with pytest.raises(GaiaLoggerError, match="could not be recorded"):
        write_gaia_trace_feedback(
            config,
            interaction=1,
            rating="good",
            reasons=(),
            details="",
            submitted_at="2026-09-02T12:00:00.000Z",
        )

    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]
