"""Durable trace tests for Gaia's projected agent composition boundary."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.gaia_agent.runtime.contracts import (
    AgentCancellation,
    AgentCancelledError,
    ChatMessage,
    ModelClientExecutionError,
    ModelTurn,
    RegisteredTool,
    ToolCall,
)
from src.gaia_agent.runtime.loop import AgentRuntime
from src.gaia_agent.tools.query.execution import QueryExecutionResult
from src.gaia_agent.runtime.service import GaiaAgent, build_gaia_agent
from src.gaia_agent.tools import GaiaToolset
from src.logger.gaia_logger import GaiaLoggerConfig

from tests.gaia_agent.support import RecordingClient, canadian_bird_plan, query_result


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _trace(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    assert list(document) == [
        "run",
        "runtime",
        "user_prompt",
        "reasoning_traces",
        "tools_requested",
        "returned_result",
    ]
    return document


def _registered_with_query_result(
    internal_result: QueryExecutionResult,
) -> tuple[RegisteredTool, ...]:
    query_handler = AsyncMock(return_value=internal_result)

    return tuple(
        RegisteredTool(
            tool.definition,
            query_handler
            if tool.definition.name == "query_occurrences"
            else tool.handler,
            observation_projector=tool.observation_projector,
        )
        for tool in GaiaToolset.load().registered_tools()
    )


def test_query_trace_preserves_turns_reasoning_two_tools_and_outcome(
    tmp_path: Path,
) -> None:
    internal_result = query_result()
    client = RecordingClient(
        (
            ModelTurn(
                reasoning_content="I need trusted schema metadata first.",
                tool_calls=(
                    ToolCall(
                        "schema-call",
                        "retrieve_schema",
                        {"datasets": ["birds"]},
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ModelTurn(
                reasoning_content=(
                    "The schema observation includes the QueryPlan structure, "
                    "so I can construct a complete valid plan."
                ),
                tool_calls=(
                    ToolCall(
                        "occurrence-call",
                        "query_occurrences",
                        canadian_bird_plan(("country",)),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ModelTurn(
                content="The occurrence query completed.",
                reasoning_content="I used the projected query observation.",
                finish_reason="stop",
            ),
        )
    )
    registered = _registered_with_query_result(internal_result)
    trace_path = tmp_path / "query.json"
    agent = GaiaAgent(
        AgentRuntime(client, registered),
        system_prompt="Use Gaia's trusted database tools.",
        trace_config=GaiaLoggerConfig(
            path=trace_path,
            run_id="trace-run",
            repository_root=REPOSITORY_ROOT,
            invocation={
                "model": "test-model",
                "system_prompt_path": str(
                    REPOSITORY_ROOT / "src/gaia_agent/runtime/system_prompt.md"
                ),
            },
        ),
    )

    result = asyncio.run(
        agent.run(
            "Find bird records from Canada and show the country field."
        )
    )

    assert result.final_response == "The occurrence query completed."
    trace = _trace(trace_path)
    assert trace["run"]["status"] == "completed"
    assert trace["run"]["trace_version"] == 5
    assert [item["turn"] for item in trace["runtime"]["model_turns"]] == [
        1,
        2,
        3,
    ]
    assert all(
        item["status"] == "completed"
        and item["client_round_trip_seconds"] >= 0
        for item in trace["runtime"]["model_turns"]
    )
    assert [
        (item["turn"], item["tool_calls"])
        for item in trace["runtime"]["tool_batches"]
    ] == [(1, 1), (2, 1)]
    assert all(
        item["wall_elapsed_seconds"] >= 0
        for item in trace["runtime"]["tool_batches"]
    )
    assert trace["run"]["invocation"]["system_prompt_path"] == (
        "<repository>/src/gaia_agent/runtime/system_prompt.md"
    )
    assert trace["user_prompt"] == (
        "Find bird records from Canada and show the country field."
    )
    assert [
        item["content"] for item in trace["reasoning_traces"]
    ] == [
        "I need trusted schema metadata first.",
        (
            "The schema observation includes the QueryPlan structure, so I can "
            "construct a complete valid plan."
        ),
        "I used the projected query observation.",
    ]
    assert [item["name"] for item in trace["tools_requested"]] == [
        "retrieve_schema",
        "query_occurrences",
    ]
    assert all(
        item["returned_result"]["ok"] is True
        for item in trace["tools_requested"]
    )
    assert all(
        item["elapsed_seconds"] >= 0 for item in trace["tools_requested"]
    )
    assert trace["returned_result"] == {
        "status": "completed",
        "content": "The occurrence query completed.",
    }
    assert "Use Gaia's trusted database tools." not in trace_path.read_text(
        encoding="utf-8"
    )


def test_query_projection_precedes_model_messages_result_and_durable_trace(
    tmp_path: Path,
) -> None:
    private_marker = "UNPROJECTED-PRIVATE-OCCURRENCE-VALUE"
    internal_result = query_result(
        records=(
            {
                "dataset_key": "birds",
                "occurrenceID": "bird-1",
                "country": "Canada",
                "scientificName": private_marker,
                "dynamicProperties": private_marker,
            },
        ),
        return_columns=("country",),
    )

    registered_tools = _registered_with_query_result(internal_result)
    client = RecordingClient(
        (
            ModelTurn(
                reasoning_content="I have a complete validated QueryPlan.",
                tool_calls=(
                    ToolCall("occurrence-call", "query_occurrences", {}),
                ),
                finish_reason="tool_calls",
            ),
            ModelTurn(
                content="Found one Canadian bird record.",
                reasoning_content="I used only the projected observation.",
                finish_reason="stop",
            ),
        )
    )
    trace_path = tmp_path / "projected-occurrence.json"
    agent = GaiaAgent(
        AgentRuntime(client, registered_tools),
        system_prompt="Use only trusted projected occurrence observations.",
        trace_config=GaiaLoggerConfig(
            path=trace_path,
            run_id="projection-run",
            repository_root=REPOSITORY_ROOT,
        ),
    )

    result = asyncio.run(agent.run("Find a Canadian bird occurrence."))

    expected_record = {
        "dataset_key": "birds",
        "occurrenceID": "bird-1",
        "country": "Canada",
    }
    model_payload = json.loads(client.requests[1][0][-1].content)
    transcript_payload = json.loads(result.transcript[-2].content)
    for records in (
        model_payload["result"]["records"],
        result.tool_results[0].result["records"],
        transcript_payload["result"]["records"],
    ):
        assert records == [expected_record]

    trace_text = trace_path.read_text(encoding="utf-8")
    all_persisted_views = "\n".join(
        (
            repr(client.requests),
            repr(result.transcript),
            repr(result.tool_results),
            trace_text,
        )
    )
    assert private_marker not in all_persisted_views
    assert "Canada" in trace_text

    trace = _trace(trace_path)
    assert len(trace["tools_requested"]) == 1
    assert trace["tools_requested"][0]["returned_result"]["result"][
        "records"
    ] == [expected_record]
    assert trace_text.count('"Canada"') == 1


def test_query_trace_persists_failed_model_turn_for_backtracking(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "failed.json"
    agent = build_gaia_agent(
        REPOSITORY_ROOT,
        RecordingClient((RuntimeError("endpoint stopped"),)),
        trace_config=GaiaLoggerConfig(
            path=trace_path,
            run_id="failed-run",
            repository_root=REPOSITORY_ROOT,
        ),
    )

    with pytest.raises(ModelClientExecutionError, match="model client failed"):
        asyncio.run(agent.run("Find birds."))

    trace = _trace(trace_path)
    assert trace["user_prompt"] == "Find birds."
    assert trace["runtime"]["model_turns"][0]["turn"] == 1
    assert trace["runtime"]["model_turns"][0]["status"] == "failed"
    assert trace["runtime"]["model_turns"][0]["error_type"] == "RuntimeError"
    assert trace["runtime"]["model_turns"][0]["client_round_trip_seconds"] >= 0
    assert trace["runtime"]["tool_batches"] == []
    assert trace["reasoning_traces"] == []
    assert trace["tools_requested"] == []
    assert trace["returned_result"] == {
        "status": "failed",
        "error": {
            "type": "ModelClientExecutionError",
            "message": "model client failed: RuntimeError",
            "cause_type": "RuntimeError",
        },
    }


def test_query_trace_records_cancellation_as_its_terminal_status(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "cancelled.json"
    cancellation = AgentCancellation()
    cancellation.cancel()
    agent = build_gaia_agent(
        REPOSITORY_ROOT,
        RecordingClient(()),
        trace_config=GaiaLoggerConfig(
            path=trace_path,
            run_id="cancelled-run",
            repository_root=REPOSITORY_ROOT,
        ),
    )

    with pytest.raises(AgentCancelledError, match="Stopped thinking"):
        asyncio.run(agent.run("Find birds.", cancellation=cancellation))

    trace = _trace(trace_path)
    assert trace["run"]["status"] == "cancelled"
    assert trace["run"]["counts"]["model_turns"] == 0
    assert trace["returned_result"] == {"status": "cancelled"}


def test_continued_conversation_trace_contains_only_the_current_prompt(
    tmp_path: Path,
) -> None:
    client = RecordingClient(
        (
            ModelTurn(
                content="I found three bird species.",
                reasoning_content="I answered the first prompt.",
                finish_reason="stop",
            ),
            ModelTurn(
                content="They were recorded in Canada.",
                reasoning_content="I resolved 'they' from the conversation.",
                finish_reason="stop",
            ),
        )
    )
    agent = GaiaAgent(
        AgentRuntime(client),
        system_prompt="Use trusted observations.",
    )
    chat_directory = tmp_path / "gaia_chat_test_user_20260901T000000_000000Z"
    first_path = chat_directory / "interaction_0001.json"
    first_result = asyncio.run(
        agent.run(
            "Find birds.",
            trace_config=GaiaLoggerConfig(
                path=first_path,
                run_id="first-turn",
                repository_root=REPOSITORY_ROOT,
            ),
        )
    )
    second_path = chat_directory / "interaction_0002.json"

    second_result = asyncio.run(
        agent.run(
            "Where were they recorded?",
            history=first_result.context,
            trace_config=GaiaLoggerConfig(
                path=second_path,
                run_id="second-turn",
                repository_root=REPOSITORY_ROOT,
            ),
        )
    )

    assert client.requests[1][0][:-1] == first_result.transcript
    assert second_result.transcript[: len(first_result.transcript)] == (
        first_result.transcript
    )
    assert first_path.is_file()
    assert second_path.is_file()
    trace = _trace(second_path)
    assert trace["user_prompt"] == "Where were they recorded?"
    assert trace["run"]["counts"]["messages"] == 2
    assert trace["run"]["counts"]["user_messages"] == 1
    assert trace["reasoning_traces"] == [
        {
            "turn": 1,
            "content": "I resolved 'they' from the conversation.",
        }
    ]
    assert trace["returned_result"]["content"] == (
        "They were recorded in Canada."
    )
    assert "Find birds." not in second_path.read_text(encoding="utf-8")


def test_failed_continuation_trace_excludes_completed_history(
    tmp_path: Path,
) -> None:
    client = RecordingClient(
        (
            ModelTurn(
                content="I found birds.",
                finish_reason="stop",
            ),
            RuntimeError("endpoint stopped"),
        )
    )

    agent = GaiaAgent(
        AgentRuntime(client),
        system_prompt="Use trusted observations.",
    )
    first_result = asyncio.run(agent.run("Find birds."))
    trace_path = tmp_path / "failed-follow-up.json"

    with pytest.raises(ModelClientExecutionError, match="model client failed"):
        asyncio.run(
            agent.run(
                "Where were they found?",
                history=first_result.context,
                trace_config=GaiaLoggerConfig(
                    path=trace_path,
                    run_id="failed-follow-up",
                    repository_root=REPOSITORY_ROOT,
                ),
            )
        )

    trace = _trace(trace_path)
    assert trace["user_prompt"] == "Where were they found?"
    assert trace["run"]["counts"]["messages"] == 1
    assert trace["run"]["counts"]["user_messages"] == 1
    assert trace["runtime"]["model_turns"][0]["status"] == "failed"
    assert trace["returned_result"]["error"]["cause_type"] == "RuntimeError"
    assert "Find birds." not in trace_path.read_text(encoding="utf-8")
