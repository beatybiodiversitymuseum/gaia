"""Deterministic tests for the model-agnostic agent loop."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from src.gaia_agent.runtime.contracts import (
    AgentCancellation,
    AgentCancelledError,
    ToolAttemptsExhaustedError,
    AgentProgressEvent,
    AgentResult,
    ChatMessage,
    ContextCompactionError,
    ContextInfo,
    InvalidModelTurnError,
    ModelClientExecutionError,
    ModelRequestMetrics,
    ModelTurn,
    RegisteredTool,
    ResponseDisclosureError,
    ToolCall,
    ToolCallLimitExceededError,
    ToolDefinition,
    ToolExecutionError,
    TokenUsage,
)
from src.gaia_agent.runtime.defaults import CONTEXT_CONTINUATION_PROMPT, MAX_TOOL_CALLS_PER_TURN
from src.gaia_agent.runtime.disclosure import ResponseDisclosureGuard
from src.gaia_agent.runtime.loop import AgentRuntime

from src.logger.gaia_logger import build_trace_record, serialize_agent_trace


pytestmark = pytest.mark.anyio


class _TestContextClient:
    async def context_info(self, *, messages, tools, enable_thinking=None):
        return ContextInfo(prompt_tokens=100, context_limit_tokens=1000000)

    async def count_text_tokens(self, text):
        return len(text)


class FakeClient(_TestContextClient):
    """Return predefined turns and retain every request for assertions."""

    def __init__(self, turns: Sequence[object]) -> None:
        self._turns = list(turns)
        self.requests: list[
            tuple[tuple[ChatMessage, ...], tuple[ToolDefinition, ...]]
        ] = []

    async def complete(
        self, *, messages, tools, max_completion_tokens=None, enable_thinking=None,
    ):
        self.requests.append((tuple(messages), tuple(tools)))
        turn = self._turns.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        return turn


class _BudgetClient(FakeClient):
    """Provide exact rendered-token counts at each runtime checkpoint."""

    def __init__(self, turns: Sequence[object], prompt_sizes: Sequence[int]) -> None:
        super().__init__(turns)
        self.prompt_sizes = list(prompt_sizes)
        self.context_requests = []
        self.output_limits = []
        self.thinking = []

    async def context_info(self, *, messages, tools, enable_thinking=None):
        if not any(message.role == "user" for message in messages):
            raise ValueError("No user query found in messages.")
        self.context_requests.append((tuple(messages), tuple(tools)))
        tokens = self.prompt_sizes.pop(0) if len(self.prompt_sizes) > 1 else self.prompt_sizes[0]
        return ContextInfo(prompt_tokens=tokens, context_limit_tokens=1000)

    async def complete(
        self, *, messages, tools, max_completion_tokens=None, enable_thinking=None,
    ):
        self.output_limits.append(max_completion_tokens)
        self.thinking.append(enable_thinking)
        return await super().complete(messages=messages, tools=tools)


async def _model_compaction(client, messages, *, context_limit_tokens, cancellation):
    """Exercise the loop's compaction accounting independently of summarization."""
    turn = await client.complete(
        messages=messages, tools=(), max_completion_tokens=100, enable_thinking=False,
    )
    return turn.content


def registered_tool(
    name: str,
    handler,
    projector=None,
    *,
    strict: bool = False,
) -> RegisteredTool:
    """Build one small object-argument tool for a test."""
    return RegisteredTool(
        ToolDefinition(
            name=name,
            description=f"Run {name}.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            strict=strict,
        ),
        AsyncMock(wraps=handler),
        observation_projector=projector,
    )


def _double(
    arguments: dict[str, object],
    *,
    handled: list[tuple[str, dict[str, object]]],
) -> object:
    handled.append(("double", arguments))
    return {"value": arguments["value"] * 2}


def _increment(
    arguments: dict[str, object],
    *,
    handled: list[tuple[str, dict[str, object]]],
) -> object:
    handled.append(("increment", arguments))
    return {"value": arguments["value"] + 1}


def _cancel_after_progress(
    event: AgentProgressEvent,
    *,
    kind: str,
    cancellation: AgentCancellation,
    events: list[AgentProgressEvent] | None = None,
) -> None:
    if events is not None:
        events.append(event)
    if event.kind == kind:
        cancellation.cancel()


@dataclass
class _PendingModelClient(_TestContextClient):
    started: asyncio.Event
    cleanup_started: asyncio.Event
    finish_cleanup: asyncio.Event

    async def complete(self, *, messages, tools, **kwargs):
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cleanup_started.set()
            await self.finish_cleanup.wait()


@dataclass
class _FollowupClient(_TestContextClient):
    started: asyncio.Event
    release: asyncio.Event

    async def complete(self, *, messages, tools, **kwargs):
        if messages[-1].content == "First request.":
            return ModelTurn(content="First answer.")
        self.started.set()
        await self.release.wait()
        return ModelTurn(content="Followup answer.")


async def _run_followup(
    runtime: AgentRuntime, first_cancellation: AgentCancellation,
) -> AgentResult:
    await runtime.run(
        (ChatMessage(role="user", content="First request."),),
        cancellation=first_cancellation,
    )
    return await runtime.run(
        (ChatMessage(role="user", content="Followup request."),),
        cancellation=AgentCancellation(),
    )


class _YieldingFollowupClient(_TestContextClient):
    async def complete(self, *, messages, tools, **kwargs):
        await asyncio.sleep(0)
        return ModelTurn(content="Followup answer.")


@dataclass
class _PendingRequestClient(_TestContextClient):
    started: asyncio.Queue[None]

    async def complete(self, *, messages, tools, **kwargs):
        self.started.put_nowait(None)
        await asyncio.Event().wait()


async def _cancel_repeated_runs(
    started: asyncio.Queue[None],
    cancellations: tuple[AgentCancellation, ...],
    errors: list[AgentCancelledError],
) -> None:
    for cancellation in cancellations:
        with pytest.raises(AgentCancelledError) as captured:
            await AgentRuntime(_PendingRequestClient(started)).run(
                (ChatMessage(role="user", content="Cancel this request."),),
                cancellation=cancellation,
            )
        errors.append(captured.value)


async def _coordinate_calls(
    arguments: dict[str, object],
    *,
    all_started: asyncio.Barrier,
    can_complete: list[asyncio.Event],
    completion_order: list[int],
) -> dict[str, object]:
    value = arguments["value"]
    assert isinstance(value, int)
    if value == 0:
        raise ToolExecutionError("unavailable", "Try again.")
    await asyncio.wait_for(all_started.wait(), timeout=5)
    await asyncio.wait_for(can_complete[value].wait(), timeout=5)
    completion_order.append(value)
    can_complete[value - 1].set()
    return {"value": value}


@dataclass
class _ResultProjection:
    internal_result: object
    projected_result: object
    events: list[str] = field(default_factory=list)

    def handle(self, arguments: dict[str, object]) -> object:
        self.events.append("handler")
        return self.internal_result

    def project(self, value: object) -> object:
        self.events.append("projector")
        assert value is self.internal_result
        return self.projected_result


@dataclass
class _ConcurrentClient(_TestContextClient):
    first_turns_started: asyncio.Barrier
    call_count: int

    async def complete(self, *, messages, tools, **kwargs):
        user_message = next(
            message.content for message in messages if message.role == "user"
        )
        assert user_message is not None
        value = int(user_message.rsplit(" ", 1)[-1])
        if not any(message.role == "assistant" for message in messages):
            await asyncio.wait_for(self.first_turns_started.wait(), timeout=5)
            return ModelTurn(
                tool_calls=tuple(
                    ToolCall(f"shared-call-id-{index}", "echo", {"value": value})
                    for index in range(self.call_count)
                ),
            )
        assert bool(tools) is (self.call_count < 15)
        return ModelTurn(content=f"Completed {value}.")


async def _fail_after_barrier(
    arguments: dict[str, object],
    *,
    attempts: list[tuple[int, int]],
    all_started: asyncio.Barrier,
) -> object:
    attempts.append((arguments["round"], arguments["value"]))
    await asyncio.wait_for(all_started.wait(), timeout=5)
    raise ToolExecutionError("unavailable", "The tool is unavailable.")


def _reject_zero(arguments: dict[str, object]) -> object:
    if arguments["value"] == 0:
        raise ToolExecutionError("invalid_input", "Correct the arguments.")
    return arguments


@dataclass
class _RetryClient(_TestContextClient):
    both_started: asyncio.Barrier

    async def complete(self, *, messages, tools, **kwargs):
        chat = messages[0].content
        completed_rounds = sum(message.role == "tool" for message in messages)
        if completed_rounds == 0:
            await asyncio.wait_for(self.both_started.wait(), timeout=5)
        if chat == "second" and completed_rounds == 2:
            return ModelTurn(content="Explained the unavailable result.")
        return ModelTurn(tool_calls=(
            ToolCall(f"call-{completed_rounds}", "lookup", {}),
        ))


async def _exhaustion_fail(
    arguments: dict[str, object],
    *,
    healthy_started: asyncio.Event,
    release_healthy: asyncio.Event,
) -> object:
    if arguments["value"] != 0:
        await asyncio.wait_for(healthy_started.wait(), timeout=5)
    if arguments["value"] == 3:
        release_healthy.set()
    raise ToolExecutionError("invalid_input", "Correct the arguments.")


async def _exhaustion_succeed(
    arguments: dict[str, object],
    *,
    healthy_started: asyncio.Event,
    release_healthy: asyncio.Event,
) -> object:
    healthy_started.set()
    await asyncio.wait_for(release_healthy.wait(), timeout=5)
    return "Completed."


@dataclass
class _PendingTools:
    pending_started: asyncio.Event
    cleanup_started: asyncio.Event
    finish_cleanup: asyncio.Event
    closed: list[int]
    pending_count: int = 0

    async def handle(self, arguments):
        if arguments["value"] == 0:
            return "Completed."
        self.pending_count += 1
        if self.pending_count == 2:
            self.pending_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            if arguments["value"] == 2:
                self.cleanup_started.set()
                await self.finish_cleanup.wait()
            self.closed.append(arguments["value"])


def _mutate_arguments(arguments: dict[str, object], *, returned_value: object) -> object:
    arguments["nested"]["value"] = "changed"
    return returned_value


async def test_returns_final_response_without_calling_a_tool() -> None:
    client = FakeClient((ModelTurn(content="Finished.", finish_reason="stop"),))
    runtime = AgentRuntime(client)

    result = await runtime.run(
        (
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(role="user", content="Say when you are done."),
        )
    )

    assert result.final_response == "Finished."
    assert result.model_turns == 1
    assert result.tool_calls == 0
    assert [message.role for message in result.transcript] == [
        "system",
        "user",
        "assistant",
    ]
    assert client.requests[0][1] == ()
    assert result.context == result.transcript


async def test_context_below_trigger_retains_history_and_reserves_output() -> None:
    messages = (ChatMessage(role="user", content="Continue."),)
    client = _BudgetClient((ModelTurn(content="Finished."),), (854,))

    result = await AgentRuntime(client).run(messages)

    assert client.output_limits == [96]
    assert client.requests[0][0] == messages
    assert result.context == result.transcript
    assert result.context_info == ContextInfo(prompt_tokens=854, context_limit_tokens=1000)


async def test_trigger_compacts_before_request_and_records_model_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    markdown = "## Chat summary\nContinue the current request.\n\n## Tools called\nNone.\n"
    usage = TokenUsage(prompt_tokens=855, completion_tokens=40, total_tokens=895)
    client = _BudgetClient(
        (ModelTurn(content=markdown, usage=usage), ModelTurn(content="Finished.")),
        (855, 100, 110),
    )
    messages = (
        ChatMessage(role="system", content="System instructions."),
        ChatMessage(role="user", content="Continue the current request."),
    )
    events = []
    history_path = tmp_path / "history.md"
    history_path.write_text("Previous summary.", encoding="utf-8")
    monkeypatch.setattr("src.gaia_agent.runtime.loop.compact_context", _model_compaction)

    result = await AgentRuntime(client).run(
        messages, history_path=history_path, progress_callback=events.append,
    )

    assert history_path.read_text(encoding="utf-8") == markdown
    assert client.requests[1][0] == (
        messages[0],
        ChatMessage(role="user", content=CONTEXT_CONTINUATION_PROMPT),
        ChatMessage(role="assistant", content=markdown),
    )
    assert result.transcript == (*messages, ChatMessage(role="assistant", content="Finished."))
    assert result.context == (*client.requests[1][0], result.transcript[-1])
    assert result.model_turns == 2
    assert [item.purpose for item in result.model_turn_runtime] == ["compaction", "response"]
    assert result.model_turn_runtime[0].usage is usage
    assert client.output_limits == [100, 850]
    assert client.thinking == [False, None]
    assert [event.kind for event in events] == [
        "context_compacting", "context_compacted", "model_turn_started", "response_ready",
    ]


async def test_compaction_waits_for_complete_tool_batch_and_preserves_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = ToolCall("call-1", "lookup", {"value": 1})
    markdown = "## Chat summary\nFinish the response.\n\n## Tools called\nlookup returned 2.\n"
    client = _BudgetClient(
        (
            ModelTurn(tool_calls=(call,)), ModelTurn(content=markdown),
            ModelTurn(content="The value is 2."),
        ),
        (100, 855, 100, 110),
    )
    tool = registered_tool("lookup", lambda arguments: {"value": 2})
    monkeypatch.setattr("src.gaia_agent.runtime.loop.compact_context", _model_compaction)

    result = await AgentRuntime(client, (tool,)).run(
        (ChatMessage(role="user", content="Look it up."),),
        history_path=tmp_path / "history.md",
    )

    compaction_input = client.requests[1][0]
    assert compaction_input[-2].tool_calls == (call,)
    assert compaction_input[-1].tool_call_id == call.id
    assert json.loads(compaction_input[-1].content) == {"ok": True, "result": {"value": 2}}
    assert client.requests[2][0] == (
        ChatMessage(role="user", content=CONTEXT_CONTINUATION_PROMPT),
        ChatMessage(role="assistant", content=markdown),
    )
    assert result.transcript == (*compaction_input, ChatMessage(role="assistant", content="The value is 2."))
    assert result.tool_calls == 1
    tool.handler.assert_awaited_once()
    assert [item.purpose for item in result.model_turn_runtime] == [
        "response", "compaction", "response",
    ]
    assert all(tools for _, tools in client.context_requests)


async def test_final_answer_can_trigger_compaction_without_changing_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    markdown = "## Chat summary\nThe request is complete.\n\n## Tools called\nNone.\n"
    client = _BudgetClient(
        (ModelTurn(content="Original final answer."), ModelTurn(content=markdown)),
        (100, 855, 100),
    )
    monkeypatch.setattr("src.gaia_agent.runtime.loop.compact_context", _model_compaction)

    result = await AgentRuntime(client).run(
        (ChatMessage(role="system", content="Instructions."), ChatMessage(role="user", content="Answer.")),
        history_path=tmp_path / "history.md",
    )

    assert result.final_response == "Original final answer."
    assert result.transcript[-1].content == result.final_response
    assert result.context == (
        result.transcript[0],
        ChatMessage(role="user", content=CONTEXT_CONTINUATION_PROMPT),
        ChatMessage(role="assistant", content=markdown),
    )
    assert client.requests[1][0][-1].content == result.final_response


async def test_repeated_compaction_combines_prior_summary_and_exact_new_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_summary = "## Chat summary\nFirst request.\n\n## Tools called\nNone.\n"
    second_summary = "## Chat summary\nFirst and second requests.\n\n## Tools called\nNone.\n"
    client = _BudgetClient(
        (
            ModelTurn(content=first_summary), ModelTurn(content="First answer."),
            ModelTurn(content=second_summary), ModelTurn(content="Second answer."),
        ),
        (855, 100, 110, 855, 100, 110),
    )
    monkeypatch.setattr("src.gaia_agent.runtime.loop.compact_context", _model_compaction)
    runtime = AgentRuntime(client)
    history_path = tmp_path / "history.md"

    first = await runtime.run(
        (ChatMessage(role="user", content="First request."),), history_path=history_path,
    )
    second_messages = (*first.context, ChatMessage(role="user", content="Second request."))
    second = await runtime.run(second_messages, history_path=history_path)

    assert client.requests[2][0] == second_messages
    assert second_messages[0].content == CONTEXT_CONTINUATION_PROMPT
    assert second_messages[1].content == first_summary
    assert second.context == (
        ChatMessage(role="user", content=CONTEXT_CONTINUATION_PROMPT),
        ChatMessage(role="assistant", content=second_summary),
        ChatMessage(role="assistant", content="Second answer."),
    )
    assert history_path.read_text(encoding="utf-8") == second_summary


@pytest.mark.parametrize("failure", ("oversized", "write", "model"))
async def test_failed_compaction_preserves_history_and_original_messages(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    markdown = "## Chat summary\nSummary.\n\n## Tools called\nNone.\n"
    client = _BudgetClient(
        (RuntimeError("Unavailable") if failure == "model" else ModelTurn(content=markdown),),
        (855, 855 if failure == "oversized" else 100),
    )
    history_path = tmp_path / "history.md"
    history_path.write_text("Saved history.", encoding="utf-8")
    messages = (ChatMessage(role="user", content="Continue."),)
    monkeypatch.setattr("src.gaia_agent.runtime.loop.compact_context", _model_compaction)
    if failure == "write":
        monkeypatch.setattr("src.gaia_agent.runtime.loop.write_history", Mock(side_effect=OSError("Full")))

    with pytest.raises(ContextCompactionError) as captured:
        await AgentRuntime(client).run(messages, history_path=history_path)

    assert history_path.read_text(encoding="utf-8") == "Saved history."
    assert captured.value.transcript == messages
    assert captured.value.model_turns == 1
    assert captured.value.model_turn_runtime[0].purpose == "compaction"
    assert captured.value.model_turn_runtime[0].status == ("failed" if failure == "model" else "completed")


async def test_compaction_requires_chat_storage_before_generating_summary() -> None:
    client = _BudgetClient((), (855,))

    with pytest.raises(ContextCompactionError) as captured:
        await AgentRuntime(client).run((ChatMessage(role="user", content="Continue."),))

    assert captured.value.model_turns == 0
    assert client.requests == []


async def test_compaction_rejects_private_content_before_saving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    markdown = "## Chat summary\nRead /private/runtime/data.\n\n## Tools called\nNone.\n"
    client = _BudgetClient((ModelTurn(content=markdown),), (855,))
    history_path = tmp_path / "history.md"
    history_path.write_text("Saved history.", encoding="utf-8")
    guard = ResponseDisclosureGuard(
        system_prompt="System instructions.", tools=(), private_locations=("/private/runtime",),
    )
    monkeypatch.setattr("src.gaia_agent.runtime.loop.compact_context", _model_compaction)

    with pytest.raises(ContextCompactionError):
        await AgentRuntime(client, response_guard=guard).run(
            (ChatMessage(role="user", content="Continue."),), history_path=history_path,
        )

    assert history_path.read_text(encoding="utf-8") == "Saved history."
    assert len(client.context_requests) == 1
    assert len(client.requests) == 1


async def test_rejected_final_response_is_never_compacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _BudgetClient((ModelTurn(content="Read /private/runtime/data."),), (100, 855))
    guard = ResponseDisclosureGuard(
        system_prompt="System instructions.", tools=(), private_locations=("/private/runtime",),
    )
    compactor = AsyncMock()
    monkeypatch.setattr("src.gaia_agent.runtime.loop.compact_context", compactor)
    history_path = tmp_path / "history.md"

    with pytest.raises(ResponseDisclosureError):
        await AgentRuntime(client, response_guard=guard).run(
            (ChatMessage(role="user", content="Answer."),), history_path=history_path,
        )

    compactor.assert_not_awaited()
    assert not history_path.exists()


async def test_compaction_preserves_current_turn_tool_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    markdown = "## Chat summary\nContinue.\n\n## Tools called\n14 lookup calls completed.\n"
    initial_calls = tuple(ToolCall(f"call-{index}", "lookup", {"value": index}) for index in range(14))
    client = _BudgetClient(
        (
            ModelTurn(tool_calls=initial_calls), ModelTurn(content=markdown),
            ModelTurn(tool_calls=(ToolCall("call-14", "lookup", {"value": 14}), ToolCall("call-15", "lookup", {"value": 15}))),
        ),
        (100, 855, 100),
    )
    tool = registered_tool("lookup", lambda arguments: arguments)
    monkeypatch.setattr("src.gaia_agent.runtime.loop.compact_context", _model_compaction)

    with pytest.raises(ToolCallLimitExceededError) as captured:
        await AgentRuntime(client, (tool,)).run(
            (ChatMessage(role="user", content="Look up records."),),
            history_path=tmp_path / "history.md",
        )

    assert captured.value.tool_calls_used == 14
    assert len(captured.value.tool_results) == 14
    assert tool.handler.await_count == 14


async def test_cancelling_compaction_preserves_previous_history(
    tmp_path: Path,
) -> None:
    cancellation = AgentCancellation()
    client = _BudgetClient((), (855,))
    history_path = tmp_path / "history.md"
    history_path.write_text("Saved history.", encoding="utf-8")

    with pytest.raises(AgentCancelledError):
        await AgentRuntime(client).run(
            (ChatMessage(role="user", content="Continue."),),
            history_path=history_path,
            cancellation=cancellation,
            progress_callback=partial(
                _cancel_after_progress, kind="context_compacting", cancellation=cancellation,
            ),
        )

    assert history_path.read_text(encoding="utf-8") == "Saved history."
    assert client.requests == []


@pytest.mark.parametrize("call_count", (15, 16))
async def test_tool_call_budget_accepts_fifteen_and_rejects_sixteen_before_dispatch(
    call_count: int,
) -> None:
    handled: list[int] = []
    calls = tuple(
        ToolCall(f"schema-{index}", "retrieve_schema", {"value": index})
        for index in range(call_count)
    )
    client = FakeClient((ModelTurn(tool_calls=calls), ModelTurn(content="Finished.")))
    runtime = AgentRuntime(client, (
        registered_tool("retrieve_schema", lambda arguments: handled.append(arguments["value"])),
    ))
    events: list[AgentProgressEvent] = []
    assert MAX_TOOL_CALLS_PER_TURN == 15

    if call_count == 15:
        result = await runtime.run(
            (ChatMessage(role="user", content="Retrieve schemas."),),
            progress_callback=events.append,
        )
        assert handled == list(range(15))
        assert result.tool_calls == 15
        assert result.final_response == "Finished."
        assert result.model_turns == 2
        assert client.requests[-1][1] == ()
        assert events[-1].kind == "response_ready"
    else:
        with pytest.raises(ToolCallLimitExceededError, match="at most 15") as captured:
            await runtime.run(
                (ChatMessage(role="user", content="Retrieve schemas."),),
                progress_callback=events.append,
            )
        error = captured.value
        assert handled == []
        assert error.tool_calls_used == 0
        assert error.requested_tool_calls == 16
        assert error.tool_results == error.tool_batch_runtime == ()
        assert error.transcript[-1].tool_calls == calls
        assert len(client.requests) == error.model_turns == 1
        assert [event.kind for event in events] == ["model_turn_started"]


@pytest.mark.parametrize(("completed", "requested"), ((10, 6), (15, 1)))
async def test_shared_budget_rejects_later_query_batch_and_preserves_exports(
    completed: int,
    requested: int,
) -> None:
    exported: list[int] = []
    queried: list[int] = []
    calls = tuple(
        ToolCall(f"export-{index}", "generate_excel", {"value": index})
        for index in range(completed)
    )
    rejected = tuple(
        ToolCall(f"query-{index}", "query_occurrences", {"value": index})
        for index in range(requested)
    )
    client = FakeClient((ModelTurn(tool_calls=calls), ModelTurn(tool_calls=rejected)))
    events: list[AgentProgressEvent] = []
    runtime = AgentRuntime(client, (
        registered_tool("generate_excel", lambda arguments: exported.append(arguments["value"])),
        registered_tool("query_occurrences", lambda arguments: queried.append(arguments["value"])),
    ))

    with pytest.raises(ToolCallLimitExceededError) as captured:
        await runtime.run(
            (ChatMessage(role="user", content="Generate files, then query."),),
            progress_callback=events.append,
        )

    error = captured.value
    assert exported == list(range(completed))
    assert queried == []
    assert error.tool_calls_used == completed
    assert error.requested_tool_calls == requested
    assert tuple(result.call for result in error.tool_results) == calls
    assert all(result.ok for result in error.tool_results)
    assert error.transcript[-1].tool_calls == rejected
    assert [batch.tool_calls for batch in error.tool_batch_runtime] == [completed]
    assert [event.kind for event in events].count("tool_batch_started") == 1
    assert len(client.requests) == error.model_turns == 2
    assert bool(client.requests[-1][1]) is (completed < 15)


async def test_budget_accumulates_across_model_rounds_and_all_tools() -> None:
    batches = (("retrieve_schema", 1), ("query_occurrences", 9), ("generate_excel", 5))
    client = FakeClient((
        *(ModelTurn(tool_calls=tuple(
            ToolCall(f"{name}-{index}", name, {"value": index})
            for index in range(count)
        )) for name, count in batches),
        ModelTurn(content="Finished."),
    ))
    runtime = AgentRuntime(client, tuple(
        registered_tool(name, lambda arguments: arguments) for name, _ in batches
    ))

    result = await runtime.run((ChatMessage(role="user", content="Use the tools."),))

    assert result.tool_calls == 15
    assert result.model_turns == 4
    assert [batch.tool_calls for batch in result.tool_batch_runtime] == [1, 9, 5]
    assert all(definitions for _, definitions in client.requests[:-1])
    assert client.requests[-1][1] == ()


async def test_failed_and_unknown_calls_consume_the_shared_budget() -> None:
    fail = Mock(side_effect=ToolExecutionError("unavailable", "Try again later."))

    calls = tuple(
        ToolCall(f"{name}-{index}", name, {"value": index})
        for name in ("fragile", "missing", "healthy")
        for index in range(5)
    )
    client = FakeClient((
        ModelTurn(tool_calls=calls),
        ModelTurn(tool_calls=(ToolCall("over-budget", "healthy", {"value": 99}),)),
    ))
    healthy = registered_tool("healthy", lambda arguments: arguments)
    runtime = AgentRuntime(client, (registered_tool("fragile", fail), healthy))

    with pytest.raises(ToolCallLimitExceededError) as captured:
        await runtime.run((ChatMessage(role="user", content="Try these tools."),))

    error = captured.value
    assert error.tool_calls_used == 15
    assert len(error.tool_results) == 15
    assert [result.error_code for result in error.tool_results] == (
        ["unavailable"] * 5 + ["unknown_tool"] * 5 + [None] * 5
    )
    assert healthy.handler.await_count == 5
    assert client.requests[-1][1] == ()


async def test_tool_budget_resets_for_reused_runtime_and_ignores_history_calls() -> None:
    client = FakeClient(tuple(
        turn
        for label in ("first", "second")
        for turn in (
            ModelTurn(tool_calls=tuple(
                ToolCall(f"{label}-{index}", "echo", {"value": index})
                for index in range(15)
            )),
            ModelTurn(content=f"Finished {label}."),
        )
    ))
    runtime = AgentRuntime(client, (registered_tool("echo", lambda arguments: arguments),))

    first = await runtime.run((ChatMessage(role="user", content="First request."),))
    second = await runtime.run((
        *first.transcript, ChatMessage(role="user", content="Second request."),
    ))

    assert first.tool_calls == second.tool_calls == 15
    assert sum(message.role == "tool" for message in second.transcript) == 30
    assert client.requests[1][1] == client.requests[3][1] == ()
    assert client.requests[0][1] == client.requests[2][1]
    assert client.requests[2][1]


async def test_executes_multiple_tool_calls_and_returns_results_to_model() -> None:
    handled: list[tuple[str, dict[str, object]]] = []

    double = partial(_double, handled=handled)

    increment = partial(_increment, handled=handled)

    client = FakeClient(
        (
            ModelTurn(
                tool_calls=(
                    ToolCall("call-1", "double", {"value": 4}),
                    ToolCall("call-2", "increment", {"value": 9}),
                ),
                finish_reason="tool_calls",
            ),
            ModelTurn(content="The results are 8 and 10.", finish_reason="stop"),
        )
    )
    runtime = AgentRuntime(
        client,
        (
            registered_tool("double", double, strict=True),
            registered_tool("increment", increment),
        ),
    )

    result = await runtime.run((ChatMessage(role="user", content="Calculate both."),))

    assert {name: arguments for name, arguments in handled} == {
        "double": {"value": 4},
        "increment": {"value": 9},
    }
    assert result.final_response == "The results are 8 and 10."
    assert result.model_turns == 2
    assert result.tool_calls == 2
    assert [message.role for message in result.transcript] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    second_request = client.requests[1][0]
    assert json.loads(second_request[-2].content) == {
        "ok": True,
        "result": {"value": 8},
    }
    assert json.loads(second_request[-1].content) == {
        "ok": True,
        "result": {"value": 10},
    }
    assert [
        [(definition.name, definition.strict) for definition in definitions]
        for _, definitions in client.requests
    ] == [
        [("double", True), ("increment", False)],
        [("double", True), ("increment", False)],
    ]
    assert result.registered_tool_count == 2


async def test_reports_high_level_model_tool_and_response_progress() -> None:
    client = FakeClient(
        (
            ModelTurn(
                tool_calls=(
                    ToolCall("call-1", "double", {"value": 4}),
                    ToolCall("call-2", "double", {"value": 5}),
                ),
                finish_reason="tool_calls",
            ),
            ModelTurn(content="Finished.", finish_reason="stop"),
        )
    )
    events: list[AgentProgressEvent] = []

    result = await AgentRuntime(
        client,
        (registered_tool("double", lambda arguments: arguments["value"] * 2),),
    ).run(
        (ChatMessage(role="user", content="Calculate both."),),
        progress_callback=events.append,
    )

    assert result.final_response == "Finished."
    assert events == [
        AgentProgressEvent(kind="model_turn_started", turn=1),
        AgentProgressEvent(
            kind="tool_batch_started",
            turn=1,
            tool_names=("double", "double"),
        ),
        AgentProgressEvent(
            kind="tool_batch_completed",
            turn=1,
            tool_results=result.tool_results,
        ),
        AgentProgressEvent(kind="model_turn_started", turn=2),
        AgentProgressEvent(kind="response_ready", turn=2),
    ]


async def test_progress_replaces_unregistered_model_tool_names() -> None:
    client = FakeClient(
        (
            ModelTurn(
                tool_calls=(
                    ToolCall("call-1", "model_controlled_status", {}),
                ),
                finish_reason="tool_calls",
            ),
            ModelTurn(content="Finished.", finish_reason="stop"),
        )
    )
    events: list[AgentProgressEvent] = []

    await AgentRuntime(client).run(
        (ChatMessage(role="user", content="Try an invalid tool."),),
        progress_callback=events.append,
    )

    assert events[1] == AgentProgressEvent(
        kind="tool_batch_started",
        turn=1,
        tool_names=("unknown_tool",),
    )
    assert events[2].kind == "tool_batch_completed"
    assert events[2].tool_results[0].call.name == "model_controlled_status"


async def test_cancellation_stops_after_a_completed_tool_batch() -> None:
    client = FakeClient(
        (
            ModelTurn(
                tool_calls=(ToolCall("call-1", "double", {"value": 4}),),
                finish_reason="tool_calls",
            ),
            ModelTurn(content="Unused.", finish_reason="stop"),
        )
    )
    events: list[AgentProgressEvent] = []
    cancellation = AgentCancellation()

    request_cancellation = partial(
        _cancel_after_progress,
        kind="tool_batch_completed",
        cancellation=cancellation,
        events=events,
    )

    with pytest.raises(AgentCancelledError, match="Stopped thinking") as captured:
        await AgentRuntime(
            client,
            (registered_tool("double", lambda arguments: arguments["value"] * 2),),
        ).run(
            (ChatMessage(role="user", content="Calculate once."),),
            progress_callback=request_cancellation,
            cancellation=cancellation,
        )

    assert [event.kind for event in events] == [
        "model_turn_started",
        "tool_batch_started",
        "tool_batch_completed",
    ]
    assert len(client.requests) == 1
    assert len(captured.value.tool_results) == 1


async def test_cancellation_stops_before_model_execution() -> None:
    client = FakeClient((ModelTurn(content="Unused.", finish_reason="stop"),))
    cancellation = AgentCancellation()
    cancellation.cancel()

    with pytest.raises(AgentCancelledError, match="Stopped thinking"):
        await AgentRuntime(client).run(
            (ChatMessage(role="user", content="Do not execute."),),
            cancellation=cancellation,
        )

    assert client.requests == []
    await asyncio.sleep(0)


async def test_cancellation_interrupts_a_pending_model_from_another_thread() -> None:
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()
    cancellation = AgentCancellation()


    task = asyncio.create_task(AgentRuntime(
        _PendingModelClient(started, cleanup_started, finish_cleanup),
    ).run(
        (ChatMessage(role="user", content="Wait for the model."),),
        cancellation=cancellation,
    ))
    await asyncio.wait_for(started.wait(), timeout=5)
    await asyncio.to_thread(cancellation.cancel)
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=5)
        await asyncio.to_thread(cancellation.cancel)
        assert not task.done()
    finally:
        finish_cleanup.set()
    with pytest.raises(AgentCancelledError) as captured:
        await asyncio.wait_for(task, timeout=5)

    error = captured.value
    assert error.transcript == (
        ChatMessage(role="user", content="Wait for the model."),
    )
    assert error.tool_results == ()
    assert error.model_turns == 1
    assert error.model_turn_runtime[0].status == "failed"
    assert error.model_turn_runtime[0].error_type == "AgentCancelledError"


async def test_completed_binding_cannot_cancel_a_later_run_on_the_same_task() -> None:
    first_cancellation = AgentCancellation()
    started = asyncio.Event()
    release = asyncio.Event()


    runtime = AgentRuntime(_FollowupClient(started, release))


    task = asyncio.create_task(_run_followup(runtime, first_cancellation))
    await asyncio.wait_for(started.wait(), timeout=5)
    try:
        await asyncio.to_thread(first_cancellation.cancel)
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    result = await asyncio.wait_for(task, timeout=5)
    assert result.final_response == "Followup answer."


async def test_queued_cancellation_cannot_affect_a_later_run_on_the_same_task() -> None:
    first_cancellation = AgentCancellation()

    cancel_before_return = partial(
        _cancel_after_progress, kind="response_ready", cancellation=first_cancellation,
    )

    with pytest.raises(AgentCancelledError):
        await AgentRuntime(FakeClient((ModelTurn(content="Cancelled answer."),))).run(
            (ChatMessage(role="user", content="First request."),),
            cancellation=first_cancellation,
            progress_callback=cancel_before_return,
        )


    result = await AgentRuntime(_YieldingFollowupClient()).run(
        (ChatMessage(role="user", content="Followup request."),),
        cancellation=AgentCancellation(),
    )
    assert result.final_response == "Followup answer."


async def test_repeated_runs_on_one_task_can_each_be_cancelled() -> None:
    started: asyncio.Queue[None] = asyncio.Queue()
    cancellations = (AgentCancellation(), AgentCancellation())
    errors: list[AgentCancelledError] = []


    task = asyncio.create_task(_cancel_repeated_runs(started, cancellations, errors))
    for cancellation in cancellations:
        await asyncio.wait_for(started.get(), timeout=5)
        await asyncio.to_thread(cancellation.cancel)
    await asyncio.wait_for(task, timeout=5)
    assert len(errors) == 2


async def test_keyboard_interrupt_becomes_an_explicit_cancellation() -> None:

    cancellation = AgentCancellation()

    with pytest.raises(AgentCancelledError, match="Stopped thinking") as captured:
        await AgentRuntime(FakeClient((KeyboardInterrupt(),))).run(
            (ChatMessage(role="user", content="Stop this."),),
            cancellation=cancellation,
        )

    assert cancellation.is_cancelled
    assert captured.value.transcript == (
        ChatMessage(role="user", content="Stop this."),
    )


async def test_records_model_tool_and_batch_runtime_with_a_deterministic_clock() -> None:
    clock_values = iter((0.0, 2.0, 2.5, 3.0, 5.0, 6.0, 6.5, 7.5))
    usage = TokenUsage(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        cached_prompt_tokens=60,
        reasoning_tokens=8,
    )
    provider_metrics = ModelRequestMetrics(
        provider="vllm",
        time_to_first_token_seconds=0.25,
        generation_seconds=1.5,
        queue_seconds=0.01,
        mean_inter_token_latency_seconds=0.05,
        tokens_per_second=20,
    )
    client = FakeClient(
        (
            ModelTurn(
                tool_calls=(ToolCall("call-1", "echo", {"value": 1}),),
                response_id="chatcmpl-1",
                usage=usage,
                provider_metrics=provider_metrics,
            ),
            ModelTurn(content="Finished.", response_id="chatcmpl-2"),
        )
    )

    result = await AgentRuntime(
        client,
        (registered_tool("echo", lambda arguments: arguments),),
        clock=lambda: next(clock_values),
    ).run((ChatMessage(role="user", content="Echo."),))

    assert [item.client_round_trip_seconds for item in result.model_turn_runtime] == [
        2.0,
        1.0,
    ]
    assert result.model_turn_runtime[0].response_id == "chatcmpl-1"
    assert result.model_turn_runtime[0].usage is usage
    assert result.model_turn_runtime[0].provider_metrics is provider_metrics
    assert result.tool_results[0].elapsed_seconds == 2.0
    assert result.tool_batch_runtime[0].turn == 1
    assert result.tool_batch_runtime[0].tool_calls == 1
    assert result.tool_batch_runtime[0].wall_elapsed_seconds == 3.5


async def test_model_client_failure_retains_failed_turn_runtime() -> None:

    clock_values = iter((10.0, 12.5))
    with pytest.raises(ModelClientExecutionError) as captured:
        await AgentRuntime(
            FakeClient((RuntimeError("endpoint stopped"),)),
            clock=lambda: next(clock_values),
        ).run((ChatMessage(role="user", content="Try."),))

    assert captured.value.model_turn_runtime[0].turn == 1
    assert captured.value.model_turn_runtime[0].status == "failed"
    assert captured.value.model_turn_runtime[0].client_round_trip_seconds == 2.5
    assert captured.value.model_turn_runtime[0].error_type == "RuntimeError"
    assert captured.value.tool_batch_runtime == ()


@pytest.mark.parametrize("tool_names", (("first", "second"), ("lookup",) * 8))
@pytest.mark.parametrize("prior_failures", (0, 2))
async def test_independent_calls_overlap_but_results_retain_model_call_order(
    tool_names: tuple[str, ...],
    prior_failures: int,
) -> None:
    call_count = len(tool_names)
    all_started = asyncio.Barrier(call_count)
    can_complete = [asyncio.Event() for _ in range(call_count + 1)]
    can_complete[-1].set()
    completion_order: list[int] = []

    coordinate = partial(
        _coordinate_calls,
        all_started=all_started,
        can_complete=can_complete,
        completion_order=completion_order,
    )

    client = FakeClient(
        (
            *(ModelTurn(tool_calls=(
                ToolCall(f"failed-{i}", tool_names[0], {"value": 0}),
            )) for i in range(prior_failures)),
            ModelTurn(
                tool_calls=tuple(
                    ToolCall(f"call-{i}", name, {"value": i})
                    for i, name in enumerate(tool_names, start=1)
                ),
            ),
            ModelTurn(content="All completed."),
        )
    )

    result = await AgentRuntime(
        client,
        tuple(registered_tool(name, coordinate) for name in dict.fromkeys(tool_names)),
    ).run((ChatMessage(role="user", content="Run the independent calls."),))

    assert completion_order == list(range(call_count, 0, -1))
    concurrent_results = result.tool_results[prior_failures:]
    call_ids = [f"call-{i}" for i in range(1, call_count + 1)]
    assert [item.call.id for item in concurrent_results] == call_ids
    assert [item.result for item in concurrent_results] == [
        {"value": i} for i in range(1, call_count + 1)
    ]
    individual_elapsed = [
        item.elapsed_seconds for item in concurrent_results
    ]
    assert all(value is not None for value in individual_elapsed)
    assert result.tool_batch_runtime[-1].tool_calls == call_count
    assert result.tool_batch_runtime[-1].wall_elapsed_seconds >= max(
        value for value in individual_elapsed if value is not None
    )
    assert [
        message.tool_call_id
        for message in result.transcript
        if message.role == "tool"
    ][prior_failures:] == call_ids


async def test_projects_internal_result_before_messages_results_and_trace() -> None:
    events: list[str] = []
    private_marker = "UNPROJECTED-INTERNAL-FIELD"
    internal_result = {
        "records": [{"visible": "kept", "unused": private_marker}],
    }


    projection = _ResultProjection(
        internal_result,
        {"records": [{"visible": internal_result["records"][0]["visible"]}]},
        events,
    )

    client = FakeClient(
        (
            ModelTurn(
                tool_calls=(ToolCall("call-1", "projected", {"value": 1}),)
            ),
            ModelTurn(content="Used the projected result."),
        )
    )
    result = await AgentRuntime(
        client,
        (registered_tool("projected", projection.handle, projection.project),),
    ).run((ChatMessage(role="user", content="Run it."),))

    assert events == ["handler", "projector"]
    projected_payload = {
        "ok": True,
        "result": {
            "records": [{"visible": "kept"}],
        },
    }
    assert json.loads(client.requests[1][0][-1].content) == projected_payload
    assert result.tool_results[0].payload() == projected_payload
    assert json.loads(result.transcript[-2].content) == projected_payload

    trace = serialize_agent_trace(
        build_trace_record(
            result,
            run_id="projection-run",
            started_at="2026-08-25T12:00:00Z",
            ended_at="2026-08-25T12:00:01Z",
            elapsed_seconds=1,
        ),
    )
    assert private_marker not in trace


async def test_bounded_tool_error_is_returned_so_model_can_recover() -> None:
    long_message = "invalid input\n" + ("x" * 1_000)

    fail = Mock(side_effect=ToolExecutionError("invalid_value", long_message))

    client = FakeClient(
        (
            ModelTurn(tool_calls=(ToolCall("call-1", "fragile", {"value": 2}),)),
            ModelTurn(content="Please provide another value."),
        )
    )
    await AgentRuntime(client, (registered_tool("fragile", fail),)).run(
        (ChatMessage(role="user", content="Try it."),)
    )

    payload = json.loads(client.requests[1][0][-1].content)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_value"
    assert "\n" not in payload["error"]["message"]
    assert len(payload["error"]["message"]) == 512


async def test_unknown_and_invalid_results_are_recoverable_tool_failures() -> None:
    invalid_result = Mock(return_value={"not_json": object()})

    unexpected_failure = Mock(side_effect=RuntimeError(
        "internal detail must not reach the model",
    ))

    client = FakeClient(
        (
            ModelTurn(
                tool_calls=(
                    ToolCall("call-1", "missing", {}),
                    ToolCall("call-2", "invalid", {"value": 1}),
                    ToolCall("call-3", "crash", {"value": 1}),
                )
            ),
            ModelTurn(content="Recovered."),
        )
    )
    result = await AgentRuntime(
        client,
        (
            registered_tool("invalid", invalid_result),
            registered_tool("crash", unexpected_failure),
        ),
    ).run((ChatMessage(role="user", content="Use tools."),))

    assert [tool_result.error_code for tool_result in result.tool_results] == [
        "unknown_tool",
        "invalid_tool_result",
        "tool_execution_error",
    ]
    assert all(not tool_result.ok for tool_result in result.tool_results)
    assert result.tool_results[-1].error_message == "This operation could not be completed."


@pytest.mark.parametrize("projected", (False, True))
async def test_large_tool_results_are_returned_to_model_intact(projected: bool) -> None:
    expected_result = {
        "records": [
            {"id": index, "text": "海藻" * 2_000}
            for index in range(20)
        ],
    }
    internal_result = object() if projected else expected_result

    projection = _ResultProjection(internal_result, expected_result)

    client = FakeClient(
        (
            ModelTurn(tool_calls=(ToolCall("call-1", "large", {"value": 1}),)),
            ModelTurn(content="Used the complete result."),
        )
    )
    result = await AgentRuntime(
        client,
        (registered_tool(
            "large",
            lambda arguments: internal_result,
            projection.project if projected else None,
        ),),
    ).run((ChatMessage(role="user", content="Return all records."),))

    expected_payload = {"ok": True, "result": expected_result}
    tool_message = client.requests[1][0][-1]
    assert len(tool_message.content.encode("utf-8")) > 65_536
    assert json.loads(tool_message.content) == expected_payload
    assert result.tool_results[0].payload() == expected_payload
    assert json.loads(result.transcript[-2].content) == expected_payload
    assert result.final_response == "Used the complete result."


async def test_duplicate_registrations_and_malformed_calls_are_rejected() -> None:
    tool = registered_tool("duplicate", lambda arguments: None)
    with pytest.raises(ValueError, match="duplicate registered tool"):
        AgentRuntime(FakeClient(()), (tool, tool))

    with pytest.raises(TypeError, match="JSON object"):
        ToolCall("call-1", "duplicate", [])  # type: ignore[arg-type]

    duplicate_call = ToolCall("same-id", "duplicate", {"value": 1})
    with pytest.raises(ValueError, match="duplicate tool call ids"):
        ModelTurn(tool_calls=(duplicate_call, duplicate_call))


async def test_invalid_client_turn_and_reused_call_id_preserve_transcript() -> None:
    invalid_client = FakeClient(({"content": "not a ModelTurn"},))
    with pytest.raises(InvalidModelTurnError) as invalid_error:
        await AgentRuntime(invalid_client).run(
            (ChatMessage(role="user", content="Hello."),)
        )
    assert [message.role for message in invalid_error.value.transcript] == ["user"]

    tool = registered_tool("echo", lambda arguments: arguments)
    reused_call_client = FakeClient(
        (
            ModelTurn(tool_calls=(ToolCall("same-id", "echo", {"value": 1}),)),
            ModelTurn(tool_calls=(ToolCall("same-id", "echo", {"value": 2}),)),
        )
    )
    with pytest.raises(InvalidModelTurnError, match="reused tool call") as error:
        await AgentRuntime(reused_call_client, (tool,)).run(
            (ChatMessage(role="user", content="Echo."),)
        )
    assert len(error.value.tool_results) == 1
    assert error.value.model_turns == 2


@pytest.mark.parametrize(
    "turn, message",
    (
        (ModelTurn(content="partial", finish_reason="length"), "finish reason"),
        (ModelTurn(content="", finish_reason="stop"), "empty final response"),
        (
            ModelTurn(content="no call", finish_reason="tool_calls"),
            "without requesting a tool",
        ),
    ),
)
async def test_rejects_incomplete_or_empty_final_turns(turn, message) -> None:
    with pytest.raises(InvalidModelTurnError, match=message) as error:
        await AgentRuntime(FakeClient((turn,))).run(
            (ChatMessage(role="user", content="Finish the task."),)
        )

    assert [item.role for item in error.value.transcript] == ["user", "assistant"]
    assert error.value.model_turns == 1


@pytest.mark.parametrize("call_count", (1, 15))
async def test_concurrent_runs_keep_transcripts_call_ids_and_budgets_isolated(
    call_count: int,
) -> None:
    first_turns_started = asyncio.Barrier(4)


    runtime = AgentRuntime(
        _ConcurrentClient(first_turns_started, call_count),
        (registered_tool("echo", lambda arguments: arguments),),
    )


    results = dict(zip(range(1, 5), await asyncio.gather(*(
        runtime.run((ChatMessage(role="user", content=f"Run value {value}"),))
        for value in range(1, 5)
    ))))

    for value, result in results.items():
        assert result.final_response == f"Completed {value}."
        assert [message.role for message in result.transcript] == [
            "user",
            "assistant",
            *(["tool"] * call_count),
            "assistant",
        ]
        assert [
            message.content
            for message in result.transcript
            if message.role == "user"
        ] == [f"Run value {value}"]
        assert result.tool_calls == call_count
        assert [item.call.id for item in result.tool_results] == [
            f"shared-call-id-{index}" for index in range(call_count)
        ]
        assert all(item.result == {"value": value} for item in result.tool_results)


async def test_process_control_exception_is_not_wrapped() -> None:

    with pytest.raises(KeyboardInterrupt):
        await AgentRuntime(FakeClient((KeyboardInterrupt(),))).run(
            (ChatMessage(role="user", content="Stop."),)
        )


@pytest.mark.parametrize("calls_per_round", (1, 5))
async def test_stops_after_three_failed_rounds_regardless_of_batch_size(
    calls_per_round: int,
) -> None:
    attempts: list[tuple[int, int]] = []
    all_started = asyncio.Barrier(calls_per_round)

    fail = partial(_fail_after_barrier, attempts=attempts, all_started=all_started)

    calls_by_round = tuple(tuple(
        ToolCall(f"call-{round_index}-{i}", "fragile", {"round": round_index, "value": i})
        for i in range(calls_per_round)
    ) for round_index in range(3))
    client = FakeClient((
        *(ModelTurn(tool_calls=calls) for calls in calls_by_round),
        ModelTurn(content="This turn must not execute."),
    ))
    events: list[AgentProgressEvent] = []
    with pytest.raises(ToolAttemptsExhaustedError, match="three consecutive rounds") as captured:
        await AgentRuntime(client, (registered_tool("fragile", fail),)).run(
            (ChatMessage(role="user", content="Try the tool."),),
            progress_callback=events.append,
        )

    error = captured.value
    assert attempts == [(r, i) for r in range(3) for i in range(calls_per_round)]
    assert len(client.requests) == error.model_turns == 3
    assert error.failed_result is error.tool_results[-1]
    assert error.failed_result.call.id == f"call-2-{calls_per_round - 1}"
    assert error.registered_tool_count == 1
    assert [message.tool_call_id for message in error.transcript if message.role == "tool"] == [
        call.id for calls in calls_by_round for call in calls
    ]
    assert [batch.tool_calls for batch in error.tool_batch_runtime] == [calls_per_round] * 3
    assert events[-1].kind == "tool_batch_completed"
    assert "Please check or clarify your request" in str(error)


async def test_fully_successful_round_resets_the_tools_retry_streak() -> None:

    round_values = ((0, 0), (0,), (1, 1), (0,), (0, 0), (1,))
    turns = tuple(ModelTurn(tool_calls=tuple(
        ToolCall(f"call-{round_index}-{i}", "fragile", {"value": value})
        for i, value in enumerate(values)
    )) for round_index, values in enumerate(round_values))
    result = await AgentRuntime(
        FakeClient((*turns, ModelTurn(content="Recovered."))),
        (registered_tool("fragile", _reject_zero),),
    ).run((ChatMessage(role="user", content="Correct the failures."),))

    assert result.final_response == "Recovered."
    assert result.model_turns == 7
    assert [item.ok for item in result.tool_results] == [
        bool(value) for values in round_values for value in values
    ]


async def test_partial_success_cannot_hide_repeated_failed_rounds() -> None:

    client = FakeClient(tuple(ModelTurn(tool_calls=(
        ToolCall(f"failed-{i}", "lookup", {"value": 0}),
        ToolCall(f"succeeded-{i}", "lookup", {"value": 1}),
    )) for i in range(3)))
    with pytest.raises(ToolAttemptsExhaustedError) as captured:
        await AgentRuntime(client, (registered_tool("lookup", _reject_zero),)).run(
            (ChatMessage(role="user", content="Use both queries."),),
        )

    assert captured.value.model_turns == 3
    assert captured.value.failed_result.call.id == "failed-2"
    assert [item.ok for item in captured.value.tool_results] == [False, True] * 3


async def test_another_tools_success_does_not_reset_a_failing_tools_streak() -> None:
    fail = Mock(side_effect=ToolExecutionError("invalid_input", "Correct the arguments."))

    client = FakeClient((
        ModelTurn(tool_calls=(ToolCall("a-1", "fragile", {}),)),
        ModelTurn(tool_calls=(ToolCall("a-2", "fragile", {}),)),
        ModelTurn(tool_calls=(ToolCall("b-1", "healthy", {}),)),
        ModelTurn(tool_calls=(ToolCall("a-3", "fragile", {}),)),
    ))
    with pytest.raises(ToolAttemptsExhaustedError) as captured:
        await AgentRuntime(client, (
            registered_tool("fragile", fail),
            registered_tool("healthy", lambda arguments: "Succeeded."),
        )).run((ChatMessage(role="user", content="Use both tools."),))

    assert captured.value.failed_result.call.id == "a-3"
    assert [item.ok for item in captured.value.tool_results] == [False, False, True, False]
    assert len(client.requests) == 4


async def test_retry_rounds_are_isolated_between_concurrent_chats() -> None:
    both_started = asyncio.Barrier(2)


    fail = Mock(side_effect=ToolExecutionError("unavailable", "Try again later."))

    runtime = AgentRuntime(_RetryClient(both_started), (registered_tool("lookup", fail),))
    first, second = await asyncio.gather(
        runtime.run((ChatMessage(role="user", content="first"),)),
        runtime.run((ChatMessage(role="user", content="second"),)),
        return_exceptions=True,
    )

    assert isinstance(first, ToolAttemptsExhaustedError)
    assert first.model_turns == 3
    assert len(first.tool_results) == 3
    assert second.final_response == "Explained the unavailable result."
    assert len(second.tool_results) == 2


async def test_exhaustion_preserves_another_tools_in_flight_result() -> None:
    healthy_started = asyncio.Event()
    release_healthy = asyncio.Event()

    fail = partial(
        _exhaustion_fail, healthy_started=healthy_started, release_healthy=release_healthy,
    )

    succeed = partial(
        _exhaustion_succeed, healthy_started=healthy_started, release_healthy=release_healthy,
    )

    client = FakeClient((
        ModelTurn(tool_calls=(ToolCall("prior-1", "fragile", {"value": 0}),)),
        ModelTurn(tool_calls=(ToolCall("prior-2", "fragile", {"value": 0}),)),
        ModelTurn(tool_calls=(
            *(ToolCall(f"a-{value}", "fragile", {"value": value}) for value in range(1, 5)),
            ToolCall("b-1", "healthy", {}),
        )),
    ))
    with pytest.raises(ToolAttemptsExhaustedError) as captured:
        await AgentRuntime(client, (
            registered_tool("fragile", fail),
            registered_tool("healthy", succeed),
        )).run((ChatMessage(role="user", content="Run both tools."),))

    error = captured.value
    assert error.failed_result.call.id == "a-4"
    assert [result.call.id for result in error.tool_results] == [
        "prior-1", "prior-2", "a-1", "a-2", "a-3", "a-4", "b-1",
    ]
    assert error.tool_results[-1].result == "Completed."
    assert error.tool_batch_runtime[-1].tool_calls == 5
    assert len(client.requests) == 3


@pytest.mark.parametrize("cancel_from_token", (False, True))
async def test_cancellation_closes_pending_tools_and_preserves_completed_work(
    cancel_from_token: bool,
) -> None:
    pending_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()
    closed: list[int] = []
    cancellation = AgentCancellation()

    pending = _PendingTools(pending_started, cleanup_started, finish_cleanup, closed)

    client = FakeClient((ModelTurn(tool_calls=(
        ToolCall("done", "lookup", {"value": 0}),
        ToolCall("pending-1", "lookup", {"value": 1}),
        ToolCall("pending-2", "lookup", {"value": 2}),
    )),))
    runtime = AgentRuntime(client, (registered_tool("lookup", pending.handle),))
    task = asyncio.create_task(runtime.run(
        (ChatMessage(role="user", content="Look up both."),),
        cancellation=cancellation,
    ))
    await asyncio.wait_for(pending_started.wait(), timeout=5)
    if cancel_from_token:
        await asyncio.to_thread(cancellation.cancel)
    else:
        task.cancel()
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=5)
        await asyncio.to_thread(cancellation.cancel)
        assert not task.done()
    finally:
        finish_cleanup.set()
    with pytest.raises(AgentCancelledError) as captured:
        await asyncio.wait_for(task, timeout=5)

    assert sorted(closed) == [1, 2]
    assert cancellation.is_cancelled
    assert [item.call.id for item in captured.value.tool_results] == ["done"]
    assert captured.value.transcript[-1].tool_call_id == "done"
    assert captured.value.tool_batch_runtime[0].tool_calls == 1
    assert len(client.requests) == 1


async def test_handler_mutation_does_not_change_recorded_arguments_or_results() -> None:
    returned_value = {"records": [{"name": "Algues — 海藻"}]}

    mutate_arguments = partial(_mutate_arguments, returned_value=returned_value)

    call = ToolCall("call-1", "mutate", {"nested": {"value": "original"}})
    client = FakeClient((ModelTurn(tool_calls=(call,)), ModelTurn(content="Finished.")))
    result = await AgentRuntime(client, (registered_tool("mutate", mutate_arguments),)).run(
        (ChatMessage(role="user", content="Preserve the execution record."),)
    )
    returned_value["records"][0]["name"] = "changed"

    assert call.arguments == {"nested": {"value": "original"}}
    expected = {"ok": True, "result": {"records": [{"name": "Algues — 海藻"}]}}
    assert result.tool_results[0].payload() == expected
    assert json.loads(result.transcript[-2].content) == expected
