"""Run model/tool conversations with shared call and per-tool retry limits."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.gaia_agent.runtime.contracts import (
    AgentCancellation,
    AgentCancelledError,
    AgentRuntimeError,
    AgentProgressEvent,
    AgentProgressEventKind,
    AgentProgressHandler,
    AgentResult,
    ChatMessage,
    ContextCompactionError,
    ContextInfo,
    InvalidModelTurnError,
    ModelClient,
    ModelClientExecutionError,
    ModelTurn,
    ModelTurnRuntime,
    RegisteredTool,
    ResponseDisclosureError,
    ToolAttemptsExhaustedError,
    ToolBatchRuntime,
    ToolCall,
    ToolCallLimitExceededError,
    ToolDefinition,
    ToolExecutionError,
    ToolResult,
)
from src.gaia_agent.runtime.context import ContextBudget, compact_context, write_history
from src.gaia_agent.runtime.defaults import CONTEXT_CONTINUATION_PROMPT, MAX_TOOL_CALLS_PER_TURN
from src.gaia_agent.runtime.disclosure import ResponseDisclosureGuard


MAX_CONSECUTIVE_FAILED_TOOL_ROUNDS = 3
MAX_ERROR_CHARACTERS = 512


def _bounded_message(message: str) -> str:
    """Clip tool error text to a printable message."""
    printable = "".join(
        character if character.isprintable() else " " for character in message.strip()
    )
    if len(printable) <= MAX_ERROR_CHARACTERS:
        return printable
    return f"{printable[: MAX_ERROR_CHARACTERS - 3]}..."


@dataclass(slots=True)
class _RunState:
    """Keep one interaction's transcript, allowance, and progress together."""

    transcript: list[ChatMessage]
    context: list[ChatMessage]
    registered_tool_count: int
    progress_callback: AgentProgressHandler | None
    cancellation: AgentCancellation | None
    history_path: Path | None
    tool_results: list[ToolResult] = field(default_factory=list)
    model_turn_runtime: list[ModelTurnRuntime] = field(default_factory=list)
    tool_batch_runtime: list[ToolBatchRuntime] = field(default_factory=list)
    failed_tool_rounds: dict[str, int] = field(default_factory=dict)
    seen_call_ids: set[str] = field(default_factory=set, init=False)
    model_turns: int = 0
    tool_calls_used: int = 0

    def __post_init__(self) -> None:
        self.seen_call_ids.update(
            call.id for message in self.transcript for call in message.tool_calls
        )
        self.seen_call_ids.update(
            message.tool_call_id
            for message in self.transcript
            if message.role == "tool" and message.tool_call_id is not None
        )

    def error(
        self,
        error_type: type[AgentRuntimeError],
        message: str,
        **details: Any,
    ) -> AgentRuntimeError:
        return error_type(
            message,
            transcript=self.transcript,
            tool_results=self.tool_results,
            model_turns=self.model_turns,
            registered_tool_count=self.registered_tool_count,
            model_turn_runtime=self.model_turn_runtime,
            tool_batch_runtime=self.tool_batch_runtime,
            **details,
        )

    def append_message(self, message: ChatMessage) -> None:
        """Retain exact audit messages separately from replaceable model context."""
        self.transcript.append(message)
        self.context.append(message)

    def ensure_active(self, cause: BaseException | None = None) -> None:
        if self.cancellation is not None and self.cancellation.is_cancelled:
            raise self.error(AgentCancelledError, "Stopped thinking") from cause

    def report_progress(
        self,
        kind: AgentProgressEventKind,
        *,
        tool_names: tuple[str, ...] = (),
        batch_results: tuple[ToolResult, ...] = (),
    ) -> None:
        if self.progress_callback is not None:
            self.progress_callback(AgentProgressEvent(
                kind=kind,
                turn=self.model_turns,
                tool_names=tool_names,
                tool_results=batch_results,
            ))

    def validate_tool_calls(self, calls: tuple[ToolCall, ...]) -> None:
        if len(calls) > MAX_TOOL_CALLS_PER_TURN - self.tool_calls_used:
            raise self.error(
                ToolCallLimitExceededError,
                f"This interaction allows at most {MAX_TOOL_CALLS_PER_TURN} "
                f"tool calls. {self.tool_calls_used} have already been used; the "
                f"model requested {len(calls)} more. Please narrow "
                "the request or continue in a new message.",
                tool_calls_used=self.tool_calls_used,
                requested_tool_calls=len(calls),
            )
        duplicate_ids = [call.id for call in calls if call.id in self.seen_call_ids]
        if duplicate_ids:
            rendered = ", ".join(repr(identifier) for identifier in duplicate_ids)
            raise self.error(
                InvalidModelTurnError, f"model reused tool call id(s): {rendered}",
            )
        self.seen_call_ids.update(call.id for call in calls)


def _cancel_active_task(
    cancellation: AgentCancellation,
    error: BaseException,
) -> None:
    cancellation.cancel()
    task = asyncio.current_task()
    if isinstance(error, asyncio.CancelledError) and task is not None:
        task.uncancel()


def _tool_failure(call: ToolCall, code: str, message: str) -> ToolResult:
    return ToolResult(
        call=call,
        ok=False,
        error_code=code,
        error_message=_bounded_message(message),
    )


@dataclass(slots=True)
class _CompactionClient:
    """Account for summarization requests in the interaction's model trace."""

    runtime: AgentRuntime
    state: _RunState

    async def context_info(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        enable_thinking: bool | None = None,
    ) -> ContextInfo:
        return await self.runtime._client.context_info(
            messages=messages, tools=tools, enable_thinking=enable_thinking,
        )

    async def count_text_tokens(self, text: str) -> int:
        return await self.runtime._client.count_text_tokens(text)

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        max_completion_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> ModelTurn:
        return await self.runtime._request_model_turn(
            self.state,
            messages=messages,
            tools=tools,
            max_completion_tokens=max_completion_tokens,
            enable_thinking=enable_thinking,
            purpose="compaction",
        )


class AgentRuntime:
    """Run within one user interaction's tool-call budget and retry guards.

    Independent calls execute asynchronously, including calls to the same tool.
    A batch counts as one round per tool, regardless of how many calls it has.
    A fully successful round resets that tool's retry streak. Retry state is
    local to this conversation's current request. The shared call budget counts
    every attempted call, including failures and unknown tools; an over-budget
    batch is rejected before any of its calls execute.
    """

    def __init__(
        self,
        client: ModelClient,
        tools: Iterable[RegisteredTool] = (),
        *,
        clock: Callable[[], float] = time.perf_counter,
        response_guard: ResponseDisclosureGuard | None = None,
    ) -> None:
        registered: dict[str, RegisteredTool] = {}
        for tool in tools:
            name = tool.definition.name
            if name in registered:
                raise ValueError(f"duplicate registered tool: {name!r}")
            registered[name] = tool

        self._client = client
        self._tools = registered
        self._definitions = tuple(tool.definition for tool in registered.values())
        self._clock = clock
        self._response_guard = response_guard

    @property
    def registered_tool_count(self) -> int:
        """Return the catalog size recorded in execution traces."""
        return len(self._tools)

    async def run(
        self,
        messages: Iterable[ChatMessage],
        *,
        progress_callback: AgentProgressHandler | None = None,
        cancellation: AgentCancellation | None = None,
        history_path: Path | None = None,
    ) -> AgentResult:
        """Execute a conversation, preserving completed work on failure."""
        unbind = None
        if cancellation is not None:
            task = asyncio.current_task()
            if task is None:
                raise RuntimeError("an agent run requires an asyncio task")
            unbind = cancellation.bind_task(task)
        try:
            return await self._run(
                messages,
                progress_callback=progress_callback,
                cancellation=cancellation,
                history_path=history_path,
            )
        finally:
            if unbind is not None:
                unbind()

    async def _run(
        self,
        messages: Iterable[ChatMessage],
        *,
        progress_callback: AgentProgressHandler | None,
        cancellation: AgentCancellation | None,
        history_path: Path | None,
    ) -> AgentResult:
        """Run model and tool rounds within the active task binding."""
        transcript = list(messages)
        if not transcript:
            raise ValueError("an agent run requires at least one initial message")

        state = _RunState(
            transcript=transcript,
            context=list(transcript),
            registered_tool_count=self.registered_tool_count,
            progress_callback=progress_callback,
            cancellation=cancellation,
            history_path=history_path,
        )
        state.ensure_active()
        while True:
            turn = await self._complete_model_turn(state)
            if not turn.tool_calls:
                return await self._finish_turn(state, turn)
            await self._run_tool_round(state, turn.tool_calls)

    async def _complete_model_turn(self, state: _RunState) -> ModelTurn:
        """Request and record one model response before interpreting its work."""
        info = await self._prepare_context(state)
        turn = await self._request_model_turn(
            state,
            messages=tuple(state.context),
            tools=self._available_definitions(state),
            max_completion_tokens=(
                ContextBudget(info.context_limit_tokens).usable_tokens - info.prompt_tokens
            ),
        )
        state.append_message(ChatMessage(
            role="assistant",
            content=turn.content,
            reasoning_content=turn.reasoning_content,
            reasoning_field=turn.reasoning_field,
            tool_calls=turn.tool_calls,
        ))
        if turn.finish_reason not in {None, "stop", "tool_calls"}:
            raise state.error(
                InvalidModelTurnError,
                f"model turn ended with unusable finish reason {turn.finish_reason!r}",
            )
        return turn

    def _available_definitions(self, state: _RunState) -> tuple[ToolDefinition, ...]:
        return self._definitions if state.tool_calls_used < MAX_TOOL_CALLS_PER_TURN else ()

    async def _prepare_context(
        self,
        state: _RunState,
        *,
        tools: tuple[ToolDefinition, ...] | None = None,
    ) -> ContextInfo:
        """Compact complete exchanges before a request can exceed its budget."""
        state.ensure_active()
        try:
            if tools is None:
                tools = self._available_definitions(state)
            info = await self._client.context_info(messages=tuple(state.context), tools=tools)
            state.ensure_active()
            budget = ContextBudget(info.context_limit_tokens)
            if info.prompt_tokens < budget.trigger_tokens:
                return info
            if state.history_path is None:
                raise ValueError("the chat has no compaction history location")
            state.report_progress("context_compacting")
            state.ensure_active()
            markdown = await compact_context(
                _CompactionClient(self, state),
                tuple(state.context),
                context_limit_tokens=info.context_limit_tokens,
                cancellation=state.cancellation,
            )
            if (
                self._response_guard is not None
                and not self._response_guard.allows_compaction(markdown)
            ):
                raise ValueError("the compacted history contains private runtime information")
            replacement = [message for message in state.context if message.role == "system"]
            # The deployed chat template requires a user turn before assistant history.
            replacement.append(ChatMessage(role="user", content=CONTEXT_CONTINUATION_PROMPT))
            replacement.append(ChatMessage(role="assistant", content=markdown))
            replacement_info = await self._client.context_info(
                messages=tuple(replacement), tools=tools,
            )
            replacement_budget = ContextBudget(replacement_info.context_limit_tokens)
            if replacement_info.prompt_tokens >= replacement_budget.trigger_tokens:
                raise ValueError("the compacted context leaves insufficient room to continue")
            state.ensure_active()
            write_history(state.history_path, markdown)
            state.context = replacement
            state.report_progress("context_compacted")
            state.ensure_active()
            return replacement_info
        except (KeyboardInterrupt, asyncio.CancelledError) as error:
            if state.cancellation is None:
                raise
            _cancel_active_task(state.cancellation, error)
            state.ensure_active(error)
            raise
        except AgentCancelledError:
            raise
        except Exception as error:
            raise state.error(
                ContextCompactionError,
                "Gaia could not compact this conversation. Please try again.",
            ) from error

    async def _request_model_turn(
        self,
        state: _RunState,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        max_completion_tokens: int | None,
        enable_thinking: bool | None = None,
        purpose: Literal["response", "compaction"] = "response",
    ) -> ModelTurn:
        """Record every response or compaction request with its provider usage."""
        state.model_turns += 1
        if purpose == "response":
            state.report_progress("model_turn_started")
        state.ensure_active()
        model_started = self._clock()
        turn = None
        client_error: BaseException | None = None
        try:
            turn = await self._client.complete(
                messages=messages,
                tools=tools,
                max_completion_tokens=max_completion_tokens,
                enable_thinking=enable_thinking,
            )
        except (KeyboardInterrupt, asyncio.CancelledError) as error:
            if state.cancellation is None:
                raise
            _cancel_active_task(state.cancellation, error)
            client_error = error
        except Exception as error:
            client_error = error

        valid_turn = turn if isinstance(turn, ModelTurn) else None
        error_type = None
        if isinstance(client_error, (KeyboardInterrupt, asyncio.CancelledError)):
            error_type = "AgentCancelledError"
        elif client_error is not None:
            error_type = type(client_error).__name__
        elif valid_turn is None:
            error_type = "InvalidModelTurnError"
        state.model_turn_runtime.append(ModelTurnRuntime(
            turn=state.model_turns,
            status="completed" if valid_turn is not None else "failed",
            client_round_trip_seconds=self._clock() - model_started,
            response_id=valid_turn.response_id if valid_turn else None,
            usage=valid_turn.usage if valid_turn else None,
            provider_metrics=valid_turn.provider_metrics if valid_turn else None,
            error_type=error_type,
            purpose=purpose,
        ))
        state.ensure_active(client_error)
        if client_error is not None:
            raise state.error(
                ModelClientExecutionError,
                f"model client failed: {type(client_error).__name__}",
            ) from client_error
        if valid_turn is None:
            raise state.error(InvalidModelTurnError, "model client must return a ModelTurn")

        return valid_turn

    async def _finish_turn(self, state: _RunState, turn: ModelTurn) -> AgentResult:
        """Validate the final answer before exposing a completed interaction."""
        if turn.finish_reason == "tool_calls":
            raise state.error(
                InvalidModelTurnError,
                "model reported tool_calls without requesting a tool",
            )
        if turn.content is None or not turn.content.strip():
            raise state.error(
                InvalidModelTurnError, "model returned an empty final response",
            )
        if (
            self._response_guard is not None
            and not self._response_guard.allows(turn.content)
        ):
            raise state.error(
                ResponseDisclosureError,
                "Gaia could not prepare a response. Please try rephrasing your request.",
            )
        context_info = await self._prepare_context(state, tools=self._definitions)
        state.report_progress("response_ready")
        state.ensure_active()
        return AgentResult(
            final_response=turn.content,
            transcript=tuple(state.transcript),
            context=tuple(state.context),
            context_info=context_info,
            tool_results=tuple(state.tool_results),
            model_turns=state.model_turns,
            registered_tool_count=self.registered_tool_count,
            model_turn_runtime=tuple(state.model_turn_runtime),
            tool_batch_runtime=tuple(state.tool_batch_runtime),
        )

    async def _run_tool_round(
        self, state: _RunState, calls: tuple[ToolCall, ...],
    ) -> None:
        """Reserve one batch's allowance, execute it, and retain ordered results."""
        state.validate_tool_calls(calls)
        state.report_progress(
            "tool_batch_started",
            tool_names=tuple(
                call.name if call.name in self._tools else "unknown_tool"
                for call in calls
            ),
        )
        state.ensure_active()
        state.tool_calls_used += len(calls)
        batch_started = self._clock()
        batch_results, exhausted = await self._execute_tool_batch(
            calls, state.failed_tool_rounds, state.cancellation,
        )
        state.tool_batch_runtime.append(ToolBatchRuntime(
            turn=state.model_turns,
            tool_calls=len(batch_results),
            wall_elapsed_seconds=self._clock() - batch_started,
        ))
        state.tool_results.extend(batch_results)
        for result in batch_results:
            state.append_message(ChatMessage(
                role="tool",
                content=result.to_json(),
                tool_call_id=result.call.id,
                name=result.call.name,
            ))
        state.report_progress("tool_batch_completed", batch_results=batch_results)
        state.ensure_active()
        if exhausted is not None:
            raise state.error(
                ToolAttemptsExhaustedError,
                f"Tool {exhausted.call.name!r} failed in three consecutive rounds "
                f"({exhausted.error_code}). Please check or clarify your request; "
                "if the tool is unavailable, try again later.",
                failed_result=exhausted,
            )

    async def _execute_tool_batch(
        self,
        calls: tuple[ToolCall, ...],
        failed_rounds: dict[str, int],
        cancellation: AgentCancellation | None,
    ) -> tuple[tuple[ToolResult, ...], ToolResult | None]:
        """Run the whole batch concurrently, then account for one tool round."""
        by_call_id: dict[str, ToolResult] = {}

        try:
            async with asyncio.TaskGroup() as tasks:
                for call in calls:
                    tasks.create_task(self._record_tool_call(
                        call, by_call_id, cancellation,
                    ))
        except (KeyboardInterrupt, asyncio.CancelledError) as error:
            if cancellation is None:
                raise
            _cancel_active_task(cancellation, error)
        results = tuple(
            by_call_id[call.id] for call in calls if call.id in by_call_id
        )
        round_failures: dict[str, ToolResult | None] = {}
        for result in results:
            round_failures.setdefault(result.call.name, None)
            if not result.ok:
                round_failures[result.call.name] = result
        exhausted = None
        for name, failure in round_failures.items():
            failed_rounds[name] = failed_rounds.get(name, 0) + 1 if failure else 0
            if failed_rounds[name] == MAX_CONSECUTIVE_FAILED_TOOL_ROUNDS:
                exhausted = exhausted or failure
        return results, exhausted

    async def _record_tool_call(
        self,
        call: ToolCall,
        by_call_id: dict[str, ToolResult],
        cancellation: AgentCancellation | None,
    ) -> None:
        if cancellation is not None and cancellation.is_cancelled:
            return
        started = self._clock()
        result = await self._execute_tool(call)
        object.__setattr__(result, "elapsed_seconds", self._clock() - started)
        by_call_id[call.id] = result

    async def _execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute and project a tool result, encoding its observation once."""
        registered = self._tools.get(call.name)
        if registered is None:
            return _tool_failure(
                call, "unknown_tool", f"No tool named {call.name!r} is registered.",
            )
        try:
            # Handlers may mutate arguments; preserve the recorded model request.
            value = await registered.handler(deepcopy(call.arguments))
            if registered.observation_projector is not None:
                value = registered.observation_projector(value)
        except ToolExecutionError as error:
            return _tool_failure(call, error.code, str(error))
        except Exception:
            return _tool_failure(
                call, "tool_execution_error", "This operation could not be completed.",
            )

        try:
            return ToolResult(
                call=call,
                ok=True,
                result=value,
            )
        except (TypeError, ValueError, RecursionError):
            return _tool_failure(
                call, "invalid_tool_result", "Tool result is not valid JSON.",
            )
