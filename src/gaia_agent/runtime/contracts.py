"""Shared model, tool, progress, and execution-result contracts."""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal, Protocol


_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

Role = Literal["system", "user", "assistant", "tool"]
ModelTurnRuntimeStatus = Literal["completed", "failed"]
ToolHandler = Callable[[dict[str, object]], Awaitable[object]]
ToolObservationProjector = Callable[[object], object]


def _compact_json(value: object) -> str:
    """Encode model-visible JSON without changing supplied field order."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# Model and tool contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Describe an available tool and its JSON Schema arguments."""

    name: str
    description: str
    parameters: dict[str, object]
    strict: bool = False


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Store an invocation's public purpose and executable arguments."""

    id: str
    name: str
    arguments: dict[str, object]
    purpose: str | None = None
    emitted_arguments: dict[str, object] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    arguments_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("tool call id must be a non-empty string")
        if not isinstance(self.name, str) or not _TOOL_NAME_PATTERN.fullmatch(
            self.name
        ):
            raise ValueError("tool call name must match [A-Za-z0-9_-]{1,64}")
        if not isinstance(self.arguments, dict):
            raise TypeError("tool call arguments must be a JSON object")
        if self.purpose is not None and (
            not isinstance(self.purpose, str)
            or not self.purpose.strip()
            or len(self.purpose) > 300
        ):
            raise ValueError(
                "tool call purpose must be a non-empty string of at most 300 characters"
            )
        emitted_arguments = self.emitted_arguments
        if emitted_arguments is not None and not isinstance(emitted_arguments, dict):
            raise TypeError("emitted tool call arguments must be a JSON object")
        object.__setattr__(
            self,
            "arguments_json",
            _compact_json(emitted_arguments or self.arguments),
        )


AgentProgressEventKind = Literal[
    "model_turn_started",
    "tool_batch_started",
    "tool_batch_completed",
    "response_ready",
    "context_compacting",
    "context_compacted",
]


@dataclass(frozen=True, slots=True)
class AgentProgressEvent:
    """Report execution progress to an optional callback."""

    kind: AgentProgressEventKind
    turn: int
    tool_names: tuple[str, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()


AgentProgressHandler = Callable[[AgentProgressEvent], None]


class AgentCancellation:
    """Cancel bound asyncio tasks from the thread accepting a stop request."""

    def __init__(self) -> None:
        self._requested = threading.Event()
        self._lock = threading.Lock()
        self._tasks: dict[object, asyncio.Task[Any]] = {}

    def cancel(self) -> None:
        """Record cancellation and interrupt each active task once."""
        with self._lock:
            if self._requested.is_set():
                return
            self._requested.set()
            bindings = tuple(self._tasks.items())
        for binding, task in bindings:
            self._schedule_cancellation(binding, task)

    def bind_task(self, task: asyncio.Task[Any]) -> Callable[[], None]:
        """Bind a running task and return a callback that removes its binding."""
        binding = object()
        with self._lock:
            self._tasks[binding] = task
            already_cancelled = self._requested.is_set()
        if already_cancelled:
            self._schedule_cancellation(binding, task)

        return partial(self._unbind_task, binding)

    def _unbind_task(self, binding: object) -> None:
        with self._lock:
            self._tasks.pop(binding, None)

    def _schedule_cancellation(
        self, binding: object, task: asyncio.Task[Any],
    ) -> None:
        loop = task.get_loop()
        try:
            loop.call_soon_threadsafe(self._cancel_task, binding, task)
        except RuntimeError:
            # The run may have unbound its task and closed its loop meanwhile.
            if not loop.is_closed():
                raise

    def _cancel_task(self, binding: object, task: asyncio.Task[Any]) -> None:
        with self._lock:
            if self._tasks.get(binding) is not task:
                return
            # A second cancellation can interrupt the first one's cleanup.
            if not task.done() and not task.cancelling():
                task.cancel()

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._requested.is_set()


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """Connect a tool definition to its handler and optional result projector."""

    definition: ToolDefinition
    handler: ToolHandler = field(repr=False, compare=False)
    observation_projector: ToolObservationProjector | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Store token counts parsed from provider metadata."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_prompt_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ContextInfo:
    """Count a rendered request against the deployed model's context limit."""

    prompt_tokens: int
    context_limit_tokens: int

    def __post_init__(self) -> None:
        if type(self.prompt_tokens) is not int or self.prompt_tokens < 0:
            raise ValueError("prompt tokens must be a nonnegative integer")
        if type(self.context_limit_tokens) is not int or self.context_limit_tokens <= 0:
            raise ValueError("context limit must be a positive integer")


@dataclass(frozen=True, slots=True)
class ModelRequestMetrics:
    """Store provider timings in seconds and token rate."""

    provider: str | None = None
    time_to_first_token_seconds: float | None = None
    generation_seconds: float | None = None
    queue_seconds: float | None = None
    mean_inter_token_latency_seconds: float | None = None
    tokens_per_second: float | None = None


@dataclass(frozen=True, slots=True)
class ModelTurn:
    """Store an assistant response and any requested tool calls."""

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    reasoning_content: str | None = None
    reasoning_field: str | None = None
    response_id: str | None = None
    usage: TokenUsage | None = None
    provider_metrics: ModelRequestMetrics | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.content, "model turn content"),
            (self.reasoning_content, "model turn reasoning content"),
            (self.finish_reason, "finish reason"),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{label} must be a string or None")
        if (
            self.content is None
            and self.reasoning_content is None
            and not self.tool_calls
        ):
            raise ValueError(
                "model turn must contain content, reasoning, or at least one tool call"
            )
        call_ids = [call.id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("model turn contains duplicate tool call ids")


@dataclass(frozen=True, slots=True)
class ModelTurnRuntime:
    """Record model-request duration, status, and provider metadata."""

    turn: int
    status: ModelTurnRuntimeStatus
    client_round_trip_seconds: float
    response_id: str | None = None
    usage: TokenUsage | None = None
    provider_metrics: ModelRequestMetrics | None = None
    error_type: str | None = None
    purpose: Literal["response", "compaction"] = "response"


@dataclass(frozen=True, slots=True)
class ToolBatchRuntime:
    """Record a tool batch's executed-call count and wall duration."""

    turn: int
    tool_calls: int
    wall_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Store transcript data already checked at the user or model boundary."""

    role: Role
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None
    reasoning_field: str | None = None


class ModelClient(Protocol):
    """Transport boundary for the agent loop."""

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        max_completion_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> ModelTurn:
        """Return the next assistant turn for the supplied conversation."""

    async def context_info(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        enable_thinking: bool | None = None,
    ) -> ContextInfo:
        """Count the complete rendered prompt and return its server limit."""

    async def count_text_tokens(self, text: str) -> int:
        """Count standalone text without chat-template or special tokens."""


# ---------------------------------------------------------------------------
# Execution results and errors
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Store a tool invocation's outcome and its encoded model observation."""

    call: ToolCall
    ok: bool
    result: object = None
    error_code: str | None = None
    error_message: str | None = None
    elapsed_seconds: float | None = None
    _serialized: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        serialized = _compact_json(self.payload())
        object.__setattr__(self, "_serialized", serialized)
        if self.ok:
            object.__setattr__(self, "result", json.loads(serialized)["result"])

    def payload(self) -> dict[str, object]:
        """Return the JSON object sent back to the model."""
        if self.ok:
            return {"ok": True, "result": self.result}
        return {
            "ok": False,
            "error": {
                "code": self.error_code,
                "message": self.error_message,
            },
        }

    def to_json(self) -> str:
        """Reuse the payload encoded at the tool-result boundary."""
        return self._serialized


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Return a final response with its complete structured execution record."""

    final_response: str
    transcript: tuple[ChatMessage, ...]
    context: tuple[ChatMessage, ...]
    tool_results: tuple[ToolResult, ...]
    model_turns: int
    registered_tool_count: int = 0
    model_turn_runtime: tuple[ModelTurnRuntime, ...] = ()
    tool_batch_runtime: tuple[ToolBatchRuntime, ...] = ()
    context_info: ContextInfo | None = None

    @property
    def tool_calls(self) -> int:
        """Return the number of tool requests executed or rejected."""
        return len(self.tool_results)


class ToolExecutionError(RuntimeError):
    """Let a handler return an expected, machine-readable failure to the model."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class AgentRuntimeError(RuntimeError):
    """Base error preserving work completed before an agent run stopped."""

    def __init__(
        self,
        message: str,
        *,
        transcript: Iterable[ChatMessage],
        tool_results: Iterable[ToolResult],
        model_turns: int,
        registered_tool_count: int = 0,
        model_turn_runtime: Iterable[ModelTurnRuntime] = (),
        tool_batch_runtime: Iterable[ToolBatchRuntime] = (),
    ) -> None:
        super().__init__(message)
        self.transcript = tuple(transcript)
        self.tool_results = tuple(tool_results)
        self.model_turns = model_turns
        self.registered_tool_count = registered_tool_count
        self.model_turn_runtime = tuple(model_turn_runtime)
        self.tool_batch_runtime = tuple(tool_batch_runtime)


class AgentCancelledError(AgentRuntimeError):
    """Indicate that an explicit cancellation stopped an incomplete run."""


class InvalidModelTurnError(AgentRuntimeError):
    """Indicate that a model client violated the turn contract."""


class ContextCompactionError(AgentRuntimeError):
    """Preserve a conversation when context compaction cannot complete."""


class ResponseDisclosureError(AgentRuntimeError):
    """Keep a rejected assistant response inside the private failure trace."""


class ToolCallLimitExceededError(AgentRuntimeError):
    """Reject a batch that exceeds the current user interaction's tool budget."""

    def __init__(
        self,
        message: str,
        *,
        tool_calls_used: int,
        requested_tool_calls: int,
        **run_state: Any,
    ) -> None:
        super().__init__(message, **run_state)
        self.tool_calls_used = tool_calls_used
        self.requested_tool_calls = requested_tool_calls


class ToolAttemptsExhaustedError(AgentRuntimeError):
    """Stop a run after one tool has failures in three consecutive rounds."""

    def __init__(
        self,
        message: str,
        *,
        failed_result: ToolResult,
        **run_state: Any,
    ) -> None:
        super().__init__(message, **run_state)
        self.failed_result = failed_result


class ModelClientExecutionError(AgentRuntimeError):
    """Preserve a run when an ordinary model-client exception stops execution."""
