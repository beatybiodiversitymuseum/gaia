"""Manage web conversation history, cancellation, and interaction feedback."""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import threading
import time
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import BinaryIO, Protocol, cast

from src.logger import create_chat_directory, normalize_chat_username
from src.logger.gaia_logger import (
    MAX_FEEDBACK_DETAILS_CHARACTERS,
    MAX_FEEDBACK_REASON_CHARACTERS,
    MAX_FEEDBACK_REASONS,
    GaiaLoggerConfig,
    create_chat_trace_config,
    write_gaia_trace_feedback,
)

from src.gaia_agent.runtime.context import history_transaction
from src.gaia_agent.runtime.defaults import (
    CONTEXT_COMPACTED_NOTICE,
    DEFAULT_MAX_ACTIVE_CHATS,
    DEFAULT_MAX_WAITING_CHATS,
)
from src.gaia_agent.runtime.contracts import (
    AgentCancellation,
    AgentCancelledError,
    AgentProgressEvent,
    AgentProgressHandler,
    AgentResult,
    ChatMessage,
)
from src.gaia_agent.tools.file_generation.artifacts import (
    GeneratedFile,
    collect_generated_files,
    open_generated_file,
)
from src.gaia_agent.tools.concurrency import WorkloadBusyError
from src.gaia_agent.web.verification import project_verification_steps


_CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_MAX_CONVERSATION_ID_GENERATION_ATTEMPTS = 8


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------


class GaiaAgentRunner(Protocol):
    """Describe the async Gaia boundary invoked by each HTTP worker."""

    async def run(
        self,
        user_query: str,
        *,
        history: Iterable[ChatMessage] = (),
        trace_config: GaiaLoggerConfig | None = None,
        progress_callback: AgentProgressHandler | None = None,
        cancellation: AgentCancellation | None = None,
    ) -> AgentResult:
        """Run one Gaia interaction with optional high-level progress events."""


class UnknownConversationError(LookupError):
    """Indicate that a supplied opaque conversation identifier is unknown."""


class ConversationNotRunningError(RuntimeError):
    """Indicate that a conversation has no cancellable interaction."""


class UnknownInteractionError(LookupError):
    """Indicate that an interaction is not a completed conversation turn."""


class UnknownGeneratedFileError(LookupError):
    """Indicate that a requested conversation attachment is unavailable."""


@dataclass(frozen=True, slots=True)
class GaiaChatResponse:
    """Return the public response and bounded execution metrics for one turn."""

    conversation_id: str
    interaction: int
    content: str
    model_turns: int
    tool_calls: int
    elapsed_seconds: float
    attachments: tuple[GeneratedFile, ...] = ()
    notices: tuple[str, ...] = ()
    verification: tuple[dict[str, object], ...] = ()

    def payload(self) -> dict[str, object]:
        """Return the stable JSON response contract."""
        return {
            "conversation_id": self.conversation_id,
            "interaction": self.interaction,
            "message": {
                "role": "assistant",
                "content": self.content,
                "attachments": [
                    {
                        "file_id": attachment.file_id,
                        "filename": attachment.filename,
                        "media_type": attachment.media_type,
                        "size_bytes": attachment.size_bytes,
                    }
                    for attachment in self.attachments
                ],
                **({"notices": list(self.notices)} if self.notices else {}),
            },
            "metrics": {
                "elapsed_seconds": self.elapsed_seconds,
            },
            "verification": list(self.verification),
        }


@dataclass(slots=True)
class _ActiveInteraction:
    cancellation: AgentCancellation
    finished: threading.Event = field(default_factory=threading.Event)
    accepting_cancellation: bool = True


@dataclass(slots=True)
class _InteractionProgress:
    callback: AgentProgressHandler | None
    compacted: bool = False

    def report(self, event: AgentProgressEvent) -> None:
        if event.kind == "context_compacted":
            self.compacted = True
        if self.callback is not None:
            self.callback(event)


@dataclass(slots=True)
class _ChatWaiter:
    capacity: threading.BoundedSemaphore
    acquired: bool = False

    def acquire(self) -> None:
        if not self.acquired:
            if not self.capacity.acquire(blocking=False):
                raise WorkloadBusyError("Gaia is busy. Please try again shortly.")
            self.acquired = True

    def release(self) -> None:
        if self.acquired:
            self.acquired = False
            self.capacity.release()


@dataclass(slots=True)
class _Conversation:
    conversation_id: str
    username: str
    trace_directory: Path
    history: tuple[ChatMessage, ...] = ()
    interactions: int = 0
    completed_traces: dict[int, GaiaLoggerConfig] = field(default_factory=dict)
    generated_files: dict[str, GeneratedFile] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    state_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    active_interaction: _ActiveInteraction | None = field(
        default=None,
        repr=False,
    )


class GaiaChatService:
    """Keep web histories and serialize turns within each conversation."""

    def __init__(
        self,
        agent: GaiaAgentRunner,
        *,
        repository_root: Path,
        trace_root: Path | None = None,
        trace_invocation: Mapping[str, object] | None = None,
        conversation_id_factory: Callable[[], str] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        max_active_chats: int = DEFAULT_MAX_ACTIVE_CHATS,
        max_waiting_chats: int = DEFAULT_MAX_WAITING_CHATS,
    ) -> None:
        require_positive_integer(max_active_chats, "active chat limit")
        if type(max_waiting_chats) is not int or max_waiting_chats < 0:
            raise ValueError("waiting chat limit must be a nonnegative integer")
        self._admitted_chats = threading.BoundedSemaphore(
            max_active_chats + max_waiting_chats,
        )
        self._active_chats = threading.BoundedSemaphore(max_active_chats)
        self._waiting_chats = threading.BoundedSemaphore(max_waiting_chats)

        self._agent = agent
        self._repository_root = Path(repository_root)
        self._trace_root = Path(trace_root or self._repository_root / "logs")
        self._trace_invocation = _detached_json_object(
            trace_invocation or {},
            label="trace invocation",
        )
        self._conversation_id_factory = (
            conversation_id_factory or partial(secrets.token_urlsafe, 24)
        )
        self._clock = clock
        self._sessions: dict[str, _Conversation] = {}
        self._sessions_lock = threading.Lock()

    def chat(
        self,
        message: str,
        *,
        username: str,
        conversation_id: str | None = None,
        progress_callback: AgentProgressHandler | None = None,
        conversation_callback: Callable[[str], None] | None = None,
        response_callback: Callable[[GaiaChatResponse], None] | None = None,
        cancellation: AgentCancellation | None = None,
    ) -> GaiaChatResponse:
        """Admit bounded work across request threads before allocating a chat."""
        if not self._admitted_chats.acquire(blocking=False):
            raise WorkloadBusyError("Gaia is busy. Please try again shortly.")
        waiting = _ChatWaiter(self._waiting_chats)
        try:
            return self._chat(
                message,
                username=username,
                conversation_id=conversation_id,
                progress_callback=progress_callback,
                conversation_callback=conversation_callback,
                response_callback=response_callback,
                cancellation=cancellation,
                waiting=waiting,
            )
        finally:
            waiting.release()
            self._admitted_chats.release()

    def _chat(
        self,
        message: str,
        *,
        username: str,
        conversation_id: str | None,
        progress_callback: AgentProgressHandler | None,
        conversation_callback: Callable[[str], None] | None,
        response_callback: Callable[[GaiaChatResponse], None] | None,
        cancellation: AgentCancellation | None,
        waiting: _ChatWaiter,
    ) -> GaiaChatResponse:
        """Run one turn and commit only after optional response delivery."""
        normalized_username = normalize_chat_username(username)
        is_new_conversation = conversation_id is None
        conversation = (
            self._create_conversation(normalized_username)
            if is_new_conversation
            else self._find_conversation(
                conversation_id,
                username=normalized_username,
            )
        )

        interaction_published = not is_new_conversation
        active_interaction = _ActiveInteraction(
            cancellation=cancellation or AgentCancellation()
        )
        try:
            with self._conversation_turn(conversation, waiting):
                conversation.interactions += 1
                interaction = conversation.interactions
                trace_config = self._trace_config(conversation, interaction)
                history = conversation.history
                with conversation.state_lock:
                    conversation.active_interaction = active_interaction

                try:
                    if conversation_callback is not None:
                        conversation_callback(conversation.conversation_id)
                        interaction_published = True
                    return self._run_turn(
                        message,
                        conversation=conversation,
                        interaction=interaction,
                        trace_config=trace_config,
                        history=history,
                        active_interaction=active_interaction,
                        progress_callback=progress_callback,
                        response_callback=response_callback,
                        waiting=waiting,
                    )
                finally:
                    with conversation.state_lock:
                        conversation.active_interaction = None
                        active_interaction.accepting_cancellation = False
        except BaseException:
            if is_new_conversation and not interaction_published:
                self._remove_conversation(conversation)
            raise
        finally:
            # A cancellation acknowledgement follows conversation cleanup.
            active_interaction.finished.set()

    def _run_turn(
        self,
        message: str,
        *,
        conversation: _Conversation,
        interaction: int,
        trace_config: GaiaLoggerConfig,
        history: tuple[ChatMessage, ...],
        active_interaction: _ActiveInteraction,
        progress_callback: AgentProgressHandler | None,
        response_callback: Callable[[GaiaChatResponse], None] | None,
        waiting: _ChatWaiter,
    ) -> GaiaChatResponse:
        with history_transaction(conversation.trace_directory / "history.md"):
            started = self._clock()
            progress = _InteractionProgress(progress_callback)
            with self._execution_slot(active_interaction.cancellation, history, waiting):
                result = asyncio.run(
                    self._agent.run(
                        message,
                        history=history,
                        trace_config=trace_config,
                        progress_callback=progress.report,
                        cancellation=active_interaction.cancellation,
                    )
                )
            elapsed_seconds = self._clock() - started
            with conversation.state_lock:
                if active_interaction.cancellation.is_cancelled:
                    raise AgentCancelledError(
                        "Stopped thinking",
                        transcript=result.transcript,
                        tool_results=result.tool_results,
                        model_turns=result.model_turns,
                        registered_tool_count=result.registered_tool_count,
                        model_turn_runtime=result.model_turn_runtime,
                        tool_batch_runtime=result.tool_batch_runtime,
                    )
                active_interaction.accepting_cancellation = False
            response = self._build_response(
                conversation, interaction, result, elapsed_seconds,
                notices=(CONTEXT_COMPACTED_NOTICE,) if progress.compacted else (),
            )
            _deliver_response(conversation, trace_config, response, response_callback)
            conversation.history = result.context
            return response

    def _build_response(
        self,
        conversation: _Conversation,
        interaction: int,
        result: AgentResult,
        elapsed_seconds: float,
        *,
        notices: tuple[str, ...],
    ) -> GaiaChatResponse:
        return GaiaChatResponse(
            conversation_id=conversation.conversation_id,
            interaction=interaction,
            content=result.final_response,
            model_turns=result.model_turns,
            tool_calls=result.tool_calls,
            elapsed_seconds=max(0.0, elapsed_seconds),
            notices=notices,
            attachments=collect_generated_files(
                result.tool_results,
                export_directory=conversation.trace_directory / "exports",
            ),
            verification=project_verification_steps(result.tool_results),
        )

    @contextmanager
    def _conversation_turn(
        self, conversation: _Conversation, waiting: _ChatWaiter,
    ) -> Generator[None, None, None]:
        """Count conversation-lock contention against the shared waiting limit."""
        if not conversation.lock.acquire(blocking=False):
            waiting.acquire()
            conversation.lock.acquire()
        try:
            yield
        finally:
            conversation.lock.release()

    @contextmanager
    def _execution_slot(
        self, cancellation: AgentCancellation, history: tuple[ChatMessage, ...],
        waiting: _ChatWaiter,
    ) -> Generator[None, None, None]:
        """Wait on the HTTP worker while keeping a queued chat cancellable."""
        _check_chat_cancellation(cancellation, history)
        if not self._active_chats.acquire(blocking=False):
            waiting.acquire()
            while not self._active_chats.acquire(timeout=0.05):
                _check_chat_cancellation(cancellation, history)
        try:
            waiting.release()
            _check_chat_cancellation(cancellation, history)
            yield
        finally:
            self._active_chats.release()

    def open_file(
        self,
        *,
        username: str,
        conversation_id: str,
        file_id: str,
    ) -> tuple[GeneratedFile, BinaryIO]:
        """Open a registered attachment from the requesting user's conversation."""
        normalized_username = normalize_chat_username(username)
        if not is_valid_conversation_id(conversation_id):
            raise ValueError("conversation ID has an invalid format")
        if not is_valid_conversation_id(file_id):
            raise ValueError("file ID has an invalid format")
        conversation = self._find_conversation(
            conversation_id,
            username=normalized_username,
        )
        with conversation.state_lock:
            generated_file = conversation.generated_files.get(file_id)
        if generated_file is None:
            raise UnknownGeneratedFileError("unknown generated file")
        try:
            stream = open_generated_file(
                generated_file,
                export_directory=conversation.trace_directory / "exports",
            )
        except (OSError, ValueError):
            raise UnknownGeneratedFileError("generated file is unavailable") from None
        return generated_file, stream

    def record_feedback(
        self,
        *,
        username: object,
        conversation_id: object,
        interaction: object,
        rating: object,
        reasons: object,
        details: object,
    ) -> dict[str, object]:
        """Replace feedback for one owned, completed interaction trace."""
        normalized_username = normalize_chat_username(username)
        if not is_valid_conversation_id(conversation_id):
            raise ValueError("conversation ID has an invalid format")
        require_positive_integer(interaction, "interaction")
        normalized_rating, normalized_reasons, normalized_details = (
            _validate_feedback(rating, reasons, details)
        )
        owned_conversation_id = cast(str, conversation_id)
        interaction_number = cast(int, interaction)
        conversation = self._find_conversation(
            owned_conversation_id,
            username=normalized_username,
        )

        with conversation.state_lock:
            trace_config = conversation.completed_traces.get(interaction_number)
            if trace_config is None:
                raise UnknownInteractionError("unknown completed interaction")
            write_gaia_trace_feedback(
                trace_config,
                interaction=interaction_number,
                rating=normalized_rating,
                reasons=normalized_reasons,
                details=normalized_details,
                submitted_at=_utc_timestamp(),
            )

        return {
            "status": "recorded",
            "conversation_id": conversation.conversation_id,
            "interaction": interaction_number,
            "rating": normalized_rating,
        }

    def cancel(self, conversation_id: str, *, username: str) -> str:
        """Cancel one owned active interaction and wait for it to unwind."""
        normalized_username = normalize_chat_username(username)
        conversation = self._find_conversation(
            conversation_id,
            username=normalized_username,
        )
        with conversation.state_lock:
            active_interaction = conversation.active_interaction
            if (
                active_interaction is None
                or not active_interaction.accepting_cancellation
            ):
                raise ConversationNotRunningError(
                    "conversation has no active interaction"
                )
            active_interaction.accepting_cancellation = False
            active_interaction.cancellation.cancel()
        active_interaction.finished.wait()
        return conversation.conversation_id

    def _create_conversation(self, username: str) -> _Conversation:
        with self._sessions_lock:
            for _attempt in range(_MAX_CONVERSATION_ID_GENERATION_ATTEMPTS):
                conversation_id = self._conversation_id_factory()
                if not is_valid_conversation_id(conversation_id):
                    raise ValueError(
                        "conversation ID factory returned an invalid value"
                    )
                if conversation_id in self._sessions:
                    continue
                try:
                    trace_directory = create_chat_directory(
                        self._trace_root, username
                    )
                except FileExistsError as error:
                    if error.filename == str(self._trace_root):
                        raise
                    continue
                conversation = _Conversation(
                    conversation_id=conversation_id,
                    username=username,
                    trace_directory=trace_directory,
                )
                self._sessions[conversation_id] = conversation
                return conversation

        raise RuntimeError("could not allocate a unique conversation ID")

    def _find_conversation(
        self,
        conversation_id: str,
        *,
        username: str,
    ) -> _Conversation:
        with self._sessions_lock:
            conversation = self._sessions.get(conversation_id)
        if conversation is None or conversation.username != username:
            raise UnknownConversationError("unknown conversation ID")
        return conversation

    def _remove_conversation(self, conversation: _Conversation) -> None:
        with self._sessions_lock:
            if self._sessions.get(conversation.conversation_id) is conversation:
                del self._sessions[conversation.conversation_id]

    def _trace_config(
        self,
        conversation: _Conversation,
        interaction: int,
    ) -> GaiaLoggerConfig:
        return create_chat_trace_config(
            chat_directory=conversation.trace_directory,
            interaction=interaction,
            repository_root=self._repository_root,
            invocation={
                **self._trace_invocation,
                "command": "serve",
                "transport": "http",
                "username": conversation.username,
            },
            feedback_enabled=True,
        )


def _deliver_response(
    conversation: _Conversation,
    trace_config: GaiaLoggerConfig,
    response: GaiaChatResponse,
    response_callback: Callable[[GaiaChatResponse], None] | None,
) -> None:
    with conversation.state_lock:
        conversation.completed_traces[response.interaction] = trace_config
        previous_files = {
            attachment.file_id: conversation.generated_files.get(attachment.file_id)
            for attachment in response.attachments
        }
        conversation.generated_files.update(
            (attachment.file_id, attachment)
            for attachment in response.attachments
        )
    try:
        if response_callback is not None:
            response_callback(response)
    except BaseException:
        with conversation.state_lock:
            for file_id, previous_file in previous_files.items():
                if previous_file is None:
                    conversation.generated_files.pop(file_id, None)
                else:
                    conversation.generated_files[file_id] = previous_file
        raise


def _check_chat_cancellation(
    cancellation: AgentCancellation,
    history: tuple[ChatMessage, ...],
) -> None:
    if cancellation.is_cancelled:
        raise AgentCancelledError(
            "Stopped thinking", transcript=history, tool_results=(), model_turns=0,
        )


def is_valid_conversation_id(value: object) -> bool:
    """Check the opaque conversation identifier accepted by the chat API."""
    return (
        isinstance(value, str)
        and _CONVERSATION_ID_PATTERN.fullmatch(value) is not None
    )


def require_positive_integer(value: object, label: str) -> None:
    """Reject booleans and non-positive values at the API boundary."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _validate_feedback(
    rating: object,
    reasons: object,
    details: object,
) -> tuple[str, tuple[str, ...], str]:
    if not isinstance(rating, str) or rating not in {"good", "bad"}:
        raise ValueError("rating must be 'good' or 'bad'")
    if not isinstance(reasons, list) or len(reasons) > MAX_FEEDBACK_REASONS:
        raise ValueError(
            f"reasons must be a list of at most {MAX_FEEDBACK_REASONS} items"
        )
    if any(
        not isinstance(reason, str)
        or not reason.strip()
        or len(reason) > MAX_FEEDBACK_REASON_CHARACTERS
        for reason in reasons
    ):
        raise ValueError(
            "reasons must contain non-empty strings of at most "
            f"{MAX_FEEDBACK_REASON_CHARACTERS} characters"
        )
    if len(set(reasons)) != len(reasons):
        raise ValueError("reasons must not contain duplicates")
    if not isinstance(details, str):
        raise TypeError("details must be text")
    if len(details) > MAX_FEEDBACK_DETAILS_CHARACTERS:
        raise ValueError(
            "details must be at most "
            f"{MAX_FEEDBACK_DETAILS_CHARACTERS} characters"
        )
    return rating, tuple(reasons), details


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )


def _detached_json_object(
    value: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    try:
        return json.loads(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain only JSON values") from error
