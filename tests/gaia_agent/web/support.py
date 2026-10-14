"""Shared deterministic agents and builders for web backend tests."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

from src.gaia_agent.runtime.contracts import (
    AgentCancellation,
    AgentCancelledError,
    AgentProgressEvent,
    AgentResult,
    ChatMessage,
    ToolCall,
    ToolResult,
)
from src.gaia_agent.tools.file_generation.excel_export import ExcelExportWriter
from src.gaia_agent.web.chat_service import GaiaChatService


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class FakeAgent:
    """Return complete transcripts or configured failures without I/O."""

    def __init__(
        self,
        outcomes: list[str | BaseException],
        *,
        progress_events: tuple[AgentProgressEvent, ...] = (),
    ) -> None:
        self.outcomes = list(outcomes)
        self.progress_events = progress_events
        self.calls: list[
            tuple[str, tuple[ChatMessage, ...], object]
        ] = []
        self.results: list[AgentResult] = []

    async def run(
        self,
        message,
        *,
        history=(),
        trace_config=None,
        progress_callback=None,
        cancellation=None,
    ):
        del cancellation
        if progress_callback is not None:
            for event in self.progress_events:
                progress_callback(event)
        history_messages = tuple(history)
        self.calls.append((message, history_messages, trace_config))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome

        interaction = len(self.calls)
        current_messages = (
            ChatMessage(role="user", content=message),
            ChatMessage(role="assistant", content=outcome),
        )
        transcript = (
            history_messages + current_messages
            if history_messages
            else (
                ChatMessage(role="system", content="test system prompt"),
                *current_messages,
            )
        )
        result = AgentResult(
            final_response=outcome,
            transcript=transcript,
            context=transcript,
            tool_results=(
                ToolResult(
                    call=ToolCall(
                        f"schema-{interaction}",
                        "retrieve_schema",
                        {"datasets": ["birds"]},
                        purpose="Check which fields are available for the request.",
                    ),
                    ok=True,
                    result={"datasets": []},
                ),
            ),
            model_turns=2,
        )
        self.results.append(result)
        return result


class GatedAgent(FakeAgent):
    """Observe concurrent calls and release selected messages explicitly."""

    def __init__(
        self,
        outcomes: list[str | BaseException],
        *,
        started: Mapping[str, threading.Event] | None = None,
        releases: Mapping[str, threading.Event] | None = None,
    ) -> None:
        super().__init__(outcomes)
        self.started = dict(started or {})
        self.releases = dict(releases or {})
        self.active_lock = threading.Lock()
        self.active_calls = 0
        self.maximum_active_calls = 0

    async def run(self, message, **kwargs):
        with self.active_lock:
            self.active_calls += 1
            self.maximum_active_calls = max(self.maximum_active_calls, self.active_calls)
        try:
            if message in self.started:
                self.started[message].set()
            if message in self.releases:
                await _wait_for_thread_event(self.releases[message])
            return await super().run(message, **kwargs)
        finally:
            with self.active_lock:
                self.active_calls -= 1


class DownloadAgent(FakeAgent):
    """Return real workbooks in tool results without a model or database."""

    def __init__(self, *, failed_tool: bool = False, failed_turn: bool = False) -> None:
        super().__init__(["The workbooks are ready."] * 8)
        self.failed_tool = failed_tool
        self.failed_turn = failed_turn
        self.files: list[dict[str, str]] = []

    async def run(self, message, **kwargs):
        result = await super().run(message, **kwargs)
        directory = kwargs["trace_config"].path.parent / "exports"
        directory.mkdir(exist_ok=True)
        interaction = len(self.calls)
        files = []
        for index, stem in enumerate(('Herons – 鹭 "review"', "Mammals")):
            with ExcelExportWriter(directory, filename=f"{stem}-{interaction}.xlsx") as writer:
                writer.add_sheet("records", ("catalogNumber", "scientificName"))
                writer.append("records", ("0001", "鹭"))
                path = writer.finalize().path
            files.append({
                "file_id": f"file_{interaction}_{index}",
                "filename": path.name,
                "path": str(path),
            })
        self.files.extend(files)
        public_files = [
            {"file_id": item["file_id"], "filename": item["filename"]}
            for item in files
        ]
        tool_results = (
            ToolResult(
                call=ToolCall(
                    f"export_{interaction}",
                    "generate_excel",
                    {},
                    purpose="Create the requested workbook for download.",
                ),
                ok=not self.failed_tool,
                result={"files": public_files},
                error_code="export_failed" if self.failed_tool else None,
                error_message="Export failed." if self.failed_tool else None,
            ),
        )
        if self.failed_turn:
            raise AgentCancelledError(
                "Stopped thinking",
                transcript=result.transcript,
                tool_results=tool_results,
                model_turns=2,
            )
        return replace(result, tool_results=tool_results)


async def _wait_for_cancellation(cancellation: AgentCancellation) -> None:
    async with asyncio.timeout(2):
        while not cancellation.is_cancelled:
            await asyncio.sleep(0.01)


async def _wait_for_thread_event(event: threading.Event) -> None:
    async with asyncio.timeout(2):
        while not event.is_set():
            await asyncio.sleep(0.01)


def _feedback_document(
    *,
    username: object = "Museum Researcher",
    conversation_id: object = "conversation_one",
    interaction: object = 1,
    rating: object = "bad",
    reasons: object = ("incorrect",),
    details: object = "The response needs correction.",
) -> dict[str, object]:
    return {
        "username": username,
        "conversation_id": conversation_id,
        "interaction": interaction,
        "rating": rating,
        "reasons": reasons,
        "details": details,
    }


def _service(
    tmp_path: Path,
    agent: FakeAgent,
    **options: object,
) -> GaiaChatService:
    return GaiaChatService(
        agent,
        repository_root=REPOSITORY_ROOT,
        trace_root=tmp_path,
        conversation_id_factory=_fixed_conversation_id,
        **options,
    )


def _raise_callback_failure(failure: BaseException, *_args: object) -> None:
    raise failure


def _observe_call(
    event: threading.Event,
    callback: Callable[..., object],
    *args: object,
    only_truthy: bool = False,
    **kwargs: object,
) -> object:
    result = callback(*args, **kwargs)
    if not only_truthy or result:
        event.set()
    return result


def _publish_conversation(
    identifiers: list[str], event: threading.Event, identifier: str,
) -> None:
    identifiers.append(identifier)
    event.set()


def _fixed_conversation_id() -> str:
    return "conversation_one"


def _signal_event(event: threading.Event, *_args: object) -> None:
    event.set()
