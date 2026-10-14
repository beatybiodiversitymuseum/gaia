"""Exercise shared chat admission, overload delivery, and cancellation recovery."""

from __future__ import annotations

import asyncio
import itertools
import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import pytest

from src.gaia_agent.runtime.contracts import (
    AgentCancelledError,
    ChatMessage,
    ContextInfo,
    ModelTurn,
    ToolAttemptsExhaustedError,
    ToolCall,
    ToolResult,
)
from src.gaia_agent.runtime.loop import AgentRuntime
from src.gaia_agent.runtime.service import GaiaAgent
from src.gaia_agent.tools.concurrency import WorkloadBusyError
from src.gaia_agent.web.chat_service import GaiaChatService

from tests.gaia_agent.web.support import (
    REPOSITORY_ROOT,
    FakeAgent,
    GatedAgent,
    _observe_call,
    _publish_conversation,
    _raise_callback_failure,
    _signal_event,
)
from tests.gaia_agent.web.test_http_api import (
    STREAM_HEADERS,
    _chat_body,
    _post_chat,
    _running_server,
    _stream_request,
)


@dataclass
class _AllocationObserver:
    create_conversation: Callable[[str], object]
    expected_count: int
    completed: threading.Event
    count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, username: str) -> object:
        conversation = self.create_conversation(username)
        with self.lock:
            self.count += 1
            if self.count == self.expected_count:
                self.completed.set()
        return conversation


class _StalledClient:
    def __init__(self, started: threading.Event, cleaned_up: threading.Event) -> None:
        self.started = started
        self.cleaned_up = cleaned_up
        self.requests = []
        self.stalled_task = None

    async def complete(
        self, *, messages, tools, max_completion_tokens=None, enable_thinking=None,
    ):
        self.requests.append(tuple(messages))
        if messages[-1].content == "stall":
            self.stalled_task = asyncio.current_task()
            self.started.set()
            try:
                await asyncio.Future()
            finally:
                self.cleaned_up.set()
        return ModelTurn(content="Completed.")

    async def context_info(self, *, messages, tools, enable_thinking=None):
        return ContextInfo(prompt_tokens=100, context_limit_tokens=131_072)

    async def count_text_tokens(self, text):
        return len(text)


def _service(tmp_path: Path, agent: object, **options: object) -> GaiaChatService:
    identifiers = map("conversation_{}".format, itertools.count(1))
    return GaiaChatService(
        agent,
        repository_root=REPOSITORY_ROOT,
        trace_root=tmp_path,
        conversation_id_factory=identifiers.__next__,
        **options,
    )


@pytest.mark.parametrize(
    ("options", "active_limit", "waiting_limit"),
    (
        ({"max_active_chats": 1, "max_waiting_chats": 1}, 1, 1),
        ({"max_active_chats": 2, "max_waiting_chats": 1}, 2, 1),
        ({}, 8, 16),
    ),
    ids=("one-active-one-waiting", "two-active-one-waiting", "defaults"),
)
def test_chat_admission_bounds_work_and_rejects_before_allocating_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, int],
    active_limit: int,
    waiting_limit: int,
) -> None:
    admitted_limit = active_limit + waiting_limit
    release = threading.Event()
    all_allocated = threading.Event()
    started = [threading.Event() for _ in range(admitted_limit)]
    agent = GatedAgent(
        ["Finished."] * (admitted_limit + 1),
        started={str(index): event for index, event in enumerate(started)},
        releases={str(index): release for index in range(admitted_limit)},
    )
    service = _service(tmp_path, agent, **options)
    allocation = _AllocationObserver(
        service._create_conversation, admitted_limit, all_allocated,
    )
    monkeypatch.setattr(service, "_create_conversation", allocation)
    with ThreadPoolExecutor(max_workers=admitted_limit) as executor:
        requests = []
        try:
            for index in range(active_limit):
                requests.append(executor.submit(service.chat, str(index), username=str(index)))
                assert started[index].wait(timeout=2)
            for index in range(active_limit, admitted_limit):
                requests.append(executor.submit(service.chat, str(index), username=str(index)))
            assert all_allocated.wait(timeout=2)
            assert not any(event.is_set() for event in started[active_limit:])
            existing_paths = set(tmp_path.rglob("*"))
            published = []
            with pytest.raises(WorkloadBusyError, match="try again"):
                service.chat(
                    "overflow", username="rejected_user",
                    conversation_callback=published.append,
                )
            assert published == []
            assert allocation.count == admitted_limit
            assert set(tmp_path.rglob("*")) == existing_paths
            assert agent.maximum_active_calls == active_limit
        finally:
            release.set()
        for request in requests:
            assert request.result(timeout=2).content == "Finished."
    assert agent.active_calls == 0
    assert agent.maximum_active_calls == active_limit
    assert len(agent.calls) == admitted_limit
    # Both the active permit and total admission place are reusable.
    assert service.chat("0", username="recovered_user").content == "Finished."


@pytest.mark.parametrize("setting", ("max_active_chats", "max_waiting_chats"))
@pytest.mark.parametrize("value", (-1, True, False, "2", 1.5, None))
def test_chat_capacity_rejects_invalid_configuration(
    tmp_path: Path, setting: str, value: object,
) -> None:
    with pytest.raises(ValueError, match="chat limit"):
        _service(tmp_path, FakeAgent([]), **{setting: value})
    assert list(tmp_path.iterdir()) == []


def test_active_chat_capacity_cannot_be_zero(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="active chat limit"):
        _service(tmp_path, FakeAgent([]), max_active_chats=0)


def test_same_conversation_queue_does_not_occupy_an_active_chat_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = threading.Event()
    queued_resolved = threading.Event()
    release = threading.Event()
    queued_started = threading.Event()

    agent = GatedAgent(
        ["Initial.", "Independent.", "First.", "Queued."],
        started={"hold": first_started, "queued": queued_started},
        releases={"hold": release},
    )
    service = _service(tmp_path, agent, max_active_chats=2, max_waiting_chats=1)
    initial = service.chat("initial", username="owner")
    observe_lookup = partial(_observe_call, queued_resolved, service._find_conversation)

    with ThreadPoolExecutor(max_workers=3) as executor:
        first = executor.submit(
            service.chat, "hold", username="owner", conversation_id=initial.conversation_id,
        )
        try:
            assert first_started.wait(timeout=2)
            monkeypatch.setattr(service, "_find_conversation", observe_lookup)
            queued = executor.submit(
                service.chat, "queued", username="owner",
                conversation_id=initial.conversation_id,
            )
            assert queued_resolved.wait(timeout=2)
            independent = executor.submit(service.chat, "independent", username="other")
            assert independent.result(timeout=1).content == "Independent."
            assert not queued_started.is_set()
        finally:
            release.set()
        assert first.result(timeout=2).content == "First."
        assert queued.result(timeout=2).content == "Queued."
    assert agent.calls[-1][1] == agent.results[-2].transcript


def test_same_conversation_waiters_obey_waiting_limit_with_idle_active_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    queued = threading.Event()
    release = threading.Event()

    agent = GatedAgent(
        ["Initial."] + ["Finished."] * 6,
        started={"hold": started}, releases={"hold": release},
    )
    service = _service(tmp_path, agent, max_active_chats=4, max_waiting_chats=1)
    initial = service.chat("initial", username="owner")
    monkeypatch.setattr(
        service._waiting_chats,
        "acquire",
        partial(_observe_call, queued, service._waiting_chats.acquire, only_truthy=True),
    )
    with ThreadPoolExecutor(max_workers=3) as executor:
        # A second round verifies that the sole waiting place was released.
        for round_number in range(2):
            started.clear()
            queued.clear()
            release.clear()
            first = executor.submit(
                service.chat, "hold", username="owner",
                conversation_id=initial.conversation_id,
            )
            try:
                assert started.wait(timeout=2)
                follow_up = executor.submit(
                    service.chat, "queued", username="owner",
                    conversation_id=initial.conversation_id,
                )
                assert queued.wait(timeout=2)
                overflow = executor.submit(
                    service.chat, "overflow", username="owner",
                    conversation_id=initial.conversation_id,
                )
                with pytest.raises(WorkloadBusyError):
                    overflow.result(timeout=1)
                independent = executor.submit(
                    service.chat, "independent", username=f"other_{round_number}",
                )
                assert independent.result(timeout=1).content == "Finished."
                assert not first.done()
                assert not follow_up.done()
            finally:
                release.set()
            assert first.result(timeout=2).content == "Finished."
            assert follow_up.result(timeout=2).content == "Finished."
    assert [call[0] for call in agent.calls] == [
        "initial", "independent", "hold", "queued", "independent", "hold", "queued",
    ]


@pytest.mark.parametrize(
    "failure_stage", ("agent", "trace_setup", "conversation_callback", "response_callback"),
)
def test_failed_chat_returns_active_and_admission_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str,
) -> None:
    failure = RuntimeError("failed interaction")
    agent = FakeAgent([failure, "Recovered."] if failure_stage == "agent" else ["Recovered."] * 2)
    service = _service(tmp_path, agent, max_active_chats=1, max_waiting_chats=0)

    fail = partial(_raise_callback_failure, failure)

    with monkeypatch.context() as patch:
        callbacks = {}
        if failure_stage == "trace_setup":
            patch.setattr(service, "_trace_config", fail)
        elif failure_stage != "agent":
            callbacks[failure_stage] = fail
        with pytest.raises(RuntimeError, match="failed interaction"):
            service.chat("fail", username="owner", **callbacks)
    response = service.chat("retry", username="owner")
    assert response.content == "Recovered."
    assert response.interaction == 1
    assert agent.calls[-1][1] == ()


def test_stop_interrupts_a_stalled_model_and_preserves_history_and_traces(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    cleaned_up = threading.Event()

    client = _StalledClient(started, cleaned_up)
    agent = GaiaAgent(AgentRuntime(client), system_prompt="Test system prompt.")
    service = _service(tmp_path, agent, max_active_chats=1, max_waiting_chats=0)
    first = service.chat("initial", username="owner")
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(
            service.chat, "stall", username="owner", conversation_id=first.conversation_id,
        )
        try:
            assert started.wait(timeout=2)
            stopped = executor.submit(service.cancel, first.conversation_id, username="owner")
            assert stopped.result(timeout=2) == first.conversation_id
            assert cleaned_up.is_set()
            with pytest.raises(AgentCancelledError, match="Stopped thinking"):
                pending.result(timeout=2)
        finally:
            if not pending.done() and client.stalled_task is not None:
                client.stalled_task.get_loop().call_soon_threadsafe(client.stalled_task.cancel)
    resumed = service.chat("resume", username="owner", conversation_id=first.conversation_id)
    assert resumed.interaction == 3
    assert resumed.content == "Completed."
    assert client.requests[-1] == (
        ChatMessage(role="system", content="Test system prompt."),
        ChatMessage(role="user", content="initial"),
        ChatMessage(role="assistant", content="Completed."),
        ChatMessage(role="user", content="resume"),
    )
    traces = [json.loads(path.read_text()) for path in sorted(tmp_path.glob("*/interaction_*.json"))]
    assert [trace["run"]["status"] for trace in traces] == ["completed", "cancelled", "completed"]
    assert traces[1]["returned_result"] == {"status": "cancelled"}


def test_queued_independent_chat_can_be_stopped_and_its_place_reused(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    release = threading.Event()
    queued_published = threading.Event()
    replacement_published = threading.Event()
    queued_ids = []

    agent = GatedAgent(
        ["First.", "Reused."],
        started={"hold": started}, releases={"hold": release},
    )
    service = _service(tmp_path, agent, max_active_chats=1, max_waiting_chats=1)

    publish = partial(_publish_conversation, queued_ids, queued_published)

    with ThreadPoolExecutor(max_workers=3) as executor:
        first = executor.submit(service.chat, "hold", username="first")
        try:
            assert started.wait(timeout=2)
            queued = executor.submit(
                service.chat, "queued", username="second", conversation_callback=publish,
            )
            assert queued_published.wait(timeout=1)
            stopped = executor.submit(service.cancel, queued_ids[0], username="second")
            assert stopped.result(timeout=1) == queued_ids[0]
            with pytest.raises(AgentCancelledError):
                queued.result(timeout=1)
            assert not first.done()
            assert agent.calls == []
            replacement = executor.submit(
                service.chat, "replacement", username="second", conversation_id=queued_ids[0],
                conversation_callback=partial(_signal_event, replacement_published),
            )
            assert replacement_published.wait(timeout=1)
            assert not replacement.done()
        finally:
            release.set()
        assert first.result(timeout=2).content == "First."
        assert replacement.result(timeout=2).content == "Reused."
    assert [call[0] for call in agent.calls] == ["hold", "replacement"]
    assert agent.calls[-1][1] == ()


@pytest.mark.parametrize("streaming", (False, True), ids=("json", "ndjson"))
@pytest.mark.parametrize("source", ("admission", "tool"))
def test_http_busy_failures_have_safe_public_json_and_ndjson_errors(
    tmp_path: Path, streaming: bool, source: str,
) -> None:
    private_detail = "PRIVATE-CONCURRENCY-AND-BACKEND-DETAILS"
    if source == "tool":
        result = ToolResult(
            call=ToolCall("query", "query_occurrences", {}),
            ok=False, error_code="server_busy", error_message=private_detail,
        )
        failure = ToolAttemptsExhaustedError(
            private_detail, failed_result=result, transcript=(), tool_results=(result,),
            model_turns=3,
        )
    else:
        failure = WorkloadBusyError(private_detail)
    service = _service(tmp_path, FakeAgent([failure]))
    with _running_server(service) as server:
        if streaming:
            status, events, _headers = _stream_request(
                server, body=_chat_body("Find birds."), headers=STREAM_HEADERS,
            )
            assert status == 200
            assert events[-1]["type"] == "error"
            assert events[-1]["status"] == 503
            problem = events[-1]["error"]
            public_content = json.dumps(events)
        else:
            status, document, _headers, raw = _post_chat(server, _chat_body("Find birds."))
            assert status == 503
            problem = document["error"]
            public_content = raw.decode()
    assert problem == {"code": "server_busy", "message": "Gaia is busy. Please try again shortly."}
    assert private_detail not in public_content
