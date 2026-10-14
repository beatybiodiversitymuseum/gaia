"""Deterministic tests for web conversation state and interaction lifecycle."""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

import src.gaia_agent.web.chat_service as chat_service_module
from src.gaia_agent.runtime.contracts import (
    AgentCancellation,
    AgentCancelledError,
    AgentProgressEvent,
    ChatMessage,
)
from src.gaia_agent.runtime.defaults import CONTEXT_COMPACTED_NOTICE
from src.gaia_agent.web.chat_service import (
    ConversationNotRunningError,
    GaiaChatResponse,
    GaiaChatService,
    UnknownConversationError,
    UnknownGeneratedFileError,
    UnknownInteractionError,
)
from src.logger.gaia_logger import GaiaLoggerConfig

from tests.gaia_agent.web.support import (
    REPOSITORY_ROOT,
    DownloadAgent,
    FakeAgent,
    GatedAgent,
    _observe_call,
    _raise_callback_failure,
    _service,
    _wait_for_cancellation,
)


class _CancellableAgent(FakeAgent):
    def __init__(
        self, outcomes: list[str | BaseException], *, started: threading.Event,
    ) -> None:
        super().__init__(outcomes)
        self.started = started

    async def run(
        self,
        message,
        *,
        history=(),
        trace_config=None,
        progress_callback=None,
        cancellation=None,
    ):
        if message != "Cancel this request.":
            return await super().run(
                message,
                history=history,
                trace_config=trace_config,
                progress_callback=progress_callback,
                cancellation=cancellation,
            )
        assert isinstance(cancellation, AgentCancellation)
        history_messages = tuple(history)
        self.calls.append((message, history_messages, trace_config))
        self.started.set()
        await _wait_for_cancellation(cancellation)
        raise AgentCancelledError(
            "Stopped thinking",
            transcript=history_messages
            + (ChatMessage(role="user", content=message),),
            tool_results=(),
            model_turns=1,
        )


class _CompactingAgent(FakeAgent):
    async def run(self, message, **kwargs):
        result = await super().run(message, **kwargs)
        return replace(
            result,
            context=(
                result.transcript[0],
                ChatMessage(role="assistant", content="Cumulative chat summary."),
                result.transcript[-1],
            ),
        )


class _HistoryWritingAgent(_CompactingAgent):
    async def run(self, message, **kwargs):
        result = await super().run(message, **kwargs)
        history_path = kwargs["trace_config"].path.parent / "history.md"
        history_path.write_text(
            f"## Chat summary\n{message}\n\n## Tools called\nNone.\n",
            encoding="utf-8",
        )
        return result


def _capture_feedback(
    writes: list[tuple[GaiaLoggerConfig, dict[str, object]]],
    config: GaiaLoggerConfig,
    **feedback: object,
) -> None:
    writes.append((config, feedback))


def _reject_delivery(service: GaiaChatService, response: GaiaChatResponse) -> None:
    assert len(response.attachments) == 2
    metadata, stream = service.open_file(
        username="Museum Researcher",
        conversation_id=response.conversation_id,
        file_id=response.attachments[0].file_id,
    )
    with stream:
        assert stream.read() == metadata.path.read_bytes()
    raise ConnectionError("Browser disconnected.")


def _chat_after_signal(
    requested: threading.Event,
    service: GaiaChatService,
    message: str,
    **kwargs: object,
) -> GaiaChatResponse:
    requested.set()
    return service.chat(message, **kwargs)


def _ignore_feedback(*_args: object, **_kwargs: object) -> None:
    pass


def test_chat_service_commits_history_and_builds_interaction_traces(
    tmp_path: Path,
) -> None:
    agent = FakeAgent(["First answer.", "Second answer."])
    clock_values = iter((10.0, 10.25, 20.0, 20.75))
    service = _service(
        tmp_path,
        agent,
        clock=clock_values.__next__,
        trace_invocation={"model": "test-model"},
    )

    first = service.chat(
        "Find birds.",
        username=" Field Researcher #42 ",
    )
    second = service.chat(
        "Where were they found?",
        username="field-researcher 42",
        conversation_id=first.conversation_id,
    )

    assert first.payload() == {
        "conversation_id": "conversation_one",
        "interaction": 1,
        "message": {"role": "assistant", "content": "First answer.", "attachments": []},
        "metrics": {
            "elapsed_seconds": 0.25,
        },
        "verification": [{
            "sequence": 1,
            "capability": "retrieve_schema",
            "purpose": "Check which fields are available for the request.",
            "arguments": {"datasets": ["birds"]},
            "status": "completed",
            "evidence": {"datasets": []},
        }],
    }
    assert second.elapsed_seconds == 0.75
    assert second.interaction == 2
    assert agent.calls[0][1] == ()
    assert agent.calls[1][1] == agent.results[0].transcript
    trace_configs = [call[2] for call in agent.calls]
    assert [config.path.name for config in trace_configs] == [
        "interaction_0001.json",
        "interaction_0002.json",
    ]
    assert [config.invocation["interaction"] for config in trace_configs] == [
        1,
        2,
    ]
    assert trace_configs[0].path.parent == trace_configs[1].path.parent
    assert trace_configs[0].path.parent.parent == tmp_path
    assert re.fullmatch(
        r"gaia_chat_field_researcher_42_\d{8}T\d{6}_\d{6}Z",
        trace_configs[0].path.parent.name,
    )
    assert trace_configs[0].invocation == {
        "command": "serve",
        "interaction": 1,
        "model": "test-model",
        "transport": "http",
        "username": "field_researcher_42",
    }
    assert trace_configs[1].invocation["username"] == "field_researcher_42"


def test_chat_service_carries_compacted_context_and_reports_one_notice(
    tmp_path: Path,
) -> None:
    progress_events = (
        AgentProgressEvent(kind="context_compacting", turn=1),
        AgentProgressEvent(kind="context_compacted", turn=1),
        AgentProgressEvent(kind="context_compacting", turn=2),
        AgentProgressEvent(kind="context_compacted", turn=2),
    )
    agent = _CompactingAgent(
        ["First answer.", "Second answer."], progress_events=progress_events,
    )
    service = _service(tmp_path, agent)
    observed: list[AgentProgressEvent] = []
    first = service.chat(
        "Find birds.", username="researcher", progress_callback=observed.append,
    )
    second = service.chat(
        "Where were they found?",
        username="researcher",
        conversation_id=first.conversation_id,
    )

    assert observed == list(progress_events)
    assert agent.calls[1][1] == (
        ChatMessage(role="system", content="test system prompt"),
        ChatMessage(role="assistant", content="Cumulative chat summary."),
        ChatMessage(role="assistant", content="First answer."),
    )
    assert agent.calls[1][1] != agent.results[0].transcript
    assert first.notices == second.notices == (CONTEXT_COMPACTED_NOTICE,)
    assert first.payload()["message"]["notices"] == [CONTEXT_COMPACTED_NOTICE]


def test_feedback_targets_only_the_owned_completed_interaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    writes: list[tuple[GaiaLoggerConfig, dict[str, object]]] = []

    monkeypatch.setattr(
        chat_service_module,
        "write_gaia_trace_feedback",
        partial(_capture_feedback, writes),
    )
    agent = FakeAgent(["First answer.", RuntimeError("failed turn")])
    service = _service(tmp_path, agent)
    response = service.chat("Find birds.", username="Museum Researcher")

    with pytest.raises(RuntimeError, match="failed turn"):
        service.chat(
            "Try again.",
            username="Museum Researcher",
            conversation_id=response.conversation_id,
        )

    confirmation = service.record_feedback(
        username="museum-researcher",
        conversation_id=response.conversation_id,
        interaction=response.interaction,
        rating="bad",
        reasons=["incorrect", "missing_context"],
        details="The locality was wrong.",
    )

    assert confirmation == {
        "status": "recorded",
        "conversation_id": "conversation_one",
        "interaction": 1,
        "rating": "bad",
    }
    assert response.interaction == 1
    trace_config = agent.calls[0][2]
    assert isinstance(trace_config, GaiaLoggerConfig)
    assert trace_config.path.name == "interaction_0001.json"
    assert trace_config.invocation["interaction"] == response.interaction
    assert writes[0][0] is trace_config
    assert writes[0][1]["interaction"] == response.interaction

    with pytest.raises(UnknownConversationError):
        service.record_feedback(
            username="Another Researcher",
            conversation_id=response.conversation_id,
            interaction=1,
            rating="good",
            reasons=[],
            details="",
        )
    with pytest.raises(UnknownInteractionError):
        service.record_feedback(
            username="Museum Researcher",
            conversation_id=response.conversation_id,
            interaction=2,
            rating="bad",
            reasons=["incorrect"],
            details="This failed turn must not accept feedback.",
        )
    assert len(writes) == 1


def test_feedback_does_not_wait_for_the_next_interaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    next_turn_started = threading.Event()
    release_next_turn = threading.Event()
    monkeypatch.setattr(
        chat_service_module,
        "write_gaia_trace_feedback",
        _ignore_feedback,
    )
    agent = GatedAgent(
        ["First answer.", "Second answer."],
        started={"Keep working.": next_turn_started},
        releases={"Keep working.": release_next_turn},
    )
    service = _service(tmp_path, agent)
    first = service.chat("First request.", username="web_user")
    with ThreadPoolExecutor(max_workers=2) as executor:
        turn = executor.submit(
            service.chat, "Keep working.", username="web_user",
            conversation_id=first.conversation_id,
        )
        try:
            assert next_turn_started.wait(timeout=2)
            feedback = executor.submit(
                service.record_feedback,
                username="web_user",
                conversation_id=first.conversation_id,
                interaction=first.interaction,
                rating="good",
                reasons=[],
                details="",
            )
            feedback.result(timeout=0.2)
        finally:
            release_next_turn.set()
        turn.result(timeout=2)


def test_failed_turn_does_not_commit_history_or_leak_an_unpublished_session(
    tmp_path: Path,
) -> None:
    new_failure = FakeAgent(
        [RuntimeError("private failure detail"), "Answer after retry."]
    )
    new_service = _service(tmp_path / "new", new_failure)

    with pytest.raises(RuntimeError, match="private failure detail"):
        new_service.chat("Start a conversation.", username="web_user")

    with pytest.raises(UnknownConversationError):
        new_service.chat(
            "Continue unpublished conversation.",
            username="web_user",
            conversation_id="conversation_one",
        )
    assert new_service.chat("Retry.", username="web_user").interaction == 1

    agent = FakeAgent(
        [
            "Committed answer.",
            RuntimeError("private retry detail"),
            "Answer after retry.",
        ]
    )
    service = _service(tmp_path / "existing", agent)
    first = service.chat("First request.", username="web_user")

    with pytest.raises(RuntimeError, match="private retry detail"):
        service.chat(
            "Failed request.",
            username="web_user",
            conversation_id=first.conversation_id,
        )

    service.chat(
        "Retry request.",
        username="web_user",
        conversation_id=first.conversation_id,
    )

    assert agent.calls[1][1] == agent.results[0].transcript
    assert agent.calls[2][1] == agent.results[0].transcript
    assert [call[2].path.name for call in agent.calls] == [
        "interaction_0001.json",
        "interaction_0002.json",
        "interaction_0003.json",
    ]


@pytest.mark.parametrize("previous", (None, "Earlier summary – 鹭\r\n".encode("utf-8")))
def test_failed_delivery_restores_saved_history_and_keeps_committed_context(
    tmp_path: Path, previous: bytes | None,
) -> None:
    agent = _HistoryWritingAgent(["First answer.", "Undelivered answer.", "Retry answer."])
    service = _service(tmp_path, agent)
    first = service.chat("First request.", username="researcher")
    history_path = agent.calls[0][2].path.parent / "history.md"
    assert history_path.read_text(encoding="utf-8").startswith("## Chat summary\nFirst request.")
    if previous is None:
        history_path.unlink()
    else:
        history_path.write_bytes(previous)

    with pytest.raises(ConnectionError, match="Browser disconnected"):
        service.chat(
            "Undelivered request.",
            username="researcher",
            conversation_id=first.conversation_id,
            response_callback=partial(
                _raise_callback_failure, ConnectionError("Browser disconnected"),
            ),
        )

    if previous is None:
        assert not history_path.exists()
    else:
        assert history_path.read_bytes() == previous
    service.chat("Retry request.", username="researcher", conversation_id=first.conversation_id)
    assert agent.calls[2][1] == agent.calls[1][1]
    assert all(message.content != "Undelivered answer." for message in agent.calls[2][1])
    assert "Retry request." in history_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "failure_stage",
    ("trace_setup", "conversation_callback", "response_callback"),
)
def test_setup_or_delivery_failure_discards_unpublished_conversation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str,
) -> None:
    agent = FakeAgent(["Answer.", "Answer."])
    service = _service(tmp_path, agent)
    failure = (
        ValueError("trace setup failed")
        if failure_stage == "trace_setup"
        else ConnectionError("client disconnected")
    )

    fail = partial(_raise_callback_failure, failure)

    with monkeypatch.context() as patch:
        callbacks = {}
        if failure_stage == "trace_setup":
            patch.setattr(service, "_trace_config", fail)
        else:
            callbacks[failure_stage] = fail
        with pytest.raises(type(failure), match=str(failure)):
            service.chat(
                "Start a conversation.",
                username="web_user",
                **callbacks,
            )

    assert len(agent.calls) == (1 if failure_stage == "response_callback" else 0)
    with pytest.raises(UnknownConversationError):
        service.chat(
            "Continue unpublished conversation.",
            username="web_user",
            conversation_id="conversation_one",
        )
    response = service.chat("Start again.", username="web_user")
    assert response.interaction == 1
    assert agent.calls[-1][1] == ()


def test_chat_service_cancels_without_committing_the_incomplete_turn(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    agent = _CancellableAgent(
        ["First answer.", "Answer after cancellation."], started=started,
    )
    service = _service(tmp_path, agent)
    first = service.chat("First request.", username="web_user")
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            service.chat, "Cancel this request.", username="web_user",
            conversation_id=first.conversation_id,
        )
        assert started.wait(timeout=2)
        cancelled_id = service.cancel(first.conversation_id, username="web_user")
        third = service.chat(
            "Continue after cancellation.",
            username="web_user",
            conversation_id=first.conversation_id,
        )
        with pytest.raises(AgentCancelledError):
            pending.result(timeout=2)

    assert cancelled_id == first.conversation_id
    assert third.content == "Answer after cancellation."
    assert third.interaction == 3
    assert agent.calls[1][1] == agent.results[0].transcript
    assert agent.calls[2][1] == agent.results[0].transcript
    with pytest.raises(ConversationNotRunningError):
        service.cancel(first.conversation_id, username="web_user")


def test_retained_histories_and_long_conversations_remain_usable(
    tmp_path: Path,
) -> None:
    identifiers = iter(("conversation_one", "conversation_two", "conversation_three"))
    agent = FakeAgent(["Answer."] * 103)
    service = GaiaChatService(
        agent,
        repository_root=REPOSITORY_ROOT,
        trace_root=tmp_path,
        conversation_id_factory=identifiers.__next__,
    )
    first = service.chat("First request.", username="web_user")
    service.chat("Another conversation.", username="second_user")
    service.chat("A third conversation.", username="third_user")

    for interaction in range(2, 102):
        response = service.chat(
            f"Follow-up {interaction}.",
            username="web_user",
            conversation_id=first.conversation_id,
        )
        assert response.interaction == interaction

    assert response.interaction == 101
    assert agent.calls[3][1] == agent.results[0].transcript
    assert agent.calls[-1][1] == agent.results[-2].transcript
    assert len(agent.calls) == 103

    with pytest.raises(UnknownConversationError):
        service.chat(
            "Unknown.",
            username="web_user",
            conversation_id="missing_conversation",
        )


def test_independent_conversations_overlap_and_faster_request_finishes_first(
    tmp_path: Path,
) -> None:
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    agent = GatedAgent(
        ["Second.", "First."],
        started={"First request.": first_entered, "Second request.": second_entered},
        releases={"First request.": release_first},
    )
    identifiers = iter(("conversation_one", "conversation_two"))
    service = GaiaChatService(
        agent,
        repository_root=REPOSITORY_ROOT,
        trace_root=tmp_path,
        conversation_id_factory=identifiers.__next__,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.chat, "First request.", username="First request.")
        try:
            assert first_entered.wait(timeout=2)
            second = executor.submit(
                service.chat, "Second request.", username="Second request.",
            )
            second_response = second.result(timeout=2)
            assert not first.done()
            assert second_response.content == "Second."
        finally:
            release_first.set()
        first_response = first.result(timeout=2)

    assert second_entered.is_set()
    assert agent.maximum_active_calls == 2
    assert first_response.content == "First."


def test_published_chat_survives_first_turn_failure_with_a_queued_follow_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_entered = threading.Event()
    follow_up_resolved = threading.Event()
    release_first = threading.Event()
    agent = GatedAgent(
        [RuntimeError("temporary failure"), "Recovered.", "Continued."],
        started={"First request.": first_entered},
        releases={"First request.": release_first},
    )
    service = _service(tmp_path, agent)
    monkeypatch.setattr(
        service, "_find_conversation",
        partial(_observe_call, follow_up_resolved, service._find_conversation),
    )
    published = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            service.chat, "First request.", username="web_user",
            conversation_callback=published.append,
        )
        try:
            assert first_entered.wait(timeout=2)
            second = executor.submit(
                service.chat, "Queued follow-up.", username="web_user",
                conversation_id=published[0],
            )
            assert follow_up_resolved.wait(timeout=2)
        finally:
            release_first.set()
        with pytest.raises(RuntimeError, match="^temporary failure$"):
            first.result(timeout=2)
        response = second.result(timeout=2)

    assert response.interaction == 2
    assert response.content == "Recovered."
    assert agent.calls[1][1] == ()
    continued = service.chat(
        "Continue.", username="web_user", conversation_id=published[0],
    )
    assert continued.interaction == 3
    assert continued.content == "Continued."
    assert agent.calls[2][1] == agent.results[0].transcript


def test_same_conversation_waits_for_prior_turn_and_uses_its_history(
    tmp_path: Path,
) -> None:
    first_entered = threading.Event()
    release_first = threading.Event()
    second_requested = threading.Event()
    second_entered = threading.Event()
    agent = GatedAgent(
        ["Initial answer.", "Independent answer.", "First answer.", "Second answer."],
        started={"First follow-up.": first_entered, "Second follow-up.": second_entered},
        releases={"First follow-up.": release_first},
    )
    identifiers = iter(("conversation_one", "conversation_two"))
    service = GaiaChatService(
        agent,
        repository_root=REPOSITORY_ROOT,
        trace_root=tmp_path,
        conversation_id_factory=identifiers.__next__,
    )
    initial = service.chat("Start.", username="web_user")
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            service.chat, "First follow-up.", username="web_user",
            conversation_id=initial.conversation_id,
        )
        try:
            assert first_entered.wait(timeout=2)
            second = executor.submit(
                _chat_after_signal, second_requested, service, "Second follow-up.",
                username="web_user", conversation_id=initial.conversation_id,
            )
            assert second_requested.wait(timeout=2)
            assert not second_entered.wait(timeout=0.1)
            independent = service.chat("Independent request.", username="other_user")
            assert independent.content == "Independent answer."
        finally:
            release_first.set()
        first_response = first.result(timeout=2)
        second_response = second.result(timeout=2)

    assert [call[0] for call in agent.calls] == [
        "Start.", "Independent request.", "First follow-up.", "Second follow-up."
    ]
    assert agent.calls[3][1] == agent.results[2].transcript
    assert first_response.interaction == 2
    assert second_response.interaction == 3
    assert second_response.content == "Second answer."


@pytest.mark.parametrize("failure", ("tool", "turn", "delivery"))
def test_failed_work_does_not_publish_downloadable_attachments(
    tmp_path: Path,
    failure: str,
) -> None:
    agent = DownloadAgent(failed_tool=failure == "tool", failed_turn=failure == "turn")
    service = _service(tmp_path, agent)
    published_ids = []

    arguments = {
        "username": "Museum Researcher",
        "conversation_callback": published_ids.append,
    }
    if failure == "delivery":
        arguments["response_callback"] = partial(_reject_delivery, service)
    if failure == "tool":
        response = service.chat("Export.", **arguments)
        assert response.attachments == ()
        assert response.payload()["message"]["attachments"] == []
    else:
        with pytest.raises(AgentCancelledError if failure == "turn" else ConnectionError):
            service.chat("Export.", **arguments)
    assert published_ids == ["conversation_one"]
    for metadata in agent.files:
        with pytest.raises(UnknownGeneratedFileError):
            service.open_file(
                username="Museum Researcher",
                conversation_id=published_ids[0],
                file_id=metadata["file_id"],
            )
