"""Compaction budgets, cumulative source coverage, validation, and persistence."""

from __future__ import annotations

import asyncio
import json
import stat
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.gaia_agent.runtime.context import (
    COMPACTION_TOKEN_LIMIT,
    CompactionError,
    ContextBudget,
    compact_context,
    history_transaction,
    write_history,
)
from src.gaia_agent.runtime.contracts import (
    AgentCancellation,
    ChatMessage,
    ModelTurn,
    ToolCall,
    ToolDefinition,
)
from src.gaia_agent.runtime.defaults import CONTEXT_CONTINUATION_PROMPT


SUMMARY = "## Chat summary\nFind café specimens; export is pending.\n\n## Tools called\nquery: two matching records.\n"


class CompactionClient:
    """Use characters as deterministic tokens to exercise real budget decisions."""

    def __init__(self, turns: Sequence[ModelTurn] = (), limit: int = 100_000) -> None:
        self.turns = list(turns)
        self.limit = limit
        self.requests: list[dict[str, object]] = []
        self.preflights: list[tuple[ChatMessage, ...]] = []

    async def count_text_tokens(self, text: str) -> int:
        return len(text)

    async def context_info(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        enable_thinking: bool | None = None,
    ) -> SimpleNamespace:
        assert not tools
        assert enable_thinking is False
        self.preflights.append(tuple(messages))
        return SimpleNamespace(
            prompt_tokens=sum(len(message.content or "") for message in messages) + 12,
            context_limit_tokens=self.limit,
        )

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        max_completion_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> ModelTurn:
        assert not tools
        assert enable_thinking is False
        assert max_completion_tokens is not None
        prompt_tokens = sum(len(message.content or "") for message in messages) + 12
        assert prompt_tokens + max_completion_tokens <= ContextBudget(self.limit).usable_tokens
        self.requests.append({
            "messages": tuple(messages),
            "data": json.loads(messages[-1].content or ""),
            "max_completion_tokens": max_completion_tokens,
        })
        return self.turns.pop(0) if self.turns else ModelTurn(content=SUMMARY, finish_reason="stop")


@pytest.mark.parametrize(
    ("limit", "usable", "trigger"),
    [(100_000, 95_000, 85_500), (32_768, 31_129, 28_016), (101, 95, 85), (1, 0, 0)],
)
def test_budget_uses_integer_thresholds(limit: int, usable: int, trigger: int) -> None:
    budget = ContextBudget(limit)
    assert budget.usable_tokens == usable
    assert budget.trigger_tokens == trigger


@pytest.mark.parametrize("limit", [0, -1, True, 10.5])
def test_budget_rejects_invalid_limits(limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ContextBudget(limit)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_compaction_quotes_full_history_preserving_tool_order_and_unicode() -> None:
    call = ToolCall("q1", "query", {"country": "Canada", "nested": {"z": "é", "a": ""}})
    messages = (
        ChatMessage("system", "PRIVATE_SYSTEM_PROMPT"),
        ChatMessage("assistant", SUMMARY),
        ChatMessage("user", "Now export those café records; keep Unicode."),
        ChatMessage("assistant", None, tool_calls=(call,), reasoning_content="PRIVATE_REASONING"),
        ChatMessage("tool", '{"ok":true,"result":{"count":2}}', tool_call_id="q1", name="query"),
    )
    client = CompactionClient()
    result = await compact_context(client, messages, context_limit_tokens=client.limit)
    assert result == SUMMARY
    assert len(client.requests) == 1
    data = client.requests[0]["data"]
    source = data["source_fragment"]
    assert "PRIVATE_SYSTEM_PROMPT" not in source
    assert "PRIVATE_REASONING" not in source
    assert "café" in source
    assert '"country":"Canada","nested":{"z":"é","a":""}' in source
    records = [json.loads(line) for line in source.splitlines()]
    assert [record["role"] for record in records] == ["assistant", "user", "assistant", "tool"]
    assert records[0]["content"] == SUMMARY
    assert records[3]["tool_call_id"] == "q1"
    assert messages[0].content == "PRIVATE_SYSTEM_PROMPT"


@pytest.mark.anyio
@pytest.mark.parametrize("system_messages", [0, 1, 2])
async def test_new_user_correction_keeps_prior_calls_in_cumulative_compaction_input(
    system_messages: int,
) -> None:
    previous = (
        "## Chat summary\nFind Canadian café specimens; export is pending.\n\n"
        '## Tools called\nquery(country="Canada", locality="café"): '
        "two matching records, occurrenceIDs C-1 and C-2.\n"
    )
    replacement = previous.replace(
        "export is pending.", "export is pending; use the corrected basename cafe_records.xlsx.",
    )
    messages = (ChatMessage("system", "PRIVATE_SYSTEM_PROMPT"),) * system_messages + (
        ChatMessage("user", CONTEXT_CONTINUATION_PROMPT),
        ChatMessage("assistant", previous),
        ChatMessage("assistant", "The two matching records are ready for export."),
        ChatMessage("user", "Correction: name the workbook cafe_records.xlsx."),
    )
    client = CompactionClient([ModelTurn(content=replacement)])
    result = await compact_context(client, messages, context_limit_tokens=client.limit)
    assert result == replacement
    assert len(client.requests) == 1
    data = client.requests[0]["data"]
    assert data["previous_compaction"] == previous
    records = [json.loads(line) for line in data["source_fragment"].splitlines()]
    assert records == [
        {"role": "assistant", "content": messages[-2].content},
        {"role": "user", "content": messages[-1].content},
    ]
    assert CONTEXT_CONTINUATION_PROMPT not in data["source_fragment"]
    assert "query" not in data["source_fragment"]
    assert 'query(country="Canada", locality="café")' in data["previous_compaction"]
    assert messages[-3].content == previous


@pytest.mark.anyio
async def test_previous_compaction_without_new_records_can_be_compacted_again() -> None:
    messages = (
        ChatMessage("system", "PRIVATE_SYSTEM_PROMPT"),
        ChatMessage("user", CONTEXT_CONTINUATION_PROMPT),
        ChatMessage("assistant", SUMMARY),
    )
    client = CompactionClient()
    assert await compact_context(client, messages, context_limit_tokens=client.limit) == SUMMARY
    assert len(client.requests) == 1
    data = client.requests[0]["data"]
    assert data["previous_compaction"] == SUMMARY
    assert data["source_fragment"] == ""
    assert data["source_continues"] is False


@pytest.mark.anyio
async def test_harness_prefix_seed_survives_oversized_newer_history() -> None:
    newer_content = "New café observations 🐢 " * 1_000
    messages = (
        ChatMessage("system", "PRIVATE_SYSTEM_PROMPT"),
        ChatMessage("user", CONTEXT_CONTINUATION_PROMPT),
        ChatMessage("assistant", SUMMARY),
        ChatMessage("user", newer_content),
    )
    client = CompactionClient(limit=6_000)
    assert await compact_context(client, messages, context_limit_tokens=client.limit) == SUMMARY
    assert len(client.requests) > 1
    data = [request["data"] for request in client.requests]
    assert all(item["previous_compaction"] == SUMMARY for item in data)
    recovered = json.loads("".join(item["source_fragment"] for item in data))
    assert recovered == {"role": "user", "content": newer_content}


@pytest.mark.anyio
async def test_erasing_prior_tools_with_none_receives_a_correction_attempt() -> None:
    messages = (
        ChatMessage("system", "System"),
        ChatMessage("user", CONTEXT_CONTINUATION_PROMPT),
        ChatMessage("assistant", SUMMARY),
        ChatMessage("user", "Correction: use cafe_records.xlsx."),
    )
    erased = "## Chat summary\nExport cafe_records.xlsx.\n\n## Tools called\nNone.\n"
    client = CompactionClient([ModelTurn(content=erased), ModelTurn(content=SUMMARY)])
    assert await compact_context(client, messages, context_limit_tokens=client.limit) == SUMMARY
    assert len(client.requests) == 2
    assert all(request["data"]["previous_compaction"] == SUMMARY for request in client.requests)


@pytest.mark.anyio
async def test_prior_no_tools_can_remain_none_in_next_compaction() -> None:
    previous = "## Chat summary\nFind records; no action yet.\n\n## Tools called\nNone.\n"
    messages = (
        ChatMessage("system", "System"),
        ChatMessage("user", CONTEXT_CONTINUATION_PROMPT),
        ChatMessage("assistant", previous),
        ChatMessage("user", "Use cafe_records.xlsx for the workbook."),
    )
    client = CompactionClient([ModelTurn(content=previous)])
    assert await compact_context(client, messages, context_limit_tokens=client.limit) == previous
    assert len(client.requests) == 1


@pytest.mark.anyio
async def test_oversized_history_is_summarized_in_complete_consecutive_pieces() -> None:
    content = "Record café 🐢 " * 2_000
    client = CompactionClient(limit=6_000)
    result = await compact_context(client, (ChatMessage("tool", content, tool_call_id="q"),), context_limit_tokens=client.limit)
    assert result == SUMMARY
    assert len(client.requests) > 2
    data = [request["data"] for request in client.requests]
    recovered = json.loads("".join(item["source_fragment"] for item in data))
    assert recovered == {"role": "tool", "content": content, "tool_call_id": "q"}
    assert data[0]["previous_compaction"] is None
    assert all(item["previous_compaction"] == SUMMARY for item in data[1:])
    assert all(item["source_continues"] is True for item in data[:-1])
    assert data[-1]["source_continues"] is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid",
    [
        ModelTurn(content="Not the required headings"),
        ModelTurn(content="## Chat summary\n\n## Tools called\nNone."),
        ModelTurn(content="## Chat summary\nRequest.\n## Extra heading\nBad.\n## Tools called\nNone."),
        ModelTurn(content=SUMMARY, finish_reason="length"),
        ModelTurn(content=SUMMARY, finish_reason="content_filter"),
        ModelTurn(content=SUMMARY, finish_reason="tool_calls"),
        ModelTurn(content=SUMMARY, finish_reason="unexpected"),
        ModelTurn(content=SUMMARY, tool_calls=(ToolCall("x", "query", {}),)),
        ModelTurn(reasoning_content="No summary generated"),
        ModelTurn(content="## Chat summary\n" + "s" * 12_001 + "\n## Tools called\nNone."),
        ModelTurn(content="## Chat summary\nA request.\n## Tools called\n" + "t" * 4_001),
        ModelTurn(content="## Chat summary\n" + "s" * 12_000 + "\n## Tools called\n" + "t" * 4_000),
    ],
)
async def test_invalid_compaction_receives_one_correction_attempt(invalid: ModelTurn) -> None:
    client = CompactionClient([invalid, ModelTurn(content=SUMMARY)])
    result = await compact_context(client, [ChatMessage("user", "Find records")], context_limit_tokens=client.limit)
    assert result == SUMMARY
    assert len(client.requests) == 2
    assert client.requests[0]["data"] == client.requests[1]["data"]
    assert "previous attempt failed" in client.requests[1]["messages"][0].content


@pytest.mark.anyio
async def test_summary_below_section_and_total_caps_is_accepted() -> None:
    summary = "## Chat summary\n" + "s" * 11_950 + "\n\n## Tools called\n" + "t" * 3_990 + "\n"
    assert len(summary) <= COMPACTION_TOKEN_LIMIT
    client = CompactionClient([ModelTurn(content=summary)])
    assert await compact_context(client, [ChatMessage("user", "Records")], context_limit_tokens=client.limit) == summary
    assert len(client.requests) == 1


@pytest.mark.anyio
async def test_invalid_compaction_stops_after_bounded_attempts_without_mutation() -> None:
    messages = [ChatMessage("system", "Keep system"), ChatMessage("user", "Find records")]
    before = list(messages)
    client = CompactionClient([ModelTurn(content="invalid")] * 3)
    with pytest.raises(CompactionError, match="valid compacted history"):
        await compact_context(client, messages, context_limit_tokens=client.limit)
    assert len(client.requests) == 2
    assert messages == before


@pytest.mark.anyio
async def test_smaller_context_enforces_smaller_summary_output_budget() -> None:
    client = CompactionClient([ModelTurn(content="## Chat summary\n" + "s" * 2_000 + "\n## Tools called\nNone.")], limit=6_000)
    result = await compact_context(client, [ChatMessage("user", "Find records")], context_limit_tokens=client.limit)
    assert result == SUMMARY
    assert len(client.requests) == 2
    assert client.requests[0]["max_completion_tokens"] < 2_000


@pytest.mark.anyio
async def test_too_small_context_fails_before_a_model_request() -> None:
    client = CompactionClient(limit=200)
    with pytest.raises(CompactionError, match="too small"):
        await compact_context(client, [ChatMessage("user", "Find records")], context_limit_tokens=client.limit)
    assert not client.requests


@pytest.mark.anyio
async def test_empty_conversation_fails_before_model_request() -> None:
    client = CompactionClient()
    with pytest.raises(CompactionError, match="no conversation"):
        await compact_context(client, [ChatMessage("system", "system")], context_limit_tokens=client.limit)
    assert not client.preflights
    assert not client.requests


@pytest.mark.anyio
async def test_cancelled_compaction_makes_no_model_requests() -> None:
    cancellation = AgentCancellation()
    cancellation.cancel()
    client = CompactionClient()
    with pytest.raises(asyncio.CancelledError):
        await compact_context(client, [ChatMessage("user", "Find records")], context_limit_tokens=client.limit, cancellation=cancellation)
    assert not client.preflights
    assert not client.requests


class CancellingClient(CompactionClient):
    def __init__(self, cancellation: AgentCancellation) -> None:
        super().__init__([ModelTurn(content="invalid")])
        self.cancellation = cancellation

    async def complete(self, **kwargs: object) -> ModelTurn:
        turn = await super().complete(**kwargs)
        self.cancellation.cancel()
        return turn


@pytest.mark.anyio
async def test_cancellation_after_request_prevents_retry() -> None:
    cancellation = AgentCancellation()
    client = CancellingClient(cancellation)
    with pytest.raises(asyncio.CancelledError):
        await compact_context(client, [ChatMessage("user", "Find records")], context_limit_tokens=client.limit, cancellation=cancellation)
    assert len(client.requests) == 1


def test_history_is_created_then_atomically_replaced_with_owner_only_access(tmp_path: Path) -> None:
    path = tmp_path / "session" / "history.md"
    write_history(path, SUMMARY)
    assert path.read_text(encoding="utf-8") == SUMMARY
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    replacement = SUMMARY.replace("café", "new café")
    write_history(path, replacement)
    assert path.read_text(encoding="utf-8") == replacement
    assert list(path.parent.iterdir()) == [path]


def test_failed_history_replace_preserves_previous_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "history.md"
    path.write_bytes(b"old history\n")
    monkeypatch.setattr("src.gaia_agent.runtime.context.os.replace", fail_replace)
    with pytest.raises(CompactionError, match="could not be saved"):
        write_history(path, SUMMARY)
    assert path.read_bytes() == b"old history\n"
    assert list(tmp_path.iterdir()) == [path]


def fail_replace(source: object, destination: object) -> None:
    raise OSError("replace failed")


@pytest.mark.parametrize("error", [RuntimeError("delivery failed"), asyncio.CancelledError()])
def test_history_transaction_restores_existing_exact_bytes_on_failure(
    tmp_path: Path, error: BaseException,
) -> None:
    path = tmp_path / "history.md"
    original = "Earlier café history\r\n".encode("utf-8") + b"\xff"
    path.write_bytes(original)
    with pytest.raises(type(error)) as caught:
        with history_transaction(path):
            write_history(path, SUMMARY)
            raise error
    assert caught.value is error
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("error", [RuntimeError("delivery failed"), asyncio.CancelledError()])
def test_history_transaction_removes_new_file_on_failure(
    tmp_path: Path, error: BaseException,
) -> None:
    path = tmp_path / "history.md"
    with pytest.raises(type(error)) as caught:
        with history_transaction(path):
            write_history(path, SUMMARY)
            raise error
    assert caught.value is error
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_history_transaction_retains_replacement_on_success(tmp_path: Path) -> None:
    path = tmp_path / "history.md"
    path.write_bytes(b"old history")
    with history_transaction(path):
        write_history(path, SUMMARY)
    assert path.read_text(encoding="utf-8") == SUMMARY


def test_history_transaction_skips_restoration_for_unchanged_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "history.md"
    path.write_bytes(b"old history")
    monkeypatch.setattr("src.gaia_agent.runtime.context.os.replace", fail_replace)
    with pytest.raises(RuntimeError, match="delivery failed"):
        with history_transaction(path):
            raise RuntimeError("delivery failed")
    assert path.read_bytes() == b"old history"
    assert list(tmp_path.iterdir()) == [path]


def test_history_transaction_preserves_file_when_compaction_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "history.md"
    path.write_bytes(b"old history")
    monkeypatch.setattr("src.gaia_agent.runtime.context.os.replace", fail_replace)
    with pytest.raises(CompactionError, match="could not be saved"):
        with history_transaction(path):
            write_history(path, SUMMARY)
    assert path.read_bytes() == b"old history"
    assert list(tmp_path.iterdir()) == [path]


def test_history_transaction_without_path_preserves_exception() -> None:
    with pytest.raises(RuntimeError, match="delivery failed"):
        with history_transaction(None):
            raise RuntimeError("delivery failed")
