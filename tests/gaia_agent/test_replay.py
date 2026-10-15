"""Tests for terminal replay of a public verification step."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.gaia_agent import replay as replay_tool_call


def test_parse_call_accepts_verification_step() -> None:
    assert replay_tool_call.parse_call({
        "sequence": 1,
        "capability": "query_statistics",
        "purpose": "Count records",
        "arguments": {"dataset": "example"},
        "evidence": {"rows": []},
    }) == ("query_statistics", {"dataset": "example"})


@pytest.mark.parametrize("capability", ["generate_excel", "unknown", None])
def test_parse_call_rejects_unreplayable_capability(capability: object) -> None:
    with pytest.raises(ValueError, match="supported read-only"):
        replay_tool_call.parse_call({"capability": capability, "arguments": {}})


def test_replay_call_uses_registered_handler_and_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(arguments: dict[str, object]) -> object:
        arguments["changed"] = True
        return {"internal": "result"}

    tool = SimpleNamespace(
        definition=SimpleNamespace(name="query_statistics"),
        handler=handler,
        observation_projector=lambda result: {"visible": result},
    )
    toolset = SimpleNamespace(registered_tools=lambda: (tool,))
    monkeypatch.setattr(replay_tool_call.GaiaToolset, "load", lambda: toolset)
    arguments = {"dataset": "example"}

    assert asyncio.run(replay_tool_call.replay_call("query_statistics", arguments)) == {
        "visible": {"internal": "result"},
    }
    assert arguments == {"dataset": "example"}
