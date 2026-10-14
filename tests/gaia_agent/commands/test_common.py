"""Tests for shared command construction and trace metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from src.gaia_agent.commands import common


@pytest.mark.parametrize("api_key", (None, "secret-token"))
def test_build_command_agent_passes_optional_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    api_key: str | None,
) -> None:
    captured_client_arguments: dict[str, object] = {}
    client = object()

    def create_client(**arguments: object) -> object:
        captured_client_arguments.update(arguments)
        return client

    monkeypatch.setattr(common, "OpenAIChatClient", create_client)
    monkeypatch.setattr(
        common,
        "build_gaia_agent",
        lambda repository_root, model_client, **arguments: (
            repository_root,
            model_client,
            arguments,
        ),
    )
    arguments = argparse.Namespace(
        endpoint="https://models.example/v1",
        model="example-model",
        api_key=api_key,
        system_prompt=tmp_path / "system_prompt.md",
        max_active_queries=1,
        max_waiting_queries=2,
        max_active_exports=3,
        max_waiting_exports=4,
    )

    built_agent = common.build_command_agent(arguments, tmp_path)

    assert captured_client_arguments == {
        "base_url": "https://models.example/v1",
        "model": "example-model",
        "api_key": api_key,
    }
    assert built_agent[0:2] == (tmp_path, client)


def test_agent_trace_invocation_excludes_api_key(tmp_path: Path) -> None:
    arguments = argparse.Namespace(
        command="query",
        endpoint="https://models.example/v1",
        model="example-model",
        api_key="secret-token",
        system_prompt=tmp_path / "system_prompt.md",
    )

    invocation = common.agent_trace_invocation(arguments)

    assert invocation == {
        "command": "query",
        "endpoint": "https://models.example/v1",
        "model": "example-model",
        "system_prompt_path": str(tmp_path / "system_prompt.md"),
    }
