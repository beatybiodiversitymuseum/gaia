"""Shared construction and trace metadata for Gaia commands."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlsplit

from src.gaia_agent.runtime.openai_client import OpenAIChatClient
from src.gaia_agent.runtime.service import GaiaAgent, build_gaia_agent
from src.logger import bounded_log_detail

_MAX_INVOCATION_VALUE_LENGTH = 160


def agent_trace_invocation(arguments: argparse.Namespace) -> dict[str, object]:
    """Return bounded, non-secret runtime settings for an agent trace."""
    return {
        "command": arguments.command,
        "endpoint": endpoint_identity(arguments.endpoint),
        "model": bounded_log_detail(
            arguments.model, limit=_MAX_INVOCATION_VALUE_LENGTH
        ),
        "system_prompt_path": str(arguments.system_prompt),
    }


def endpoint_identity(endpoint: str) -> str:
    """Remove credentials, query parameters, and fragments from an endpoint."""
    parsed = urlsplit(endpoint)
    host = parsed.hostname or "<invalid-host>"
    if ":" in host:
        host = f"[{host}]"
    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        port = ":<invalid-port>"
    return f"{parsed.scheme}://{host}{port}{parsed.path}"


def build_command_agent(
    arguments: argparse.Namespace, repository_root: Path
) -> GaiaAgent:
    """Build the Gaia agent shared by interactive and HTTP commands."""
    client = OpenAIChatClient(
        base_url=arguments.endpoint,
        model=arguments.model,
        api_key=arguments.api_key,
    )
    return build_gaia_agent(
        repository_root,
        client,
        system_prompt_path=arguments.system_prompt,
        max_active_queries=arguments.max_active_queries,
        max_waiting_queries=arguments.max_waiting_queries,
        max_active_exports=arguments.max_active_exports,
        max_waiting_exports=arguments.max_waiting_exports,
    )
