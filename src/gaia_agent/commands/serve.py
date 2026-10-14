"""HTTP server lifecycle for the Gaia serve command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.gaia_agent.commands.common import agent_trace_invocation, build_command_agent
from src.gaia_agent.runtime.openai_client import OpenAIClientError
from src.gaia_agent.web.chat_service import GaiaChatService
from src.gaia_agent.web.http_api import (
    create_gaia_http_server,
    load_cors_allowed_origins,
    validate_cors_allowed_origins,
)


def run(arguments: argparse.Namespace, *, repository_root: Path) -> int:
    """Build Gaia once and serve its loopback HTTP bridge."""
    try:
        allowed_origins = (
            validate_cors_allowed_origins(arguments.allowed_origins)
            if arguments.allowed_origins is not None
            else load_cors_allowed_origins()
        )
        agent = build_command_agent(arguments, repository_root)
        chat_service = GaiaChatService(
            agent,
            repository_root=repository_root,
            trace_root=arguments.trace_directory,
            max_active_chats=arguments.max_active_chats,
            max_waiting_chats=arguments.max_waiting_chats,
            trace_invocation=agent_trace_invocation(arguments),
        )
        server = create_gaia_http_server(
            chat_service,
            host=arguments.host,
            port=arguments.port,
            allowed_origins=allowed_origins,
        )
    except (OpenAIClientError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    host, port = server.server_address[:2]
    print(f"Gaia HTTP bridge: http://{host}:{port}")
    print("Press Control-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0
