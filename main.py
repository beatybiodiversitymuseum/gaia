#!/usr/bin/env python3
"""Gaia command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path

from src.gaia_agent.runtime.defaults import (
    DEFAULT_GAIA_HTTP_HOST,
    DEFAULT_GAIA_HTTP_PORT,
    DEFAULT_MAX_ACTIVE_CHATS,
    DEFAULT_MAX_ACTIVE_EXPORTS,
    DEFAULT_MAX_ACTIVE_QUERIES,
    DEFAULT_MAX_WAITING_CHATS,
    DEFAULT_MAX_WAITING_EXPORTS,
    DEFAULT_MAX_WAITING_QUERIES,
)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

QUERY_COMMAND = "query"
SERVE_COMMAND = "serve"
REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_DIRECTORY = REPOSITORY_ROOT / "logs"
DEFAULT_AGENT_ENDPOINT = "http://127.0.0.1:8000/v1"
DEFAULT_AGENT_MODEL = "default_model"
DEFAULT_SYSTEM_PROMPT_PATH = REPOSITORY_ROOT / "src/gaia_agent/runtime/system_prompt.md"
COMMAND_MODULES = {
    QUERY_COMMAND: "src.gaia_agent.commands.query",
    SERVE_COMMAND: "src.gaia_agent.commands.serve",
}


def _add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the model connection and prompt settings shared by query and serve."""
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_AGENT_ENDPOINT,
        help="OpenAI-compatible API base (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_AGENT_MODEL,
        help="model identifier sent to the server (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        help="optional bearer token sent to the model server",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=DEFAULT_SYSTEM_PROMPT_PATH,
        help="system prompt placed at the start of each chat (default: %(default)s)",
    )
    for resource, active, waiting in (
        ("queries", DEFAULT_MAX_ACTIVE_QUERIES, DEFAULT_MAX_WAITING_QUERIES),
        ("exports", DEFAULT_MAX_ACTIVE_EXPORTS, DEFAULT_MAX_WAITING_EXPORTS),
    ):
        parser.add_argument(
            f"--max-active-{resource}", type=int, default=active,
            help=f"maximum simultaneous {resource} across chats (default: %(default)s)",
        )
        parser.add_argument(
            f"--max-waiting-{resource}", type=int, default=waiting,
            help=f"maximum queued {resource}; zero rejects excess work (default: %(default)s)",
        )


def create_parser() -> argparse.ArgumentParser:
    """Create the Gaia command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser(
        QUERY_COMMAND,
        help="query PostgreSQL through Gaia's schema and QueryPlan tools",
        description=(
            "Start an interactive biodiversity chat using an already-running "
            "OpenAI-compatible model and loaded PostgreSQL tables. Gaia asks "
            "for a username and writes each interaction trace to its chat "
            "folder under logs/. Type /exit or /quit to end the chat."
        ),
    )
    query_parser.add_argument(
        "query",
        nargs="?",
        help=(
            "optional first biodiversity request; when omitted Gaia prompts "
            "for it interactively"
        ),
    )
    _add_agent_arguments(query_parser)

    serve_parser = subparsers.add_parser(
        SERVE_COMMAND,
        help="serve Gaia's local JSON chat bridge",
        description=(
            "Serve Gaia's JSON and NDJSON-progress chat API on loopback for "
            "the approved web interface. Active conversation context stays "
            "in server memory, while each "
            "interaction writes a repository-local Gaia trace."
        ),
    )
    _add_agent_arguments(serve_parser)
    _add_serve_arguments(serve_parser)
    return parser


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-active-chats", type=int, default=DEFAULT_MAX_ACTIVE_CHATS,
        help="maximum simultaneous chat interactions (default: %(default)s)",
    )
    parser.add_argument(
        "--max-waiting-chats", type=int, default=DEFAULT_MAX_WAITING_CHATS,
        help="maximum queued chat interactions (default: %(default)s)",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_GAIA_HTTP_HOST,
        help="HTTP bind host; keep this on loopback (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_GAIA_HTTP_PORT,
        help="HTTP bind port (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        dest="allowed_origins",
        metavar="ORIGIN",
        help=(
            "exact browser origin allowed by CORS; repeat to replace the "
            "default or GAIA_HTTP_ALLOWED_ORIGINS allowlist"
        ),
    )
    parser.add_argument(
        "--trace-directory",
        type=Path,
        default=DEFAULT_LOG_DIRECTORY,
        help="directory for per-conversation web traces (default: %(default)s)",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a Gaia command and return its process exit status."""
    arguments = create_parser().parse_args(argv)
    # Load only the selected command so query remains independent of web transport.
    command = import_module(COMMAND_MODULES[arguments.command])
    return command.run(arguments, repository_root=REPOSITORY_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
