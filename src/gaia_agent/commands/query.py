"""Interactive terminal conversation for the Gaia query command."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.gaia_agent.commands.common import agent_trace_invocation, build_command_agent
from src.gaia_agent.runtime.context import CompactionError
from src.gaia_agent.runtime.contracts import (
    AgentCancellation,
    AgentCancelledError,
    AgentProgressEvent,
    AgentRuntimeError,
    ChatMessage,
)
from src.gaia_agent.runtime.defaults import CONTEXT_COMPACTED_NOTICE
from src.gaia_agent.runtime.openai_client import OpenAIClientError
from src.gaia_agent.runtime.service import GaiaAgent
from src.gaia_agent.tools.file_generation.artifacts import collect_generated_files
from src.logger import create_chat_directory, normalize_chat_username
from src.logger.gaia_logger import GaiaLoggerError, create_chat_trace_config

QUERY_EXIT_COMMANDS = frozenset({"/exit", "/quit"})


def _report_query_progress(event: AgentProgressEvent) -> None:
    """Print context maintenance notices during a terminal request."""
    if event.kind == "context_compacting":
        print("Compacting earlier conversation…", flush=True)
    elif event.kind == "context_compacted":
        print(CONTEXT_COMPACTED_NOTICE, flush=True)


def request_query_username() -> str:
    """Request and normalize the username used in a query chat folder."""
    while True:
        value = input("Username: ").strip()
        if not value:
            print("ERROR: username is required", file=sys.stderr)
            continue
        try:
            return normalize_chat_username(value)
        except ValueError as error:
            print(f"ERROR: {error}", file=sys.stderr)


def run(arguments: argparse.Namespace, *, repository_root: Path) -> int:
    """Run a continuous, in-memory Gaia chat until the user exits."""
    try:
        username = request_query_username()
    except EOFError:
        print()
        print("ERROR: username is required", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 130

    try:
        chat_directory = create_chat_directory(repository_root / "logs", username)
        agent = build_command_agent(arguments, repository_root)
    except (OpenAIClientError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    return _run_conversation(
        agent, arguments, repository_root, chat_directory, username
    )


def _run_conversation(
    agent: GaiaAgent,
    arguments: argparse.Namespace,
    repository_root: Path,
    chat_directory: Path,
    username: str,
) -> int:
    """Read requests and retain only completed turns in the terminal history."""
    print("Gaia chat started. Type /exit or /quit to end.")
    print("Press Control-C while Gaia is working to stop that request.")
    history: tuple[ChatMessage, ...] = ()
    next_query: str | None = arguments.query
    interaction = 0
    had_error = False

    while True:
        try:
            if next_query is None:
                user_query = input("You: ")
            else:
                user_query = next_query
                next_query = None
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            return 130

        normalized_query = user_query.strip()
        if normalized_query.casefold() in QUERY_EXIT_COMMANDS:
            break
        if not normalized_query:
            continue

        interaction += 1
        cancellation = AgentCancellation()
        try:
            trace_config = create_chat_trace_config(
                chat_directory=chat_directory,
                interaction=interaction,
                repository_root=repository_root,
                invocation={
                    **agent_trace_invocation(arguments),
                    "username": username,
                },
            )
            result = asyncio.run(agent.run(
                user_query,
                history=history,
                trace_config=trace_config,
                progress_callback=_report_query_progress,
                cancellation=cancellation,
            ))
        except AgentCancelledError:
            print("Stopped thinking")
            continue
        except KeyboardInterrupt:
            cancellation.cancel()
            print("Stopped thinking")
            continue
        except (
            AgentRuntimeError,
            CompactionError,
            GaiaLoggerError,
            OpenAIClientError,
            OSError,
            TypeError,
            ValueError,
        ):
            had_error = True
            print("ERROR: Gaia could not complete this request.", file=sys.stderr)
            continue

        history = result.context
        print(f"Gaia: {result.final_response}")
        for file in collect_generated_files(
            result.tool_results, export_directory=chat_directory / "exports"
        ):
            print(f"File ready: {file.filename}")

    return 1 if had_error else 0
