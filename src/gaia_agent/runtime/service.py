"""Compose the agent loop with Gaia's domain tools and prompt."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from src.logger import normalize_chat_username
from src.logger.gaia_logger import (
    GaiaLoggerError,
    GaiaLoggerConfig,
    build_trace_record,
    write_gaia_trace,
)

from src.gaia_agent.runtime.contracts import (
    AgentCancellation,
    AgentProgressHandler,
    AgentResult,
    ChatMessage,
    ModelClient,
)
from src.gaia_agent.runtime.loop import AgentRuntime
from src.gaia_agent.runtime.context import history_transaction
from src.gaia_agent.runtime.disclosure import ResponseDisclosureGuard
from src.gaia_agent.runtime.defaults import (
    DEFAULT_MAX_ACTIVE_EXPORTS,
    DEFAULT_MAX_ACTIVE_QUERIES,
    DEFAULT_MAX_WAITING_EXPORTS,
    DEFAULT_MAX_WAITING_QUERIES,
)
from src.gaia_agent.tools import GaiaToolset
from src.gaia_agent.tools.concurrency import WorkloadLimiter
from src.gaia_agent.tools.file_generation.workers import ExcelExportWorkers


class GaiaAgent:
    """Run natural-language turns through Gaia's tool-using runtime."""

    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        system_prompt: str,
        trace_config: GaiaLoggerConfig | None = None,
        runtime_factory: Callable[
            [AgentCancellation | None, Path | None, str], AgentRuntime
        ] | None = None,
    ) -> None:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system prompt must be a non-empty string")
        self._runtime = runtime
        self._system_prompt = system_prompt
        self._trace_config = trace_config
        self._runtime_factory = runtime_factory

    async def run(
        self,
        user_query: str,
        *,
        history: Iterable[ChatMessage] = (),
        trace_config: GaiaLoggerConfig | None = None,
        progress_callback: AgentProgressHandler | None = None,
        cancellation: AgentCancellation | None = None,
    ) -> AgentResult:
        """Run one query, optionally continuing a completed conversation."""
        if not isinstance(user_query, str) or not user_query.strip():
            raise ValueError("user query must be a non-empty string")
        active_trace_config = (
            self._trace_config if trace_config is None else trace_config
        )
        if active_trace_config is not None and not isinstance(
            active_trace_config, GaiaLoggerConfig,
        ):
            raise TypeError("trace config must be a GaiaLoggerConfig value")
        export_directory = (
            active_trace_config.path.parent / "exports"
            if active_trace_config is not None else None
        )
        username = (
            normalize_chat_username(active_trace_config.invocation.get("username", "gaia"))
            if active_trace_config is not None else "gaia"
        )
        runtime = (
            self._runtime_factory(cancellation, export_directory, username)
            if self._runtime_factory is not None
            else self._runtime
        )
        history_messages = tuple(history)
        self._validate_history(history_messages)
        initial_messages = (
            history_messages
            or (ChatMessage(role="system", content=self._system_prompt),)
        ) + (ChatMessage(role="user", content=user_query),)

        if active_trace_config is None:
            return await runtime.run(
                initial_messages,
                progress_callback=progress_callback,
                cancellation=cancellation,
                history_path=None,
            )
        with history_transaction(active_trace_config.path.parent / "history.md"):
            started_at = _utc_timestamp()
            started = time.perf_counter()
            trace_arguments = {
                "run_id": active_trace_config.run_id,
                "started_at": started_at,
                "invocation": active_trace_config.invocation,
                "initial_messages": initial_messages,
                "registered_tool_count": runtime.registered_tool_count,
                "history_message_count": len(history_messages),
            }
            try:
                result = await runtime.run(
                    initial_messages,
                    progress_callback=progress_callback,
                    cancellation=cancellation,
                    history_path=active_trace_config.path.parent / "history.md",
                )
            except Exception as error:
                record = build_trace_record(
                    error,
                    ended_at=_utc_timestamp(),
                    elapsed_seconds=time.perf_counter() - started,
                    **trace_arguments,
                )
                try:
                    write_gaia_trace(active_trace_config, record)
                except Exception as trace_error:
                    raise GaiaLoggerError(
                        "agent run stopped and its trace could not be persisted: "
                        f"{type(trace_error).__name__}"
                    ) from error
                raise

            record = build_trace_record(
                result,
                ended_at=_utc_timestamp(),
                elapsed_seconds=time.perf_counter() - started,
                **trace_arguments,
            )
            write_gaia_trace(active_trace_config, record)
            return result

    def _validate_history(
        self,
        history: tuple[ChatMessage, ...],
    ) -> None:
        """Accept only a completed transcript created with this system prompt."""
        if any(not isinstance(message, ChatMessage) for message in history):
            raise TypeError("conversation history must contain ChatMessage values")
        if not history:
            return
        if (
            history[0].role != "system"
            or history[0].content != self._system_prompt
        ):
            raise ValueError(
                "conversation history must begin with this agent's system prompt"
            )
        if any(message.role == "system" for message in history[1:]):
            raise ValueError(
                "conversation history cannot contain another system message"
            )
        final_message = history[-1]
        if (
            final_message.role != "assistant"
            or final_message.tool_calls
            or final_message.content is None
            or not final_message.content.strip()
        ):
            raise ValueError(
                "conversation history must end with a completed assistant response"
            )


def build_gaia_agent(
    repository_root: Path,
    client: ModelClient,
    *,
    system_prompt_path: Path | None = None,
    trace_config: GaiaLoggerConfig | None = None,
    max_active_queries: int = DEFAULT_MAX_ACTIVE_QUERIES,
    max_waiting_queries: int = DEFAULT_MAX_WAITING_QUERIES,
    max_active_exports: int = DEFAULT_MAX_ACTIVE_EXPORTS,
    max_waiting_exports: int = DEFAULT_MAX_WAITING_EXPORTS,
) -> GaiaAgent:
    """Load Gaia's tools and system prompt into one agent."""
    root = Path(repository_root)
    prompt_path = system_prompt_path or root / "src/gaia_agent/runtime/system_prompt.md"
    system_prompt = prompt_path.read_text(encoding="utf-8")
    toolset = GaiaToolset.load(
        repository_root=root,
        database_limiter=WorkloadLimiter(
            max_active_queries, max_waiting_queries, label="Queries",
        ),
        export_workers=ExcelExportWorkers(
            max_workers=max_active_exports, max_waiting=max_waiting_exports,
        ),
    )

    make_runtime = partial(
        _make_runtime,
        root=root,
        client=client,
        toolset=toolset,
        system_prompt=system_prompt,
    )

    return GaiaAgent(
        make_runtime(None, None, "gaia"),
        system_prompt=system_prompt,
        trace_config=trace_config,
        runtime_factory=make_runtime,
    )


def _make_runtime(
    cancellation: AgentCancellation | None,
    export_directory: Path | None,
    username: str,
    *,
    root: Path,
    client: ModelClient,
    toolset: GaiaToolset,
    system_prompt: str,
) -> AgentRuntime:
    """Bind one interaction's tools and disclosure guard to its session."""
    registered_tools = toolset.registered_tools(
        cancellation=cancellation,
        export_directory=export_directory,
        username=username,
    )
    private_locations = [str(root.resolve())]
    if export_directory is not None:
        private_locations.append(str(export_directory.resolve().parent))
    return AgentRuntime(
        client,
        registered_tools,
        response_guard=ResponseDisclosureGuard(
            system_prompt=system_prompt,
            tools=(tool.definition for tool in registered_tools),
            private_locations=private_locations,
        ),
    )


def _utc_timestamp() -> str:
    """Return a stable UTC timestamp for one trace boundary."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )
