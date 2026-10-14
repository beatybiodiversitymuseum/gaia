"""Bound simultaneous exports and serialize each export's blocking file work."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import Event
from typing import TypeVar

from src.gaia_agent.runtime.contracts import AgentCancellation
from src.gaia_agent.runtime.defaults import DEFAULT_MAX_ACTIVE_EXPORTS, DEFAULT_MAX_WAITING_EXPORTS
from src.gaia_agent.tools.concurrency import WorkloadLimiter


_Result = TypeVar("_Result")


class ExcelExportWorker:
    """Own one export thread and drain its work before cancellation returns."""

    def __init__(self, cancellation: AgentCancellation | None) -> None:
        self._cancellation = cancellation
        self._cancelled = Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gaia-excel")

    def check_cancellation(self) -> None:
        """Let synchronous row processing stop without abandoning its thread."""
        if self._cancelled.is_set() or (
            self._cancellation is not None and self._cancellation.is_cancelled
        ):
            raise asyncio.CancelledError

    async def run(self, operation: Callable[..., _Result], *args: object) -> _Result:
        """Await one file operation without allowing cancellation to orphan it."""
        future = asyncio.wrap_future(self._executor.submit(operation, *args))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            self._cancelled.set()
            # Cancelling a coroutine cannot stop a running worker thread. Its
            # owner must wait before closing spools or removing workbook files.
            while not future.done():
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            if not future.cancelled():
                future.exception()
            raise

    def close(self) -> None:
        """Join the already-drained executor; no file operation remains active."""
        self._executor.shutdown(wait=True)


class ExcelExportWorkers:
    """Share bounded export admission across chats and independent event loops."""

    def __init__(
        self,
        *,
        max_workers: int = DEFAULT_MAX_ACTIVE_EXPORTS,
        max_waiting: int = DEFAULT_MAX_WAITING_EXPORTS,
    ) -> None:
        self._limiter = WorkloadLimiter(
            max_active=max_workers, max_waiting=max_waiting, label="Excel export",
        )

    @asynccontextmanager
    async def slot(
        self, cancellation: AgentCancellation | None = None,
    ) -> AsyncGenerator[ExcelExportWorker, None]:
        """Allocate one serial worker only after bounded admission succeeds."""
        async with self._limiter.slot():
            worker = ExcelExportWorker(cancellation)
            try:
                worker.check_cancellation()
                yield worker
            finally:
                worker.close()
