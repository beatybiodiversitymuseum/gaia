"""Bound shared workloads without tying admission to one event loop."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Literal


class WorkloadBusyError(RuntimeError):
    """The workload's active and waiting capacities are both occupied."""


@dataclass(eq=False)
class _Waiter:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[None]
    state: Literal["queued", "granted", "released"] = "queued"


class WorkloadLimiter:
    """Share bounded, FIFO admission across request threads and event loops."""

    def __init__(self, max_active: int, max_waiting: int, *, label: str) -> None:
        if isinstance(max_active, bool) or not isinstance(max_active, int) or max_active < 1:
            raise ValueError("max_active must be a positive integer.")
        if isinstance(max_waiting, bool) or not isinstance(max_waiting, int) or max_waiting < 0:
            raise ValueError("max_waiting must be a nonnegative integer.")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("label must be a nonempty string.")
        self._max_active = max_active
        self._max_waiting = max_waiting
        self._label = label.strip()
        self._active = 0
        self._waiters: deque[_Waiter] = deque()
        self._lock = Lock()

    @asynccontextmanager
    async def slot(self) -> AsyncGenerator[None, None]:
        """Wait for capacity, rejecting excess queued work immediately."""
        waiter = await self._acquire()
        try:
            yield
        finally:
            self._release(waiter)

    async def _acquire(self) -> _Waiter:
        loop = asyncio.get_running_loop()
        waiter = _Waiter(loop, loop.create_future())
        with self._lock:
            if self._active < self._max_active:
                self._active += 1
                waiter.state = "granted"
                return waiter
            if len(self._waiters) >= self._max_waiting:
                raise WorkloadBusyError(
                    f"Too many {self._label} operations are running or waiting. "
                    "Try again shortly."
                )
            self._waiters.append(waiter)
        try:
            await waiter.future
            return waiter
        except BaseException:
            # Cancellation can arrive after admission but before this task resumes.
            self._release(waiter)
            raise

    def _release(self, waiter: _Waiter) -> None:
        with self._lock:
            self._release_locked(waiter)

    def _release_locked(self, waiter: _Waiter) -> None:
        if waiter.state == "released":
            return
        if waiter.state == "queued":
            self._waiters.remove(waiter)
        else:
            self._active -= 1
        waiter.state = "released"
        self._admit_waiters_locked()

    def _admit_waiters_locked(self) -> None:
        while self._waiters and self._active < self._max_active:
            waiter = self._waiters.popleft()
            waiter.state = "granted"
            self._active += 1
            try:
                waiter.loop.call_soon_threadsafe(self._wake, waiter)
            except RuntimeError:
                # A closing request loop must not retain shared capacity.
                waiter.state = "released"
                self._active -= 1

    def _wake(self, waiter: _Waiter) -> None:
        # Futures are inspected and completed only on their owning event loop.
        with self._lock:
            if waiter.state != "granted":
                return
            if waiter.future.cancelled():
                self._release_locked(waiter)
            elif not waiter.future.done():
                waiter.future.set_result(None)
