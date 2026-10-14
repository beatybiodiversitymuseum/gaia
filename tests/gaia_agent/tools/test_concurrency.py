"""Verify bounded admission across cancellation races and request event loops."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import pytest

from src.gaia_agent.tools.concurrency import WorkloadBusyError, WorkloadLimiter


async def _hold_slot(
    limiter: WorkloadLimiter, entered: asyncio.Event, finish: asyncio.Event,
) -> None:
    async with limiter.slot():
        entered.set()
        await finish.wait()


async def _record_admission(
    limiter: WorkloadLimiter, order: list, value: object,
) -> None:
    async with limiter.slot():
        order.append(value)
        await asyncio.sleep(0)


async def _signal_admission(limiter: WorkloadLimiter, entered: asyncio.Event) -> None:
    async with limiter.slot():
        entered.set()


async def _count_ticks() -> int:
    ticks = 0
    for _ in range(10):
        await asyncio.sleep(0)
        ticks += 1
    return ticks


@pytest.mark.parametrize("max_active", (0, -1, True, False, 1.5, "2", None))
def test_active_capacity_requires_a_positive_integer(max_active: object) -> None:
    with pytest.raises(ValueError, match="max_active"):
        WorkloadLimiter(max_active, 1, label="database")


@pytest.mark.parametrize("max_waiting", (-1, True, False, 1.5, "2", None))
def test_waiting_capacity_requires_a_nonnegative_integer(max_waiting: object) -> None:
    with pytest.raises(ValueError, match="max_waiting"):
        WorkloadLimiter(1, max_waiting, label="database")


@pytest.mark.parametrize("label", ("", "  ", None, 1))
def test_workload_label_must_be_nonempty(label: object) -> None:
    with pytest.raises(ValueError, match="label"):
        WorkloadLimiter(1, 0, label=label)


async def _check_active_and_waiting_capacities() -> None:
    limiter = WorkloadLimiter(2, 1, label="database")
    entered = asyncio.Event()
    finish = asyncio.Event()
    async with limiter.slot():
        async with limiter.slot():
            queued = asyncio.create_task(_hold_slot(limiter, entered, finish))
            await asyncio.sleep(0)
            assert not entered.is_set()
            with pytest.raises(WorkloadBusyError, match="Try again shortly"):
                async with limiter.slot():
                    pytest.fail("Excess work was admitted.")
        await asyncio.wait_for(entered.wait(), timeout=1)
        finish.set()
        await queued
    async with limiter.slot():
        async with limiter.slot():
            pass


def test_active_and_waiting_capacities_reject_excess_work() -> None:
    asyncio.run(_check_active_and_waiting_capacities())


async def _check_zero_waiting_capacity() -> None:
    limiter = WorkloadLimiter(1, 0, label="export")
    async with limiter.slot():
        with pytest.raises(WorkloadBusyError, match="export"):
            async with limiter.slot():
                pytest.fail("Excess work was admitted.")
    async with limiter.slot():
        pass


def test_zero_waiting_capacity_rejects_without_waiting() -> None:
    asyncio.run(_check_zero_waiting_capacity())


async def _check_fifo_admission() -> None:
    limiter = WorkloadLimiter(1, 5, label="database")
    order = []
    async with limiter.slot():
        tasks = []
        for index in range(5):
            tasks.append(asyncio.create_task(_record_admission(limiter, order, index)))
            await asyncio.sleep(0)
        assert order == []
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)
    assert order == list(range(5))


def test_waiters_enter_in_fifo_order() -> None:
    asyncio.run(_check_fifo_admission())


async def _check_waiting_responsiveness() -> None:
    limiter = WorkloadLimiter(1, 1, label="database")
    entered = asyncio.Event()
    async with limiter.slot():
        waiter = asyncio.create_task(_signal_admission(limiter, entered))
        assert await asyncio.wait_for(_count_ticks(), timeout=1) == 10
        assert not entered.is_set()
    await asyncio.wait_for(waiter, timeout=1)
    assert entered.is_set()


def test_waiting_leaves_the_event_loop_responsive() -> None:
    asyncio.run(_check_waiting_responsiveness())


async def _check_cancelled_waiter() -> None:
    limiter = WorkloadLimiter(1, 1, label="database")
    entered = []
    async with limiter.slot():
        cancelled = asyncio.create_task(_record_admission(limiter, entered, "cancelled"))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        replacement = asyncio.create_task(_record_admission(limiter, entered, "replacement"))
        await asyncio.sleep(0)
        assert not replacement.done()
    await asyncio.wait_for(replacement, timeout=1)
    assert entered == ["replacement"]


def test_cancelled_waiter_frees_its_queue_place() -> None:
    asyncio.run(_check_cancelled_waiter())


async def _check_cancelled_grant(after_wakeup: bool) -> None:
    limiter = WorkloadLimiter(1, 2, label="database")
    entered = []
    async with limiter.slot():
        cancelled = asyncio.create_task(_record_admission(limiter, entered, "cancelled"))
        replacement = asyncio.create_task(_record_admission(limiter, entered, "replacement"))
        await asyncio.sleep(0)
    if after_wakeup:
        # Admission's wake callback runs first, cancellation runs before the
        # task scheduled by that callback can resume its context manager.
        asyncio.get_running_loop().call_soon(cancelled.cancel)
    else:
        cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    await asyncio.wait_for(replacement, timeout=1)
    assert entered == ["replacement"]
    async with limiter.slot():
        pass


@pytest.mark.parametrize("after_wakeup", (False, True))
def test_cancelled_grant_is_reassigned_before_the_task_resumes(
    after_wakeup: bool,
) -> None:
    asyncio.run(_check_cancelled_grant(after_wakeup))


async def _check_cancelled_owner() -> None:
    limiter = WorkloadLimiter(1, 1, label="database")
    entered = asyncio.Event()
    owner = asyncio.create_task(_hold_slot(limiter, entered, asyncio.Event()))
    await entered.wait()
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    async with limiter.slot():
        pass


def test_cancelling_an_active_owner_releases_capacity() -> None:
    asyncio.run(_check_cancelled_owner())


async def _check_failed_owner() -> None:
    limiter = WorkloadLimiter(1, 0, label="database")
    with pytest.raises(RuntimeError, match="operation failed"):
        async with limiter.slot():
            raise RuntimeError("operation failed")
    async with limiter.slot():
        pass


def test_exception_from_an_active_operation_releases_capacity() -> None:
    asyncio.run(_check_failed_owner())


class ThreadedWorkload:
    """Record one limiter's capacity across independently owned request loops."""

    def __init__(self) -> None:
        self.limiter = WorkloadLimiter(2, 6, label="database")
        self.start = Barrier(6, timeout=5)
        self.counters_lock = Lock()
        self.active = 0
        self.maximum_active = 0
        self.finished = 0

    async def run_operations(self) -> None:
        for _ in range(10):
            async with self.limiter.slot():
                with self.counters_lock:
                    self.active += 1
                    self.maximum_active = max(self.maximum_active, self.active)
                    assert self.active <= 2
                try:
                    await asyncio.sleep(0.002)
                finally:
                    with self.counters_lock:
                        self.active -= 1
                        self.finished += 1

    def request_thread(self) -> None:
        self.start.wait()
        asyncio.run(self.run_operations(), debug=True)


def test_request_threads_share_one_active_limit_across_event_loops() -> None:
    workload = ThreadedWorkload()
    with ThreadPoolExecutor(max_workers=6) as executor:
        requests = [executor.submit(workload.request_thread) for _ in range(6)]
        for request in requests:
            request.result(timeout=10)
    assert workload.active == 0
    assert workload.maximum_active == 2
    assert workload.finished == 60
