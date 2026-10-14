"""Share the occurrence retrieval allowance across one interaction's queries."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import replace

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql.schema import Table

from src.gaia_agent.runtime.contracts import ToolExecutionError
from src.gaia_agent.tools.concurrency import WorkloadLimiter
from src.gaia_agent.tools.query.execution import (
    MAX_CHAT_RECORDS,
    QueryExecutionResult,
    execute_query_plan,
)
from src.gaia_agent.tools.query.plan import ValidatedQueryPlan


class ChatQueryBudget:
    """Reserve occurrence-row slots while allowing independent queries to overlap."""

    def __init__(self) -> None:
        self._remaining = MAX_CHAT_RECORDS
        self._running = 0
        self._condition = asyncio.Condition()

    async def execute(
        self,
        plan: ValidatedQueryPlan,
        *,
        table_registry: Mapping[str, Table],
        engine_factory: Callable[[], AsyncEngine],
        database_limiter: WorkloadLimiter | None = None,
    ) -> QueryExecutionResult:
        """Retrieve occurrence rows within the shared interaction allowance."""
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._remaining > 0 or self._running == 0
            )
            if self._remaining == 0:
                raise ToolExecutionError(
                    "chat_record_limit_reached",
                    f"This interaction has returned {MAX_CHAT_RECORDS} occurrence rows. "
                    "The occurrence retrieval allowance for this interaction is exhausted.",
                )
            allowance = min(plan.limit, self._remaining)
            self._remaining -= allowance
            self._running += 1

        returned = 0
        try:
            result = await execute_query_plan(
                replace(plan, limit=allowance),
                table_registry=table_registry,
                engine_factory=engine_factory,
                database_limiter=database_limiter,
            )
            returned = len(result.records)
            return result
        finally:
            async with self._condition:
                self._remaining += allowance - returned
                self._running -= 1
                self._condition.notify_all()
