"""Verify the retrieval cap across concurrent calls and separate turns."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import src.gaia_agent.tools.query.budget as budget_module
from src.database.darwin_core import load_simple_darwin_core_profile
from src.database.datasets import DATASETS, write_dataset_catalog
from src.database.schema_profiles import write_dataset_schema_profiles
from src.gaia_agent.runtime.contracts import ToolExecutionError, ModelTurn, ToolCall
from src.gaia_agent.tools.concurrency import WorkloadBusyError, WorkloadLimiter
from src.gaia_agent.tools.query.budget import ChatQueryBudget
from src.gaia_agent.tools.query.execution import QueryExecutionError
from src.gaia_agent.tools.query.plan import ValidatedQueryPlan
from src.gaia_agent.runtime.service import build_gaia_agent
from tests.gaia_agent.support import RecordingClient, canadian_bird_plan, query_result


OCCURRENCE_HEADERS = load_simple_darwin_core_profile().headers


def _plan(limit: int) -> ValidatedQueryPlan:
    return ValidatedQueryPlan(
        ("birds",), (), (), (), (), limit,
        column_mode="default", dataset_return_columns=(),
    )


def _rows(count: int):
    return tuple({"dataset_key": "birds", "occurrenceID": str(i)} for i in range(count))


class BudgetExecutor:
    """Record reservations and control the first query's completion."""

    def __init__(
        self, *, short_first: int | None = None,
        first_error: Exception | None = None,
        limiter: WorkloadLimiter | None = None,
    ) -> None:
        self.short_first = short_first
        self.first_error = first_error
        self.limiter = limiter
        self.limits: list[int] = []

    async def execute(self, plan: ValidatedQueryPlan, **kwargs):
        self.limits.append(plan.limit)
        if self.limiter is not None:
            assert kwargs["database_limiter"] is self.limiter
        if len(self.limits) == 1 and self.first_error is not None:
            raise self.first_error
        await asyncio.sleep(0)
        count = self.short_first if len(self.limits) == 1 else None
        return query_result(
            records=_rows(plan.limit if count is None else count), limit=plan.limit,
        )


async def _check_short_results_refund() -> None:
    budget = ChatQueryBudget()
    results = await asyncio.gather(*(
        budget.execute(_plan(200), table_registry={}, engine_factory=lambda: None)
        for _ in range(2)
    ))
    assert [len(result.records) for result in results] == [75, 125]
    with pytest.raises(ToolExecutionError) as error:
        await budget.execute(_plan(1), table_registry={}, engine_factory=lambda: None)
    assert error.value.code == "chat_record_limit_reached"


def test_concurrent_queries_share_200_and_refund_short_results(monkeypatch):
    executor = BudgetExecutor(short_first=75)
    monkeypatch.setattr(budget_module, "execute_query_plan", executor.execute)
    asyncio.run(_check_short_results_refund())
    assert executor.limits == [200, 125]


async def _check_failed_reservation_refund(
    expected_error: type[Exception], limiter: WorkloadLimiter | None = None,
) -> None:
    budget = ChatQueryBudget()
    with pytest.raises(expected_error):
        await budget.execute(
            _plan(200), table_registry={}, engine_factory=lambda: None,
            database_limiter=limiter,
        )
    result = await budget.execute(
        _plan(200), table_registry={}, engine_factory=lambda: None,
        database_limiter=limiter,
    )
    assert len(result.records) == 200


def test_failed_query_releases_reserved_records(monkeypatch):
    executor = BudgetExecutor(first_error=QueryExecutionError("query_unavailable", "Unavailable"))
    monkeypatch.setattr(budget_module, "execute_query_plan", executor.execute)
    asyncio.run(_check_failed_reservation_refund(QueryExecutionError))


def test_database_busy_error_refunds_rows_and_preserves_shared_limiter(monkeypatch):
    limiter = WorkloadLimiter(1, 0, label="database")
    executor = BudgetExecutor(first_error=WorkloadBusyError("database busy"), limiter=limiter)
    monkeypatch.setattr(budget_module, "execute_query_plan", executor.execute)
    asyncio.run(_check_failed_reservation_refund(WorkloadBusyError, limiter))


async def _check_turn_budgets(agent) -> None:
    first = await agent.run("Show records")
    second = await agent.run("Show records again", history=first.transcript)
    assert [len(result.result["records"]) for result in first.tool_results] == [150, 50]
    assert len(second.tool_results[0].result["records"]) == 200


def test_agent_turns_have_independent_budgets(monkeypatch, tmp_path: Path):
    executor = BudgetExecutor()
    monkeypatch.setattr(budget_module, "execute_query_plan", executor.execute)
    client = RecordingClient((
        ModelTurn(tool_calls=(
            ToolCall("first", "query_occurrences", canadian_bird_plan((), limit=150)),
            ToolCall("second", "query_occurrences", canadian_bird_plan((), limit=150)),
        )),
        ModelTurn(content="First 200 records."),
        ModelTurn(tool_calls=(
            ToolCall("next", "query_occurrences", canadian_bird_plan((), limit=200)),
        )),
        ModelTurn(content="Another 200 records."),
    ))
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Use the supplied tools.")
    write_dataset_catalog(tmp_path, DATASETS)
    write_dataset_schema_profiles(
        tmp_path,
        {
            dataset.key: {header: 1 for header in OCCURRENCE_HEADERS}
            for dataset in DATASETS
        },
        datasets=DATASETS,
    )
    agent = build_gaia_agent(tmp_path, client, system_prompt_path=prompt)
    asyncio.run(_check_turn_budgets(agent))
