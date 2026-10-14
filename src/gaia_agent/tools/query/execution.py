"""Compile and execute validated occurrence plans with query-owned connections."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from sqlalchemy import Numeric, Select, and_, case, cast, func, select
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.schema import Column, Table

from src.database.database import DatabaseConfigurationError, load_database_url

from src.gaia_agent.runtime.contracts import AgentCancellation
from src.gaia_agent.tools.concurrency import WorkloadLimiter
from src.gaia_agent.tools.query.columns import preferred_identifier
from src.gaia_agent.tools.query.plan import (
    MAX_NUMERIC_TEXT_LENGTH,
    NUMERIC_TEXT_PATTERN,
    PlannedFilter,
    PlannedOrder,
    ValidatedQueryPlan,
    query_plan_for_dataset,
)

MAX_CHAT_RECORDS = 200
QUERY_UNAVAILABLE_MESSAGE = (
    "The biodiversity query service is unavailable. Please try again later."
)


# ---------------------------------------------------------------------------
# Execution contracts
# ---------------------------------------------------------------------------


class QueryExecutionError(RuntimeError):
    """Describe an occurrence-query failure with a safe message and error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class DatasetSkip:
    """Describe why one selected table could not execute the QueryPlan."""

    dataset_key: str
    reason: str
    missing_headers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetSelect:
    """Bind one safe SELECT to the exact output headers it returns."""

    statement: Select[tuple[object, ...]]
    output_headers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QueryExecutionResult:
    """Keep internal flat rows and metadata until Gaia projects an observation."""

    records: tuple[dict[str, str], ...]
    selected_dataset_keys: tuple[str, ...]
    compatible_dataset_keys: tuple[str, ...]
    queried_dataset_keys: tuple[str, ...]
    skipped_datasets: tuple[DatasetSkip, ...]
    return_columns: tuple[str, ...]
    limit: int
    database_truncated: bool


@dataclass(frozen=True, slots=True)
class ExportQueryResult:
    """Summarize streamed datasets without retaining their occurrence rows."""

    selected_dataset_keys: tuple[str, ...]
    compatible_dataset_keys: tuple[str, ...]
    queried_dataset_keys: tuple[str, ...]
    skipped_datasets: tuple[DatasetSkip, ...]
    dataset_row_counts: tuple[tuple[str, int], ...]
    limit: int | None
    database_truncated: bool

    @property
    def record_count(self) -> int:
        return sum(count for _key, count in self.dataset_row_counts)


# ---------------------------------------------------------------------------
# Database lifecycle
# ---------------------------------------------------------------------------


def create_query_engine() -> AsyncEngine:
    """Create a query-owned async engine without pooling across event loops."""
    return create_async_engine(
        load_database_url().set(drivername="postgresql+psycopg"),
        poolclass=NullPool,
    )


@asynccontextmanager
async def managed_query_engine(
    engine_factory: Callable[[], AsyncEngine],
    *,
    database_limiter: WorkloadLimiter | None = None,
) -> AsyncGenerator[AsyncEngine, None]:
    """Hold database admission until the query-owned engine is disposed."""
    admission = database_limiter.slot() if database_limiter is not None else nullcontext()
    async with admission:
        try:
            engine = engine_factory()
        except (DatabaseConfigurationError, SQLAlchemyError) as error:
            raise QueryExecutionError(
                "query_unavailable", QUERY_UNAVAILABLE_MESSAGE,
            ) from error
        try:
            if engine.dialect.name != "postgresql":
                raise QueryExecutionError(
                    "query_unavailable", QUERY_UNAVAILABLE_MESSAGE,
                )
            yield engine
        finally:
            try:
                await engine.dispose()
            except SQLAlchemyError as error:
                raise QueryExecutionError(
                    "query_unavailable", QUERY_UNAVAILABLE_MESSAGE,
                ) from error


async def execute_query_plan(
    plan: ValidatedQueryPlan,
    *,
    table_registry: Mapping[str, Table],
    engine_factory: Callable[[], AsyncEngine] = create_query_engine,
    database_limiter: WorkloadLimiter | None = None,
) -> QueryExecutionResult:
    """Execute a validated plan and dispose its engine on every exit path."""
    async with managed_query_engine(
        engine_factory, database_limiter=database_limiter,
    ) as engine:
        try:
            async with engine.connect() as connection:
                return await connection.run_sync(
                    _execute_on_connection,
                    plan,
                    table_registry=table_registry,
                )
        except SQLAlchemyError as error:
            raise QueryExecutionError(
                "query_unavailable", QUERY_UNAVAILABLE_MESSAGE,
            ) from error
        except ValueError as error:
            raise QueryExecutionError(
                "invalid_query_data",
                "Occurrence query data failed validation.",
            ) from error


async def stream_query_plans(
    plans: Sequence[ValidatedQueryPlan],
    *,
    table_registry: Mapping[str, Table],
    on_batch: Callable[
        [str, tuple[str, ...], tuple[tuple[str, ...], ...]], Awaitable[None] | None,
    ],
    row_limit: int | None = None,
    limit_scope: Literal["total", "per_dataset"] = "total",
    batch_size: int = 1000,
    engine_factory: Callable[[], AsyncEngine] = create_query_engine,
    database_limiter: WorkloadLimiter | None = None,
    cancellation: AgentCancellation | None = None,
    on_dataset_complete: Callable[[str], Awaitable[None] | None] | None = None,
) -> ExportQueryResult:
    """Stream export rows from one consistent, read-only PostgreSQL snapshot.

    Plan limits apply to chat only. ``limit_scope`` applies ``row_limit`` across
    all datasets or independently to each one; ``None`` exports every matching
    row. Each compatible dataset receives an empty initialization batch, bounded
    row batches, then a completion callback after its cursor closes. Datasets
    after a shared limit receive empty callbacks. Batch and completion callbacks
    may be synchronous or awaitable; each finishes before the next cursor read.
    """
    if row_limit is not None and (type(row_limit) is not int or row_limit < 1):
        raise ValueError("row_limit must be a positive integer or None")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if limit_scope not in ("total", "per_dataset"):
        raise ValueError("limit_scope must be 'total' or 'per_dataset'")
    _check_export_cancellation(cancellation)
    async with managed_query_engine(
        engine_factory, database_limiter=database_limiter,
    ) as engine:
        try:
            async with engine.connect() as connection:
                _check_export_cancellation(cancellation)
                await connection.execution_options(
                    isolation_level="REPEATABLE READ",
                    postgresql_readonly=True,
                )
                _check_export_cancellation(cancellation)
                async with connection.begin():
                    return await _stream_on_connection(
                        connection,
                        plans,
                        table_registry=table_registry,
                        on_batch=on_batch,
                        on_dataset_complete=on_dataset_complete,
                        row_limit=row_limit,
                        limit_scope=limit_scope,
                        batch_size=batch_size,
                        cancellation=cancellation,
                    )
        except SQLAlchemyError as error:
            raise QueryExecutionError(
                "query_unavailable", QUERY_UNAVAILABLE_MESSAGE,
            ) from error


# ---------------------------------------------------------------------------
# SQLAlchemy compilation and execution
# ---------------------------------------------------------------------------


def exact_table_columns(table: Table) -> dict[str, Column[object]]:
    """Resolve physical source names without accepting ORM keys as aliases."""
    return {column.name: column for column in table.columns}


def numeric_value(column: ColumnElement[object]) -> ColumnElement[object]:
    """Convert only bounded plain decimal source text, leaving other text NULL."""
    trimmed = func.trim(column)
    valid = and_(
        func.length(trimmed) <= MAX_NUMERIC_TEXT_LENGTH,
        trimmed.regexp_match(NUMERIC_TEXT_PATTERN),
    )
    # CASE guards the cast itself: SQL predicate evaluation order is unspecified.
    return case((valid, cast(trimmed, Numeric())))


def build_filter_predicates(
    columns: Mapping[str, ColumnElement[object]],
    filters: tuple[PlannedFilter, ...],
) -> tuple[ColumnElement[bool], ...]:
    """Compile parameterized literal predicates shared by all occurrence tools."""
    predicates: list[ColumnElement[bool]] = []
    for planned_filter in filters:
        column = columns[planned_filter.header]
        if planned_filter.exact_any:
            predicates.append(column.in_(planned_filter.exact_any))
        if planned_filter.exclude_any:
            predicates.append(column.not_in(planned_filter.exclude_any))
        if planned_filter.contains is not None:
            predicates.append(column.contains(planned_filter.contains, autoescape=True))
        if planned_filter.starts_with is not None:
            predicates.append(column.startswith(planned_filter.starts_with, autoescape=True))
        if planned_filter.is_empty is not None:
            predicates.append(column == "" if planned_filter.is_empty else column != "")
        comparable = numeric_value(column) if planned_filter.value_type == "numeric" else column
        for comparison in planned_filter.comparisons:
            literal = (
                Decimal(comparison.value)
                if planned_filter.value_type == "numeric"
                else comparison.value
            )
            if comparison.operator == "gt":
                predicates.append(comparable > literal)
            elif comparison.operator == "gte":
                predicates.append(comparable >= literal)
            elif comparison.operator == "lt":
                predicates.append(comparable < literal)
            else:
                predicates.append(comparable <= literal)
    return tuple(predicates)


def _ordered_column(
    column: ColumnElement[object], ordering: PlannedOrder,
) -> ColumnElement[object]:
    if ordering.value_type == "numeric":
        column = numeric_value(column)
    expression = column.asc() if ordering.direction == "asc" else column.desc()
    return expression.nulls_last() if ordering.value_type == "numeric" else expression


def build_dataset_select(
    table: Table,
    plan: ValidatedQueryPlan,
    *,
    fetch_limit: int | None,
) -> DatasetSelect:
    """Compile one parameterized SELECT from prevalidated identifiers only."""
    columns = exact_table_columns(table)
    identifier = preferred_identifier(columns)
    identifier_column = columns[identifier]

    output_headers = tuple(
        dict.fromkeys(
            (
                identifier,
                *(
                    header for header in plan.return_columns
                    if header in columns and (header != "id" or identifier == "id")
                ),
            )
        )
    )

    predicates = list(build_filter_predicates(columns, plan.filters))
    for header in plan.distinct_by:
        predicates.append(func.trim(columns[header]) != "")

    if plan.distinct_by:
        auxiliary_headers = tuple(
            header
            for header in dict.fromkeys(
                (*plan.distinct_by, *(ordering.header for ordering in plan.order_by))
            )
            if header not in output_headers
        )
        selected_headers = (*output_headers, *auxiliary_headers)
        aliases = {
            header: f"_gaia_column_{index}"
            for index, header in enumerate(selected_headers)
        }
        ranked_columns = tuple(
            columns[header].label(aliases[header]) for header in selected_headers
        )
        representative_order: list[ColumnElement[object]] = [
            _ordered_column(columns[ordering.header], ordering)
            for ordering in plan.order_by
        ]
        if identifier not in {
            ordering.header for ordering in plan.order_by
        }:
            representative_order.append(identifier_column.asc())
        rank_label = "_gaia_distinct_rank"
        representative_rank = func.row_number().over(
            partition_by=tuple(columns[header] for header in plan.distinct_by),
            order_by=tuple(representative_order),
        ).label(rank_label)
        ranked_statement = select(*ranked_columns, representative_rank)
        if predicates:
            ranked_statement = ranked_statement.where(and_(*predicates))
        ranked = ranked_statement.subquery()
        ranked_by_header = {
            header: ranked.c[aliases[header]] for header in selected_headers
        }
        output_columns = tuple(
            ranked_by_header[header] for header in output_headers
        )
        statement = select(*output_columns).where(ranked.c[rank_label] == 1)

        ordering_columns = ranked_by_header
    else:
        output_columns = tuple(columns[header] for header in output_headers)
        statement = select(*output_columns)
        if predicates:
            statement = statement.where(and_(*predicates))
        ordering_columns = columns

    ordered_headers: set[str] = set()
    for ordering in plan.order_by:
        column = ordering_columns[ordering.header]
        statement = statement.order_by(_ordered_column(column, ordering))
        ordered_headers.add(ordering.header)
    for header in plan.distinct_by:
        if header not in ordered_headers:
            statement = statement.order_by(ordering_columns[header].asc())
            ordered_headers.add(header)
    if identifier not in ordered_headers:
        statement = statement.order_by(ordering_columns[identifier].asc())

    if fetch_limit is not None:
        statement = statement.limit(fetch_limit)
    return DatasetSelect(statement, output_headers)


def _execute_on_connection(
    connection: Connection,
    plan: ValidatedQueryPlan,
    *,
    table_registry: Mapping[str, Table],
) -> QueryExecutionResult:
    """Query compatible tables in registry order with one bounded chat limit."""
    compatible, skipped = resolve_query_datasets(
        plan, table_registry=table_registry
    )
    limit = min(plan.limit, MAX_CHAT_RECORDS)
    records: list[dict[str, str]] = []
    queried_dataset_keys: list[str] = []
    for dataset_key, table in compatible:
        remaining_probe = limit + 1 - len(records)
        dataset_select = build_dataset_select(
            table,
            query_plan_for_dataset(plan, dataset_key),
            fetch_limit=remaining_probe,
        )
        queried_dataset_keys.append(dataset_key)
        rows = connection.execute(dataset_select.statement).fetchall()
        for row in rows:
            record: dict[str, str] = {"dataset_key": dataset_key}
            for header, value in zip(
                dataset_select.output_headers,
                row,
                strict=True,
            ):
                if not isinstance(value, str):
                    raise ValueError(
                        f"{table.name}.{header}: expected a TEXT string result"
                    )
                record[header] = value
            records.append(record)
        if len(records) == limit + 1:
            break

    database_truncated = len(records) > limit
    return QueryExecutionResult(
        records=tuple(records[:limit]),
        selected_dataset_keys=plan.dataset_keys,
        compatible_dataset_keys=tuple(key for key, _table in compatible),
        queried_dataset_keys=tuple(queried_dataset_keys),
        skipped_datasets=tuple(skipped),
        return_columns=plan.return_columns,
        limit=limit,
        database_truncated=database_truncated,
    )


def resolve_query_datasets(
    plan: ValidatedQueryPlan,
    *,
    table_registry: Mapping[str, Table],
) -> tuple[tuple[tuple[str, Table], ...], tuple[DatasetSkip, ...]]:
    """Resolve compatible physical tables and explain skipped dataset scopes."""
    compatible: list[tuple[str, Table]] = []
    skipped: list[DatasetSkip] = []
    required_header_groups = (
        ("missing_filter_columns", tuple(item.header for item in plan.filters)),
        ("missing_order_columns", tuple(item.header for item in plan.order_by)),
        ("missing_distinct_columns", plan.distinct_by),
    )
    for dataset_key in plan.dataset_keys:
        table = table_registry[dataset_key]
        physical_headers = frozenset(exact_table_columns(table))
        for reason, required_headers in required_header_groups:
            missing_headers = tuple(
                header
                for header in required_headers
                if header not in physical_headers
            )
            if missing_headers:
                skipped.append(DatasetSkip(dataset_key, reason, missing_headers))
                break
        else:
            compatible.append((dataset_key, table))

    return tuple(compatible), tuple(skipped)


# ---------------------------------------------------------------------------
# Export streaming
# ---------------------------------------------------------------------------


async def _stream_on_connection(
    connection: AsyncConnection,
    plans: Sequence[ValidatedQueryPlan],
    *,
    table_registry: Mapping[str, Table],
    on_batch: Callable[
        [str, tuple[str, ...], tuple[tuple[str, ...], ...]], Awaitable[None] | None,
    ],
    on_dataset_complete: Callable[[str], Awaitable[None] | None] | None,
    row_limit: int | None,
    limit_scope: Literal["total", "per_dataset"],
    batch_size: int,
    cancellation: AgentCancellation | None,
) -> ExportQueryResult:
    selected_dataset_keys: list[str] = []
    jobs: list[tuple[str, Table, ValidatedQueryPlan]] = []
    skipped_datasets: list[DatasetSkip] = []
    for plan in plans:
        selected_dataset_keys.extend(plan.dataset_keys)
        compatible, skipped = resolve_query_datasets(
            plan, table_registry=table_registry,
        )
        jobs.extend(
            (key, table, query_plan_for_dataset(plan, key))
            for key, table in compatible
        )
        skipped_datasets.extend(skipped)

    remaining = row_limit
    queried_dataset_keys: list[str] = []
    dataset_row_counts: list[tuple[str, int]] = []
    database_truncated = False
    for dataset_key, table, plan in jobs:
        _check_export_cancellation(cancellation)
        if limit_scope == "per_dataset":
            remaining = row_limit
        dataset_select = build_dataset_select(
            table,
            plan,
            fetch_limit=None if remaining is None else remaining + 1,
        )
        headers = dataset_select.output_headers
        initialization = on_batch(dataset_key, headers, ())
        if inspect.isawaitable(initialization):
            await initialization
        _check_export_cancellation(cancellation)
        row_count = 0
        if limit_scope == "per_dataset" or not database_truncated:
            queried_dataset_keys.append(dataset_key)
            async with connection.stream(
                dataset_select.statement,
                execution_options={
                    "yield_per": batch_size,
                    "max_row_buffer": batch_size,
                },
            ) as result:
                while True:
                    _check_export_cancellation(cancellation)
                    rows = await result.fetchmany(batch_size)
                    _check_export_cancellation(cancellation)
                    if not rows:
                        break
                    write_count = (
                        len(rows) if remaining is None else min(len(rows), remaining)
                    )
                    if write_count:
                        batch = _validated_export_batch(headers, rows[:write_count])
                        batch_completion = on_batch(dataset_key, headers, batch)
                        if inspect.isawaitable(batch_completion):
                            await batch_completion
                        _check_export_cancellation(cancellation)
                        row_count += write_count
                        if remaining is not None:
                            remaining -= write_count
                    if len(rows) > write_count:
                        database_truncated = True
                        break
                    await asyncio.sleep(0)
        dataset_row_counts.append((dataset_key, row_count))
        _check_export_cancellation(cancellation)
        if on_dataset_complete is not None:
            completion = on_dataset_complete(dataset_key)
            if inspect.isawaitable(completion):
                await completion
        _check_export_cancellation(cancellation)
        await asyncio.sleep(0)

    _check_export_cancellation(cancellation)
    return ExportQueryResult(
        selected_dataset_keys=tuple(selected_dataset_keys),
        compatible_dataset_keys=tuple(key for key, _table, _plan in jobs),
        queried_dataset_keys=tuple(queried_dataset_keys),
        skipped_datasets=tuple(skipped_datasets),
        dataset_row_counts=tuple(dataset_row_counts),
        limit=row_limit,
        database_truncated=database_truncated,
    )


def _validated_export_batch(
    headers: tuple[str, ...],
    rows: Sequence[Sequence[object]],
) -> tuple[tuple[str, ...], ...]:
    batch: list[tuple[str, ...]] = []
    for row in rows:
        if len(row) != len(headers):
            raise QueryExecutionError(
                "invalid_query_data", "Occurrence query data failed validation.",
            )
        values: list[str] = []
        for value in row:
            if not isinstance(value, str):
                raise QueryExecutionError(
                    "invalid_query_data", "Occurrence query data failed validation.",
                )
            values.append(value)
        batch.append(tuple(values))
    return tuple(batch)


def _check_export_cancellation(cancellation: AgentCancellation | None) -> None:
    if cancellation is not None and cancellation.is_cancelled:
        raise asyncio.CancelledError
