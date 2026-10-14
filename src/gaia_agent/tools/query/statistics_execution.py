"""Compute statistics over complete compatible occurrence sets."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from sqlalchemy import Select, Text, cast, func, literal, select, union_all
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.schema import Table

from src.gaia_agent.tools.concurrency import WorkloadLimiter
from src.gaia_agent.tools.query.statistics_plan import PlannedMetric, ValidatedStatisticsPlan
from src.gaia_agent.tools.query.execution import (
    QUERY_UNAVAILABLE_MESSAGE,
    DatasetSelect,
    DatasetSkip,
    QueryExecutionError,
    build_filter_predicates,
    create_query_engine,
    exact_table_columns,
    managed_query_engine,
    numeric_value,
)

STATISTICS_BATCH_SIZE = 1000


@dataclass(frozen=True, slots=True)
class StatisticsExecutionResult:
    """Keep typed statistics rows and complete dataset-coverage metadata."""

    rows: tuple[dict[str, str | int | None], ...]
    selected_dataset_keys: tuple[str, ...]
    compatible_dataset_keys: tuple[str, ...]
    queried_dataset_keys: tuple[str, ...]
    skipped_datasets: tuple[DatasetSkip, ...]
    output_columns: tuple[str, ...]
    group_by: tuple[str, ...]
    metrics: tuple[PlannedMetric, ...]
    limit: int | None
    database_truncated: bool


async def execute_statistics_plan(
    plan: ValidatedStatisticsPlan,
    *,
    table_registry: Mapping[str, Table],
    engine_factory: Callable[[], AsyncEngine] = create_query_engine,
    database_limiter: WorkloadLimiter | None = None,
) -> StatisticsExecutionResult:
    """Compute complete inputs and collect optional output groups in async batches."""
    async with managed_query_engine(
        engine_factory, database_limiter=database_limiter,
    ) as engine:
        try:
            async with engine.connect() as connection:
                await connection.execution_options(postgresql_readonly=True)
                return await _execute_statistics_on_connection(
                    connection, plan, table_registry=table_registry,
                )
        except SQLAlchemyError as error:
            raise QueryExecutionError(
                "query_unavailable", QUERY_UNAVAILABLE_MESSAGE,
            ) from error
        except ValueError as error:
            raise QueryExecutionError(
                "invalid_query_data", "Occurrence statistics data failed validation.",
            ) from error


def resolve_statistics_datasets(
    plan: ValidatedStatisticsPlan,
    *,
    table_registry: Mapping[str, Table],
) -> tuple[tuple[tuple[str, Table], ...], tuple[DatasetSkip, ...]]:
    """Skip tables missing any filter, grouping, or metric input column."""
    required_header_groups = (
        ("missing_filter_columns", tuple(item.header for item in plan.filters)),
        ("missing_group_columns", plan.group_by),
        ("missing_metric_columns", tuple(dict.fromkeys(
            item.header for item in plan.metrics if item.header is not None
        ))),
    )
    compatible: list[tuple[str, Table]] = []
    skipped: list[DatasetSkip] = []
    for dataset_key in plan.dataset_keys:
        table = table_registry[dataset_key]
        columns = exact_table_columns(table)
        for reason, headers in required_header_groups:
            missing = tuple(header for header in headers if header not in columns)
            if missing:
                skipped.append(DatasetSkip(dataset_key, reason, missing))
                break
        else:
            compatible.append((dataset_key, table))
    return tuple(compatible), tuple(skipped)


def build_statistics_select(
    datasets: Sequence[tuple[str, Table]],
    plan: ValidatedStatisticsPlan,
    *,
    fetch_limit: int | None,
) -> DatasetSelect:
    """Compute statistics over UNION ALL inputs before ordering and limiting output."""
    if not datasets:
        raise ValueError("A statistics SELECT requires at least one compatible dataset")
    source_headers = tuple(dict.fromkeys((
        *plan.group_by,
        *(metric.header for metric in plan.metrics if metric.header is not None),
    )))
    source_aliases = {header: f"_gaia_column_{index}" for index, header in enumerate(source_headers)}
    statements: list[Select[tuple[object, ...]]] = []
    for _dataset_key, table in datasets:
        columns = exact_table_columns(table)
        selected = tuple(columns[header].label(source_aliases[header]) for header in source_headers)
        statement = select(*(selected or (literal(1).label("_gaia_presence"),))).select_from(table)
        predicates = build_filter_predicates(columns, plan.filters)
        if predicates:
            statement = statement.where(*predicates)
        statements.append(statement)
    source = (
        statements[0] if len(statements) == 1 else union_all(*statements)
    ).subquery("_gaia_statistics_source")
    source_columns = {header: source.c[alias] for header, alias in source_aliases.items()}
    output_columns = [source_columns[header].label(header) for header in plan.group_by]
    ordering_columns: dict[str, ColumnElement[object]] = {
        header: source_columns[header] for header in plan.group_by
    }
    for metric in plan.metrics:
        expression = _metric_expression(metric, source_columns)
        ordering_columns[metric.alias] = expression
        # Serialize database NUMERIC directly, preserving arbitrary decimal precision.
        serialized = cast(expression, Text()) if metric.value_type == "numeric" else expression
        output_columns.append(serialized.label(metric.alias))
    statement = select(*output_columns).select_from(source)
    if plan.group_by:
        statement = statement.group_by(*(source_columns[header] for header in plan.group_by))
    ordered_headers: set[str] = set()
    for ordering in plan.order_by:
        expression = ordering_columns[ordering.header]
        order = expression.asc() if ordering.direction == "asc" else expression.desc()
        statement = statement.order_by(order.nulls_last())
        ordered_headers.add(ordering.header)
    for header in plan.group_by:
        if header not in ordered_headers:
            statement = statement.order_by(source_columns[header].asc())
    return DatasetSelect(
        statement.limit(fetch_limit),
        (*plan.group_by, *(metric.alias for metric in plan.metrics)),
    )


def _metric_expression(
    metric: PlannedMetric,
    columns: Mapping[str, ColumnElement[object]],
) -> ColumnElement[object]:
    if metric.operation == "count":
        return func.count()
    assert metric.header is not None
    column = columns[metric.header]
    if metric.operation == "count_distinct":
        return func.count(column.distinct())
    if metric.value_type == "numeric":
        column = numeric_value(column)
    if metric.operation == "min":
        return func.min(column)
    if metric.operation == "max":
        return func.max(column)
    if metric.operation == "sum":
        return func.sum(column)
    return func.avg(column)


async def _execute_statistics_on_connection(
    connection: AsyncConnection,
    plan: ValidatedStatisticsPlan,
    *,
    table_registry: Mapping[str, Table],
) -> StatisticsExecutionResult:
    compatible, skipped = resolve_statistics_datasets(plan, table_registry=table_registry)
    limit = plan.limit
    rows: list[dict[str, str | int | None]] = []
    output_columns = (*plan.group_by, *(metric.alias for metric in plan.metrics))
    if compatible:
        compiled = build_statistics_select(
            compatible, plan, fetch_limit=None if limit is None else limit + 1,
        )
        count_aliases = {
            metric.alias for metric in plan.metrics
            if metric.operation in {"count", "count_distinct"}
        }
        group_headers = frozenset(plan.group_by)
        async with connection.stream(
            compiled.statement,
            execution_options={
                "yield_per": STATISTICS_BATCH_SIZE,
                "max_row_buffer": STATISTICS_BATCH_SIZE,
            },
        ) as result:
            while True:
                batch = await result.fetchmany(STATISTICS_BATCH_SIZE)
                if not batch:
                    break
                for row in batch:
                    record: dict[str, str | int | None] = {}
                    for header, value in zip(output_columns, row, strict=True):
                        if header in count_aliases:
                            valid = type(value) is int and value >= 0
                        elif header in group_headers:
                            valid = isinstance(value, str)
                        else:
                            valid = value is None or isinstance(value, str)
                        if not valid:
                            raise ValueError(f"{header}: unexpected statistics result type")
                        record[header] = value
                    rows.append(record)
                await asyncio.sleep(0)
    compatible_keys = tuple(key for key, _table in compatible)
    return StatisticsExecutionResult(
        rows=tuple(rows if limit is None else rows[:limit]),
        selected_dataset_keys=plan.dataset_keys,
        compatible_dataset_keys=compatible_keys,
        queried_dataset_keys=compatible_keys,
        skipped_datasets=skipped,
        output_columns=output_columns,
        group_by=plan.group_by,
        metrics=plan.metrics,
        limit=limit,
        database_truncated=limit is not None and len(rows) > limit,
    )
