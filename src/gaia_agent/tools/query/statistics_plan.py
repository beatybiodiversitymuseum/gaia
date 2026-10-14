"""Validate statistics plans against trusted occurrence schemas."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from sqlalchemy.sql.schema import Table

from src.gaia_agent.tools.query.plan import (
    PlannedFilter,
    PlannedOrder,
    QueryPlanError,
    ValueType,
    _registered_headers,
    _require_exact_properties,
    _require_list,
    _require_object,
    _validate_scope,
    validate_filters,
    validate_limit,
    validate_order_by,
    validate_value_type,
)

MAX_GROUP_BY_COLUMNS = 8
MAX_METRICS = 8
METRIC_ALIAS_MAX_LENGTH = 63
METRIC_ALIAS_PATTERN = (
    rf"^[A-Za-z_][A-Za-z0-9_]{{0,{METRIC_ALIAS_MAX_LENGTH - 1}}}$"
)

MetricOperation = Literal["count", "count_distinct", "min", "max", "sum", "avg"]


@dataclass(frozen=True, slots=True)
class PlannedMetric:
    """Describe one statistic and its exact output alias and source semantics."""

    operation: MetricOperation
    alias: str
    header: str | None
    value_type: ValueType


@dataclass(frozen=True, slots=True)
class ValidatedStatisticsPlan:
    """Hold a trusted statistics plan over all compatible datasets in one scope."""

    dataset_keys: tuple[str, ...]
    filters: tuple[PlannedFilter, ...]
    group_by: tuple[str, ...]
    metrics: tuple[PlannedMetric, ...]
    order_by: tuple[PlannedOrder, ...]
    limit: int | None


def validate_statistics_plan(
    value: object,
    *,
    table_registry: Mapping[str, Table],
    scope_registry: Mapping[str, Mapping[str, str]],
) -> ValidatedStatisticsPlan:
    """Validate all names and operations before constructing statistics SQL."""
    plan = _require_object(value, "StatisticsPlan")
    _require_exact_properties(
        plan, required=("scope", "query", "result"), label="StatisticsPlan",
    )
    scope = _require_object(plan["scope"], "scope")
    _require_exact_properties(
        scope, required=(), optional=("collection", "dataset"), label="scope",
    )
    dataset_keys = _validate_scope(scope, scope_registry=scope_registry)
    registered_headers = _registered_headers(table_registry)
    known_headers = frozenset(registered_headers)
    query = _require_object(plan["query"], "query")
    _require_exact_properties(
        query, required=("filters", "group_by", "metrics"), label="query",
    )
    filters = validate_filters(query["filters"], registered_headers=registered_headers)
    groups = _require_list(query["group_by"], "query.group_by")
    if len(groups) > MAX_GROUP_BY_COLUMNS:
        raise QueryPlanError(
            f"query.group_by may contain at most {MAX_GROUP_BY_COLUMNS} entries"
        )
    if any(not isinstance(header, str) or not header for header in groups):
        raise QueryPlanError("query.group_by must contain only non-empty strings")
    group_by = tuple(cast(list[str], groups))
    if len(set(group_by)) != len(group_by):
        raise QueryPlanError("query.group_by must not contain duplicates")
    unknown_groups = tuple(header for header in group_by if header not in known_headers)
    if unknown_groups:
        raise QueryPlanError("unknown group_by header(s): " + ", ".join(unknown_groups))
    metric_items = _require_list(query["metrics"], "query.metrics")
    if not metric_items or len(metric_items) > MAX_METRICS:
        raise QueryPlanError(f"query.metrics must contain 1 to {MAX_METRICS} entries")
    metrics = tuple(
        _validate_metric(item, index=index, known_headers=known_headers)
        for index, item in enumerate(metric_items)
    )
    aliases: set[str] = set(group_by)
    for metric in metrics:
        if metric.alias in aliases:
            raise QueryPlanError(
                f"metric alias {metric.alias!r} duplicates an output column"
            )
        aliases.add(metric.alias)
    result = _require_object(plan["result"], "result")
    _require_exact_properties(result, required=("order_by", "limit"), label="result")
    return ValidatedStatisticsPlan(
        dataset_keys=dataset_keys,
        filters=filters,
        group_by=group_by,
        metrics=metrics,
        order_by=validate_order_by(
            result["order_by"], known_headers=frozenset(aliases), allow_value_type=False,
        ),
        limit=None if result["limit"] is None else validate_limit(result["limit"]),
    )


def _validate_metric(
    value: object,
    *,
    index: int,
    known_headers: frozenset[str],
) -> PlannedMetric:
    label = f"query.metrics[{index}]"
    metric = _require_object(value, label)
    _require_exact_properties(
        metric,
        required=("operation", "alias"),
        optional=("header", "value_type"),
        label=label,
    )
    operation = metric["operation"]
    if not isinstance(operation, str) or operation not in {
        "count", "count_distinct", "min", "max", "sum", "avg",
    }:
        raise QueryPlanError(
            f"{label}.operation must be count, count_distinct, min, max, sum, or avg"
        )
    alias = metric["alias"]
    if not isinstance(alias, str) or re.fullmatch(METRIC_ALIAS_PATTERN, alias) is None:
        raise QueryPlanError(
            f"{label}.alias must be 1 to {METRIC_ALIAS_MAX_LENGTH} ASCII letters, "
            "digits, or underscores, "
            "starting with a letter or underscore"
        )
    value_type = validate_value_type(metric.get("value_type", "text"), label)
    header: str | None = None
    if operation == "count":
        if "header" in metric or "value_type" in metric:
            raise QueryPlanError(f"{label}: count accepts no header or value_type; it counts rows")
    else:
        requested_header = metric.get("header")
        if not isinstance(requested_header, str) or not requested_header:
            raise QueryPlanError(f"{label}.header must be a non-empty string")
        if requested_header not in known_headers:
            raise QueryPlanError(f"unknown metric header: {requested_header}")
        header = requested_header
        if operation == "count_distinct" and value_type != "text":
            raise QueryPlanError(f"{label}: count_distinct requires text semantics")
        if operation in {"sum", "avg"} and value_type != "numeric":
            raise QueryPlanError(f"{label}: {operation} requires value_type 'numeric'")
    return PlannedMetric(cast(MetricOperation, operation), alias, header, value_type)
