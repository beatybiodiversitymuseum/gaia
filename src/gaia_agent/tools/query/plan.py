"""Validate flat occurrence query plans against trusted schema metadata."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

from sqlalchemy.sql.schema import Table

from src.gaia_agent.tools.query.columns import ColumnMode, resolve_return_columns

MAX_ORDER_BY_COLUMNS = 8
MAX_DISTINCT_COLUMNS = 8
MAX_EXACT_VALUES_PER_HEADER = 50
MAX_NUMERIC_TEXT_LENGTH = 128
NUMERIC_TEXT_PATTERN = r"\A[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)\Z"
NUMERIC_VALUE_POLICY = (
    "Numeric operations trim surrounding ASCII spaces and accept signed plain "
    "decimal strings of at most 128 characters. Empty, nondecimal, exponent, "
    "and longer values become NULL: numeric comparisons exclude them, numeric "
    "ordering places them last, and numeric statistics ignore them. Source "
    "text is preserved; decimal metric results are returned as strings."
)

SortDirection = Literal["asc", "desc"]
ValueType = Literal["text", "numeric"]
ComparisonOperator = Literal["gt", "gte", "lt", "lte"]


# ---------------------------------------------------------------------------
# Validated contracts
# ---------------------------------------------------------------------------


class QueryPlanError(ValueError):
    """Indicate that a model-supplied QueryPlan violates the trusted contract."""


@dataclass(frozen=True, slots=True)
class PlannedComparison:
    """Hold a validated comparison against a literal source or decimal value."""

    operator: ComparisonOperator
    value: str


@dataclass(frozen=True, slots=True)
class PlannedFilter:
    """Hold AND-combined literal conditions for one trusted exact header."""

    header: str
    exact_any: tuple[str, ...] = ()
    exclude_any: tuple[str, ...] = ()
    contains: str | None = None
    starts_with: str | None = None
    is_empty: bool | None = None
    comparisons: tuple[PlannedComparison, ...] = ()
    value_type: ValueType = "text"


@dataclass(frozen=True, slots=True)
class PlannedOrder:
    """Hold one resolved, exact-header ordering instruction."""

    header: str
    direction: SortDirection
    value_type: ValueType = "text"


@dataclass(frozen=True, slots=True)
class ValidatedQueryPlan:
    """Hold resolved columns, including per-dataset selections for collections.

    A single-dataset plan uses ``return_columns`` directly. A collection plan
    stores their first-seen union there for observation projection, with each
    dataset's actual selection in ``dataset_return_columns``.
    """

    dataset_keys: tuple[str, ...]
    filters: tuple[PlannedFilter, ...]
    return_columns: tuple[str, ...]
    distinct_by: tuple[str, ...]
    order_by: tuple[PlannedOrder, ...]
    limit: int
    column_mode: ColumnMode
    dataset_return_columns: tuple[tuple[str, tuple[str, ...]], ...]


# ---------------------------------------------------------------------------
# QueryPlan validation
# ---------------------------------------------------------------------------


def validate_query_plan(
    value: object,
    *,
    table_registry: Mapping[str, Table],
    scope_registry: Mapping[str, Mapping[str, str]],
) -> ValidatedQueryPlan:
    """Validate every QueryPlan property against trusted schema and tables."""
    plan = _require_object(value, "QueryPlan")
    structural_errors = _query_plan_structural_errors(plan)
    if structural_errors:
        raise QueryPlanError("; ".join(structural_errors))

    dataset_keys = _validate_scope(
        plan["scope"],
        scope_registry=scope_registry,
    )
    registered_headers = _registered_headers(table_registry)
    filters, requested_columns, column_mode = _validate_query(
        plan["query"],
        registered_headers=registered_headers,
    )
    distinct_by, order_by, limit = _validate_result(
        plan["result"],
        registered_headers=registered_headers,
        dataset_keys=dataset_keys,
        return_columns=requested_columns,
        column_mode=column_mode,
    )
    resolved_datasets: list[tuple[str, tuple[str, ...]]] = []
    for dataset_key in dataset_keys:
        try:
            columns = resolve_return_columns(
                physical_headers=tuple(column.name for column in table_registry[dataset_key].columns),
                column_mode=column_mode,
                requested_columns=requested_columns,
                filter_headers=tuple(item.header for item in filters),
                distinct_headers=distinct_by,
                order_headers=tuple(item.header for item in order_by),
            )
        except ValueError as error:
            raise QueryPlanError(f"{dataset_key}: {error}") from error
        resolved_datasets.append((dataset_key, columns))
    dataset_return_columns = tuple(resolved_datasets)
    return ValidatedQueryPlan(
        dataset_keys=dataset_keys,
        filters=filters,
        return_columns=tuple(dict.fromkeys(
            header for _key, columns in dataset_return_columns for header in columns
        )),
        distinct_by=distinct_by,
        order_by=order_by,
        limit=limit,
        column_mode=column_mode,
        dataset_return_columns=dataset_return_columns if len(dataset_keys) > 1 else (),
    )


def query_plan_for_dataset(
    plan: ValidatedQueryPlan,
    dataset_key: str,
) -> ValidatedQueryPlan:
    """Narrow a validated collection plan to its resolved dataset selection."""
    if dataset_key not in plan.dataset_keys:
        raise ValueError(f"Dataset {dataset_key!r} is outside this query plan's scope")
    if plan.dataset_keys == (dataset_key,):
        return plan
    return replace(
        plan,
        dataset_keys=(dataset_key,),
        return_columns=dict(plan.dataset_return_columns)[dataset_key],
        dataset_return_columns=(),
    )


def _inspect_plan_object(
    missing: list[str],
    unknown: list[str],
    value: Mapping[str, object],
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    path: str,
) -> None:
    """Collect structural diagnostics while preserving required-field order."""
    allowed = {*required, *optional}
    for name in required:
        if name not in value:
            missing.append(f"{path}.{name}" if path else name)
    for name in sorted(
        key for key in value if isinstance(key, str) and key not in allowed
    ):
        unknown.append(f"{path}.{name}" if path else name)


def _query_plan_structural_errors(
    plan: Mapping[str, object],
) -> tuple[str, ...]:
    """Collect all discoverable missing and unknown plan properties at once."""
    missing: list[str] = []
    unknown: list[str] = []

    _inspect_plan_object(
        missing, unknown,
        plan,
        required=("scope", "query", "result"),
        path="",
    )
    scope = plan.get("scope")
    if isinstance(scope, dict):
        _inspect_plan_object(
            missing, unknown,
            scope,
            required=(),
            optional=("collection", "dataset"),
            path="scope",
        )

    query = plan.get("query")
    if isinstance(query, dict):
        _inspect_plan_object(
            missing, unknown,
            query,
            required=("filters", "return_columns"),
            optional=("column_mode",),
            path="query",
        )

    result = plan.get("result")
    if isinstance(result, dict):
        _inspect_plan_object(
            missing, unknown,
            result,
            required=("distinct_by", "order_by", "limit"),
            path="result",
        )

    errors: list[str] = []
    if missing:
        errors.append(
            "QueryPlan is missing required properties: " + ", ".join(missing)
        )
    if unknown:
        errors.append(
            "QueryPlan contains unknown properties: " + ", ".join(unknown)
        )
    return tuple(errors)


def _registered_headers(
    table_registry: Mapping[str, Table],
) -> tuple[str, ...]:
    """Return the exact physical header union in registry and table order."""
    return tuple(
        dict.fromkeys(
            column.name
            for table in table_registry.values()
            for column in table.columns
        )
    )


def _validate_scope(
    value: object,
    *,
    scope_registry: Mapping[str, Mapping[str, str]],
) -> tuple[str, ...]:
    scope = _require_object(value, "scope")

    if not scope:
        raise QueryPlanError(
            "scope must contain at least one of collection or dataset"
        )

    collection: str | None = None
    if "collection" in scope:
        collection_value = scope["collection"]
        if not isinstance(collection_value, str) or not collection_value:
            raise QueryPlanError("scope.collection must be a non-empty string")
        collection = collection_value
    dataset: str | None = None
    if "dataset" in scope:
        dataset_value = scope["dataset"]
        if not isinstance(dataset_value, str) or not dataset_value:
            raise QueryPlanError("scope.dataset must be a non-empty string")
        dataset = dataset_value

    if isinstance(collection, str):
        datasets = scope_registry.get(collection)
        if datasets is None:
            raise QueryPlanError(f"unknown collection label: {collection!r}")
        if isinstance(dataset, str):
            dataset_key = datasets.get(dataset)
            if dataset_key is None:
                owners = tuple(
                    label
                    for label, members in scope_registry.items()
                    if label != collection and dataset in members
                )
                if owners:
                    raise QueryPlanError(
                        f"dataset label {dataset!r} does not belong to "
                        f"collection {collection!r}"
                    )
                raise QueryPlanError(
                    f"unknown dataset label {dataset!r} for collection "
                    f"{collection!r}"
                )
            dataset_keys = (dataset_key,)
        else:
            dataset_keys = tuple(datasets.values())
    else:
        assert isinstance(dataset, str)
        matches = tuple(
            (collection_label, members[dataset])
            for collection_label, members in scope_registry.items()
            if dataset in members
        )
        if not matches:
            raise QueryPlanError(f"unknown dataset label: {dataset!r}")
        if len(matches) > 1:
            collections = ", ".join(label for label, _key in matches)
            raise QueryPlanError(
                f"ambiguous dataset label {dataset!r}; found in collections: "
                f"{collections}"
            )
        dataset_keys = (matches[0][1],)

    return dataset_keys


def _validate_query(
    value: object,
    *,
    registered_headers: tuple[str, ...],
) -> tuple[tuple[PlannedFilter, ...], tuple[str, ...], ColumnMode]:
    query = _require_object(value, "query")
    column_mode = query.get("column_mode", "default")
    if not isinstance(column_mode, str) or column_mode not in ("default", "all", "selected"):
        raise QueryPlanError("query.column_mode must be 'default', 'all', or 'selected'")

    known_headers = frozenset(registered_headers)
    planned_filters = validate_filters(
        query["filters"], registered_headers=registered_headers,
    )

    columns = query["return_columns"]
    if isinstance(columns, str) and columns == "all":
        if "column_mode" in query and column_mode != "all":
            raise QueryPlanError(
                "query.return_columns: 'all' conflicts with "
                f"query.column_mode: {column_mode!r}; omit query.column_mode "
                "or set it to 'all' to return every physical field"
            )
        return tuple(planned_filters), (), "all"
    if not isinstance(columns, list):
        raise QueryPlanError(
            "query.return_columns must be an array of exact header names "
            "or the literal string 'all'"
        )
    if column_mode == "all":
        raise QueryPlanError(
            "query.column_mode: 'all' requires query.return_columns: 'all'; "
            "replace the array with the literal string 'all'"
        )
    if any(not isinstance(header, str) or not header for header in columns):
        raise QueryPlanError(
            "query.return_columns must contain only non-empty strings"
        )
    return_columns = tuple(columns)
    if len(set(return_columns)) != len(return_columns):
        raise QueryPlanError("query.return_columns must not contain duplicates")
    unknown_columns = tuple(
        header for header in return_columns
        if header not in known_headers and header not in ("occurrenceID", "id")
    )
    if unknown_columns:
        raise QueryPlanError(
            "unknown return column(s): " + ", ".join(unknown_columns)
        )

    if column_mode == "selected" and not return_columns:
        raise QueryPlanError(
            "query.return_columns must not be empty when query.column_mode is 'selected'; "
            "use 'default' for dataset defaults or query.return_columns: 'all' "
            "with query.column_mode omitted for every physical field"
        )

    return tuple(planned_filters), return_columns, cast(ColumnMode, column_mode)


def _validate_filter_headers(
    filters: Mapping[str, object],
    known_headers: frozenset[str],
) -> None:
    for header in filters:
        if not isinstance(header, str) or not header:
            raise QueryPlanError("filter header names must be non-empty strings")
        if header not in known_headers:
            raise QueryPlanError(f"unknown filter header {header!r}")


def validate_filters(
    value: object,
    *,
    registered_headers: tuple[str, ...],
) -> tuple[PlannedFilter, ...]:
    """Resolve shared record, export, and statistics filters in schema order."""
    filters = _require_object(value, "query.filters")
    _validate_filter_headers(filters, frozenset(registered_headers))
    return tuple(
        _validate_filter(header, filters[header])
        for header in registered_headers
        if header in filters
    )


def _validate_filter(
    header: str,
    value: object,
) -> PlannedFilter:
    label = f"filter {header!r}"
    condition = _require_object(value, label)
    operators = (
        "exact_any", "exclude_any", "contains", "starts_with", "is_empty",
        "gt", "gte", "lt", "lte",
    )
    _require_exact_properties(
        condition, required=(), optional=(*operators, "value_type"), label=label,
    )
    if not any(operator in condition for operator in operators):
        raise QueryPlanError(f"{label} must contain at least one filter operator")
    value_type = validate_value_type(condition.get("value_type", "text"), label)
    comparisons: list[PlannedComparison] = []
    for operator in ("gt", "gte", "lt", "lte"):
        if operator not in condition:
            continue
        literal = condition[operator]
        if not isinstance(literal, str):
            raise QueryPlanError(f"{label}.{operator} must be a string")
        if value_type == "numeric":
            literal = literal.strip(" ")
            if (
                len(literal) > MAX_NUMERIC_TEXT_LENGTH
                or re.fullmatch(NUMERIC_TEXT_PATTERN, literal) is None
            ):
                raise QueryPlanError(
                    f"{label}.{operator} must be a signed plain decimal string "
                    f"of at most {MAX_NUMERIC_TEXT_LENGTH} characters"
                )
        comparisons.append(PlannedComparison(cast(ComparisonOperator, operator), literal))
    if "value_type" in condition and not comparisons:
        raise QueryPlanError(f"{label}.value_type requires a comparison operator")
    for operator in ("contains", "starts_with"):
        if operator in condition and (
            not isinstance(condition[operator], str) or not condition[operator]
        ):
            raise QueryPlanError(f"{label}.{operator} must be a non-empty string")
    if "is_empty" in condition and not isinstance(condition["is_empty"], bool):
        raise QueryPlanError(f"{label}.is_empty must be a boolean")
    return PlannedFilter(
        header=header,
        exact_any=_validate_filter_values(condition, "exact_any", label),
        exclude_any=_validate_filter_values(condition, "exclude_any", label),
        contains=cast(str | None, condition.get("contains")),
        starts_with=cast(str | None, condition.get("starts_with")),
        is_empty=cast(bool | None, condition.get("is_empty")),
        comparisons=tuple(comparisons),
        value_type=value_type,
    )


def _validate_filter_values(
    condition: Mapping[str, object], operator: str, label: str,
) -> tuple[str, ...]:
    if operator not in condition:
        return ()
    values = _require_list(condition[operator], f"{label}.{operator}")
    if not values:
        raise QueryPlanError(f"{label}.{operator} must not be empty")
    if len(values) > MAX_EXACT_VALUES_PER_HEADER:
        raise QueryPlanError(
            f"{label}.{operator} may contain at most "
            f"{MAX_EXACT_VALUES_PER_HEADER} values"
        )
    if any(not isinstance(item, str) for item in values):
        raise QueryPlanError(f"{label}.{operator} must contain only strings")
    if len(set(values)) != len(values):
        raise QueryPlanError(f"{label}.{operator} must not contain duplicates")
    return tuple(cast(list[str], values))


def validate_value_type(value: object, label: str) -> ValueType:
    """Accept explicit numeric conversion or literal source-text semantics."""
    if not isinstance(value, str) or value not in {"text", "numeric"}:
        raise QueryPlanError(f"{label}.value_type must be 'text' or 'numeric'")
    return cast(ValueType, value)


def _validate_result(
    value: object,
    *,
    registered_headers: tuple[str, ...],
    dataset_keys: tuple[str, ...],
    return_columns: tuple[str, ...],
    column_mode: ColumnMode,
) -> tuple[tuple[str, ...], tuple[PlannedOrder, ...], int]:
    result = _require_object(value, "result")
    distinct_items = _require_list(result["distinct_by"], "result.distinct_by")
    if len(distinct_items) > MAX_DISTINCT_COLUMNS:
        raise QueryPlanError(
            f"result.distinct_by may contain at most {MAX_DISTINCT_COLUMNS} entries"
        )
    if any(not isinstance(header, str) or not header for header in distinct_items):
        raise QueryPlanError(
            "result.distinct_by must contain only non-empty strings"
        )
    distinct_by = tuple(distinct_items)
    if len(set(distinct_by)) != len(distinct_by):
        raise QueryPlanError("result.distinct_by must not contain duplicates")
    known_headers = frozenset(registered_headers)
    unknown_distinct = tuple(
        header for header in distinct_by if header not in known_headers
    )
    if unknown_distinct:
        raise QueryPlanError(
            "unknown distinct_by header(s): " + ", ".join(unknown_distinct)
        )
    return_header_set = frozenset(return_columns)
    missing_distinct_returns = tuple(
        header for header in distinct_by if header not in return_header_set
    )
    if column_mode == "selected" and missing_distinct_returns:
        raise QueryPlanError(
            "result.distinct_by header(s) must also appear in "
            "query.return_columns: " + ", ".join(missing_distinct_returns)
        )
    if distinct_by and len(dataset_keys) != 1:
        raise QueryPlanError(
            "result.distinct_by requires exactly one selected dataset"
        )

    order_by = validate_order_by(result["order_by"], known_headers=known_headers)
    limit = validate_limit(result["limit"])
    return distinct_by, order_by, limit


def validate_order_by(
    value: object,
    *,
    known_headers: frozenset[str],
    allow_value_type: bool = True,
) -> tuple[PlannedOrder, ...]:
    """Validate bounded ordering against source fields or statistics outputs."""
    order_items = _require_list(value, "result.order_by")
    if len(order_items) > MAX_ORDER_BY_COLUMNS:
        raise QueryPlanError(
            f"result.order_by may contain at most {MAX_ORDER_BY_COLUMNS} entries"
        )
    order_by: list[PlannedOrder] = []
    seen_headers: set[str] = set()
    for index, value in enumerate(order_items):
        label = f"result.order_by[{index}]"
        order = _require_object(value, label)
        _require_exact_properties(
            order,
            required=("header", "direction"),
            optional=("value_type",) if allow_value_type else (),
            label=label,
        )
        header = order["header"]
        if not isinstance(header, str) or not header:
            raise QueryPlanError(f"{label}.header must be a non-empty string")
        if header not in known_headers:
            raise QueryPlanError(f"unknown order_by header: {header}")
        if header in seen_headers:
            raise QueryPlanError(f"result.order_by repeats header {header!r}")
        seen_headers.add(header)
        direction = order["direction"]
        if not isinstance(direction, str) or direction not in {"asc", "desc"}:
            raise QueryPlanError(f"{label}.direction must be 'asc' or 'desc'")
        value_type = validate_value_type(order.get("value_type", "text"), label)
        order_by.append(PlannedOrder(header, direction, value_type))
    return tuple(order_by)


def validate_limit(value: object) -> int:
    """Require a positive requested limit before runtime retrieval budgeting."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QueryPlanError("result.limit must be a positive integer")
    return value


def _require_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise QueryPlanError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise QueryPlanError(f"{label} property names must be strings")
    return value


def _require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise QueryPlanError(f"{label} must be a JSON array")
    return value


def _require_exact_properties(
    value: Mapping[str, object],
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    label: str,
) -> None:
    allowed = {*required, *optional}
    missing = [name for name in required if name not in value]
    unknown = sorted(set(value) - allowed)
    if missing:
        raise QueryPlanError(
            f"{label} is missing required properties: {', '.join(missing)}"
        )
    if unknown:
        raise QueryPlanError(
            f"{label} contains unknown properties: {', '.join(unknown)}"
        )
