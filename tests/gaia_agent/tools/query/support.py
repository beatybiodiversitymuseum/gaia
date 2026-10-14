"""Shared flat-table and plan builders for query validation and execution tests."""

from __future__ import annotations

import copy
from collections.abc import Sequence

from sqlalchemy import Column, MetaData, Table, Text

from src.database.base import orm_attribute_name
from src.gaia_agent.tools.query.plan import ValidatedQueryPlan, validate_query_plan


BASE_PLAN = {
    "scope": {
        "collection": "Cowan Tetrapod Collection",
        "dataset": "Birds",
    },
    "query": {"filters": {}, "return_columns": []},
    "result": {"distinct_by": [], "order_by": [], "limit": 5},
}

BIRDS_SCOPE_REGISTRY = {
    "Cowan Tetrapod Collection": {"Birds": "birds"},
    "Herbarium": {"Algae": "algae"},
}


def make_table(
    metadata: MetaData,
    name: str,
    headers: Sequence[str],
) -> Table:
    """Build a flat TEXT table whose Python keys differ from physical headers."""
    return Table(
        name,
        metadata,
        *(
            Column(
                header,
                Text,
                key=orm_attribute_name(header),
                nullable=False,
                primary_key=(
                    header == "occurrenceID"
                    or (header == "id" and "occurrenceID" not in headers)
                ),
            )
            for header in headers
        ),
    )


def validate_bird_plan(
    plan: dict[str, object],
    table: Table,
) -> ValidatedQueryPlan:
    return validate_query_plan(
        plan,
        table_registry={"birds": table},
        scope_registry=BIRDS_SCOPE_REGISTRY,
    )


def blank_plan(
    dataset: str | None = "Birds",
    *,
    collection: str | None = "Cowan Tetrapod Collection",
    limit: int = 5,
    column_mode: str = "default",
) -> dict[str, object]:
    """Return a detached complete flat plan."""
    plan = copy.deepcopy(BASE_PLAN)
    scope = plan["scope"]
    result = plan["result"]
    assert isinstance(scope, dict)
    assert isinstance(result, dict)
    if collection is None:
        del scope["collection"]
    else:
        scope["collection"] = collection
    if dataset is None:
        del scope["dataset"]
    else:
        scope["dataset"] = dataset
    result["limit"] = limit
    query = plan["query"]
    assert isinstance(query, dict)
    query["column_mode"] = column_mode
    if column_mode == "all":
        query["return_columns"] = "all"
    return plan


def add_filter(
    plan: dict[str, object],
    header: str,
    condition: object,
) -> None:
    query = plan["query"]
    assert isinstance(query, dict)
    filters = query["filters"]
    assert isinstance(filters, dict)
    filters[header] = condition


def add_return_columns(
    plan: dict[str, object],
    *headers: str,
) -> None:
    query = plan["query"]
    assert isinstance(query, dict)
    columns = query["return_columns"]
    assert isinstance(columns, list)
    columns.extend(headers)
