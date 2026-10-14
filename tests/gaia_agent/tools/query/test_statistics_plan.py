"""Validate exact statistics names, bounded structure, and source value types."""

from __future__ import annotations

import pytest
from sqlalchemy import MetaData

from src.gaia_agent.tools.query.statistics_plan import (
    MAX_GROUP_BY_COLUMNS,
    MAX_METRICS,
    validate_statistics_plan,
)
from src.gaia_agent.tools.query.plan import QueryPlanError

from tests.gaia_agent.tools.query.support import BIRDS_SCOPE_REGISTRY, make_table


def statistics_plan(*, group_by=(), metrics=None, limit=5):
    return {
        "scope": {"dataset": "Birds"},
        "query": {
            "filters": {},
            "group_by": list(group_by),
            "metrics": metrics or [{"operation": "count", "alias": "record_count"}],
        },
        "result": {"order_by": [], "limit": limit},
    }


def validate(plan):
    table = make_table(
        MetaData(), "statistics_validation", ("occurrenceID", "scientificName", "year"),
    )
    return validate_statistics_plan(
        plan, table_registry={"birds": table}, scope_registry=BIRDS_SCOPE_REGISTRY,
    )


def test_statistics_plan_resolves_source_fields_and_statistics_ordering():
    raw = statistics_plan(
        group_by=("scientificName",),
        metrics=[
            {"operation": "count", "alias": "record_count"},
            {"operation": "max", "alias": "last_year", "header": "year", "value_type": "numeric"},
        ],
    )
    raw["query"]["filters"] = {"year": {"gte": " 1900 ", "lt": "2027", "value_type": "numeric"}}
    raw["result"]["order_by"] = [{"header": "record_count", "direction": "desc"}]
    plan = validate(raw)
    assert plan.dataset_keys == ("birds",)
    assert plan.group_by == ("scientificName",)
    assert plan.filters[0].comparisons[0].value == "1900"
    assert plan.metrics[1].value_type == "numeric"
    assert plan.order_by[0].header == "record_count"


@pytest.mark.parametrize("limit", (None, 1, 201, 500))
def test_statistics_limit_accepts_all_groups_or_a_positive_integer_without_a_cap(limit):
    assert validate(statistics_plan(limit=limit)).limit == limit


@pytest.mark.parametrize("limit", (False, True, 0, -1, 1.5, "200", [], {}))
def test_statistics_limit_rejects_non_positive_integers_and_other_json_values(limit):
    with pytest.raises(QueryPlanError, match="must be a positive integer"):
        validate(statistics_plan(limit=limit))


def test_statistics_limit_is_required_even_when_all_groups_are_requested():
    raw = statistics_plan(limit=None)
    del raw["result"]["limit"]
    with pytest.raises(QueryPlanError, match="missing required properties: limit"):
        validate(raw)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    (
        ("scope", "sql", "SELECT 1", "unknown properties: sql"),
        ("query", "group_by", ["scientific_name"], "unknown group_by"),
        ("query", "group_by", ["year", "year"], "must not contain duplicates"),
        ("query", "group_by", ["year"] * (MAX_GROUP_BY_COLUMNS + 1), "at most"),
        ("query", "metrics", [], "must contain 1 to"),
        ("query", "metrics", [{"operation": "count", "alias": "n"}] * (MAX_METRICS + 1), "must contain 1 to"),
        ("result", "order_by", [{"header": "year", "direction": "asc"}], "unknown order_by"),
        ("result", "order_by", [{"header": "record_count", "direction": "asc", "value_type": "text"}], "unknown properties: value_type"),
        ("result", "raw_sql", "DROP TABLE birds", "unknown properties: raw_sql"),
    ),
)
def test_statistics_rejects_untrusted_or_unbounded_structures(section, key, value, message):
    raw = statistics_plan()
    raw[section][key] = value
    with pytest.raises(QueryPlanError, match=message):
        validate(raw)


@pytest.mark.parametrize(
    ("metric", "message"),
    (
        ({"operation": "execute", "alias": "n"}, "operation must be"),
        ({"operation": "count", "alias": "n; SELECT 1"}, "alias must be"),
        ({"operation": "count", "alias": "a" * 64}, "alias must be"),
        ({"operation": "count", "alias": "n", "header": "year"}, "count accepts no header"),
        ({"operation": "count", "alias": "n", "value_type": "numeric"}, "count accepts no header"),
        ({"operation": "count_distinct", "alias": "n"}, "header must be"),
        ({"operation": "count_distinct", "alias": "n", "header": "year", "value_type": "numeric"}, "requires text semantics"),
        ({"operation": "max", "alias": "n", "header": "year", "value_type": "date"}, "value_type must be"),
        ({"operation": "min", "alias": "n", "header": "scientific_name"}, "unknown metric header"),
        ({"operation": "sum", "alias": "n", "header": "year"}, "requires value_type 'numeric'"),
        ({"operation": "avg", "alias": "n", "header": "year", "value_type": "text"}, "requires value_type 'numeric'"),
    ),
)
def test_metric_contract_rejects_unsafe_names_and_implicit_numeric_casts(metric, message):
    with pytest.raises(QueryPlanError, match=message):
        validate(statistics_plan(metrics=[metric]))


@pytest.mark.parametrize("collide_with_group", (False, True))
def test_metric_aliases_cannot_overwrite_other_outputs(collide_with_group):
    metrics = [{"operation": "count", "alias": "year"}]
    groups = ("year",) if collide_with_group else ()
    if not collide_with_group:
        metrics.append({"operation": "count_distinct", "header": "year", "alias": "year"})
    with pytest.raises(QueryPlanError, match="duplicates an output column"):
        validate(statistics_plan(group_by=groups, metrics=metrics))
