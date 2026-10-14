"""Exercise shared literal filters and explicit numeric ordering contracts."""

from __future__ import annotations

import pytest
from sqlalchemy import MetaData
from sqlalchemy.dialects import postgresql

from src.gaia_agent.tools.query.execution import _execute_on_connection, build_dataset_select
from src.gaia_agent.tools.query.plan import QueryPlanError

from tests.gaia_agent.tools.query.support import add_filter, add_return_columns, blank_plan, make_table, validate_bird_plan
from tests.gaia_agent.tools.query.test_execution import populated_table


@pytest.mark.parametrize(
    ("condition", "message"),
    (
        ({"value_type": "numeric"}, "at least one filter operator"),
        ({"exclude_any": []}, "must not be empty"),
        ({"exclude_any": ["a", "a"]}, "must not contain duplicates"),
        ({"exclude_any": [None]}, "must contain only strings"),
        ({"exclude_any": [str(i) for i in range(51)]}, "at most 50"),
        ({"contains": ""}, "must be a non-empty string"),
        ({"starts_with": 1}, "must be a non-empty string"),
        ({"is_empty": 0}, "must be a boolean"),
        ({"gt": 2}, "must be a string"),
        ({"lt": "2", "value_type": "date"}, "value_type must be"),
        ({"gte": "1e3", "value_type": "numeric"}, "plain decimal string"),
        ({"gte": "Infinity", "value_type": "numeric"}, "plain decimal string"),
        ({"lte": "9" * 129, "value_type": "numeric"}, "plain decimal string"),
        ({"lt": "", "value_type": "numeric"}, "plain decimal string"),
        ({"lt": "1\n", "value_type": "numeric"}, "plain decimal string"),
        ({"exact_any": ["2"], "value_type": "numeric"}, "requires a comparison operator"),
    ),
)
def test_filters_reject_malformed_or_ambiguous_numeric_conditions(condition, message):
    table = make_table(MetaData(), "filter_validation", ("occurrenceID", "year"))
    raw = blank_plan()
    add_filter(raw, "year", condition)
    with pytest.raises(QueryPlanError, match=message):
        validate_bird_plan(raw, table)


@pytest.mark.parametrize(
    ("condition", "expected"),
    (
        ({"contains": "%_"}, ["literal", "prefix"]),
        ({"starts_with": "%_"}, ["prefix"]),
        ({"exclude_any": ["", " ", "%_Canada"]}, ["literal", "ordinary"]),
        ({"is_empty": True}, ["empty"]),
        ({"is_empty": False, "starts_with": "%"}, ["prefix"]),
        ({"contains": "%_", "exclude_any": ["%_Canada"]}, ["literal"]),
        ({"gte": "B", "lt": "D"}, ["ordinary"]),
    ),
)
def test_literal_filters_combine_with_and_and_preserve_empty_and_unicode(condition, expected):
    engine, table = populated_table(
        "literal_filters", ("occurrenceID", "locality"),
        tuple(
            {"occurrenceID": key, "locality": value}
            for key, value in (
                ("empty", ""), ("space", " "), ("literal", "Élan %_ 🐦"),
                ("prefix", "%_Canada"), ("ordinary", "Canada"),
            )
        ),
    )
    raw = blank_plan(limit=20, column_mode="selected")
    add_return_columns(raw, "locality")
    add_filter(raw, "locality", condition)
    plan = validate_bird_plan(raw, table)
    try:
        with engine.connect() as connection:
            result = _execute_on_connection(connection, plan, table_registry={"birds": table})
    finally:
        engine.dispose()
    assert [row["occurrenceID"] for row in result.records] == expected


def test_numeric_comparison_excludes_invalid_source_and_keeps_literal_values():
    source_values = ("", "unknown", "1e3", "NaN", "9" * 129, "-3", "2", "10", " 20.50 ", ".75", "10\n", "\n10")
    engine, table = populated_table(
        "numeric_filters", ("occurrenceID", "year"),
        tuple({"occurrenceID": f"row-{index:02}", "year": value} for index, value in enumerate(source_values)),
    )
    raw = blank_plan(limit=20, column_mode="selected")
    add_return_columns(raw, "year")
    add_filter(raw, "year", {"gt": "1", "lte": "20.5", "value_type": "numeric"})
    raw["result"]["order_by"] = [{"header": "year", "direction": "desc", "value_type": "numeric"}]
    plan = validate_bird_plan(raw, table)
    try:
        with engine.connect() as connection:
            result = _execute_on_connection(connection, plan, table_registry={"birds": table})
    finally:
        engine.dispose()
    assert [row["year"] for row in result.records] == [" 20.50 ", "10", "2"]


@pytest.mark.parametrize("distinct", (False, True))
def test_numeric_ordering_uses_numeric_values_and_places_invalid_values_last(distinct):
    engine, table = populated_table(
        "numeric_ordering", ("occurrenceID", "scientificName", "year"),
        tuple(
            {"occurrenceID": key, "scientificName": species, "year": value}
            for key, species, value in (
                ("raven-a", "Corvus corax", "9"),
                ("raven-z", "Corvus corax", "10"),
                ("robin", "Turdus migratorius", "20"),
                ("empty", "", ""),
                ("unknown", "Unknown", "unknown"),
            )
        ),
    )
    raw = blank_plan(limit=20, column_mode="selected")
    add_return_columns(raw, "scientificName", "year")
    raw["result"]["order_by"] = [{"header": "year", "direction": "desc", "value_type": "numeric"}]
    if distinct:
        raw["result"]["distinct_by"] = ["scientificName"]
    try:
        with engine.connect() as connection:
            result = _execute_on_connection(
                connection, validate_bird_plan(raw, table), table_registry={"birds": table},
            )
    finally:
        engine.dispose()
    assert [row["occurrenceID"] for row in result.records] == (
        ["robin", "raven-z", "unknown"] if distinct
        else ["robin", "raven-z", "raven-a", "empty", "unknown"]
    )


def test_extended_filters_compile_to_parameterized_postgresql_sql():
    table = make_table(MetaData(), "safe_filter_sql", ("occurrenceID", "scientificName", "year"))
    raw = blank_plan()
    malicious = "%' OR 1=1 --_/"
    add_filter(raw, "scientificName", {"contains": malicious, "starts_with": "Larus_"})
    add_filter(raw, "year", {"gte": "1900", "value_type": "numeric"})
    raw["result"]["order_by"] = [{"header": "year", "direction": "desc", "value_type": "numeric"}]
    statement = build_dataset_select(table, validate_bird_plan(raw, table), fetch_limit=6).statement
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert malicious not in sql
    assert "LIKE" in sql and "ESCAPE '/'" in sql
    assert "CASE WHEN" in sql and " AS NUMERIC" in sql
    assert "DESC NULLS LAST" in sql
    assert any(isinstance(value, str) and "OR 1=1" in value for value in compiled.params.values())
