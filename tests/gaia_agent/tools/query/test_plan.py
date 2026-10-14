"""Trusted QueryPlan validation tests."""

from __future__ import annotations

import pytest
from sqlalchemy import MetaData

from src.gaia_agent.tools.query.plan import QueryPlanError, validate_query_plan

from tests.gaia_agent.tools.query.support import (
    BIRDS_SCOPE_REGISTRY,
    add_filter,
    add_return_columns,
    blank_plan,
    make_table,
    validate_bird_plan,
)


def invalidate_plan(plan: dict[str, object], case: str) -> str:
    """Apply one representative strict-validation violation."""
    scope = plan["scope"]
    query = plan["query"]
    result = plan["result"]
    assert isinstance(scope, dict)
    assert isinstance(query, dict)
    assert isinstance(result, dict)

    if case == "root_property":
        plan["raw_sql"] = "SELECT *"
        return "unknown properties: raw_sql"
    if case == "unknown_collection":
        scope["collection"] = "Unknown Collection"
        return "unknown collection label"
    if case == "unknown_dataset":
        scope["dataset"] = "Raw Table Name"
        return "unknown dataset label"
    if case == "mismatched_dataset":
        scope["dataset"] = "Algae"
        return "does not belong to collection"
    if case == "missing_query_property":
        del query["filters"]
        return "missing required properties"
    if case == "unknown_query_property":
        query["raw_sql"] = {}
        return "unknown properties: query.raw_sql"

    filters = query["filters"]
    assert isinstance(filters, dict)
    if case == "unknown_header":
        filters["raw_sql"] = {"exact_any": ["SELECT *"]}
        return "unknown filter header"
    if case == "unknown_operator":
        filters["country"] = {
            "exact_any": ["Canada"],
            "sql": "LIKE '%'",
        }
        return "unknown properties: sql"
    if case == "missing_operator":
        filters["country"] = {}
        return "must contain at least one filter operator"
    if case == "exact_any_value_type":
        filters["country"] = {"exact_any": [1]}
        return "must contain only strings"
    if case == "unknown_return_column":
        query["return_columns"] = ["random()"]
        return "unknown return column"
    if case == "unknown_order_header":
        result["order_by"] = [{"header": "random()", "direction": "asc"}]
        return "unknown order_by header"
    if case == "order_direction":
        result["order_by"] = [
            {"header": "occurrenceID", "direction": "sideways"}
        ]
        return "direction must be 'asc' or 'desc'"
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    (
        "root_property",
        "unknown_collection",
        "unknown_dataset",
        "mismatched_dataset",
        "missing_query_property",
        "unknown_query_property",
        "unknown_header",
        "unknown_operator",
        "missing_operator",
        "exact_any_value_type",
        "unknown_return_column",
        "unknown_order_header",
        "order_direction",
    ),
)
def test_query_plan_validation_rejects_untrusted_shapes_identifiers_and_types(
    case: str,
) -> None:
    metadata = MetaData()
    table = make_table(
        metadata,
        "ctc_birds_validation",
        ("occurrenceID", "country", "year", "scientificName"),
    )
    plan = blank_plan()
    message = invalidate_plan(plan, case)

    with pytest.raises(QueryPlanError, match=message):
        validate_bird_plan(plan, table)


@pytest.mark.parametrize(
    ("physical_identifier", "absent_identifier"),
    (("id", "occurrenceID"), ("occurrenceID", "id")),
)
@pytest.mark.parametrize("operation", ("filters", "distinct_by", "order_by"))
def test_identifier_output_aliases_do_not_alias_query_operations(
    physical_identifier, absent_identifier, operation,
) -> None:
    table = make_table(MetaData(), "physical_identity", (physical_identifier,))
    raw = blank_plan(column_mode="selected")
    add_return_columns(raw, absent_identifier)
    if operation == "filters":
        add_filter(raw, absent_identifier, {"exact_any": ["sample"]})
        message = "unknown filter header"
    elif operation == "distinct_by":
        raw["result"][operation] = [absent_identifier]
        message = "unknown distinct_by header"
    else:
        raw["result"][operation] = [{"header": absent_identifier, "direction": "asc"}]
        message = "unknown order_by header"

    with pytest.raises(QueryPlanError, match=message):
        validate_bird_plan(raw, table)


@pytest.mark.parametrize("limit", (0, -1, True, 1.5, "5", None))
def test_result_limit_requires_a_positive_integer(limit: object) -> None:
    table = make_table(MetaData(), "limit_validation", ("occurrenceID",))
    plan = blank_plan()
    result = plan["result"]
    assert isinstance(result, dict)
    result["limit"] = limit

    with pytest.raises(QueryPlanError, match="result.limit must be a positive integer"):
        validate_bird_plan(plan, table)


def test_distinct_key_must_be_returned_but_may_be_a_proper_subset() -> None:
    metadata = MetaData()
    table = make_table(
        metadata,
        "distinct_subset_validation",
        ("occurrenceID", "scientificName", "locality"),
    )
    valid_plan = blank_plan(column_mode="selected")
    add_return_columns(valid_plan, "scientificName", "locality")
    valid_result = valid_plan["result"]
    assert isinstance(valid_result, dict)
    valid_result["distinct_by"] = ["scientificName"]

    validated = validate_bird_plan(valid_plan, table)

    assert validated.distinct_by == ("scientificName",)
    assert validated.return_columns == ("scientificName", "locality")

    invalid_plan = blank_plan(column_mode="selected")
    add_return_columns(invalid_plan, "locality")
    invalid_result = invalid_plan["result"]
    assert isinstance(invalid_result, dict)
    invalid_result["distinct_by"] = ["scientificName"]

    with pytest.raises(
        QueryPlanError,
        match=(
            r"result.distinct_by header\(s\) must also appear in "
            r"query.return_columns: scientificName"
        ),
    ):
        validate_bird_plan(invalid_plan, table)


def test_query_plan_reports_all_structural_omissions_together() -> None:
    metadata = MetaData()
    table = make_table(
        metadata,
        "ctc_birds_structural_retry",
        ("occurrenceID", "country", "locality", "scientificName"),
    )
    plan = blank_plan()
    query = plan["query"]
    result = plan["result"]
    assert isinstance(query, dict)
    assert isinstance(result, dict)
    del query["filters"]
    del result["order_by"]

    with pytest.raises(QueryPlanError) as first_error:
        validate_bird_plan(plan, table)

    first_message = str(first_error.value)
    assert "query.filters" in first_message
    assert "result.order_by" in first_message

    query["filters"] = {}

    with pytest.raises(QueryPlanError) as second_error:
        validate_bird_plan(plan, table)

    assert str(second_error.value) == (
        "QueryPlan is missing required properties: result.order_by"
    )


@pytest.mark.parametrize(
    ("scope", "expected_dataset_keys"),
    (
        ({"collection": "Cowan Tetrapod Collection"}, ("birds",)),
        ({"dataset": "Algae"}, ("algae",)),
        (
            {
                "collection": "Cowan Tetrapod Collection",
                "dataset": "Birds",
            },
            ("birds",),
        ),
    ),
)
def test_scope_accepts_collection_dataset_or_both(
    scope: dict[str, object],
    expected_dataset_keys: tuple[str, ...],
) -> None:
    metadata = MetaData()
    registry = {
        "birds": make_table(metadata, "scope_birds", ("occurrenceID",)),
        "algae": make_table(metadata, "scope_algae", ("occurrenceID",)),
    }
    plan = blank_plan()
    plan["scope"] = scope

    validated = validate_query_plan(
        plan,
        table_registry=registry,
        scope_registry=BIRDS_SCOPE_REGISTRY,
    )

    assert validated.dataset_keys == expected_dataset_keys


def test_scope_rejects_neither_collection_nor_dataset() -> None:
    metadata = MetaData()
    table = make_table(metadata, "scope_missing", ("occurrenceID",))
    plan = blank_plan(None, collection=None)

    with pytest.raises(
        QueryPlanError,
        match="scope must contain at least one of collection or dataset",
    ):
        validate_bird_plan(plan, table)


def test_dataset_only_scope_rejects_an_ambiguous_display_label() -> None:
    metadata = MetaData()
    registry = {
        "birds": make_table(metadata, "scope_shared_birds", ("occurrenceID",)),
        "algae": make_table(metadata, "scope_shared_algae", ("occurrenceID",)),
    }
    plan = blank_plan("Shared", collection=None)
    scope_registry = {
        "Collection A": {"Shared": "birds"},
        "Collection B": {"Shared": "algae"},
    }

    with pytest.raises(QueryPlanError, match="ambiguous dataset label 'Shared'"):
        validate_query_plan(
            plan,
            table_registry=registry,
            scope_registry=scope_registry,
        )


def test_validation_preserves_flat_header_and_registry_dataset_order() -> None:
    metadata = MetaData()
    registry = {
        "fish": make_table(
            metadata,
            "fc_fish_validation",
            ("occurrenceID", "country", "year", "scientificName"),
        ),
        "birds": make_table(
            metadata,
            "ctc_birds_validation_order",
            ("occurrenceID", "country", "year", "scientificName"),
        ),
    }
    scope_registry = {
        "Test Collection": {
            "Fish": "fish",
            "Birds": "birds",
        }
    }
    plan = blank_plan(None, collection="Test Collection", limit=7, column_mode="selected")
    add_filter(plan, "country", {"exact_any": ["Canada", "Japan"]})
    add_filter(plan, "year", {"exact_any": ["1900", "1999"]})
    add_return_columns(plan, "scientificName", "country")
    result = plan["result"]
    assert isinstance(result, dict)
    result["order_by"] = [{"header": "year", "direction": "desc"}]

    validated = validate_query_plan(
        plan,
        table_registry=registry,
        scope_registry=scope_registry,
    )

    assert validated.dataset_keys == ("fish", "birds")
    assert [item.header for item in validated.filters] == ["country", "year"]
    assert validated.return_columns == ("scientificName", "country")
    assert [(item.header, item.direction) for item in validated.order_by] == [
        ("year", "desc")
    ]
    assert validated.limit == 7
