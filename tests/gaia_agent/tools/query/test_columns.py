"""Shared column selection modes and per-dataset resolution contracts."""

from __future__ import annotations

import pytest
from sqlalchemy import MetaData

from src.gaia_agent.tools.query.columns import DEFAULT_RETURN_COLUMNS, preferred_identifier
from src.gaia_agent.tools.query.execution import build_dataset_select, resolve_query_datasets
from src.gaia_agent.tools.query.plan import (
    QueryPlanError,
    query_plan_for_dataset,
    validate_query_plan,
)

from tests.gaia_agent.tools.query.support import add_filter, add_return_columns, blank_plan, make_table, validate_bird_plan


def test_defaults_are_catalog_independent_darwin_core_headers() -> None:
    assert DEFAULT_RETURN_COLUMNS == (
        "scientificName", "vernacularName", "kingdom", "phylum", "class",
        "order", "family",
    )


@pytest.mark.parametrize("omit_mode", (False, True))
def test_defaults_append_requested_and_query_fields_in_deterministic_order(omit_mode) -> None:
    table = make_table(MetaData(), "birds", (
        "occurrenceID", "id", "country", "recordedBy", "vernacularName",
        "scientificName", "kingdom", "phylum", "class", "locality",
        "verbatimLocality", "genus", "eventDate", "unused",
    ))
    raw = blank_plan()
    if omit_mode:
        del raw["query"]["column_mode"]
    add_return_columns(raw, "locality", "verbatimLocality", "scientificName")
    add_filter(raw, "recordedBy", {"exact_any": ["Élodie 日本語", " "]})
    add_filter(raw, "country", {"exact_any": [""]})
    raw["result"]["distinct_by"] = ["genus"]
    raw["result"]["order_by"] = [{"header": "eventDate", "direction": "desc"}]

    plan = validate_bird_plan(raw, table)

    assert plan.column_mode == "default"
    assert plan.return_columns == (
        "scientificName", "vernacularName", "kingdom", "phylum", "class",
        "locality", "verbatimLocality", "country", "recordedBy", "genus", "eventDate",
    )
    assert [(item.header, item.exact_any) for item in plan.filters] == [
        ("country", ("",)), ("recordedBy", ("Élodie 日本語", " ")),
    ]
    assert plan.dataset_return_columns == ()
    assert build_dataset_select(table, plan, fetch_limit=6).output_headers == (
        "occurrenceID", *plan.return_columns,
    )


def test_selected_columns_preserve_exact_order_without_implicit_query_fields() -> None:
    table = make_table(MetaData(), "birds", (
        "occurrenceID", "country", "scientificName", "locality", "eventDate",
    ))
    raw = blank_plan(column_mode="selected")
    add_return_columns(raw, "locality", "scientificName")
    add_filter(raw, "country", {"exact_any": ["Canada"]})
    raw["result"]["distinct_by"] = ["scientificName"]
    raw["result"]["order_by"] = [{"header": "eventDate", "direction": "desc"}]

    plan = validate_bird_plan(raw, table)

    assert plan.return_columns == ("locality", "scientificName")
    assert build_dataset_select(table, plan, fetch_limit=6).output_headers == (
        "occurrenceID", "locality", "scientificName",
    )


@pytest.mark.parametrize("omit_mode", (False, True))
def test_all_literal_selects_physical_headers_and_allows_an_implicit_distinct_key(
    omit_mode,
) -> None:
    physical = ("id", "locality", "occurrenceID", "scientificName", "country")
    table = make_table(MetaData(), "birds", physical)
    raw = blank_plan(column_mode="all")
    if omit_mode:
        del raw["query"]["column_mode"]
    raw["result"]["distinct_by"] = ["scientificName"]

    plan = validate_bird_plan(raw, table)

    assert plan.column_mode == "all"
    assert plan.return_columns == (
        "occurrenceID", "locality", "scientificName", "country",
    )
    assert build_dataset_select(table, plan, fetch_limit=6).output_headers == (
        "occurrenceID", "locality", "scientificName", "country",
    )


def test_collection_defaults_resolve_independently() -> None:
    metadata = MetaData()
    headers = (
        "occurrenceID", "id", "vernacularName", "scientificName", "kingdom",
        "phylum", "class", "order", "family", "locality",
    )
    registry = {key: make_table(metadata, key, headers) for key in ("birds", "fish")}
    raw = blank_plan(None, collection="Collection")
    add_return_columns(raw, "locality")
    plan = validate_query_plan(
        raw, table_registry=registry,
        scope_registry={"Collection": {"Birds": "birds", "Fish": "fish"}},
    )

    birds = query_plan_for_dataset(plan, "birds")
    fish = query_plan_for_dataset(plan, "fish")

    assert birds.return_columns == (
        "scientificName", "vernacularName", "kingdom", "phylum", "class",
        "order", "family", "locality",
    )
    assert fish.return_columns == birds.return_columns
    assert plan.return_columns == birds.return_columns
    assert plan.dataset_return_columns == (
        ("birds", birds.return_columns), ("fish", fish.return_columns),
    )
    assert birds.dataset_keys == ("birds",)
    assert birds.dataset_return_columns == ()
    assert query_plan_for_dataset(birds, "birds") is birds
    with pytest.raises(ValueError, match="outside this query plan's scope"):
        query_plan_for_dataset(plan, "mammals")


def test_query_tables_require_a_physical_identifier() -> None:
    table = make_table(MetaData(), "missing_identifier", ("scientificName",))
    message = "must contain an 'occurrenceID' or 'id' column"
    with pytest.raises(ValueError, match=message):
        preferred_identifier(("scientificName",))
    with pytest.raises(QueryPlanError, match=f"birds: A dataset {message}"):
        validate_bird_plan(blank_plan(), table)

    valid_table = make_table(MetaData(), "birds", ("id", "scientificName"))
    plan = validate_bird_plan(blank_plan(), valid_table)
    with pytest.raises(ValueError, match=message):
        build_dataset_select(table, plan, fetch_limit=6)


def test_unavailable_defaults_are_omitted_and_required_filter_skips_remain() -> None:
    metadata = MetaData()
    registry = {
        "birds": make_table(metadata, "birds", ("occurrenceID", "scientificName", "country")),
        "fish": make_table(metadata, "fish", ("occurrenceID", "kingdom")),
    }
    raw = blank_plan(None, collection="Collection")
    add_filter(raw, "country", {"exact_any": ["Canada"]})
    plan = validate_query_plan(
        raw, table_registry=registry,
        scope_registry={"Collection": {"Birds": "birds", "Fish": "fish"}},
    )

    assert query_plan_for_dataset(plan, "birds").return_columns == ("scientificName", "country")
    assert query_plan_for_dataset(plan, "fish").return_columns == ("kingdom",)
    compatible, skipped = resolve_query_datasets(plan, table_registry=registry)
    assert tuple(key for key, _table in compatible) == ("birds",)
    assert skipped[0].dataset_key == "fish"
    assert skipped[0].missing_headers == ("country",)


@pytest.mark.parametrize(
    ("mode", "columns", "message"),
    (
        ("all", [], "requires query.return_columns: 'all'"),
        ("all", ["scientificName"], "requires query.return_columns: 'all'"),
        ("default", "all", "conflicts with query.column_mode: 'default'"),
        ("selected", "all", "conflicts with query.column_mode: 'selected'"),
        ("selected", [], "must not be empty"),
        ("unknown", [], "column_mode must be"),
        (None, [], "column_mode must be"),
        ([], [], "column_mode must be"),
        ("default", ["scientific_name"], "unknown return column"),
        ("default", "scientificName", "array of exact header names or the literal string 'all'"),
        ("default", "ALL", "array of exact header names or the literal string 'all'"),
        ("default", None, "array of exact header names or the literal string 'all'"),
        ("default", {}, "array of exact header names or the literal string 'all'"),
        ("default", True, "array of exact header names or the literal string 'all'"),
    ),
)
def test_column_modes_reject_ambiguous_or_untrusted_selections(mode, columns, message) -> None:
    table = make_table(MetaData(), "birds", ("occurrenceID", "scientificName"))
    raw = blank_plan()
    raw["query"]["column_mode"] = mode
    raw["query"]["return_columns"] = columns
    with pytest.raises(QueryPlanError, match=message):
        validate_bird_plan(raw, table)


@pytest.mark.parametrize("mode", ("default", "all", "selected"))
def test_distinct_selection_still_requires_a_single_dataset(mode) -> None:
    metadata = MetaData()
    registry = {
        key: make_table(metadata, key, ("occurrenceID", "scientificName"))
        for key in ("birds", "fish")
    }
    raw = blank_plan(None, collection="Collection", column_mode=mode)
    if mode == "selected":
        add_return_columns(raw, "scientificName")
    raw["result"]["distinct_by"] = ["scientificName"]
    with pytest.raises(QueryPlanError, match="requires exactly one selected dataset"):
        validate_query_plan(
            raw, table_registry=registry,
            scope_registry={"Collection": {"Birds": "birds", "Fish": "fish"}},
        )
