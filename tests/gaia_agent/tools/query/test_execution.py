"""Occurrence SQL compilation, execution, and database configuration tests."""

from __future__ import annotations

import asyncio
from functools import partial
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import MetaData, Table, create_engine, insert
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool

import src.gaia_agent.tools.query.execution as query_execution_module
from src.database.database import DatabaseConfigurationError
from src.gaia_agent.tools.concurrency import WorkloadBusyError, WorkloadLimiter
from src.gaia_agent.tools.query.statistics_execution import execute_statistics_plan
from src.gaia_agent.tools.query.execution import (
    MAX_CHAT_RECORDS,
    QueryExecutionError,
    _execute_on_connection,
    build_dataset_select,
    create_query_engine,
    exact_table_columns,
    execute_query_plan,
    managed_query_engine,
    stream_query_plans,
)
from src.gaia_agent.tools.query.plan import validate_query_plan

from tests.gaia_agent.tools.query.support import (
    BIRDS_SCOPE_REGISTRY,
    add_filter,
    add_return_columns,
    blank_plan,
    make_table,
    validate_bird_plan,
)


def insert_records(
    engine: Engine,
    table: Table,
    records: Sequence[Mapping[str, str]],
) -> None:
    """Insert source-header dictionaries through the table's Python keys."""
    parameters = [
        {
            column.key: record[column.name]
            for column in table.columns
        }
        for record in records
    ]
    with engine.begin() as connection:
        connection.execute(insert(table), parameters)


def populated_table(
    name: str,
    headers: Sequence[str],
    records: Sequence[Mapping[str, str]],
) -> tuple[Engine, Table]:
    """Create one in-memory table populated with source-header records."""
    metadata = MetaData()
    table = make_table(metadata, name, headers)
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
    insert_records(engine, table, records)
    return engine, table


def test_query_engine_uses_async_psycopg_without_a_shared_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        query_execution_module,
        "load_database_url",
        lambda: URL.create("postgresql", username="gaia", database="gaia"),
    )
    engine = create_query_engine()
    try:
        assert engine.url.drivername == "postgresql+psycopg"
        assert engine.dialect.is_async
        assert isinstance(engine.pool, NullPool)
    finally:
        asyncio.run(engine.dispose())


async def _reject_failed_engine(engine_factory):
    async with managed_query_engine(engine_factory):
        pytest.fail("A failed engine must not become available.")


@pytest.mark.parametrize("failure_type", (DatabaseConfigurationError, SQLAlchemyError))
def test_query_engine_setup_errors_hide_configuration_and_backend_details(
    failure_type: type[Exception],
) -> None:
    failure = failure_type("POSTGRES_PORT: secret-token /srv/private/settings.env")

    engine_factory = Mock(side_effect=failure)

    with pytest.raises(QueryExecutionError) as captured:
        asyncio.run(_reject_failed_engine(engine_factory))

    assert captured.value.code == "query_unavailable"
    assert str(captured.value) == (
        "The biodiversity query service is unavailable. Please try again later."
    )
    assert captured.value.__cause__ is failure


async def _reject_unsupported_engine(engine):
    async with managed_query_engine(lambda: engine):
        pytest.fail("An unsupported engine must not become available.")


def test_query_engine_dialect_is_private_and_engine_is_disposed() -> None:
    engine = SimpleNamespace(
        dialect=SimpleNamespace(name="private_database_backend"),
        dispose=AsyncMock(),
    )

    with pytest.raises(QueryExecutionError) as captured:
        asyncio.run(_reject_unsupported_engine(engine))

    assert captured.value.code == "query_unavailable"
    assert str(captured.value) == (
        "The biodiversity query service is unavailable. Please try again later."
    )
    engine.dispose.assert_awaited_once()


async def _open_disposal_failure_engine(engine):
    async with managed_query_engine(lambda: engine):
        pass


def test_query_engine_teardown_failure_hides_backend_details() -> None:
    failure = SQLAlchemyError("private database address /srv/private/settings.env")
    engine = SimpleNamespace(
        dialect=SimpleNamespace(name="postgresql"),
        dispose=AsyncMock(side_effect=failure),
    )

    with pytest.raises(QueryExecutionError) as captured:
        asyncio.run(_open_disposal_failure_engine(engine))

    assert captured.value.code == "query_unavailable"
    assert str(captured.value) == (
        "The biodiversity query service is unavailable. Please try again later."
    )
    assert captured.value.__cause__ is failure
    engine.dispose.assert_awaited_once()


async def _check_busy_database_admission(limiter, operation, factory):
    async with limiter.slot():
        with pytest.raises(WorkloadBusyError):
            if operation == "occurrences":
                await execute_query_plan(
                    None, table_registry={}, engine_factory=factory,
                    database_limiter=limiter,
                )
            elif operation == "statistics":
                await execute_statistics_plan(
                    None, table_registry={}, engine_factory=factory,
                    database_limiter=limiter,
                )
            else:
                await stream_query_plans(
                    (), table_registry={}, on_batch=lambda *_args: None,
                    engine_factory=factory, database_limiter=limiter,
                )


@pytest.mark.parametrize("operation", ("occurrences", "statistics", "export"))
def test_busy_database_admission_prevents_engine_creation(operation: str) -> None:
    limiter = WorkloadLimiter(1, 0, label="database")
    factory = Mock()

    asyncio.run(_check_busy_database_admission(limiter, operation, factory))
    factory.assert_not_called()


async def _use_managed_engine(limiter, events, factory, name):
    async with managed_query_engine(factory, database_limiter=limiter):
        events.append(name)


def _record_second_engine(events, second_engine):
    events.append("second_factory")
    return second_engine


def _record_first_engine(events, first_engine):
    events.append("first_factory")
    return first_engine


async def _dispose_after_release(events, disposing, release_disposal):
    events.append("first_dispose_started")
    disposing.set()
    await release_disposal.wait()
    events.append("first_dispose_finished")


async def _check_disposal_admission(events, limiter):
    disposing = asyncio.Event()
    release_disposal = asyncio.Event()

    dispose_first = partial(_dispose_after_release, events, disposing, release_disposal)

    first_engine = SimpleNamespace(
        dialect=SimpleNamespace(name="postgresql"), dispose=dispose_first,
    )
    second_engine = SimpleNamespace(
        dialect=SimpleNamespace(name="postgresql"), dispose=AsyncMock(),
    )

    first_factory = partial(_record_first_engine, events, first_engine)

    second_factory = partial(_record_second_engine, events, second_engine)

    use = partial(_use_managed_engine, limiter, events)

    first_task = asyncio.create_task(use(first_factory, "first_body"))
    await asyncio.wait_for(disposing.wait(), timeout=2)
    second_task = asyncio.create_task(use(second_factory, "second_body"))
    await asyncio.sleep(0)
    assert "second_factory" not in events
    release_disposal.set()
    await asyncio.gather(first_task, second_task)
    second_engine.dispose.assert_awaited_once()


def test_database_admission_remains_occupied_until_engine_disposal_finishes() -> None:
    limiter = WorkloadLimiter(1, 1, label="database")
    events = []

    asyncio.run(_check_disposal_admission(events, limiter))
    assert events == [
        "first_factory", "first_body", "first_dispose_started",
        "first_dispose_finished", "second_factory", "second_body",
    ]


async def _check_failed_engine_admission(failure, factory, limiter):
    expected = asyncio.CancelledError if failure == "cancellation" else RuntimeError
    with pytest.raises(expected):
        async with managed_query_engine(factory, database_limiter=limiter):
            if failure == "body":
                raise RuntimeError("consumer failure")
            if failure == "cancellation":
                raise asyncio.CancelledError
    async with limiter.slot():
        pass


@pytest.mark.parametrize("failure", ("factory", "body", "dispose", "cancellation"))
def test_query_engine_failures_release_database_admission(failure: str) -> None:
    limiter = WorkloadLimiter(1, 0, label="database")
    database_error = SQLAlchemyError("database failure")
    engine = SimpleNamespace(
        dialect=SimpleNamespace(name="postgresql"),
        dispose=AsyncMock(side_effect=database_error if failure == "dispose" else None),
    )
    factory = Mock(side_effect=database_error) if failure == "factory" else lambda: engine

    asyncio.run(_check_failed_engine_admission(failure, factory, limiter))
    if failure != "factory":
        engine.dispose.assert_awaited_once()


@pytest.mark.parametrize("operation", ("occurrences", "export"))
def test_query_execution_failures_hide_backend_details(operation: str) -> None:
    failure = SQLAlchemyError("secret-token /srv/private/settings.env")
    engine = SimpleNamespace(
        dialect=SimpleNamespace(name="postgresql"),
        connect=Mock(side_effect=failure),
        dispose=AsyncMock(),
    )
    table = make_table(MetaData(), "private_errors", ("occurrenceID",))
    plan = validate_bird_plan(blank_plan(), table)
    if operation == "occurrences":
        execution = execute_query_plan(
            plan, table_registry={"birds": table}, engine_factory=lambda: engine,
        )
    else:
        execution = stream_query_plans(
            (plan,), table_registry={"birds": table}, on_batch=lambda *_args: None,
            engine_factory=lambda: engine,
        )

    with pytest.raises(QueryExecutionError) as captured:
        asyncio.run(execution)

    assert captured.value.code == "query_unavailable"
    assert str(captured.value) == (
        "The biodiversity query service is unavailable. Please try again later."
    )
    assert captured.value.__cause__ is failure
    engine.dispose.assert_awaited_once()


@pytest.mark.parametrize("fetch_limit", (None, 6))
def test_postgresql_compilation_uses_physical_columns_and_bound_literals(
    fetch_limit: int | None,
) -> None:
    metadata = MetaData()
    table = make_table(
        metadata,
        "physical_columns",
        ("occurrenceID", "country", "year", "scientificName"),
    )
    malicious_literal = "Canada' OR 1=1 --"
    plan = blank_plan(column_mode="selected")
    add_filter(plan, "country", {"exact_any": [malicious_literal]})
    add_return_columns(plan, "scientificName")
    validated = validate_bird_plan(plan, table)

    dataset_select = build_dataset_select(table, validated, fetch_limit=fetch_limit)
    compiled = dataset_select.statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)

    assert exact_table_columns(table) == {
        "occurrenceID": table.c.occurrence_id,
        "country": table.c.country,
        "year": table.c.year,
        "scientificName": table.c.scientific_name,
    }
    assert dataset_select.output_headers == ("occurrenceID", "scientificName")
    assert malicious_literal not in sql
    assert any(
        value == [malicious_literal] for value in compiled.params.values()
    )
    assert "CASE WHEN" not in sql
    assert "CAST" not in sql
    assert "ORDER BY physical_columns.\"occurrenceID\" ASC" in sql
    assert ("LIMIT" in sql) is (fetch_limit is not None)


def test_sqlite_preserves_literal_case_sensitive_equality() -> None:
    engine, table = populated_table(
        "literal_equality",
        ("occurrenceID", "country"),
        (
            {"occurrenceID": "exact", "country": "Canada"},
            {"occurrenceID": "different-case", "country": "canada"},
            {"occurrenceID": "partial", "country": "Canada West"},
        ),
    )
    plan = blank_plan(limit=5, column_mode="selected")
    add_filter(plan, "country", {"exact_any": ["Canada"]})
    add_return_columns(plan, "country")

    with engine.connect() as connection:
        result = _execute_on_connection(
            connection,
            validate_bird_plan(plan, table),
            table_registry={"birds": table},
        )

    assert result.records == (
        {
            "dataset_key": "birds",
            "occurrenceID": "exact",
            "country": "Canada",
        },
    )
    assert result.database_truncated is False


@pytest.mark.parametrize("mode", ("default", "selected", "all"))
@pytest.mark.parametrize(
    ("identifiers", "identifier", "requested_identifier"),
    (
        (("id", "occurrenceID"), "occurrenceID", "id"),
        (("id",), "id", "occurrenceID"),
        (("occurrenceID",), "occurrenceID", "id"),
    ),
)
def test_query_output_uses_the_preferred_physical_identifier_in_every_mode(
    mode, identifiers, identifier, requested_identifier,
) -> None:
    headers = (*identifiers, "scientificName", "locality", "unused")
    source_rows = (
        {
            **{header: "alternate-a" for header in identifiers},
            identifier: "é-2",
            "scientificName": "",
            "locality": "",
            "unused": "=literal",
        },
        {
            **{header: "alternate-z" for header in identifiers},
            identifier: "é-1",
            "scientificName": "Larus 日本語",
            "locality": "  Vancouver 🐦  ",
            "unused": "0",
        },
    )
    engine, table = populated_table("preferred_identity", headers, source_rows)
    raw = blank_plan(column_mode=mode)
    if mode != "all":
        add_return_columns(raw, requested_identifier, "locality")
    plan = validate_bird_plan(raw, table)
    expected_headers = (
        (identifier, "locality") if mode == "selected"
        else (identifier, "scientificName", "locality", "unused") if mode == "all"
        else (identifier, "scientificName", "locality")
    )

    try:
        with engine.connect() as connection:
            result = _execute_on_connection(connection, plan, table_registry={"birds": table})
    finally:
        engine.dispose()

    assert result.records == tuple(
        {"dataset_key": "birds", **{header: row[header] for header in expected_headers}}
        for row in reversed(source_rows)
    )
    assert tuple(result.records[0]) == ("dataset_key", *expected_headers)
    assert identifier in plan.return_columns
    assert result.database_truncated is False
    compiled = build_dataset_select(table, plan, fetch_limit=None)
    assert compiled.output_headers == expected_headers
    sql = str(compiled.statement.compile(dialect=postgresql.dialect()))
    quoted_identifier = '"occurrenceID"' if identifier == "occurrenceID" else "id"
    assert f"ORDER BY preferred_identity.{quoted_identifier} ASC" in sql


@pytest.mark.parametrize("explicit_order", (False, True))
def test_hidden_id_remains_available_to_filters_distinct_and_ordering(explicit_order) -> None:
    engine, table = populated_table(
        "distinct_hidden_id",
        ("occurrenceID", "id", "locality"),
        (
            {"occurrenceID": "é-2", "id": "z", "locality": "second"},
            {"occurrenceID": "é-1", "id": "z", "locality": "  日本  "},
            {"occurrenceID": "bird-a", "id": "a", "locality": ""},
            {"occurrenceID": "excluded", "id": "other", "locality": "ignored"},
        ),
    )
    raw = blank_plan(column_mode="selected")
    add_return_columns(raw, "id", "locality")
    add_filter(raw, "id", {"exact_any": ["a", "z"]})
    raw["result"]["distinct_by"] = ["id"]
    if explicit_order:
        raw["result"]["order_by"] = [{"header": "id", "direction": "desc"}]
    plan = validate_bird_plan(raw, table)

    try:
        with engine.connect() as connection:
            result = _execute_on_connection(connection, plan, table_registry={"birds": table})
    finally:
        engine.dispose()

    expected = (
        {"dataset_key": "birds", "occurrenceID": "bird-a", "locality": ""},
        {"dataset_key": "birds", "occurrenceID": "é-1", "locality": "  日本  "},
    )
    assert result.records == (tuple(reversed(expected)) if explicit_order else expected)
    assert plan.return_columns == ("occurrenceID", "locality")
    assert plan.distinct_by == ("id",)
    assert plan.filters[0].header == "id"


def test_distinct_representatives_use_id_when_occurrence_id_is_absent() -> None:
    engine, table = populated_table(
        "id_only_distinct",
        ("id", "scientificName", "locality"),
        (
            {"id": "é-2", "scientificName": "Corvus corax", "locality": "later"},
            {"id": "é-1", "scientificName": "Corvus corax", "locality": "  日本  "},
            {"id": "z-1", "scientificName": "Zosterops japonicus", "locality": ""},
        ),
    )
    raw = blank_plan(column_mode="default")
    add_return_columns(raw, "locality")
    raw["result"]["distinct_by"] = ["scientificName"]
    plan = validate_bird_plan(raw, table)

    try:
        with engine.connect() as connection:
            result = _execute_on_connection(connection, plan, table_registry={"birds": table})
    finally:
        engine.dispose()

    assert result.records == (
        {"dataset_key": "birds", "id": "é-1", "scientificName": "Corvus corax", "locality": "  日本  "},
        {"dataset_key": "birds", "id": "z-1", "scientificName": "Zosterops japonicus", "locality": ""},
    )


def test_heterogeneous_schemas_skip_missing_filters_and_omit_missing_returns() -> None:
    metadata = MetaData()
    mammals = make_table(
        metadata,
        "mammals_heterogeneous",
        ("occurrenceID", "scientificName"),
    )
    fish = make_table(
        metadata,
        "fish_heterogeneous",
        ("occurrenceID", "country"),
    )
    birds = make_table(
        metadata,
        "birds_heterogeneous",
        ("occurrenceID", "country", "scientificName"),
    )
    registry = {"mammals": mammals, "fish": fish, "birds": birds}
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
    insert_records(
        engine,
        mammals,
        ({"occurrenceID": "mammal-1", "scientificName": "Ursus americanus"},),
    )
    insert_records(
        engine,
        fish,
        ({"occurrenceID": "fish-1", "country": "Canada"},),
    )
    insert_records(
        engine,
        birds,
        (
            {
                "occurrenceID": "bird-1",
                "country": "Canada",
                "scientificName": "Corvus corax",
            },
        ),
    )
    scope_registry = {
        "Test Collection": {
            "Mammals": "mammals",
            "Fish": "fish",
            "Birds": "birds",
        }
    }
    plan = blank_plan(None, collection="Test Collection", limit=5, column_mode="selected")
    add_filter(plan, "country", {"exact_any": ["Canada"]})
    add_return_columns(plan, "scientificName")

    with engine.connect() as connection:
        result = _execute_on_connection(
            connection,
            validate_query_plan(
                plan,
                table_registry=registry,
                scope_registry=scope_registry,
            ),
            table_registry=registry,
        )

    assert result.selected_dataset_keys == ("mammals", "fish", "birds")
    assert result.compatible_dataset_keys == ("fish", "birds")
    assert result.queried_dataset_keys == ("fish", "birds")
    assert result.skipped_datasets[0].dataset_key == "mammals"
    assert result.skipped_datasets[0].reason == "missing_filter_columns"
    assert result.skipped_datasets[0].missing_headers == ("country",)
    assert result.records == (
        {"dataset_key": "fish", "occurrenceID": "fish-1"},
        {
            "dataset_key": "birds",
            "occurrenceID": "bird-1",
            "scientificName": "Corvus corax",
        },
    )
    assert all("record" not in row and "chunks" not in row for row in result.records)


def test_collection_execution_uses_each_dataset_defaults_and_filter_columns() -> None:
    metadata = MetaData()
    headers = (
        "occurrenceID", "vernacularName", "scientificName", "kingdom",
        "phylum", "class", "order", "family", "country",
    )
    registry = {
        key: make_table(metadata, key, headers) for key in ("birds", "fish")
    }
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
    record = {
        "vernacularName": "",
        "scientificName": "Corvus corax",
        "kingdom": "Animalia",
        "phylum": "Chordata",
        "class": "Aves",
        "order": "Passeriformes",
        "family": "Corvidae",
        "country": "日本",
    }
    for key, table in registry.items():
        insert_records(engine, table, (
            {"occurrenceID": f"{key}-1", **record},
            {"occurrenceID": f"{key}-2", **record, "country": "Canada"},
        ))
    raw = blank_plan(None, collection="Collection")
    add_filter(raw, "country", {"exact_any": ["日本"]})
    plan = validate_query_plan(
        raw,
        table_registry=registry,
        scope_registry={"Collection": {"Birds": "birds", "Fish": "fish"}},
    )

    try:
        with engine.connect() as connection:
            result = _execute_on_connection(connection, plan, table_registry=registry)
    finally:
        engine.dispose()

    assert result.records == (
        {
            "dataset_key": "birds", "occurrenceID": "birds-1",
            "scientificName": "Corvus corax", "vernacularName": "",
            "kingdom": "Animalia", "phylum": "Chordata", "class": "Aves",
            "order": "Passeriformes", "family": "Corvidae",
            "country": "日本",
        },
        {
            "dataset_key": "fish", "occurrenceID": "fish-1",
            "scientificName": "Corvus corax", "vernacularName": "",
            "kingdom": "Animalia", "phylum": "Chordata", "class": "Aves",
            "order": "Passeriformes", "family": "Corvidae", "country": "日本",
        },
    )
    assert tuple(result.records[0]) == (
        "dataset_key", "occurrenceID", "scientificName", "vernacularName",
        "kingdom", "phylum", "class", "order", "family", "country",
    )
    assert tuple(result.records[1]) == (
        "dataset_key", "occurrenceID", "scientificName", "vernacularName",
        "kingdom", "phylum", "class", "order", "family", "country",
    )
    assert result.queried_dataset_keys == ("birds", "fish")
    assert result.database_truncated is False


def test_collection_query_shares_one_limit_and_stops_after_the_probe() -> None:
    metadata = MetaData()
    registry = {
        key: make_table(metadata, key, ("occurrenceID",))
        for key in ("birds", "fish", "mammals")
    }
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
    for key, occurrence_ids in (
        ("birds", ("bird-1",)),
        ("fish", ("fish-2", "fish-1")),
        ("mammals", ("mammal-1",)),
    ):
        insert_records(
            engine,
            registry[key],
            tuple({"occurrenceID": value} for value in occurrence_ids),
        )
    plan = validate_query_plan(
        blank_plan(None, collection="Test Collection", limit=2),
        table_registry=registry,
        scope_registry={"Test Collection": {key.title(): key for key in registry}},
    )

    with engine.connect() as connection:
        result = _execute_on_connection(connection, plan, table_registry=registry)

    assert result.records == (
        {"dataset_key": "birds", "occurrenceID": "bird-1"},
        {"dataset_key": "fish", "occurrenceID": "fish-1"},
    )
    assert result.selected_dataset_keys == ("birds", "fish", "mammals")
    assert result.compatible_dataset_keys == ("birds", "fish", "mammals")
    assert result.queried_dataset_keys == ("birds", "fish")
    assert result.skipped_datasets == ()
    assert result.database_truncated is True


def _capture_dataset_select_build(
    fetch_limits,
    original_build,
    selected_table: Table,
    selected_plan,
    *,
    fetch_limit: int,
):
    fetch_limits.append(fetch_limit)
    return original_build(
        selected_table,
        selected_plan,
        fetch_limit=fetch_limit,
    )


@pytest.mark.parametrize("requested_limit", (2, 25, 200, 500))
def test_execution_uses_occurrence_id_tie_breaker_and_limit_plus_one(
    monkeypatch: pytest.MonkeyPatch,
    requested_limit: int,
) -> None:
    engine, table = populated_table(
        "ordered_limited",
        ("occurrenceID", "country"),
        tuple(
            {"occurrenceID": f"{index:03d}", "country": "Canada"}
            for index in range(requested_limit + 1, 0, -1)
        ),
    )
    plan = blank_plan(limit=requested_limit, column_mode="selected")
    add_return_columns(plan, "country")
    result_section = plan["result"]
    assert isinstance(result_section, dict)
    result_section["order_by"] = [{"header": "country", "direction": "asc"}]
    validated = validate_bird_plan(plan, table)
    fetch_limits: list[int] = []
    original_build = query_execution_module.build_dataset_select

    capture_build = partial(_capture_dataset_select_build, fetch_limits, original_build)

    monkeypatch.setattr(query_execution_module, "build_dataset_select", capture_build)

    with engine.connect() as connection:
        result = _execute_on_connection(
            connection,
            validated,
            table_registry={"birds": table},
        )

    effective_limit = min(requested_limit, MAX_CHAT_RECORDS)
    assert fetch_limits == [effective_limit + 1]
    assert [row["occurrenceID"] for row in result.records] == [
        f"{index:03d}" for index in range(1, effective_limit + 1)
    ]
    assert result.limit == effective_limit
    assert result.database_truncated is True
    compiled = original_build(
        table, validated, fetch_limit=requested_limit + 1
    ).statement.compile(
        dialect=postgresql.dialect()
    )
    assert (
        "ORDER BY ordered_limited.country ASC, "
        'ordered_limited."occurrenceID" ASC'
    ) in str(compiled)


def test_chat_cap_is_shared_across_dataset_tables() -> None:
    metadata = MetaData()
    registry = {
        key: make_table(metadata, key, ("occurrenceID",))
        for key in ("birds", "fish")
    }
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
    for key, table in registry.items():
        insert_records(
            engine,
            table,
            tuple({"occurrenceID": f"{key}-{index:03d}"} for index in range(150)),
        )
    plan = validate_query_plan(
        blank_plan(None, collection="Collection", limit=1000),
        table_registry=registry,
        scope_registry={"Collection": {key.title(): key for key in registry}},
    )

    with engine.connect() as connection:
        result = _execute_on_connection(connection, plan, table_registry=registry)

    assert len(result.records) == MAX_CHAT_RECORDS
    assert result.records[-1] == {
        "dataset_key": "fish", "occurrenceID": "fish-049"
    }
    assert result.limit == MAX_CHAT_RECORDS
    assert result.database_truncated is True
    engine.dispose()


def test_exactly_200_chat_rows_are_not_marked_truncated() -> None:
    engine, table = populated_table(
        "exact_cap",
        ("occurrenceID",),
        tuple({"occurrenceID": str(index)} for index in range(MAX_CHAT_RECORDS)),
    )
    plan = validate_bird_plan(blank_plan(limit=1000), table)

    with engine.connect() as connection:
        result = _execute_on_connection(
            connection, plan, table_registry={"birds": table}
        )

    assert len(result.records) == MAX_CHAT_RECORDS
    assert result.database_truncated is False
    engine.dispose()


def test_distinct_keys_choose_representative_rows_and_ignore_blank_keys() -> None:
    engine, table = populated_table(
        "distinct_algae_species",
        ("occurrenceID", "genus", "specificEpithet", "locality"),
        (
            {
                "occurrenceID": "1",
                "genus": "",
                "specificEpithet": "blank",
                "locality": "Blank genus",
            },
            {
                "occurrenceID": "2",
                "genus": "Fucus",
                "specificEpithet": "",
                "locality": "Blank epithet",
            },
            {
                "occurrenceID": "3",
                "genus": "Fucus",
                "specificEpithet": "vesiculosus",
                "locality": "Vancouver",
            },
            {
                "occurrenceID": "4",
                "genus": "Fucus",
                "specificEpithet": "vesiculosus",
                "locality": "Victoria",
            },
            {
                "occurrenceID": "5",
                "genus": "Laminaria",
                "specificEpithet": "digitata",
                "locality": "Tofino",
            },
            {
                "occurrenceID": "6",
                "genus": "Ulva",
                "specificEpithet": "lactuca",
                "locality": "Nanaimo",
            },
        ),
    )
    plan = blank_plan("Algae", collection="Herbarium", limit=2, column_mode="selected")
    add_return_columns(plan, "genus", "specificEpithet", "locality")
    result_section = plan["result"]
    assert isinstance(result_section, dict)
    result_section["distinct_by"] = ["genus", "specificEpithet"]
    validated = validate_query_plan(
        plan,
        table_registry={"algae": table},
        scope_registry=BIRDS_SCOPE_REGISTRY,
    )

    with engine.connect() as connection:
        result = _execute_on_connection(
            connection,
            validated,
            table_registry={"algae": table},
        )

    assert result.records == (
        {
            "dataset_key": "algae",
            "occurrenceID": "3",
            "genus": "Fucus",
            "specificEpithet": "vesiculosus",
            "locality": "Vancouver",
        },
        {
            "dataset_key": "algae",
            "occurrenceID": "5",
            "genus": "Laminaria",
            "specificEpithet": "digitata",
            "locality": "Tofino",
        },
    )
    assert result.database_truncated is True


def test_fixed_species_can_return_distinct_localities() -> None:
    engine, table = populated_table(
        "distinct_bird_localities",
        ("occurrenceID", "scientificName", "locality"),
        (
            {
                "occurrenceID": "bird-1",
                "scientificName": "Corvus corax",
                "locality": "Vancouver",
            },
            {
                "occurrenceID": "bird-2",
                "scientificName": "Corvus corax",
                "locality": "Vancouver",
            },
            {
                "occurrenceID": "bird-3",
                "scientificName": "Corvus corax",
                "locality": "Victoria",
            },
            {
                "occurrenceID": "bird-4",
                "scientificName": "Passer domesticus",
                "locality": "Tofino",
            },
        ),
    )
    plan = blank_plan(limit=10, column_mode="selected")
    add_filter(
        plan,
        "scientificName",
        {"exact_any": ["Corvus corax"]},
    )
    add_return_columns(plan, "scientificName", "locality")
    result_section = plan["result"]
    assert isinstance(result_section, dict)
    result_section["distinct_by"] = ["locality"]
    validated = validate_bird_plan(plan, table)

    with engine.connect() as connection:
        result = _execute_on_connection(
            connection,
            validated,
            table_registry={"birds": table},
        )

    assert [record["locality"] for record in result.records] == [
        "Vancouver",
        "Victoria",
    ]
    assert all(
        record["scientificName"] == "Corvus corax"
        for record in result.records
    )


def test_distinct_ordering_selects_and_orders_representative_occurrences() -> None:
    engine, table = populated_table(
        "distinct_bird_species",
        ("occurrenceID", "scientificName", "locality", "eventDate"),
        (
            {
                "occurrenceID": "bird-1",
                "scientificName": "Corvus corax",
                "locality": "Older raven",
                "eventDate": "2020-01-01",
            },
            {
                "occurrenceID": "bird-2",
                "scientificName": "Corvus corax",
                "locality": "Newer raven",
                "eventDate": "2022-01-01",
            },
            {
                "occurrenceID": "bird-3",
                "scientificName": "Passer domesticus",
                "locality": "Older sparrow",
                "eventDate": "2021-01-01",
            },
            {
                "occurrenceID": "bird-4",
                "scientificName": "Passer domesticus",
                "locality": "Newer sparrow",
                "eventDate": "2023-01-01",
            },
        ),
    )
    plan = blank_plan(limit=5, column_mode="selected")
    add_return_columns(plan, "scientificName", "locality", "eventDate")
    result_section = plan["result"]
    assert isinstance(result_section, dict)
    result_section["distinct_by"] = ["scientificName"]
    result_section["order_by"] = [{"header": "eventDate", "direction": "desc"}]
    validated = validate_bird_plan(plan, table)

    with engine.connect() as connection:
        result = _execute_on_connection(
            connection,
            validated,
            table_registry={"birds": table},
        )

    assert [record["occurrenceID"] for record in result.records] == [
        "bird-4",
        "bird-2",
    ]
    assert [record["locality"] for record in result.records] == [
        "Newer sparrow",
        "Newer raven",
    ]
    compiled = build_dataset_select(
        table,
        validated,
        fetch_limit=6,
    ).statement.compile(dialect=postgresql.dialect())
    compiled_sql = str(compiled)
    assert "row_number() OVER (PARTITION BY" in compiled_sql
    assert "SELECT DISTINCT" not in compiled_sql
    unlimited_sql = str(build_dataset_select(
        table, validated, fetch_limit=None,
    ).statement.compile(dialect=postgresql.dialect()))
    assert "row_number() OVER (PARTITION BY" in unlimited_sql
    assert "LIMIT" not in unlimited_sql


def test_complete_occurrence_is_explicitly_enumerated_as_physical_headers() -> None:
    physical_headers = (
        "occurrenceID",
        "country",
        "locality",
        "scientificName",
    )
    engine, table = populated_table(
        "complete_occurrence",
        physical_headers,
        (
            {
                "occurrenceID": "complete-1",
                "country": "Canada",
                "locality": "Vancouver Island",
                "scientificName": "Corvus caurinus",
            },
        ),
    )
    plan = blank_plan(limit=1, column_mode="selected")
    add_filter(plan, "occurrenceID", {"exact_any": ["complete-1"]})
    add_return_columns(plan, *physical_headers)
    validated = validate_bird_plan(plan, table)

    with engine.connect() as connection:
        result = _execute_on_connection(
            connection,
            validated,
            table_registry={"birds": table},
        )

    assert set(validated.return_columns) == set(physical_headers)
    assert result.records == (
        {
            "dataset_key": "birds",
            "occurrenceID": "complete-1",
            "country": "Canada",
            "locality": "Vancouver Island",
            "scientificName": "Corvus caurinus",
        },
    )
