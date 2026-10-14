"""Check complete SQL statistics, precision boundaries, and scope coverage."""

from __future__ import annotations

import asyncio
from functools import partial
from contextlib import asynccontextmanager
from itertools import islice
from types import SimpleNamespace

import pytest
from sqlalchemy import MetaData, create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError

from src.gaia_agent.tools.query.statistics_execution import (
    STATISTICS_BATCH_SIZE,
    _execute_statistics_on_connection,
    build_statistics_select,
    execute_statistics_plan,
)
from src.gaia_agent.tools.query.statistics_plan import validate_statistics_plan
from src.gaia_agent.tools.query.execution import QueryExecutionError

from tests.gaia_agent.tools.query.support import make_table
from tests.gaia_agent.tools.query.test_statistics_plan import statistics_plan
from tests.gaia_agent.tools.query.test_execution import insert_records, populated_table


def validate(raw, registry):
    return validate_statistics_plan(
        raw, table_registry=registry,
        scope_registry={"Collection": {key.title(): key for key in registry}},
    )


class SQLiteStatisticsCursor:
    def __init__(self, cursor) -> None:
        self.cursor = cursor

    async def fetchmany(self, size: int):
        return self.cursor.fetchmany(size)


class SQLiteStatisticsConnection:
    """Exercise actual SQL with bounded reads through an async cursor contract."""

    def __init__(self, connection):
        self.connection = connection

    @asynccontextmanager
    async def stream(self, statement, *, execution_options):
        with self.connection.execute(statement) as cursor:
            yield SQLiteStatisticsCursor(cursor)


def execute_sqlite_statistics(connection, plan, *, table_registry):
    return asyncio.run(_execute_statistics_on_connection(
        SQLiteStatisticsConnection(connection), plan, table_registry=table_registry,
    ))


def test_scalar_count_covers_all_matches_beyond_the_chat_row_limit():
    engine, table = populated_table(
        "all_matching_birds", ("occurrenceID", "scientificName"),
        tuple(
            {"occurrenceID": f"bird-{index:04}", "scientificName": "Corvus corax" if index < 733 else "Other"}
            for index in range(1000)
        ),
    )
    raw = statistics_plan(limit=1)
    raw["query"]["filters"] = {"scientificName": {"exact_any": ["Corvus corax"]}}
    plan = validate(raw, {"birds": table})
    try:
        with engine.connect() as connection:
            result = execute_sqlite_statistics(connection, plan, table_registry={"birds": table})
    finally:
        engine.dispose()
    assert result.rows == ({"record_count": 733},)
    assert result.database_truncated is False
    assert result.limit == 1
    assert result.queried_dataset_keys == ("birds",)


@pytest.mark.parametrize("limit", (None, 1))
def test_collection_statistics_combine_groups_and_distinct_inputs_before_limit(limit):
    metadata = MetaData()
    registry = {
        key: make_table(metadata, key, ("occurrenceID", "scientificName", "locality"))
        for key in ("birds", "mammals")
    }
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
    for key in registry:
        insert_records(engine, registry[key], tuple(
            {"occurrenceID": f"{key}-{index}", "scientificName": species, "locality": locality}
            for index, (species, locality) in enumerate((
                ("Corvus corax", "日本"), ("Corvus corax", ""), ("Other", "日本"),
            ))
        ))
    raw = statistics_plan(
        group_by=("scientificName",), limit=limit,
        metrics=[
            {"operation": "count", "alias": "record_count"},
            {"operation": "count_distinct", "alias": "locality_count", "header": "locality"},
        ],
    )
    raw["scope"] = {"collection": "Collection"}
    raw["result"]["order_by"] = [{"header": "record_count", "direction": "desc"}]
    plan = validate(raw, registry)
    try:
        with engine.connect() as connection:
            result = execute_sqlite_statistics(connection, plan, table_registry=registry)
    finally:
        engine.dispose()
    expected_rows = (
        {"scientificName": "Corvus corax", "record_count": 4, "locality_count": 2},
        {"scientificName": "Other", "record_count": 2, "locality_count": 1},
    )
    assert result.rows == expected_rows[:limit]
    assert result.database_truncated is (limit is not None)
    assert result.queried_dataset_keys == ("birds", "mammals")
    assert result.output_columns == ("scientificName", "record_count", "locality_count")
    compiled = build_statistics_select(
        tuple(registry.items()), plan, fetch_limit=None if limit is None else limit + 1,
    ).statement.compile(
        dialect=postgresql.dialect(),
    )
    sql = str(compiled)
    assert "UNION ALL" in sql
    assert "count(DISTINCT" in sql
    assert sql.count("LIMIT") == (0 if limit is None else 1)


@pytest.mark.parametrize("grouped", (False, True))
def test_empty_matching_input_distinguishes_scalar_zero_from_no_groups(grouped):
    engine, table = populated_table(
        "empty_birds", ("occurrenceID", "scientificName", "year"),
        ({"occurrenceID": "other", "scientificName": "Other", "year": "2000"},),
    )
    raw = statistics_plan(
        group_by=("scientificName",) if grouped else (),
        metrics=[
            {"operation": "count", "alias": "record_count"},
            {"operation": "count_distinct", "alias": "species_count", "header": "scientificName"},
            {"operation": "min", "alias": "first_name", "header": "scientificName"},
            {"operation": "sum", "alias": "year_sum", "header": "year", "value_type": "numeric"},
        ],
    )
    raw["query"]["filters"] = {"scientificName": {"exact_any": ["Missing"]}}
    try:
        with engine.connect() as connection:
            result = execute_sqlite_statistics(
                connection, validate(raw, {"birds": table}), table_registry={"birds": table},
            )
    finally:
        engine.dispose()
    assert result.rows == (() if grouped else ({
        "record_count": 0, "species_count": 0, "first_name": None, "year_sum": None,
    },))
    assert result.compatible_dataset_keys == ("birds",)
    assert result.database_truncated is False


def test_grouping_preserves_exact_empty_and_unicode_values_with_deterministic_ties():
    engine, table = populated_table(
        "group_values", ("occurrenceID", "scientificName"),
        tuple({"occurrenceID": str(index), "scientificName": value} for index, value in enumerate(("日本 🐦", "", " "))),
    )
    raw = statistics_plan(group_by=("scientificName",))
    raw["result"]["order_by"] = [{"header": "record_count", "direction": "desc"}]
    try:
        with engine.connect() as connection:
            result = execute_sqlite_statistics(
                connection, validate(raw, {"birds": table}), table_registry={"birds": table},
            )
    finally:
        engine.dispose()
    assert result.rows == tuple({"scientificName": value, "record_count": 1} for value in ("", " ", "日本 🐦"))


def test_numeric_metrics_ignore_nondecimals_and_keep_text_min_max_semantics():
    values = ("2", "10", " 20.5 ", "-.5", "", "unknown", "1e3", "9" * 129)
    engine, table = populated_table(
        "numeric_metrics", ("occurrenceID", "year"),
        tuple({"occurrenceID": str(index), "year": value} for index, value in enumerate(values)),
    )
    raw = statistics_plan(metrics=[
        {"operation": "count", "alias": "n"},
        {"operation": "min", "alias": "text_min", "header": "year"},
        *(
            {"operation": operation, "alias": operation, "header": "year", "value_type": "numeric"}
            for operation in ("min", "max", "sum", "avg")
        ),
    ])
    plan = validate(raw, {"birds": table})
    try:
        with engine.connect() as connection:
            result = execute_sqlite_statistics(connection, plan, table_registry={"birds": table})
    finally:
        engine.dispose()
    assert result.rows == ({"n": 8, "text_min": "", "min": "-0.5", "max": "20.5", "sum": "32.0", "avg": "8.0"},)
    compiled = build_statistics_select((("birds", table),), plan, fetch_limit=6).statement.compile(dialect=postgresql.dialect())
    assert "CASE WHEN" in str(compiled)
    assert " AS NUMERIC" in str(compiled)
    assert " AS TEXT" in str(compiled)


@pytest.mark.parametrize(
    ("query_change", "reason", "missing"),
    (
        ({"filters": {"year": {"gte": "1900"}}}, "missing_filter_columns", ("year",)),
        ({"group_by": ["year"]}, "missing_group_columns", ("year",)),
        ({"metrics": [{"operation": "max", "alias": "last_year", "header": "year"}]}, "missing_metric_columns", ("year",)),
    ),
)
def test_incompatible_scope_is_reported_without_fabricating_a_zero_total(query_change, reason, missing):
    metadata = MetaData()
    registry = {
        "birds": make_table(metadata, "missing_inputs", ("occurrenceID",)),
        "mammals": make_table(metadata, "registered_headers", ("occurrenceID", "year")),
    }
    raw = statistics_plan()
    raw["query"].update(query_change)
    plan = validate(raw, registry)
    result = execute_sqlite_statistics(None, plan, table_registry=registry)
    assert result.rows == ()
    assert result.selected_dataset_keys == ("birds",)
    assert result.compatible_dataset_keys == result.queried_dataset_keys == ()
    assert result.skipped_datasets[0].reason == reason
    assert result.skipped_datasets[0].missing_headers == missing


@pytest.mark.parametrize(
    ("limit", "returned_count", "truncated"),
    ((None, 250, False), (500, 250, False), (250, 250, False), (201, 201, True)),
)
def test_group_outputs_obey_only_the_requested_limit(limit, returned_count, truncated):
    engine, table = populated_table(
        "many_groups", ("occurrenceID", "scientificName"),
        tuple({"occurrenceID": str(index), "scientificName": f"species-{index:04}"} for index in range(250)),
    )
    plan = validate(statistics_plan(group_by=("scientificName",), limit=limit), {"birds": table})
    try:
        with engine.connect() as connection:
            result = execute_sqlite_statistics(connection, plan, table_registry={"birds": table})
    finally:
        engine.dispose()
    assert result.rows == tuple(
        {"scientificName": f"species-{index:04}", "record_count": 1}
        for index in range(returned_count)
    )
    assert result.limit == limit
    assert result.database_truncated is truncated
    assert result.queried_dataset_keys == ("birds",)


class StreamingStatisticsCursor:
    def __init__(self, connection, rows) -> None:
        self.connection = connection
        self.rows = rows

    async def fetchmany(self, size: int):
        self.connection.fetch_sizes.append(size)
        if self.connection.before_fetch is not None:
            await self.connection.before_fetch()
        return list(islice(self.rows, size))


class FetchProgress:
    """Measure event-loop progress between bounded statistics cursor reads."""

    def __init__(self, progress_at_fetch: list[int]) -> None:
        self.progress_at_fetch = progress_at_fetch
        self.progress = 0
        self.done = asyncio.Event()

    async def ticker(self) -> None:
        while not self.done.is_set():
            await asyncio.sleep(0)
            self.progress += 1

    async def before_fetch(self) -> None:
        self.progress_at_fetch.append(self.progress)


class StreamingStatisticsConnection:
    """Expose bounded reads without a buffered execute/fetchall path."""

    def __init__(self, rows):
        self.rows = rows
        self.events = []
        self.fetch_sizes = []
        self.before_fetch = None
        self.stream_error = None
        self.statement = None
        self.stream_options = None

    async def execution_options(self, **options):
        assert options == {"postgresql_readonly": True}
        self.events.append("readonly")
        return self

    @asynccontextmanager
    async def stream(self, statement, *, execution_options):
        self.events.append("stream")
        if self.stream_error is not None:
            raise self.stream_error
        self.statement = statement
        self.stream_options = execution_options
        rows = iter(self.rows)
        if statement._limit_clause is not None:
            rows = islice(rows, statement._limit_clause.value)

        try:
            yield StreamingStatisticsCursor(self, rows)
        finally:
            self.events.append("cursor_closed")


class StreamingStatisticsEngine:
    def __init__(self, rows):
        self.dialect = SimpleNamespace(name="postgresql")
        self.connection = StreamingStatisticsConnection(rows)

    @asynccontextmanager
    async def connect(self):
        self.connection.events.append("connect")
        try:
            yield self.connection
        finally:
            self.connection.events.append("closed")

    async def dispose(self):
        self.connection.events.append("disposed")


@pytest.mark.parametrize("failure", (None, "sql", "data", "dialect"))
def test_statistics_execution_owns_readonly_connection_and_disposes_on_every_exit(failure):
    table = make_table(MetaData(), "birds", ("occurrenceID",))
    plan = validate(statistics_plan(), {"birds": table})
    engine = StreamingStatisticsEngine((("invalid" if failure == "data" else 2,),))
    database_error = SQLAlchemyError("secret-token /srv/private/settings.env")
    if failure == "sql":
        engine.connection.stream_error = database_error
    if failure == "dialect":
        engine.dialect.name = "sqlite"

    if failure is None:
        result = asyncio.run(execute_statistics_plan(
            plan, table_registry={"birds": table}, engine_factory=lambda: engine,
        ))
        assert result.rows == ({"record_count": 2},)
        assert engine.connection.events == [
            "connect", "readonly", "stream", "cursor_closed", "closed", "disposed",
        ]
    else:
        with pytest.raises(QueryExecutionError) as error:
            asyncio.run(execute_statistics_plan(
                plan, table_registry={"birds": table}, engine_factory=lambda: engine,
            ))
        assert "private" not in str(error.value)
        assert error.value.code == ("invalid_query_data" if failure == "data" else "query_unavailable")
        if failure in {"sql", "dialect"}:
            assert str(error.value) == (
                "The biodiversity query service is unavailable. Please try again later."
            )
        if failure == "sql":
            assert error.value.__cause__ is database_error
        assert engine.connection.events[-1] == "disposed"
        if failure == "data":
            assert engine.connection.events[-3:] == ["cursor_closed", "closed", "disposed"]


async def _run_statistics_with_progress(progress_at_fetch, engine, plan, table):
    progress = FetchProgress(progress_at_fetch)
    engine.connection.before_fetch = progress.before_fetch
    ticker_task = asyncio.create_task(progress.ticker())
    try:
        return await execute_statistics_plan(
            plan, table_registry={"birds": table}, engine_factory=lambda: engine,
        )
    finally:
        progress.done.set()
        await ticker_task


def test_unlimited_groups_use_bounded_cursor_reads_and_yield_between_batches():
    table = make_table(MetaData(), "birds", ("occurrenceID", "scientificName"))
    plan = validate(statistics_plan(group_by=("scientificName",), limit=None), {"birds": table})
    count = STATISTICS_BATCH_SIZE * 2 + 7
    engine = StreamingStatisticsEngine((f"日本-{index}", 1) for index in range(count))
    progress_at_fetch = []

    result = asyncio.run(_run_statistics_with_progress(progress_at_fetch, engine, plan, table))
    assert len(result.rows) == count
    assert result.rows[0] == {"scientificName": "日本-0", "record_count": 1}
    assert result.rows[-1] == {"scientificName": f"日本-{count - 1}", "record_count": 1}
    assert result.limit is None and not result.database_truncated
    assert engine.connection.statement._limit_clause is None
    assert engine.connection.fetch_sizes == [STATISTICS_BATCH_SIZE] * 4
    assert engine.connection.stream_options == {
        "yield_per": STATISTICS_BATCH_SIZE, "max_row_buffer": STATISTICS_BATCH_SIZE,
    }
    assert progress_at_fetch[-1] > progress_at_fetch[1]
    assert engine.connection.events[-3:] == ["cursor_closed", "closed", "disposed"]


@pytest.mark.parametrize(
    "bad_row", ((None, 1), ("", True), ("", -1), ("", "1"), ("",), ("", 1, "extra")),
)
def test_invalid_streamed_statistics_rows_close_cursor_and_engine(bad_row):
    table = make_table(MetaData(), "birds", ("occurrenceID", "scientificName"))
    plan = validate(statistics_plan(group_by=("scientificName",)), {"birds": table})
    engine = StreamingStatisticsEngine((bad_row,))
    with pytest.raises(QueryExecutionError) as error:
        asyncio.run(execute_statistics_plan(
            plan, table_registry={"birds": table}, engine_factory=lambda: engine,
        ))
    assert error.value.code == "invalid_query_data"
    assert engine.connection.events[-3:] == ["cursor_closed", "closed", "disposed"]


async def _block_statistics_fetch(started):
    started.set()
    await asyncio.Event().wait()


async def _check_statistics_cancellation(engine, plan, table):
    started = asyncio.Event()

    block_fetch = partial(_block_statistics_fetch, started)

    engine.connection.before_fetch = block_fetch
    task = asyncio.create_task(execute_statistics_plan(
        plan, table_registry={"birds": table}, engine_factory=lambda: engine,
    ))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_statistics_cancellation_closes_pending_cursor_and_engine():
    table = make_table(MetaData(), "birds", ("occurrenceID",))
    plan = validate(statistics_plan(), {"birds": table})
    engine = StreamingStatisticsEngine(((1,),))

    asyncio.run(_check_statistics_cancellation(engine, plan, table))
    assert engine.connection.events[-3:] == ["cursor_closed", "closed", "disposed"]
