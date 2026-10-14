"""Offline contracts for bounded export cursors and transaction cleanup."""

from __future__ import annotations

import asyncio
from functools import partial
from contextlib import asynccontextmanager
from itertools import islice
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import MetaData
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError

from src.database.database import DatabaseConfigurationError
from src.gaia_agent.runtime.contracts import AgentCancellation
from src.gaia_agent.tools.query.execution import QueryExecutionError, stream_query_plans
from src.gaia_agent.tools.query.plan import validate_query_plan

from tests.gaia_agent.tools.query.support import add_filter, add_return_columns, blank_plan, make_table


class StreamingCursor:
    """Keep bounded cursor state separate from the connection lifecycle."""

    def __init__(self, connection, rows) -> None:
        self.connection = connection
        self.rows = rows

    async def fetchmany(self, size: int):
        self.connection.fetch_sizes.append(size)
        if self.connection.before_fetch is not None:
            await self.connection.before_fetch()
        if self.connection.fetch_error is not None:
            raise self.connection.fetch_error
        return list(islice(self.rows, size))


class StreamingConnection:
    """Expose only bounded cursor reads, retaining no materialized result set."""

    def __init__(self, rows_by_dataset):
        self.rows_by_dataset = rows_by_dataset
        self.events = []
        self.options = None
        self.statements = []
        self.fetch_sizes = []
        self.active_dataset = None
        self.options_error = None
        self.fetch_error = None
        self.before_fetch = None

    async def execution_options(self, **options):
        self.events.append("options")
        self.options = options
        if self.options_error is not None:
            raise self.options_error
        return self

    @asynccontextmanager
    async def begin(self):
        self.events.append("begin")
        try:
            yield
        except BaseException:
            self.events.append("rollback")
            raise
        else:
            self.events.append("commit")

    @asynccontextmanager
    async def stream(self, statement, *, execution_options):
        key = statement.get_final_froms()[0].name
        self.statements.append((key, statement, execution_options))
        assert self.events.count("begin") == 1
        assert "commit" not in self.events and "rollback" not in self.events
        assert self.active_dataset is None
        self.active_dataset = key
        self.events.append(f"open:{key}")
        rows = iter(self.rows_by_dataset[key])
        if statement._limit_clause is not None:
            rows = islice(rows, statement._limit_clause.value)

        try:
            yield StreamingCursor(self, rows)
        finally:
            self.events.append(f"close:{key}")
            self.active_dataset = None


class StreamingEngine:
    def __init__(self, rows_by_dataset):
        self.dialect = SimpleNamespace(name="postgresql")
        self.connection = StreamingConnection(rows_by_dataset)

    @asynccontextmanager
    async def connect(self):
        self.connection.events.append("connect")
        try:
            yield self.connection
        finally:
            self.connection.events.append("connection_closed")

    async def dispose(self):
        self.connection.events.append("disposed")


class ExportObserver:
    """Check bounded consumption and capture each worksheet's lifecycle."""

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.consumed = 0
        self.connection = None
        self.initialized = []
        self.completed = []
        self.batches = []
        self.samples = []

    def bird_rows(self):
        for index in range(5003):
            assert index - self.consumed < self.batch_size
            yield (f"bird-{index:04d}", "Vancouver", "Élan 🐦" if index else "")

    def on_batch(self, key, headers, rows) -> None:
        assert self.connection is not None
        if not rows:
            assert self.connection.active_dataset is None
            assert self.initialized == self.completed
            self.initialized.append(key)
            return
        assert len(rows) <= self.batch_size
        assert all(isinstance(row, tuple) for row in rows)
        self.batches.append((key, headers, len(rows)))
        if key == "birds":
            if self.consumed == 0:
                self.samples.extend(rows[:2])
            self.consumed += len(rows)

    async def on_complete(self, key) -> None:
        assert self.connection is not None
        assert self.connection.active_dataset is None
        assert "commit" not in self.connection.events
        await asyncio.sleep(0)
        self.completed.append(key)


def make_plans(headers_by_dataset, *, return_columns=(), filters=None):
    metadata = MetaData()
    registry = {
        key: make_table(metadata, key, headers)
        for key, headers in headers_by_dataset.items()
    }
    scope = {"Collection": {key.title(): key for key in registry}}
    plans = []
    for key in registry:
        raw_plan = blank_plan(
            key.title(), collection="Collection", limit=1,
            column_mode="selected" if return_columns else "default",
        )
        add_return_columns(raw_plan, *return_columns)
        for header, values in (filters or {}).items():
            add_filter(raw_plan, header, {"exact_any": values})
        plans.append(validate_query_plan(
            raw_plan, table_registry=registry, scope_registry=scope,
        ))
    return registry, plans


def test_streaming_uses_each_datasets_physical_identifier_and_preserves_values() -> None:
    metadata = MetaData()
    registry = {
        key: make_table(metadata, key, headers)
        for key, headers in (
            ("both", ("id", "occurrenceID", "locality")),
            ("only_id", ("id", "locality")),
            ("only_occurrence", ("occurrenceID", "locality")),
        )
    }
    plan = validate_query_plan(
        blank_plan(None, collection="Collection", column_mode="all"),
        table_registry=registry,
        scope_registry={"Collection": {key.title(): key for key in registry}},
    )
    engine = StreamingEngine({
        "both": (("", "0"), ("É-1", "  日本  ")),
        "only_id": (("id-α", ""),),
        "only_occurrence": (("occ-β", "=literal"),),
    })
    batches = []
    completed = []

    result = asyncio.run(stream_query_plans(
        (plan,),
        table_registry=registry,
        on_batch=lambda key, headers, rows: batches.append((key, headers, rows)),
        on_dataset_complete=completed.append,
        batch_size=1,
        engine_factory=lambda: engine,
    ))

    expected_headers = {
        "both": ("occurrenceID", "locality"),
        "only_id": ("id", "locality"),
        "only_occurrence": ("occurrenceID", "locality"),
    }
    assert [(key, headers) for key, headers, rows in batches if not rows] == list(
        expected_headers.items()
    )
    assert [(key, rows) for key, _headers, rows in batches if rows] == [
        ("both", (("", "0"),)),
        ("both", (("É-1", "  日本  "),)),
        ("only_id", (("id-α", ""),)),
        ("only_occurrence", (("occ-β", "=literal"),)),
    ]
    for key, statement, _options in engine.connection.statements:
        assert tuple(column.name for column in statement.selected_columns) == expected_headers[key]
    assert completed == list(registry)
    assert result.dataset_row_counts == (("both", 2), ("only_id", 1), ("only_occurrence", 1))
    assert result.record_count == 4
    assert result.database_truncated is False
    assert engine.connection.events[-3:] == ["commit", "connection_closed", "disposed"]


def test_export_streaming_reuses_literal_filters_and_numeric_ordering() -> None:
    table = make_table(MetaData(), "birds", ("occurrenceID", "scientificName", "year"))
    raw = blank_plan(column_mode="selected")
    add_return_columns(raw, "year")
    add_filter(raw, "scientificName", {"starts_with": "Larus_%"})
    add_filter(raw, "year", {"gte": "2", "lt": "21", "value_type": "numeric"})
    raw["result"]["order_by"] = [{"header": "year", "direction": "desc", "value_type": "numeric"}]
    plan = validate_query_plan(
        raw, table_registry={"birds": table},
        scope_registry={"Cowan Tetrapod Collection": {"Birds": "birds"}},
    )
    engine = StreamingEngine({"birds": (("r1", " 20.50 "), ("r2", "10"), ("r3", "2"))})
    batches = []
    result = asyncio.run(stream_query_plans(
        (plan,), table_registry={"birds": table},
        on_batch=lambda key, headers, rows: batches.extend(rows),
        batch_size=2, engine_factory=lambda: engine,
    ))
    assert batches == [("r1", " 20.50 "), ("r2", "10"), ("r3", "2")]
    assert result.dataset_row_counts == (("birds", 3),)
    compiled = engine.connection.statements[0][1].compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert "Larus_%" not in sql
    assert "LIKE" in sql and "ESCAPE '/'" in sql
    assert "CASE WHEN" in sql and " AS NUMERIC" in sql
    assert "DESC NULLS LAST" in sql
    assert "LIMIT" not in sql


@pytest.mark.parametrize("limit_scope", ("total", "per_dataset"))
def test_unlimited_export_streams_all_datasets_in_one_read_only_snapshot(
    limit_scope: str,
) -> None:
    registry, plans = make_plans(
        {
            "birds": ("occurrenceID", "locality", "scientificName"),
            "mammals": ("occurrenceID",),
            "empty": ("occurrenceID", "locality"),
            "fish": ("occurrenceID", "locality"),
        },
        return_columns=("locality", "scientificName"),
        filters={"locality": ["Vancouver"]},
    )
    batch_size = 37
    observer = ExportObserver(batch_size)
    engine = StreamingEngine({
        "birds": observer.bird_rows(), "empty": (), "fish": (("fish-1", "Vancouver"),),
    })
    observer.connection = engine.connection

    result = asyncio.run(stream_query_plans(
        plans,
        table_registry=registry,
        on_batch=observer.on_batch,
        on_dataset_complete=observer.on_complete,
        limit_scope=limit_scope,
        batch_size=batch_size,
        engine_factory=lambda: engine,
    ))

    assert observer.consumed == 5003
    assert observer.samples == [("bird-0000", "Vancouver", ""), ("bird-0001", "Vancouver", "Élan 🐦")]
    assert observer.initialized == observer.completed == ["birds", "empty", "fish"]
    assert result.selected_dataset_keys == ("birds", "mammals", "empty", "fish")
    assert result.compatible_dataset_keys == result.queried_dataset_keys == ("birds", "empty", "fish")
    assert result.dataset_row_counts == (("birds", 5003), ("empty", 0), ("fish", 1))
    assert result.record_count == 5004
    assert result.limit is None
    assert result.database_truncated is False
    assert result.skipped_datasets[0].dataset_key == "mammals"
    assert result.skipped_datasets[0].missing_headers == ("locality",)
    assert observer.batches[-1] == ("fish", ("occurrenceID", "locality"), 1)
    assert engine.connection.options == {
        "isolation_level": "REPEATABLE READ", "postgresql_readonly": True,
    }
    assert engine.connection.events[:3] == ["connect", "options", "begin"]
    assert engine.connection.events[-3:] == ["commit", "connection_closed", "disposed"]
    assert set(engine.connection.fetch_sizes) == {batch_size}
    for _key, statement, options in engine.connection.statements:
        assert statement._limit_clause is None
        assert options == {"yield_per": batch_size, "max_row_buffer": batch_size}


def _record_export_batch(callbacks, exported, key, headers, rows):
    assert headers == ("occurrenceID",)
    callbacks.append((key, "rows" if rows else "start"))
    exported.extend((key, row) for row in rows)


@pytest.mark.parametrize(
    ("limit", "counts", "queried", "truncated"),
    (
        (2, (2, 0, 0, 0), ("birds", "fish"), True),
        (3, (2, 1, 0, 0), ("birds", "fish"), True),
        (4, (2, 2, 0, 0), ("birds", "fish", "empty", "later"), True),
        (5, (2, 2, 0, 1), ("birds", "fish", "empty", "later"), False),
        (6, (2, 2, 0, 1), ("birds", "fish", "empty", "later"), False),
    ),
)
def test_export_limit_probes_across_datasets_and_completes_unqueried_sheets(
    limit, counts, queried, truncated,
) -> None:
    keys = ("birds", "fish", "empty", "later")
    registry, plans = make_plans({key: ("occurrenceID",) for key in keys})
    engine = StreamingEngine({
        key: tuple((f"{key}-{index}",) for index in range(count))
        for key, count in zip(keys, (2, 2, 0, 1), strict=True)
    })
    callbacks = []
    exported = []

    on_batch = partial(_record_export_batch, callbacks, exported)

    result = asyncio.run(stream_query_plans(
        plans,
        table_registry=registry,
        on_batch=on_batch,
        on_dataset_complete=lambda key: callbacks.append((key, "done")),
        row_limit=limit,
        batch_size=2,
        engine_factory=lambda: engine,
    ))

    assert result.dataset_row_counts == tuple(zip(keys, counts, strict=True))
    assert result.queried_dataset_keys == queried
    assert result.database_truncated is truncated
    assert result.record_count == len(exported) == min(limit, 5)
    assert result.limit == limit
    assert [entry for entry in callbacks if entry[1] != "rows"] == [
        entry for key in keys for entry in ((key, "start"), (key, "done"))
    ]
    assert engine.connection.events[-3:] == ["commit", "connection_closed", "disposed"]


def _count_export_batch(exported_counts, callbacks, key, headers, rows):
    assert headers == ("occurrenceID",)
    exported_counts[key] += len(rows)
    if not rows:
        callbacks.append((key, "start"))


@pytest.mark.parametrize(
    ("limit", "available", "expected", "truncated"),
    (
        (100, (101, 101, 101, 101), (100, 100, 100, 100), True),
        (100, (100, 100, 100, 100), (100, 100, 100, 100), False),
        (100, (0, 101, 100, 3), (0, 100, 100, 3), True),
        (100, (7, 101, 100, 3), (7, 100, 100, 3), True),
        (250, (251, 251, 250, 9), (250, 250, 250, 9), True),
    ),
)
def test_per_dataset_limit_resets_for_every_dataset_in_a_collection(
    limit, available, expected, truncated,
) -> None:
    keys = ("birds", "fish", "mammals", "algae")
    registry, _plans = make_plans({key: ("occurrenceID",) for key in keys})
    collection_plan = validate_query_plan(
        blank_plan(None, collection="Collection", limit=1),
        table_registry=registry,
        scope_registry={"Collection": {key.title(): key for key in keys}},
    )
    engine = StreamingEngine({
        key: tuple((f"{key}-{index}",) for index in range(count))
        for key, count in zip(keys, available, strict=True)
    })
    exported_counts = {key: 0 for key in keys}
    callbacks = []

    on_batch = partial(_count_export_batch, exported_counts, callbacks)

    result = asyncio.run(stream_query_plans(
        (collection_plan,),
        table_registry=registry,
        on_batch=on_batch,
        on_dataset_complete=lambda key: callbacks.append((key, "done")),
        row_limit=limit,
        limit_scope="per_dataset",
        batch_size=31,
        engine_factory=lambda: engine,
    ))

    assert result.selected_dataset_keys == result.queried_dataset_keys == keys
    assert result.dataset_row_counts == tuple(zip(keys, expected, strict=True))
    assert tuple(exported_counts.values()) == expected
    assert result.record_count == sum(expected)
    assert result.database_truncated is truncated
    assert result.limit == limit
    assert callbacks == [
        entry for key in keys for entry in ((key, "start"), (key, "done"))
    ]
    assert [statement._limit_clause.value for _key, statement, _options
            in engine.connection.statements] == [limit + 1] * len(keys)
    assert engine.connection.events.count("connect") == 1
    assert engine.connection.events[-3:] == ["commit", "connection_closed", "disposed"]


@pytest.mark.parametrize("bad_row", ((None,), (23,), (), ("id", "extra")))
def test_invalid_source_rows_close_cursor_and_roll_back(bad_row) -> None:
    registry, plans = make_plans({"birds": ("occurrenceID",)})
    engine = StreamingEngine({"birds": (bad_row,)})
    batches = []
    with pytest.raises(QueryExecutionError) as error:
        asyncio.run(stream_query_plans(
            plans, table_registry=registry,
            on_batch=lambda *batch: batches.append(batch),
            engine_factory=lambda: engine,
        ))
    assert error.value.code == "invalid_query_data"
    assert batches == [("birds", ("occurrenceID",), ())]
    assert engine.connection.events[-4:] == [
        "close:birds", "rollback", "connection_closed", "disposed",
    ]


@pytest.mark.parametrize("stage", ("options", "fetch"))
def test_database_errors_are_sanitized_and_resources_close(stage) -> None:
    registry, plans = make_plans({"birds": ("occurrenceID",)})
    engine = StreamingEngine({"birds": (("bird-1",),)})
    setattr(engine.connection, f"{stage}_error", SQLAlchemyError("PRIVATE DATABASE DETAIL"))
    with pytest.raises(QueryExecutionError) as error:
        asyncio.run(stream_query_plans(
            plans, table_registry=registry, on_batch=lambda *_args: None,
            engine_factory=lambda: engine,
        ))
    assert error.value.code == "query_unavailable"
    assert "PRIVATE" not in str(error.value)
    assert engine.connection.active_dataset is None
    assert engine.connection.events[-2:] == ["connection_closed", "disposed"]
    if stage == "fetch":
        assert "rollback" in engine.connection.events


async def _fail_completion_consumer(stage, _key):
    if stage == "completion":
        await asyncio.sleep(0)
        raise ValueError("Workbook write failed")


def _fail_batch_consumer(stage, _key, _headers, rows):
    if stage == "batch" and rows:
        raise ValueError("Workbook write failed")


@pytest.mark.parametrize("stage", ("batch", "completion"))
def test_consumer_errors_preserve_the_error_and_release_resources(stage) -> None:
    registry, plans = make_plans({"birds": ("occurrenceID",)})
    engine = StreamingEngine({"birds": (("bird-1",),)})

    on_batch = partial(_fail_batch_consumer, stage)

    on_complete = partial(_fail_completion_consumer, stage)

    with pytest.raises(ValueError, match="Workbook write failed"):
        asyncio.run(stream_query_plans(
            plans, table_registry=registry,
            on_batch=on_batch, on_dataset_complete=on_complete,
            engine_factory=lambda: engine,
        ))
    assert engine.connection.events[-4:] == [
        "close:birds", "rollback", "connection_closed", "disposed",
    ]


def _cancel_after_batch(batches, cancellation, _key, _headers, rows):
    if rows:
        batches.append(rows)
        cancellation.cancel()


def test_requested_cancellation_stops_before_another_batch() -> None:
    registry, plans = make_plans({"birds": ("occurrenceID",)})
    engine = StreamingEngine({"birds": ((str(index),) for index in range(5000))})
    cancellation = AgentCancellation()
    batches = []
    completed = []

    on_batch = partial(_cancel_after_batch, batches, cancellation)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(stream_query_plans(
            plans, table_registry=registry, on_batch=on_batch,
            on_dataset_complete=completed.append,
            cancellation=cancellation, batch_size=7,
            engine_factory=lambda: engine,
        ))
    assert len(batches) == 1 and len(batches[0]) == 7
    assert engine.connection.fetch_sizes == [7]
    assert completed == []
    assert engine.connection.events[-4:] == [
        "close:birds", "rollback", "connection_closed", "disposed",
    ]


async def _wait_before_recording_batch(stage, started, release, received, _key, _headers, rows):
    if (stage == "initialization" and not rows) or (stage == "batch" and rows):
        started.set()
        await release.wait()
    await asyncio.sleep(0)
    received.append(rows)


async def _check_batch_callback_order(stage, received, plans, registry, engine):
    started = asyncio.Event()
    release = asyncio.Event()

    on_batch = partial(_wait_before_recording_batch, stage, started, release, received)

    task = asyncio.create_task(stream_query_plans(
        plans, table_registry=registry, on_batch=on_batch, batch_size=2,
        engine_factory=lambda: engine,
    ))
    await asyncio.wait_for(started.wait(), timeout=2)
    assert engine.connection.fetch_sizes == ([] if stage == "initialization" else [2])
    assert received == ([] if stage == "initialization" else [()])
    release.set()
    result = await task
    assert result.record_count == 5


@pytest.mark.parametrize("stage", ("initialization", "batch"))
def test_async_batch_callbacks_finish_before_the_next_cursor_read(stage) -> None:
    registry, plans = make_plans({"birds": ("occurrenceID",)})
    engine = StreamingEngine({"birds": ((str(index),) for index in range(5))})
    received = []

    asyncio.run(_check_batch_callback_order(stage, received, plans, registry, engine))
    assert received == [(), (("0",), ("1",)), (("2",), ("3",)), (("4",),)]
    assert engine.connection.events[-3:] == ["commit", "connection_closed", "disposed"]


async def _block_batch_callback(stage, started, _key, _headers, rows):
    if (stage == "initialization" and not rows) or (stage == "batch" and rows):
        started.set()
        await asyncio.Event().wait()


async def _check_batch_callback_cancellation(stage, plans, registry, engine):
    started = asyncio.Event()

    on_batch = partial(_block_batch_callback, stage, started)

    task = asyncio.create_task(stream_query_plans(
        plans, table_registry=registry, on_batch=on_batch,
        engine_factory=lambda: engine,
    ))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("stage", ("initialization", "batch"))
def test_cancelling_an_awaited_batch_callback_releases_resources(stage) -> None:
    registry, plans = make_plans({"birds": ("occurrenceID",)})
    engine = StreamingEngine({"birds": (("bird-1",),)})

    asyncio.run(_check_batch_callback_cancellation(stage, plans, registry, engine))
    assert engine.connection.events[-3:] == ["rollback", "connection_closed", "disposed"]
    assert ("close:birds" in engine.connection.events) is (stage == "batch")


async def _block_stream_callback(started, *_args):
    started.set()
    await asyncio.Event().wait()


async def _check_stream_cancellation(stage):
    registry, plans = make_plans({"birds": ("occurrenceID",)})
    engine = StreamingEngine({"birds": (("bird-1",),)})
    started = asyncio.Event()

    block = partial(_block_stream_callback, started)

    if stage == "fetch":
        engine.connection.before_fetch = block
    task = asyncio.create_task(stream_query_plans(
        plans, table_registry=registry, on_batch=lambda *_args: None,
        on_dataset_complete=block if stage == "completion" else None,
        engine_factory=lambda: engine,
    ))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.connection.events[-4:] == [
        "close:birds", "rollback", "connection_closed", "disposed",
    ]


@pytest.mark.parametrize("stage", ("fetch", "completion"))
def test_task_cancellation_releases_cursor_transaction_and_engine(stage) -> None:
    asyncio.run(_check_stream_cancellation(stage))


def test_preexisting_cancellation_prevents_engine_creation() -> None:
    cancellation = AgentCancellation()
    cancellation.cancel()
    factory = Mock()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(stream_query_plans(
            (), table_registry={}, on_batch=lambda *_args: None,
            engine_factory=factory, cancellation=cancellation,
        ))
    factory.assert_not_called()


@pytest.mark.parametrize(
    "arguments", ({"row_limit": 0}, {"row_limit": True}, {"batch_size": 0}, {"batch_size": True}),
)
def test_invalid_export_limits_prevent_engine_creation(arguments) -> None:
    factory = Mock()
    with pytest.raises(ValueError):
        asyncio.run(stream_query_plans(
            (), table_registry={}, on_batch=lambda *_args: None,
            engine_factory=factory, **arguments,
        ))
    factory.assert_not_called()


@pytest.mark.parametrize("limit_scope", ("unknown", None, True, []))
def test_invalid_limit_scope_prevents_engine_creation_and_callbacks(limit_scope) -> None:
    factory = Mock()
    on_batch = Mock()
    on_complete = Mock()
    with pytest.raises(ValueError, match="limit_scope must be"):
        asyncio.run(stream_query_plans(
            (), table_registry={}, on_batch=on_batch,
            on_dataset_complete=on_complete,
            engine_factory=factory, limit_scope=limit_scope,
        ))
    factory.assert_not_called()
    on_batch.assert_not_called()
    on_complete.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    (DatabaseConfigurationError("Missing database setting"), SQLAlchemyError("PRIVATE CONFIG")),
)
def test_export_engine_setup_errors_are_mapped(failure) -> None:
    with pytest.raises(QueryExecutionError) as error:
        asyncio.run(stream_query_plans(
            (), table_registry={}, on_batch=lambda *_args: None,
            engine_factory=Mock(side_effect=failure),
        ))
    assert error.value.code == "query_unavailable"
    assert "PRIVATE" not in str(error.value)


def test_export_rejects_and_disposes_non_postgresql_engine() -> None:
    engine = StreamingEngine({})
    engine.dialect.name = "sqlite"
    with pytest.raises(QueryExecutionError, match="query service is unavailable"):
        asyncio.run(stream_query_plans(
            (), table_registry={}, on_batch=lambda *_args: None,
            engine_factory=lambda: engine,
        ))
    assert engine.connection.events == ["disposed"]
