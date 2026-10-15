"""Deterministic tests for Gaia's model-visible domain tools."""

from __future__ import annotations

import asyncio
from functools import partial
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

import src.gaia_agent.tools.registry as tools_module
import src.gaia_agent.tools.query.budget as budget_module
from src.database.database import DatabaseConfigurationError
from src.database.darwin_core import load_simple_darwin_core_profile
from src.database.datasets import DATASETS
from src.database.schema_profiles import write_dataset_schema_profiles
from src.database.tablesetup import TABLE_MODELS
from src.gaia_agent.tools.query.execution import (
    DatasetSkip,
    MAX_CHAT_RECORDS,
    QUERY_UNAVAILABLE_MESSAGE,
    QueryExecutionResult,
    build_dataset_select,
)
from src.gaia_agent.tools.query.statistics_execution import StatisticsExecutionResult
from src.gaia_agent.tools.query.statistics_plan import PlannedMetric
from src.gaia_agent.tools import GaiaToolset
from src.gaia_agent.tools.concurrency import WorkloadLimiter
from src.gaia_agent.tools.definitions import build_tool_definitions


OCCURRENCE_HEADERS = load_simple_darwin_core_profile().headers
from src.gaia_agent.tools.file_generation.workers import ExcelExportWorkers
from src.gaia_agent.tools.observations import (
    project_query_statistics_observation,
    project_query_occurrences_observation,
)
from src.gaia_agent.runtime.contracts import ToolExecutionError


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TOOL_NAMES = [
    "retrieve_schema",
    "search_darwin_core_terms",
    "get_darwin_core_terms",
    "query_occurrences",
    "query_statistics",
    "generate_excel",
]


def test_tool_definitions_use_only_the_active_scope_registry() -> None:
    definitions = build_tool_definitions({
        "Arbitrary Collection": {"Arbitrary Dataset": "arbitrary_dataset"},
    })
    by_name = {definition.name: definition for definition in definitions}
    serialized = json.dumps([
        {
            "description": definition.description,
            "parameters": definition.parameters,
        }
        for definition in definitions
    ])

    schema_properties = by_name["retrieve_schema"].parameters["properties"]
    assert "Arbitrary Dataset (arbitrary_dataset)" in (
        schema_properties["datasets"]["description"]
    )
    assert "Arbitrary Collection" in schema_properties["collections"]["description"]
    assert "Vancouver" not in serialized
    assert "identifier fields" not in by_name["retrieve_schema"].description
    assert "non-whitespace value" in by_name["retrieve_schema"].description
    assert "empty or whitespace-only" in by_name["retrieve_schema"].description
    assert (
        f"{MAX_CHAT_RECORDS}-row allowance"
        in by_name["query_occurrences"].description
    )
    assert (
        f"{MAX_CHAT_RECORDS}-row allowance"
        in by_name["query_occurrences"].parameters["properties"]["result"]
        ["properties"]["limit"]["description"]
    )
    assert (
        "maxItems"
        not in by_name["generate_excel"].parameters["properties"]["queries"]
    )


@pytest.fixture(autouse=True)
def isolate_example_dataset_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Keep deterministic tool tests independent of the local generated catalog."""
    profile_root = tmp_path_factory.mktemp("schema-profile")
    write_dataset_schema_profiles(
        profile_root,
        {
            dataset.key: {header: 1 for header in OCCURRENCE_HEADERS}
            for dataset in DATASETS
        },
        datasets=DATASETS,
    )
    original_load = GaiaToolset.load.__func__

    def load_example(cls, **kwargs):
        kwargs["repository_root"] = profile_root
        kwargs.setdefault("datasets", DATASETS)
        kwargs.setdefault("table_models", TABLE_MODELS)
        return original_load(cls, **kwargs)

    monkeypatch.setattr(GaiaToolset, "load", classmethod(load_example))


@pytest.fixture
def toolset() -> GaiaToolset:
    """Load the trusted JSON schema and physical table registry once."""
    return GaiaToolset.load(datasets=DATASETS, table_models=TABLE_MODELS)


def _handler(toolset: GaiaToolset, name: str):
    return next(
        tool.handler
        for tool in toolset.registered_tools()
        if tool.definition.name == name
    )


async def _check_shared_database_admission(limiter, configured, tool_name, arguments):
    async with limiter.slot():
        for _ in range(2):
            with pytest.raises(ToolExecutionError) as captured:
                await _handler(configured, tool_name)(arguments)
            assert captured.value.code == "server_busy"
    async with limiter.slot():
        pass


@pytest.mark.parametrize("tool_name", (
    "query_occurrences", "query_statistics", "generate_excel",
))
def test_database_admission_is_shared_across_tools_and_registrations(
    toolset: GaiaToolset, tmp_path: Path, tool_name: str,
) -> None:
    factory = MagicMock(side_effect=AssertionError("Busy work must not open a database"))
    limiter = WorkloadLimiter(1, 0, label="Queries")
    configured = replace(
        toolset, engine_factory=factory, database_limiter=limiter,
        export_directory=tmp_path,
    )
    arguments = {
        "query_occurrences": _blank_plan(),
        "query_statistics": {
            "scope": {"dataset": "Birds"},
            "query": {
                "filters": {}, "group_by": [],
                "metrics": [{"operation": "count", "alias": "record_count"}],
            },
            "result": {"order_by": [], "limit": None},
        },
        "generate_excel": _export_arguments(),
    }[tool_name]

    asyncio.run(_check_shared_database_admission(limiter, configured, tool_name, arguments))
    factory.assert_not_called()
    assert list(tmp_path.iterdir()) == []


async def _check_shared_export_admission(workers, configured):
    async with workers.slot():
        for _ in range(2):
            with pytest.raises(ToolExecutionError) as captured:
                await _handler(configured, "generate_excel")(_export_arguments())
            assert captured.value.code == "server_busy"


def test_export_admission_is_shared_across_registrations(
    toolset: GaiaToolset, tmp_path: Path,
) -> None:
    factory = MagicMock(side_effect=AssertionError("Busy exports must not query"))
    workers = ExcelExportWorkers(max_workers=1, max_waiting=0)
    configured = replace(
        toolset, engine_factory=factory, export_workers=workers,
        export_directory=tmp_path,
    )

    asyncio.run(_check_shared_export_admission(workers, configured))
    factory.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def _with_database_factory(toolset: GaiaToolset, factory) -> GaiaToolset:
    return GaiaToolset(
        toolset.rag_schema,
        toolset.table_registry,
        toolset.scope_registry,
        factory,
    )


def _query_engine(*, dialect: str = "postgresql"):
    engine = MagicMock(spec=AsyncEngine)
    engine.dialect.name = dialect
    connection = MagicMock(spec=AsyncConnection)
    connection.run_sync = AsyncMock()
    engine.connect.return_value.__aenter__.return_value = connection
    engine.dispose = AsyncMock()
    return engine, connection


def _blank_plan() -> dict[str, object]:
    return {
        "scope": {
            "collection": "Cowan Tetrapod Collection",
            "dataset": "Birds",
        },
        "query": {"filters": {}, "return_columns": []},
        "result": {"distinct_by": [], "order_by": [], "limit": 5},
    }


def _export_arguments() -> dict[str, object]:
    query = _blank_plan()
    query["result"] = {"distinct_by": [], "order_by": []}
    return {
        "queries": [query],
        "limit": None,
        "limit_scope": "per_dataset",
        "layout": "single_workbook",
    }


def _export_result(export_directory: Path) -> dict[str, object]:
    return {
        "layout": "single_workbook",
        "files": [{
            "file_id": "test-file",
            "filename": "workbook.xlsx",
            "path": str(export_directory / "workbook.xlsx"),
            "record_count": 1,
            "datasets": [{
                "dataset_key": "birds",
                "sheet_name": "birds",
                "row_count": 1,
                "skipped_empty_columns": [],
            }],
        }],
        "record_count": 1,
        "skipped_datasets": [],
        "unqueried_dataset_keys": [],
        "limit": None,
        "limit_scope": "per_dataset",
        "database_truncated": False,
    }


def _internal_result(
    *,
    records: tuple[dict[str, str], ...],
    return_columns: tuple[str, ...] = ("country",),
    database_truncated: bool = False,
) -> QueryExecutionResult:
    return QueryExecutionResult(
        records=records,
        selected_dataset_keys=("birds", "mammals"),
        compatible_dataset_keys=("birds",),
        queried_dataset_keys=("birds",),
        skipped_datasets=(
            DatasetSkip(
                dataset_key="mammals",
                reason="missing_filter_columns",
                missing_headers=("country",),
            ),
        ),
        return_columns=return_columns,
        limit=5,
        database_truncated=database_truncated,
    )


def test_toolset_loads_the_json_once_and_reuses_it_for_requests(
    toolset: GaiaToolset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_once = MagicMock(return_value=dict(toolset.rag_schema))
    monkeypatch.setattr(tools_module, "load_schema", load_once)
    loaded = GaiaToolset.load()
    handler = _handler(loaded, "retrieve_schema")

    asyncio.run(handler({"datasets": ["birds"]}))
    asyncio.run(handler({"datasets": ["fish"]}))

    load_once.assert_called_once()
    assert load_once.call_args.args[1] == DATASETS


def test_registers_strict_tools_in_workflow_order(
    toolset: GaiaToolset,
) -> None:
    registered = toolset.registered_tools()

    assert [tool.definition.name for tool in registered] == TOOL_NAMES
    assert all(tool.definition.strict for tool in registered)
    assert registered[3].observation_projector is not None
    definitions = {tool.definition.name: tool.definition for tool in registered}
    for definition in definitions.values():
        purpose = definition.parameters["properties"]["purpose"]
        assert purpose["type"] == "string"
        assert purpose["minLength"] == 1
        assert purpose["maxLength"] == 300
        assert "purpose" in definition.parameters["required"]
        interpretation = definition.parameters["properties"]["interpretation"]
        assert interpretation["type"] == "string"
        assert interpretation["minLength"] == 1
        assert interpretation["maxLength"] == 300
        assert "interpretation" in definition.parameters["required"]
    schema_parameters = definitions["retrieve_schema"].parameters
    assert schema_parameters["required"] == ["purpose", "interpretation"]
    assert schema_parameters["additionalProperties"] is False
    assert set(schema_parameters["properties"]) == {
        "datasets",
        "collections",
        "purpose",
        "interpretation",
    }
    query_parameters = definitions["query_occurrences"].parameters
    assert query_parameters["required"] == [
        "scope",
        "query",
        "result",
        "purpose",
        "interpretation",
    ]
    assert query_parameters["additionalProperties"] is False
    scope_property = query_parameters["properties"]["scope"]
    assert "required" not in scope_property
    assert set(scope_property["properties"]) == {"collection", "dataset"}
    assert scope_property["additionalProperties"] is False
    query_property = query_parameters["properties"]["query"]
    assert query_property["required"] == ["filters", "column_mode", "return_columns"]
    assert query_property["additionalProperties"] is False
    column_mode = query_property["properties"]["column_mode"]
    assert column_mode["enum"] == ["default", "all", "selected"]
    assert "default" not in column_mode
    return_columns = query_property["properties"]["return_columns"]["anyOf"]
    assert return_columns == [
        {"type": "string", "enum": ["all"]},
        {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
    ]
    filter_schema = query_property["properties"]["filters"][
        "additionalProperties"
    ]
    assert filter_schema["minProperties"] == 1
    assert set(filter_schema["properties"]) == {
        "exact_any", "exclude_any", "contains", "starts_with", "is_empty",
        "gt", "gte", "lt", "lte", "value_type",
    }
    assert filter_schema["additionalProperties"] is False
    result_property = query_parameters["properties"]["result"]
    assert result_property["required"] == ["distinct_by", "order_by", "limit"]
    distinct_property = result_property["properties"]["distinct_by"]
    assert "query.return_columns" in distinct_property["description"]
    assert "representative occurrence" in distinct_property["description"]
    assert "exactly one resolved dataset" in distinct_property["description"]
    order_property = result_property["properties"]["order_by"]
    assert (
        "chooses the representative occurrence"
        in order_property["description"]
    )
    limit_property = result_property["properties"]["limit"]
    assert limit_property["type"] == "integer"
    assert limit_property["minimum"] == 1
    assert limit_property["maximum"] == MAX_CHAT_RECORDS
    assert "after deduplication" in limit_property["description"]

    statistics_result = definitions["query_statistics"].parameters["properties"]["result"]
    assert statistics_result["required"] == ["order_by", "limit"]
    statistics_limit = statistics_result["properties"]["limit"]
    assert statistics_limit["type"] == ["integer", "null"]
    assert statistics_limit["minimum"] == 1
    assert "maximum" not in statistics_limit

    file_parameters = definitions["generate_excel"].parameters
    assert file_parameters["required"] == [
        "queries", "limit", "limit_scope", "layout", "purpose", "interpretation",
    ]
    assert set(file_parameters["properties"]) == set(file_parameters["required"]) | {"filename"}
    assert file_parameters["properties"]["filename"]["type"] == "string"
    assert "filename" not in file_parameters["required"]
    assert file_parameters["additionalProperties"] is False
    file_query = file_parameters["properties"]["queries"]["items"]
    assert file_query["properties"]["query"] == query_property
    assert file_query["required"] == ["scope", "query", "result"]
    assert file_query["additionalProperties"] is False
    file_result = file_query["properties"]["result"]
    assert file_result["required"] == ["distinct_by", "order_by"]
    assert set(file_result["properties"]) == {"distinct_by", "order_by"}
    assert file_result["additionalProperties"] is False
    file_limit = file_parameters["properties"]["limit"]
    assert file_limit["type"] == ["integer", "null"]
    assert file_limit["minimum"] == 1
    assert "maximum" not in file_limit
    limit_scope = file_parameters["properties"]["limit_scope"]
    assert limit_scope["type"] == "string"
    assert limit_scope["enum"] == ["per_dataset", "total"]
    layout = file_parameters["properties"]["layout"]
    assert layout["type"] == "string"
    assert layout["enum"] == ["single_workbook", "separate_files"]


def test_explicit_export_directory_is_used_without_a_session_override(
    toolset: GaiaToolset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert toolset.export_directory is None
    destination = tmp_path / "direct_exports"
    configured = GaiaToolset.load(export_directory=destination)
    expected = _export_result(destination)
    generation = AsyncMock(return_value=expected)
    monkeypatch.setattr(tools_module, "generate_excel", generation)
    arguments = _export_arguments()

    result = asyncio.run(_handler(configured, "generate_excel")(arguments))

    assert configured.export_directory == destination
    assert result is expected
    generation.assert_awaited_once_with(
        arguments,
        table_registry=configured.table_registry,
        scope_registry=configured.scope_registry,
        engine_factory=configured.engine_factory,
        export_directory=destination,
        username="gaia",
        cancellation=None,
        database_limiter=configured.database_limiter,
        export_workers=configured.export_workers,
    )
    assert list(tmp_path.iterdir()) == []


async def _generate_concurrent_export(
    received,
    both_started,
    arguments,
    *,
    export_directory,
    username,
    **kwargs,
):
    assert arguments == _export_arguments()
    received.append((export_directory, username))
    if len(received) == 2:
        both_started.set()
    await both_started.wait()
    return _export_result(export_directory)


async def _check_concurrent_export_destinations(
    monkeypatch,
    handlers,
    destinations,
    shared,
    instance_destination,
) -> None:
    both_started = asyncio.Event()
    received = []

    generate = partial(_generate_concurrent_export, received, both_started)

    monkeypatch.setattr(tools_module, "generate_excel", generate)
    results = await asyncio.wait_for(
        asyncio.gather(*(handler(_export_arguments()) for handler in handlers)),
        timeout=2,
    )
    assert results == [_export_result(destination) for destination in destinations]
    assert received == list(zip(destinations, ("alice", "bob"), strict=True))
    assert shared.export_directory == instance_destination
    if instance_destination is not None:
        fallback = await _handler(shared, "generate_excel")(_export_arguments())
        assert fallback == _export_result(instance_destination)
        assert received[-1] == (instance_destination, "gaia")
    else:
        with pytest.raises(ToolExecutionError) as error:
            await _handler(shared, "generate_excel")(_export_arguments())
        assert error.value.code == "export_session_required"


@pytest.mark.parametrize("has_instance_destination", (False, True))
def test_concurrent_registrations_keep_session_export_destinations_isolated(
    toolset: GaiaToolset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_instance_destination: bool,
) -> None:
    instance_destination = tmp_path / "direct_exports" if has_instance_destination else None
    shared = replace(toolset, export_directory=instance_destination)
    destinations = (
        tmp_path / "alice" / "chat_one" / "exports",
        tmp_path / "bob" / "chat_two" / "exports",
    )
    handlers = tuple(
        next(
            tool.handler
            for tool in shared.registered_tools(
                export_directory=destination, username=username,
            )
            if tool.definition.name == "generate_excel"
        )
        for destination, username in zip(destinations, ("alice", "bob"), strict=True)
    )

    asyncio.run(_check_concurrent_export_destinations(
        monkeypatch, handlers, destinations, shared, instance_destination,
    ))
    assert shared.export_directory == instance_destination
    assert list(tmp_path.iterdir()) == []


def test_export_without_destination_fails_before_generation_or_database_access(
    toolset: GaiaToolset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = MagicMock(side_effect=AssertionError("A sessionless export cannot open the database"))
    generation = AsyncMock(side_effect=AssertionError("A sessionless export cannot write files"))
    monkeypatch.setattr(tools_module, "generate_excel", generation)
    monkeypatch.chdir(tmp_path)
    isolated = replace(toolset, engine_factory=factory)

    with pytest.raises(ToolExecutionError, match="unavailable in this conversation") as error:
        asyncio.run(_handler(isolated, "generate_excel")(_export_arguments()))

    assert isolated.export_directory is None
    assert error.value.code == "export_session_required"
    generation.assert_not_awaited()
    factory.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_schema_tool_returns_exact_dataset_headers_and_scope_key(
    toolset: GaiaToolset,
) -> None:
    result = asyncio.run(
        _handler(toolset, "retrieve_schema")(
            {"datasets": ["birds"]}
        )
    )

    assert set(result) == {
        "query plan structure", "statistics plan structure", "datasets",
    }
    assert set(result["query plan structure"]) == {"scope", "query", "result"}
    assert result["query plan structure"]["query"]["column_mode"] == "default"
    assert result["query plan structure"]["result"] == {
        "distinct_by": [],
        "order_by": [],
        "limit": 0,
    }
    birds = result["datasets"][0]
    assert birds["key"] == "birds"
    assert birds["collection"] == "Cowan Tetrapod Collection"
    assert birds["dataset"] == "Birds"
    assert list(birds["headers"]) == list(
        toolset.table_registry["birds"].info["source_headers"]
    )
    assert birds["headers"]["occurrenceID"] == 1
    assert birds["headers"]["occurrenceID"] == 1
    assert birds["headers"]["modified"] == 1


@pytest.mark.parametrize("tool_name", ("query_occurrences", "generate_excel"))
def test_tool_column_examples_resolve_the_documented_output(
    toolset: GaiaToolset,
    tool_name: str,
) -> None:
    definition = next(
        tool.definition
        for tool in toolset.registered_tools()
        if tool.definition.name == tool_name
    )
    parameters = definition.parameters
    if tool_name == "generate_excel":
        parameters = parameters["properties"]["queries"]["items"]
    examples = parameters["properties"]["query"]["examples"]
    birds = toolset.table_registry["birds"]
    default_columns = (
        "occurrenceID", "scientificName", "vernacularName", "kingdom", "phylum",
        "class", "order", "family", "locality",
    )
    expected_columns = (
        default_columns,
        default_columns,
        ("occurrenceID", "scientificName"),
        (
            "occurrenceID",
            *(column.name for column in birds.columns if column.name not in {"occurrenceID", "id"}),
        ),
    )
    for example, columns in zip(examples, expected_columns, strict=True):
        plan = tools_module.validate_query_plan(
            {
                "scope": {"dataset": "Birds"},
                "query": example,
                "result": {"distinct_by": [], "order_by": [], "limit": 5},
            },
            table_registry=toolset.table_registry,
            scope_registry=toolset.scope_registry,
        )
        assert plan.column_mode == example["column_mode"]
        assert build_dataset_select(birds, plan, fetch_limit=5).output_headers == columns


def test_retrieved_statistics_template_counts_records_without_occurrence_columns(
    toolset: GaiaToolset,
) -> None:
    schema = asyncio.run(_handler(toolset, "retrieve_schema")({"datasets": ["birds"]}))
    raw = schema["statistics plan structure"]
    raw["scope"] = {"dataset": "Birds"}
    plan = tools_module.validate_statistics_plan(
        raw, table_registry=toolset.table_registry, scope_registry=toolset.scope_registry,
    )
    assert plan.dataset_keys == ("birds",)
    assert plan.group_by == ()
    assert plan.metrics == (PlannedMetric("count", "record_count", None, "text"),)
    assert plan.limit is None


async def _execute_blocked_occurrences(query_started, query_release, plan, **kwargs):
    assert plan.limit == 200
    query_started.set()
    await query_release.wait()
    return replace(
        _internal_result(records=tuple(
            {"dataset_key": "birds", "occurrenceID": str(index)}
            for index in range(plan.limit)
        )),
        limit=plan.limit,
    )


async def _check_independent_statistics_budget(
    monkeypatch,
    handlers,
    statistics_arguments,
    record_arguments,
):
    query_started = asyncio.Event()
    query_release = asyncio.Event()

    records = partial(_execute_blocked_occurrences, query_started, query_release)

    monkeypatch.setattr(budget_module, "execute_query_plan", records)
    initial_summary = await handlers["query_statistics"](statistics_arguments)
    assert len(initial_summary.rows) == 250
    record_task = asyncio.create_task(handlers["query_occurrences"](record_arguments))
    try:
        await asyncio.wait_for(query_started.wait(), timeout=1)
        concurrent_summary = await asyncio.wait_for(
            handlers["query_statistics"](statistics_arguments), timeout=1,
        )
        assert concurrent_summary.rows == initial_summary.rows
    finally:
        query_release.set()
        records_result = await record_task
    assert len(records_result.records) == 200

    final_summary = await handlers["query_statistics"](statistics_arguments)
    assert final_summary.rows == initial_summary.rows
    with pytest.raises(ToolExecutionError) as error:
        await handlers["query_occurrences"](record_arguments)
    assert error.value.code == "chat_record_limit_reached"


async def _execute_unbounded_statistics(statistics_limit, plan, **kwargs):
    assert plan.limit == statistics_limit
    return StatisticsExecutionResult(
        rows=tuple({"scientificName": str(index), "record_count": 1_000_000} for index in range(250)),
        selected_dataset_keys=plan.dataset_keys,
        compatible_dataset_keys=plan.dataset_keys,
        queried_dataset_keys=plan.dataset_keys,
        skipped_datasets=(),
        output_columns=("scientificName", "record_count"),
        group_by=plan.group_by,
        metrics=plan.metrics,
        limit=plan.limit,
        database_truncated=False,
    )


@pytest.mark.parametrize("statistics_limit", [None, 500])
def test_statistics_run_during_and_after_exhausted_occurrence_budget(
    toolset: GaiaToolset,
    monkeypatch: pytest.MonkeyPatch,
    statistics_limit: int | None,
) -> None:
    handlers = {tool.definition.name: tool.handler for tool in toolset.registered_tools()}
    statistics_arguments = {
        "scope": {"dataset": "Birds"},
        "query": {
            "filters": {},
            "group_by": ["scientificName"],
            "metrics": [{"operation": "count", "alias": "record_count"}],
        },
        "result": {"order_by": [], "limit": statistics_limit},
    }
    record_arguments = _blank_plan()
    record_arguments["result"]["limit"] = 200

    statistics = partial(_execute_unbounded_statistics, statistics_limit)

    monkeypatch.setattr(tools_module, "execute_statistics_plan", statistics)

    asyncio.run(_check_independent_statistics_budget(
        monkeypatch, handlers, statistics_arguments, record_arguments,
    ))


def test_statistics_tool_rejects_invalid_metrics_before_database_access(
    toolset: GaiaToolset,
) -> None:
    factory = MagicMock(side_effect=AssertionError("Invalid plans must not open the database"))
    configured = _with_database_factory(toolset, factory)
    with pytest.raises(ToolExecutionError) as error:
        asyncio.run(_handler(configured, "query_statistics")({
            "scope": {"dataset": "Birds"},
            "query": {"filters": {}, "group_by": [], "metrics": [{
                "operation": "raw_sql", "alias": "total",
            }]},
            "result": {"order_by": [], "limit": 1},
        }))
    assert error.value.code == "invalid_query_plan"
    factory.assert_not_called()


def test_statistics_projector_preserves_exact_metrics_and_coverage() -> None:
    missing_headers = tuple(f"field_{index}" for index in range(30, 0, -1))
    result = StatisticsExecutionResult(
        rows=({"scientificName": "Corvus corax", "record_count": 50001, "total": "123456789012345.123456789"},),
        selected_dataset_keys=("birds", "mammals"),
        compatible_dataset_keys=("birds",),
        queried_dataset_keys=("birds",),
        skipped_datasets=(DatasetSkip("mammals", "missing_filter_columns", missing_headers),),
        output_columns=("scientificName", "record_count", "total"),
        group_by=("scientificName",),
        metrics=(
            PlannedMetric("count", "record_count", None, "text"),
            PlannedMetric("sum", "total", "year", "numeric"),
        ),
        limit=1,
        database_truncated=True,
    )
    observed = project_query_statistics_observation(result)
    assert observed["rows"] == list(result.rows)
    assert type(observed["rows"][0]["record_count"]) is int
    assert observed["metadata"]["groups_returned"] == 1
    assert observed["metadata"]["database_truncated"] is True
    assert observed["metadata"]["skipped_datasets"] == [{
        "dataset_key": "mammals",
        "reason": "missing_filter_columns",
        "missing_headers": list(missing_headers),
        "missing_header_count": 30,
    }]
    assert "numeric_value_policy" in observed["metadata"]
    assert "occurrenceID" not in observed["rows"][0]
    json.dumps(observed, allow_nan=False)


def test_schema_tool_accepts_multiple_selector_categories_and_values(
    toolset: GaiaToolset,
) -> None:
    result = asyncio.run(
        _handler(toolset, "retrieve_schema")(
            {
                "datasets": ["birds", "fish"],
                "collections": ["Herbarium"],
            }
        )
    )

    assert [entry["key"] for entry in result["datasets"]] == ["birds", "fish"]
    assert result["collections"][0]["collection"] == "Herbarium"
    assert [
        entry["key"] for entry in result["collections"][0]["datasets"]
    ] == ["algae", "bryophytes", "fungi", "lichen", "vascular_plants"]


def test_schema_tool_defaults_to_all_collections(
    toolset: GaiaToolset,
) -> None:
    schema_handler = _handler(toolset, "retrieve_schema")
    default_result = asyncio.run(schema_handler({}))
    explicit_result = asyncio.run(
        schema_handler({"collections": list(toolset.scope_registry)})
    )

    assert default_result == explicit_result
    assert [entry["collection"] for entry in default_result["collections"]] == list(
        toolset.scope_registry
    )


def test_schema_tool_rejects_unknown_or_unbounded_arguments(
    toolset: GaiaToolset,
) -> None:
    schema_handler = _handler(toolset, "retrieve_schema")

    with pytest.raises(ToolExecutionError, match="Unknown argument"):
        asyncio.run(schema_handler({"datasets": ["birds"], "extra": True}))
    with pytest.raises(ToolExecutionError, match="Unknown argument"):
        asyncio.run(schema_handler({"chunk_structure": ["identification"]}))
    with pytest.raises(ToolExecutionError, match="non-empty array"):
        asyncio.run(schema_handler({"datasets": []}))
    with pytest.raises(ToolExecutionError, match="unknown datasets"):
        asyncio.run(schema_handler({"datasets": ["dragons"]}))


def test_occurrence_tool_validates_before_opening_the_database(
    toolset: GaiaToolset,
) -> None:
    forbidden_factory = MagicMock(
        side_effect=AssertionError("invalid plans must not open a database"),
    )
    isolated = _with_database_factory(toolset, forbidden_factory)
    plan = _blank_plan()
    plan["raw_sql"] = "SELECT * FROM ctc_birds"

    with pytest.raises(ToolExecutionError) as error:
        asyncio.run(_handler(isolated, "query_occurrences")(plan))

    assert error.value.code == "invalid_query_plan"
    assert "raw_sql" in str(error.value)
    forbidden_factory.assert_not_called()


def test_occurrence_tool_reports_missing_postgresql_configuration(
    toolset: GaiaToolset,
) -> None:
    unavailable_factory = MagicMock(
        side_effect=DatabaseConfigurationError("missing database setting: POSTGRES_DB"),
    )

    isolated = _with_database_factory(toolset, unavailable_factory)

    with pytest.raises(ToolExecutionError) as error:
        asyncio.run(_handler(isolated, "query_occurrences")(_blank_plan()))

    assert error.value.code == "query_unavailable"
    assert str(error.value) == QUERY_UNAVAILABLE_MESSAGE


def test_occurrence_tool_sanitizes_engine_setup_errors(
    toolset: GaiaToolset,
) -> None:
    unavailable_factory = MagicMock(side_effect=SQLAlchemyError("RAW-DATABASE-CONFIGURATION"))

    isolated = _with_database_factory(toolset, unavailable_factory)

    with pytest.raises(ToolExecutionError) as error:
        asyncio.run(_handler(isolated, "query_occurrences")(_blank_plan()))

    assert error.value.code == "query_unavailable"
    assert str(error.value) == QUERY_UNAVAILABLE_MESSAGE


def test_occurrence_tool_disposes_engine_after_connection_failure(
    toolset: GaiaToolset,
) -> None:
    engine, connection = _query_engine()
    engine.connect.return_value.__aenter__.side_effect = SQLAlchemyError(
        "RAW-DATABASE-CONNECTION-DETAILS"
    )
    isolated = _with_database_factory(toolset, lambda: engine)

    with pytest.raises(ToolExecutionError) as error:
        asyncio.run(_handler(isolated, "query_occurrences")(_blank_plan()))

    assert error.value.code == "query_unavailable"
    assert str(error.value) == QUERY_UNAVAILABLE_MESSAGE
    connection.run_sync.assert_not_awaited()
    engine.dispose.assert_awaited_once()


def test_occurrence_tool_rejects_a_non_postgresql_runtime_engine(
    toolset: GaiaToolset,
) -> None:
    sqlite_engine, _connection = _query_engine(dialect="sqlite")
    isolated = _with_database_factory(toolset, lambda: sqlite_engine)

    with pytest.raises(ToolExecutionError) as error:
        asyncio.run(_handler(isolated, "query_occurrences")(_blank_plan()))

    assert error.value.code == "query_unavailable"
    assert str(error.value) == QUERY_UNAVAILABLE_MESSAGE
    sqlite_engine.connect.assert_not_called()
    sqlite_engine.dispose.assert_awaited_once()


def test_occurrence_tool_disposes_postgresql_engine_after_success(
    toolset: GaiaToolset,
) -> None:
    engine, connection = _query_engine()
    expected = _internal_result(records=())
    connection.run_sync.return_value = expected
    isolated = _with_database_factory(toolset, lambda: engine)

    result = asyncio.run(_handler(isolated, "query_occurrences")(_blank_plan()))

    assert result is expected
    connection.run_sync.assert_awaited_once()
    arguments, keywords = connection.run_sync.call_args
    assert arguments[1].dataset_keys == ("birds",)
    assert arguments[1].column_mode == "default"
    assert arguments[1].return_columns == (
        "scientificName", "vernacularName", "kingdom", "phylum", "class",
        "order", "family",
    )
    assert keywords["table_registry"] is toolset.table_registry
    engine.connect.return_value.__aexit__.assert_awaited_once()
    engine.dispose.assert_awaited_once()


@pytest.mark.parametrize(
    ("execution_error", "expected_code", "expected_message"),
    (
        (
            SQLAlchemyError(
                f"{REPOSITORY_ROOT}/dataset: RAW-SQLALCHEMY-ROW-VALUE"
            ),
            "query_unavailable",
            QUERY_UNAVAILABLE_MESSAGE,
        ),
        (
            ValueError(f"{REPOSITORY_ROOT}/dataset: RAW-INVALID-ROW-VALUE"),
            "invalid_query_data",
            None,
        ),
    ),
)
def test_occurrence_tool_sanitizes_execution_errors_and_disposes_engine(
    toolset: GaiaToolset,
    execution_error: Exception,
    expected_code: str,
    expected_message: str | None,
) -> None:
    engine, connection = _query_engine()
    connection.run_sync.side_effect = execution_error
    isolated = _with_database_factory(toolset, lambda: engine)

    with pytest.raises(ToolExecutionError) as error:
        asyncio.run(_handler(isolated, "query_occurrences")(_blank_plan()))

    assert error.value.code == expected_code
    if expected_message is not None:
        assert str(error.value) == expected_message
    assert str(REPOSITORY_ROOT) not in str(error.value)
    assert "RAW-" not in str(error.value)
    engine.connect.return_value.__aexit__.assert_awaited_once()
    engine.dispose.assert_awaited_once()


async def _block_occurrence_query(first_started, pending, *_args, **_kwargs):
    first_started.set()
    await pending.wait()


async def _check_query_overlap_and_cancellation(toolset) -> None:
    first_started = asyncio.Event()
    pending = asyncio.Event()
    first_engine, first_connection = _query_engine()
    second_engine, second_connection = _query_engine()
    expected = _internal_result(records=())

    blocked_query = partial(_block_occurrence_query, first_started, pending)

    first_connection.run_sync.side_effect = blocked_query
    second_connection.run_sync.return_value = expected
    engines = iter((first_engine, second_engine))
    isolated = _with_database_factory(toolset, lambda: next(engines))
    handler = _handler(isolated, "query_occurrences")
    first = asyncio.create_task(handler(_blank_plan()))
    try:
        await asyncio.wait_for(first_started.wait(), timeout=2)
        second = await asyncio.wait_for(handler(_blank_plan()), timeout=2)
        assert second is expected
        assert not first.done()
        second_engine.dispose.assert_awaited_once()
        first_engine.dispose.assert_not_awaited()
    finally:
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

    first_engine.connect.return_value.__aexit__.assert_awaited_once()
    first_engine.dispose.assert_awaited_once()


def test_occurrence_queries_overlap_and_cancelled_query_disposes_its_engine(
    toolset: GaiaToolset,
) -> None:
    asyncio.run(_check_query_overlap_and_cancellation(toolset))


def test_projector_allows_only_identity_and_requested_return_columns() -> None:
    secret = "UNUSED-PRIVATE-OCCURRENCE-FIELD"
    internal = _internal_result(
        records=(
            {
                "dataset_key": "birds",
                "occurrenceID": "bird-1",
                "country": "Canada",
                "scientificName": "Corvus corax",
                "dynamicProperties": secret,
            },
        ),
        return_columns=("scientificName",),
    )

    observation = project_query_occurrences_observation(internal)

    assert observation["records"] == [
        {
            "dataset_key": "birds",
            "occurrenceID": "bird-1",
            "scientificName": "Corvus corax",
        }
    ]
    assert secret not in repr(observation)
    assert "country" not in observation["records"][0]
    assert observation["metadata"] == {
        "selected_dataset_keys": ["birds", "mammals"],
        "compatible_dataset_keys": ["birds"],
        "queried_dataset_keys": ["birds"],
        "skipped_datasets": [
            {
                "dataset_key": "mammals",
                "reason": "missing_filter_columns",
                "missing_headers": ["country"],
                "missing_header_count": 1,
            }
        ],
        "limit": 5,
        "database_records_returned": 1,
        "database_truncated": False,
    }


@pytest.mark.parametrize(
    ("identifiers", "selected_identifier"),
    (
        ({"id": "0001", "occurrenceID": "bird-1"}, "occurrenceID"),
        ({"occurrenceID": "bird-1"}, "occurrenceID"),
        ({"id": "0001"}, "id"),
        ({"id": "0001", "occurrenceID": ""}, "occurrenceID"),
    ),
)
def test_projector_keeps_one_available_identifier(
    identifiers: dict[str, str], selected_identifier: str,
) -> None:
    internal = _internal_result(
        records=({
            "dataset_key": "birds", **identifiers, "scientificName": "Corvus corax",
        },),
        return_columns=("scientificName",),
    )

    observation = project_query_occurrences_observation(internal)

    assert observation["records"] == [{
        "dataset_key": "birds", selected_identifier: identifiers[selected_identifier],
        "scientificName": "Corvus corax",
    }]


def test_projector_preserves_each_datasets_resolved_column_order() -> None:
    records = (
        {
            "dataset_key": "birds", "occurrenceID": "bird-1",
            "scientificName": "Corvus corax", "kingdom": "Animalia",
        },
        {
            "dataset_key": "herpetology", "occurrenceID": "herp-1",
            "higherClassification": "Amphibia", "kingdom": "Animalia",
            "scientificName": "Rana aurora",
        },
    )
    dataset_keys = ("birds", "herpetology")
    internal = QueryExecutionResult(
        records=records,
        selected_dataset_keys=dataset_keys,
        compatible_dataset_keys=dataset_keys,
        queried_dataset_keys=dataset_keys,
        skipped_datasets=(),
        return_columns=("scientificName", "kingdom", "higherClassification"),
        limit=5,
        database_truncated=False,
    )

    observation = project_query_occurrences_observation(internal)

    assert [list(row) for row in observation["records"]] == [list(row) for row in records]


def test_projector_preserves_every_record_with_large_unicode_values() -> None:
    missing_headers = tuple(f"field_{index}" for index in range(40, 0, -1))
    records = tuple(
        {
            "dataset_key": "birds",
            "occurrenceID": f"bird-{index}",
            "locality": "海岸 — " * 10_000,
        }
        for index in range(3)
    )
    internal = _internal_result(
        records=records,
        return_columns=("locality",),
    )
    internal = replace(internal, skipped_datasets=(
        DatasetSkip("mammals", "missing_filter_columns", missing_headers),
    ))

    observation = project_query_occurrences_observation(internal)

    assert observation["records"] == list(records)
    assert observation["metadata"]["database_records_returned"] == len(records)
    assert observation["metadata"]["skipped_datasets"] == [{
        "dataset_key": "mammals",
        "reason": "missing_filter_columns",
        "missing_headers": list(missing_headers),
        "missing_header_count": 40,
    }]
    assert len(json.dumps(observation, ensure_ascii=False).encode("utf-8")) > 65_536


def test_projector_preserves_database_truncation_metadata() -> None:
    internal = _internal_result(
        records=(
            {
                "dataset_key": "birds",
                "occurrenceID": "bird-1",
                "country": "Canada",
            },
        ),
        database_truncated=True,
    )

    observation = project_query_occurrences_observation(internal)

    assert observation["metadata"]["database_truncated"] is True
