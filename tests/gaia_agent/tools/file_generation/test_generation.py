"""Exercise workbook generation through the model-visible tool contract."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial, partialmethod
from pathlib import Path
from threading import Event, enumerate as threads, get_ident
from typing import TextIO
from zipfile import ZipFile

import pytest
from sqlalchemy import MetaData
from xlsxwriter import Workbook

import src.gaia_agent.tools.file_generation.generation as generation_module
import src.gaia_agent.tools.query.budget as budget_module
from src.gaia_agent.runtime.contracts import (
    AgentCancellation,
    ModelTurn,
    ToolCall,
    ToolExecutionError,
)
from src.gaia_agent.runtime.loop import AgentRuntime
from src.gaia_agent.runtime.service import GaiaAgent
from src.gaia_agent.tools import GaiaToolset
from src.gaia_agent.tools.file_generation.workers import ExcelExportWorkers
from src.gaia_agent.tools.query.columns import preferred_identifier
from src.gaia_agent.tools.query.execution import ExportQueryResult
from tests.gaia_agent.support import RecordingClient, query_result
from tests.gaia_agent.tools.file_generation.test_excel_export import worksheet_rows
from tests.gaia_agent.tools.query.support import make_table
from tests.gaia_agent.tools.query.test_streaming import StreamingEngine


BIRD_DEFAULT_COLUMNS = ("vernacularName", "scientificName", "kingdom", "phylum", "class")


def _query(
    dataset: str,
    *,
    return_columns: tuple[str, ...] | str = (),
    column_mode: str | None = None,
) -> dict[str, object]:
    query = {
        "scope": {"dataset": dataset},
        "query": {
            "filters": {},
            "return_columns": return_columns if isinstance(return_columns, str) else list(return_columns),
        },
        "result": {"distinct_by": [], "order_by": []},
    }
    if column_mode is not None:
        query["query"]["column_mode"] = column_mode
    return query


def _arguments(
    queries: list[dict[str, object]],
    *,
    limit: int | None = None,
    limit_scope: str = "total",
    layout: str = "single_workbook",
    filename: str | None = None,
) -> dict[str, object]:
    arguments = {
        "queries": queries,
        "limit": limit,
        "limit_scope": limit_scope,
        "layout": layout,
    }
    if filename is not None:
        arguments["filename"] = filename
    return arguments


@pytest.fixture
def toolset(tmp_path: Path) -> GaiaToolset:
    metadata = MetaData()
    return GaiaToolset(
        {},
        {
            "birds": make_table(
                metadata, "birds", ("occurrenceID", "catalogNumber", "locality"),
            ),
            "fish": make_table(metadata, "fish", ("occurrenceID", "scientificName")),
        },
        {"Collection": {"Birds": "birds", "Fish": "fish"}},
        lambda: None,
        tmp_path,
    )


def _handler(
    toolset: GaiaToolset,
    cancellation: AgentCancellation | None = None,
    *,
    username: str = "gaia",
):
    return next(
        tool.handler
        for tool in toolset.registered_tools(cancellation=cancellation, username=username)
        if tool.definition.name == "generate_excel"
    )


class _StreamSource:
    """Supply deterministic callback batches without opening a database."""

    def __init__(
        self,
        source_rows: Mapping[str, Sequence[Mapping[str, str]]],
        batch_size: int,
    ) -> None:
        self.source_rows = source_rows
        self.batch_size = batch_size
        self.calls: list[dict[str, object]] = []

    async def stream(
        self,
        plans,
        *,
        table_registry,
        on_batch,
        on_dataset_complete,
        row_limit=None,
        limit_scope="total",
        engine_factory,
        cancellation=None,
        database_limiter=None,
    ):
        self.calls.append({
            "plans": plans, "row_limit": row_limit, "limit_scope": limit_scope,
            "cancellation": cancellation,
        })
        remaining = row_limit
        queried = []
        counts = []
        truncated = False
        for plan in plans:
            if limit_scope == "per_dataset":
                remaining = row_limit
            key = plan.dataset_keys[0]
            headers = _stream_headers(plan, table_registry[key])
            await on_batch(key, headers, ())
            records = self.source_rows.get(key, ())
            if remaining == 0:
                exported = ()
            else:
                queried.append(key)
                exported = records if remaining is None else records[:remaining]
                truncated = truncated or len(exported) < len(records)
                if remaining is not None:
                    remaining -= len(exported)
            await self.emit_batches(key, headers, exported, on_batch)
            counts.append((key, len(exported)))
            await on_dataset_complete(key)
        keys = tuple(plan.dataset_keys[0] for plan in plans)
        return ExportQueryResult(
            selected_dataset_keys=keys,
            compatible_dataset_keys=keys,
            queried_dataset_keys=tuple(queried),
            skipped_datasets=(),
            dataset_row_counts=tuple(counts),
            limit=row_limit,
            database_truncated=truncated,
        )

    async def emit_batches(self, key, headers, records, on_batch) -> None:
        for start in range(0, len(records), self.batch_size):
            await on_batch(key, headers, tuple(
                tuple(record[header] for header in headers)
                for record in records[start:start + self.batch_size]
            ))


def _stream_headers(plan, table) -> tuple[str, ...]:
    available = {column.name for column in table.columns}
    return tuple(dict.fromkeys((
        preferred_identifier(available),
        *(header for header in plan.return_columns if header in available),
    )))


def _install_stream(
    monkeypatch: pytest.MonkeyPatch,
    source_rows: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    batch_size: int = 100,
) -> list[dict[str, object]]:
    source = _StreamSource(source_rows, batch_size)
    monkeypatch.setattr(generation_module, "stream_query_plans", source.stream)
    return source.calls


class _SpoolTracker:
    """Observe real temporary handles so anonymous-spool leaks remain visible."""

    def __init__(self) -> None:
        self.opened: list[TextIO] = []
        self.temporary_file = generation_module.TemporaryFile

    def create_spool(self, *args, **kwargs) -> TextIO:
        spool = self.temporary_file(*args, **kwargs)
        self.opened.append(spool)
        return spool


def _track_spools(monkeypatch: pytest.MonkeyPatch) -> list[TextIO]:
    tracker = _SpoolTracker()
    monkeypatch.setattr(generation_module, "TemporaryFile", tracker.create_spool)
    return tracker.opened


async def _forbidden_stream(*args, **kwargs):
    raise AssertionError("Invalid requests must be validated before streaming")


class _OverlappingStream:
    def __init__(self, stream) -> None:
        self.stream = stream
        self.active = 0
        self.maximum_active = 0

    async def run(self, *args, **kwargs):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            await asyncio.sleep(0.02)
            return await self.stream(*args, **kwargs)
        finally:
            self.active -= 1


async def _run_five_exports(agent: GaiaAgent):
    return await asyncio.wait_for(agent.run("Create five Excel files."), timeout=5)


def test_model_invokes_workbook_tool_and_receives_only_file_metadata(toolset, monkeypatch):
    calls = _install_stream(monkeypatch, {
        "birds": ({"occurrenceID": "bird-1", "catalogNumber": "0001", "locality": ""},),
        "fish": ({"occurrenceID": "fish-1", "scientificName": "日本語"},),
    })
    client = RecordingClient((
        ModelTurn(tool_calls=(ToolCall("file", "generate_excel", {
            "layout": "single_workbook", "limit_scope": "total", "queries": [_query("Birds", return_columns="all"), _query("Fish", return_columns="all")], "limit": None,
        }),)),
        ModelTurn(content="Workbook created."),
    ))
    agent = GaiaAgent(AgentRuntime(client, toolset.registered_tools()), system_prompt="Use tools.")
    result = asyncio.run(agent.run("Create an Excel workbook with birds and fish."))
    assert result.tool_results[0].ok
    metadata = result.tool_results[0].result
    assert metadata["record_count"] == 2
    assert metadata["limit"] is None
    assert set(metadata) == {
        "layout", "files", "record_count", "skipped_datasets",
        "unqueried_dataset_keys", "limit", "limit_scope", "database_truncated",
    }
    assert metadata["layout"] == "single_workbook"
    assert metadata["limit_scope"] == "total"
    assert len(metadata["files"]) == 1
    assert set(metadata["files"][0]) == {
        "file_id", "filename", "record_count", "datasets",
    }
    assert metadata["files"][0]["record_count"] == 2
    assert metadata["files"][0]["datasets"] == [
        {
            "dataset_key": "birds", "sheet_name": "birds", "row_count": 1,
            "skipped_empty_columns": [],
        },
        {
            "dataset_key": "fish", "sheet_name": "fish", "row_count": 1,
            "skipped_empty_columns": [],
        },
    ]
    assert "records" not in metadata
    observation = client.requests[-1][0][-1].content
    assert "日本語" not in observation
    assert str(toolset.export_directory) not in observation
    assert metadata["files"][0]["filename"] in observation
    assert len(calls) == 1 and calls[0]["row_limit"] is None
    path = toolset.export_directory / metadata["files"][0]["filename"]
    assert worksheet_rows(path) == [
        ("occurrenceID", "catalogNumber", "locality"), ("bird-1", "0001", ""),
    ]
    assert worksheet_rows(path, 2) == [("occurrenceID", "scientificName"), ("fish-1", "日本語")]


@pytest.mark.parametrize("filename", (None, "helloXXX"))
def test_five_exports_in_one_model_round_respect_capacity_and_publish_unique_workbooks(
    toolset: GaiaToolset,
    monkeypatch: pytest.MonkeyPatch,
    filename: str | None,
) -> None:
    calls = _install_stream(monkeypatch, {
        "birds": ({"occurrenceID": "bird-1", "catalogNumber": "0001", "locality": "日本語"},),
    })
    stream = generation_module.stream_query_plans
    client = RecordingClient((
        ModelTurn(tool_calls=tuple(
            ToolCall(f"export-{index}", "generate_excel", _arguments(
                [_query("Birds", return_columns="all")], filename=filename,
            ))
            for index in range(5)
        )),
        ModelTurn(content="Created five workbooks."),
    ))
    agent = GaiaAgent(
        AgentRuntime(client, toolset.registered_tools(username="atlas")),
        system_prompt="Use tools.",
    )
    overlapping = _OverlappingStream(stream)
    monkeypatch.setattr(generation_module, "stream_query_plans", overlapping.run)
    result = asyncio.run(_run_five_exports(agent))
    assert overlapping.maximum_active == 2
    assert len(calls) == len(result.tool_results) == 5
    assert all(tool_result.ok for tool_result in result.tool_results)
    files = [tool_result.result["files"][0] for tool_result in result.tool_results]
    expected_names = (
        {f"atlas_{index}.xlsx" for index in range(1, 6)}
        if filename is None
        else {"helloXXX.xlsx", *(f"helloXXX-{index}.xlsx" for index in range(2, 6))}
    )
    assert {file["filename"] for file in files} == expected_names
    assert len({file["file_id"] for file in files}) == 5
    for file in files:
        assert file["record_count"] == 1
        assert worksheet_rows(toolset.export_directory / file["filename"]) == [
            ("occurrenceID", "catalogNumber", "locality"), ("bird-1", "0001", "日本語"),
        ]
    assert set(toolset.export_directory.iterdir()) == {toolset.export_directory / file["filename"] for file in files}


def test_default_sequence_continues_across_tool_registrations_and_separate_files(
    toolset: GaiaToolset, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stream(monkeypatch, {
        "birds": ({"occurrenceID": "bird-1", "catalogNumber": "0001", "locality": ""},),
        "fish": ({"occurrenceID": "fish-1", "scientificName": "日本語"},),
    })
    bird_query = _query("Birds", return_columns="all")
    requests = (
        _arguments([bird_query], filename="specimens"),
        _arguments(
            [bird_query, _query("Fish", return_columns="all")], layout="separate_files",
        ),
        _arguments([bird_query]),
    )
    results = [
        asyncio.run(_handler(toolset, username="atlas")(arguments))
        for arguments in requests
    ]

    assert [[file["filename"] for file in result["files"]] for result in results] == [
        ["specimens.xlsx"], ["atlas_1.xlsx", "atlas_2.xlsx"], ["atlas_3.xlsx"],
    ]
    files = [file for result in results for file in result["files"]]
    assert len({file["file_id"] for file in files}) == 4
    assert set(toolset.export_directory.iterdir()) == {
        toolset.export_directory / file["filename"] for file in files
    }
    assert worksheet_rows(toolset.export_directory / "atlas_2.xlsx") == [
        ("occurrenceID", "scientificName"), ("fish-1", "日本語"),
    ]


@pytest.mark.parametrize(
    "filename",
    (None, "", "../escape.xlsx", "/tmp/escape.xlsx", r"nested\file", "bad\tname", "a.xlsx.xlsx", "é" * 126),
)
def test_invalid_filenames_fail_before_files_or_database_access(
    toolset: GaiaToolset,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    filename: object,
) -> None:
    destination = tmp_path / "not_created"
    configured = replace(toolset, export_directory=destination)

    monkeypatch.setattr(generation_module, "stream_query_plans", _forbidden_stream)
    arguments = _arguments([_query("Birds", return_columns="all")])
    arguments["filename"] = filename
    with pytest.raises(ToolExecutionError, match="filename") as error:
        asyncio.run(_handler(configured)(arguments))
    assert error.value.code == "invalid_query_plan"
    assert not destination.exists()


def test_expanded_filenames_are_validated_before_any_output_or_database_work(
    toolset: GaiaToolset,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "not_created"
    configured = replace(toolset, export_directory=destination)

    monkeypatch.setattr(generation_module, "stream_query_plans", _forbidden_stream)
    with pytest.raises(ToolExecutionError, match="filename") as error:
        asyncio.run(_handler(configured)(_arguments(
            [_query("Birds", return_columns="all"), _query("Fish", return_columns="all")],
            layout="separate_files", filename="a" * 249,
        )))
    assert error.value.code == "invalid_query_plan"
    assert not destination.exists()


@pytest.mark.parametrize("layout", ("single_workbook", "separate_files"))
def test_one_compatible_dataset_keeps_the_requested_filename_after_collection_expansion(
    toolset: GaiaToolset,
    tmp_path: Path,
    layout: str,
) -> None:
    engine = StreamingEngine({"birds": (("bird-1", "00001", "Vancouver"),)})
    configured = replace(toolset, engine_factory=lambda: engine)
    query = _query("Birds", return_columns="all")
    query["scope"] = {"collection": "Collection"}
    query["query"]["filters"] = {"locality": {"exact_any": ["Vancouver"]}}
    result = asyncio.run(_handler(configured)(_arguments(
        [query], layout=layout, filename="helloXXX.XLSX",
    )))
    assert len(result["files"]) == 1
    assert result["files"][0]["filename"] == "helloXXX.xlsx"
    assert "path" not in result["files"][0]
    assert result["record_count"] == 1
    assert result["skipped_datasets"] == [{
        "dataset_key": "fish", "reason": "missing_filter_columns", "missing_headers": ["locality"],
    }]
    assert worksheet_rows(tmp_path / result["files"][0]["filename"]) == [
        ("occurrenceID", "catalogNumber", "locality"), ("bird-1", "00001", "Vancouver"),
    ]


@pytest.mark.parametrize(
    ("column_mode", "requested", "filter_header", "resolved_columns"),
    (
        (None, (), None, BIRD_DEFAULT_COLUMNS),
        ("default", (), None, BIRD_DEFAULT_COLUMNS),
        (None, ("locality",), None, (*BIRD_DEFAULT_COLUMNS, "locality")),
        (None, (), "locality", (*BIRD_DEFAULT_COLUMNS, "locality")),
        (None, (), "recordedBy", (*BIRD_DEFAULT_COLUMNS, "recordedBy")),
        (None, ("id", "occurrenceID"), None, BIRD_DEFAULT_COLUMNS),
        (
            None, "all", None,
            (
                "catalogNumber", *BIRD_DEFAULT_COLUMNS, "locality",
                "recordedBy", "country", "family", "occurrenceRemarks",
            ),
        ),
        (
            "all", "all", None,
            (
                "catalogNumber", *BIRD_DEFAULT_COLUMNS, "locality",
                "recordedBy", "country", "family", "occurrenceRemarks",
            ),
        ),
        ("selected", ("locality", "scientificName"), None, ("locality", "scientificName")),
    ),
    ids=(
        "omitted-default", "explicit-default", "requested-extra", "locality-filter",
        "recordedBy-filter", "identity-dedup", "literal-all", "explicit-all", "selected",
    ),
)
def test_export_column_modes_resolve_real_worksheet_headers(
    tmp_path: Path,
    column_mode: str | None,
    requested: tuple[str, ...] | str,
    filter_header: str | None,
    resolved_columns: tuple[str, ...],
) -> None:
    record = {
        "id": "0000042",
        "occurrenceID": "bird-00001",
        "catalogNumber": "0000007",
        "vernacularName": "",
        "scientificName": "Corvus corax",
        "kingdom": "Animalia",
        "phylum": "Chordata",
        "class": "Aves",
        "locality": "Vancouver",
        "recordedBy": "A. Collector",
        "country": "Canada",
        "family": "Corvidae",
        "occurrenceRemarks": " \t\n",
    }
    queried_headers = ("occurrenceID", *resolved_columns)
    engine = StreamingEngine({
        "birds": (tuple(record[header] for header in queried_headers),),
    })
    configured = GaiaToolset(
        {},
        {"birds": make_table(MetaData(), "birds", tuple(record))},
        {"Collection": {"Birds": "birds"}},
        lambda: engine,
        tmp_path,
    )
    query = _query("Birds", return_columns=requested, column_mode=column_mode)
    if filter_header is not None:
        query["query"]["filters"] = {filter_header: {"exact_any": [record[filter_header]]}}
    result = asyncio.run(_handler(configured)(_arguments([query])))

    assert len(engine.connection.statements) == 1
    statement = engine.connection.statements[0][1]
    assert tuple(column.name for column in statement.selected_columns) == queried_headers
    kept_headers = (
        queried_headers if requested == "all"
        else tuple(header for header in queried_headers if header != "vernacularName")
    )
    assert worksheet_rows(tmp_path / result["files"][0]["filename"]) == [
        kept_headers, tuple(record[header] for header in kept_headers),
    ]
    assert result["record_count"] == result["files"][0]["record_count"] == 1
    assert result["files"][0]["datasets"] == [{
        "dataset_key": "birds", "sheet_name": "birds", "row_count": 1,
        "skipped_empty_columns": (
            ["vernacularName"] if requested != "all" and column_mode != "selected" else []
        ),
    }]
    assert result["database_truncated"] is False
    assert engine.connection.events[-3:] == ["commit", "connection_closed", "disposed"]


@pytest.mark.parametrize("identifier", ("occurrenceID", "id"))
@pytest.mark.parametrize("column_mode", ("default", "selected", "all"))
@pytest.mark.parametrize("has_records", (False, True))
def test_preferred_identifier_survives_empty_pruning_and_id_only_tables(
    tmp_path: Path,
    identifier: str,
    column_mode: str,
    has_records: bool,
) -> None:
    physical_headers = ("id", "occurrenceID", "scientificName", "locality") if identifier == "occurrenceID" else (
        "id", "scientificName", "locality",
    )
    queried_headers = (identifier, "scientificName", "locality") if column_mode == "all" else (
        identifier, "scientificName",
    )
    source_records = (
        {identifier: value, "scientificName": "", "locality": " \t"}
        for value in ("", "000012345678901234567890")
    )
    records = tuple(source_records) if has_records else ()
    engine = StreamingEngine({
        "birds": tuple(tuple(record[header] for header in queried_headers) for record in records),
    })
    configured = GaiaToolset(
        {},
        {"birds": make_table(MetaData(), "birds", physical_headers)},
        {"Collection": {"Birds": "birds"}},
        lambda: engine,
        tmp_path,
    )
    result = asyncio.run(_handler(configured)(_arguments([_query(
        "Birds", return_columns="all" if column_mode == "all" else ("scientificName",),
        column_mode=column_mode,
    )])))

    statement = engine.connection.statements[0][1]
    assert tuple(column.name for column in statement.selected_columns) == queried_headers
    kept = queried_headers if column_mode == "all" else (identifier,)
    expected = [kept]
    expected.extend(tuple(record[header] for header in kept) for record in records)
    assert worksheet_rows(tmp_path / result["files"][0]["filename"]) == expected
    assert result["record_count"] == len(records)
    assert result["files"][0]["datasets"][0]["skipped_empty_columns"] == (
        [] if column_mode == "all" else ["scientificName"]
    )


@pytest.mark.parametrize("layout", ("single_workbook", "separate_files"))
def test_mixed_column_modes_keep_empty_fields_only_for_full_column_datasets(
    tmp_path: Path,
    layout: str,
) -> None:
    metadata = MetaData()
    physical_headers = ("occurrenceID", "id", "scientificName", "family", "locality", "recordedBy")
    records = {
        "birds": {
            "occurrenceID": "bird-1", "id": "00001", "scientificName": "",
            "family": "Corvidae", "locality": "\u2003", "recordedBy": "Γιώργος",
        },
        "fish": {
            "occurrenceID": "fish-1", "id": "", "scientificName": "",
            "family": "", "locality": " \t\n", "recordedBy": "",
        },
        "mammals": {
            "occurrenceID": "mammal-1", "id": "00003", "scientificName": "",
            "family": "Felidae", "locality": "=1+2", "recordedBy": "Collector",
        },
    }
    queried_headers = {
        "birds": ("occurrenceID", "scientificName", "recordedBy", "locality"),
        "fish": tuple(header for header in physical_headers if header != "id"),
        "mammals": ("occurrenceID", "locality", "scientificName"),
    }
    kept_headers = {
        "birds": ("occurrenceID", "recordedBy"),
        "fish": tuple(header for header in physical_headers if header != "id"),
        "mammals": ("occurrenceID", "locality"),
    }
    skipped = {
        "birds": ["scientificName", "locality"],
        "fish": [],
        "mammals": ["scientificName"],
    }
    engine = StreamingEngine({
        key: (tuple(records[key][header] for header in headers),)
        for key, headers in queried_headers.items()
    })
    configured = GaiaToolset(
        {},
        {key: make_table(metadata, key, physical_headers) for key in records},
        {"Collection": {"Birds": "birds", "Fish": "fish", "Mammals": "mammals"}},
        lambda: engine,
        tmp_path,
    )
    result = asyncio.run(_handler(configured)(_arguments([
        _query("Birds", return_columns=("recordedBy", "locality")),
        _query("Fish", return_columns="all"),
        _query("Mammals", return_columns=("locality", "scientificName"), column_mode="selected"),
    ], layout=layout)))

    assert result["record_count"] == 3
    assert len(result["files"]) == (1 if layout == "single_workbook" else 3)
    assert sum(file["record_count"] for file in result["files"]) == 3
    exported_keys = []
    for file in result["files"]:
        for index, sheet in enumerate(file["datasets"], 1):
            key = sheet["dataset_key"]
            exported_keys.append(key)
            headers = kept_headers[key]
            assert worksheet_rows(tmp_path / file["filename"], index) == [
                headers, tuple(records[key][header] for header in headers),
            ]
            assert sheet["row_count"] == 1
            assert sheet["skipped_empty_columns"] == skipped[key]
    assert exported_keys == list(records)
    assert [
        (key, tuple(column.name for column in statement.selected_columns))
        for key, statement, _options in engine.connection.statements
    ] == list(queried_headers.items())
    assert engine.connection.events.count("begin") == 1
    assert engine.connection.events[-3:] == ["commit", "connection_closed", "disposed"]


def test_collection_default_columns_are_resolved_independently_for_each_worksheet(
    tmp_path: Path,
) -> None:
    metadata = MetaData()
    physical_headers = (
        "occurrenceID", "id", "vernacularName", "scientificName", "kingdom",
        "phylum", "class", "order", "family", "locality", "occurrenceRemarks",
    )
    expected = {
        "birds": ("occurrenceID", *BIRD_DEFAULT_COLUMNS, "locality"),
        "mammals": (
            "occurrenceID", "vernacularName", "kingdom", "phylum", "class",
            "order", "locality",
        ),
        "fish": ("occurrenceID", "kingdom", "phylum", "class", "order", "family", "locality"),
    }
    records = {
        key: {
            **{header: f"{key}-{header}" for header in physical_headers},
            "id": "000123", "locality": "Vancouver",
        }
        for key in expected
    }
    engine = StreamingEngine({
        key: (tuple(records[key][header] for header in headers),)
        for key, headers in expected.items()
    })
    configured = GaiaToolset(
        {},
        {
            key: make_table(
                metadata, key,
                tuple(header for header in physical_headers if key != "fish" or header != "id"),
            )
            for key in expected
        },
        {"Collection": {"Birds": "birds", "Mammals": "mammals", "Fish": "fish"}},
        lambda: engine,
        tmp_path,
    )
    query = _query("Birds", return_columns=("locality",))
    query["scope"] = {"collection": "Collection"}
    result = asyncio.run(_handler(configured)(_arguments([query])))

    assert result["record_count"] == 3
    assert len(result["files"]) == 1
    for index, (key, headers) in enumerate(expected.items(), 1):
        assert worksheet_rows(tmp_path / result["files"][0]["filename"], index) == [
            headers, tuple(records[key][header] for header in headers),
        ]
    assert [
        (key, tuple(column.name for column in statement.selected_columns))
        for key, statement, _options in engine.connection.statements
    ] == list(expected.items())
    assert [sheet["row_count"] for sheet in result["files"][0]["datasets"]] == [1, 1, 1]
    assert all(not sheet["skipped_empty_columns"] for sheet in result["files"][0]["datasets"])
    assert engine.connection.events.count("begin") == 1
    assert engine.connection.events[-3:] == ["commit", "connection_closed", "disposed"]


@pytest.mark.parametrize("limit", (200, 275))
def test_finite_generation_limit_is_shared_and_unqueried_sheets_are_reported(
    toolset, monkeypatch, limit,
):
    calls = _install_stream(monkeypatch, {
        "birds": tuple(
            {"occurrenceID": str(index), "catalogNumber": "", "locality": "Vancouver"}
            for index in range(400)
        ),
        "fish": ({"occurrenceID": "fish-1", "scientificName": "Gadus morhua"},),
    })
    result = asyncio.run(_handler(toolset)({
        "layout": "single_workbook", "limit_scope": "total", "queries": [_query("Birds", return_columns="all"), _query("Fish", return_columns="all")], "limit": limit,
    }))
    assert calls[0]["row_limit"] == limit
    assert result["record_count"] == limit
    assert result["limit"] == limit
    assert result["database_truncated"] is True
    assert result["unqueried_dataset_keys"] == ["fish"]
    assert result["files"][0]["datasets"] == [
        {
            "dataset_key": "birds", "sheet_name": "birds", "row_count": limit,
            "skipped_empty_columns": [],
        },
        {
            "dataset_key": "fish", "sheet_name": "fish", "row_count": 0,
            "skipped_empty_columns": [],
        },
    ]
    path = toolset.export_directory / result["files"][0]["filename"]
    bird_rows = worksheet_rows(path)
    assert bird_rows[0] == ("occurrenceID", "catalogNumber", "locality")
    assert len(bird_rows) == limit + 1
    assert bird_rows[-1] == (str(limit - 1), "", "Vancouver")
    assert worksheet_rows(path, 2) == [("occurrenceID", "scientificName")]


@pytest.mark.parametrize("layout", ("single_workbook", "separate_files"))
def test_collection_export_writes_100_records_per_dataset_in_requested_layout(
    tmp_path: Path,
    layout: str,
) -> None:
    metadata = MetaData()
    keys = ("birds", "mammals", "herpetology", "fish")
    headers = ("occurrenceID", "catalogNumber", "locality", "scientificName")
    configured = GaiaToolset(
        {},
        {key: make_table(metadata, key, headers) for key in keys},
        {
            "Tetrapods": {
                "Birds": "birds", "Mammals": "mammals", "Herpetology": "herpetology",
            },
            "Fish Collection": {"Fish": "fish"},
        },
        lambda: None,
        tmp_path,
    )
    source_rows = {
        key: tuple(
            {
                "occurrenceID": f"{key}-{index:05d}",
                "catalogNumber": f"{index:07d}",
                "locality": "Vancouver" if index == 124 else " \t",
                "scientificName": "Élan 日本語 🐦" if index else "=1+2",
            }
            for index in range(125)
        )
        for key in keys
    }
    requested = ("scientificName", "locality", "catalogNumber")
    engine = StreamingEngine({
        key: tuple(
            tuple(record[header] for header in ("occurrenceID", *requested))
            for record in rows
        )
        for key, rows in source_rows.items()
    })
    configured = replace(configured, engine_factory=lambda: engine)
    collection_query = _query("Birds", return_columns=requested, column_mode="selected")
    collection_query["scope"] = {"collection": "Tetrapods"}
    result = asyncio.run(_handler(configured)(_arguments(
        [collection_query, _query("Fish", return_columns=requested, column_mode="selected")],
        limit=100, limit_scope="per_dataset", layout=layout, filename="helloXXX",
    )))

    assert [key for key, _statement, _options in engine.connection.statements] == list(keys)
    assert [
        statement._limit_clause.value
        for _key, statement, _options in engine.connection.statements
    ] == [101] * 4
    assert engine.connection.events.count("connect") == 1
    assert engine.connection.options == {
        "isolation_level": "REPEATABLE READ", "postgresql_readonly": True,
    }
    assert engine.connection.events[-3:] == ["commit", "connection_closed", "disposed"]
    assert result["record_count"] == 400
    assert result["layout"] == layout
    assert result["limit"] == 100
    assert result["limit_scope"] == "per_dataset"
    assert result["unqueried_dataset_keys"] == []
    assert result["skipped_datasets"] == []
    assert result["database_truncated"] is True
    files = result["files"]
    assert len(files) == (1 if layout == "single_workbook" else 4)
    assert [file["filename"] for file in files] == (
        ["helloXXX.xlsx"] if layout == "single_workbook"
        else [f"helloXXX-{key}.xlsx" for key in keys]
    )
    assert [file["record_count"] for file in files] == (
        [400] if layout == "single_workbook" else [100] * 4
    )
    expected_headers = ("occurrenceID", "scientificName", "catalogNumber")
    exported_keys = []
    for file in files:
        assert set(file) == {"file_id", "filename", "record_count", "datasets"}
        path = tmp_path / file["filename"]
        assert path.name == file["filename"]
        with ZipFile(path) as archive:
            sheets = [name for name in archive.namelist() if name.startswith("xl/worksheets/sheet")]
            assert len(sheets) == len(file["datasets"])
        for sheet_number, sheet in enumerate(file["datasets"], 1):
            key = sheet["dataset_key"]
            exported_keys.append(key)
            assert sheet == {
                "dataset_key": key, "sheet_name": key, "row_count": 100,
                "skipped_empty_columns": ["locality"],
            }
            assert worksheet_rows(path, sheet_number) == [
                expected_headers,
                *(tuple(record[header] for header in expected_headers) for record in source_rows[key][:100]),
            ]
    assert exported_keys == list(keys)
    assert set(tmp_path.iterdir()) == {tmp_path / file["filename"] for file in files}


async def _execute_chat(plan, *, chat_limits: list[int], **kwargs):
    chat_limits.append(plan.limit)
    return query_result(
        records=tuple(
            {"dataset_key": "birds", "occurrenceID": str(index)}
            for index in range(plan.limit)
        ),
        return_columns=plan.return_columns,
        limit=plan.limit,
    )


async def _export_after_chat_limit(handlers, export_directory: Path) -> None:
    chat_arguments = _query("Birds", return_columns="all")
    chat_arguments["result"]["limit"] = 200
    chat_result = await handlers["query_occurrences"](chat_arguments)
    assert len(chat_result.records) == 200
    exported = await handlers["generate_excel"]({
        "layout": "single_workbook", "limit_scope": "total", "queries": [_query("Birds", return_columns=("locality",), column_mode="selected")], "limit": None,
    })
    assert exported["record_count"] == 350
    assert len(worksheet_rows(export_directory / exported["files"][0]["filename"])) == 351
    chat_arguments["result"]["limit"] = 1
    with pytest.raises(ToolExecutionError) as error:
        await handlers["query_occurrences"](chat_arguments)
    assert error.value.code == "chat_record_limit_reached"


def test_unlimited_export_remains_available_after_chat_budget_is_exhausted(
    toolset: GaiaToolset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_limits = []

    monkeypatch.setattr(budget_module, "execute_query_plan", partial(
        _execute_chat, chat_limits=chat_limits,
    ))
    calls = _install_stream(monkeypatch, {
        "birds": tuple(
            {"occurrenceID": str(index), "catalogNumber": "", "locality": "Vancouver"}
            for index in range(350)
        ),
    })
    handlers = {
        tool.definition.name: tool.handler for tool in toolset.registered_tools()
    }

    asyncio.run(_export_after_chat_limit(handlers, toolset.export_directory))
    assert chat_limits == [200]
    assert len(calls) == 1 and calls[0]["row_limit"] is None


@pytest.mark.parametrize("explicit_columns", (False, True))
@pytest.mark.parametrize("layout", ("single_workbook", "separate_files"))
def test_unlimited_export_keeps_late_populated_columns_beyond_200_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit_columns: bool,
    layout: str,
) -> None:
    headers = (
        "occurrenceID", "catalogNumber", "locality", "scientificName",
        "verbatimLocality", "occurrenceRemarks",
    )
    requested_columns = (
        "verbatimLocality", "occurrenceRemarks", "locality", "scientificName",
        "catalogNumber",
    ) if explicit_columns else ()
    toolset = GaiaToolset(
        {},
        {"birds": make_table(MetaData(), "birds", headers)},
        {"Collection": {"Birds": "birds"}},
        lambda: None,
        tmp_path,
    )
    records = tuple(
        {
            "occurrenceID": f"000{index:03d}",
            "catalogNumber": "0" if index == 0 else "",
            "locality": " \t\n" if index % 2 else "",
            "scientificName": (
                "Élan 日本語 🐦" if index == 349 else "\u2003" if index == 1 else ""
            ),
            "verbatimLocality": "\u00a0" if index % 2 else "",
            "occurrenceRemarks": (
                "=1+2" if index == 0 else "  preserved \t\n " if index == 349 else ""
            ),
        }
        for index in range(350)
    )
    calls = _install_stream(monkeypatch, {"birds": records}, batch_size=73)
    spools = _track_spools(monkeypatch)
    result = asyncio.run(_handler(toolset)(_arguments(
        [_query("Birds", return_columns=requested_columns if explicit_columns else "all", column_mode="selected" if explicit_columns else None)], layout=layout,
    )))

    expected_headers = (
        ("occurrenceID", "occurrenceRemarks", "scientificName", "catalogNumber")
        if explicit_columns else
        headers
    )
    expected_skipped = (
        ["verbatimLocality", "locality"]
        if explicit_columns else []
    )
    rows = worksheet_rows(tmp_path / result["files"][0]["filename"])
    assert rows[0] == expected_headers
    assert rows[1:] == [tuple(record[header] for header in expected_headers) for record in records]
    assert result["record_count"] == 350
    assert result["limit"] is None
    assert result["database_truncated"] is False
    assert result["unqueried_dataset_keys"] == []
    assert result["files"][0]["datasets"] == [{
        "dataset_key": "birds", "sheet_name": "birds", "row_count": 350,
        "skipped_empty_columns": expected_skipped,
    }]
    assert len(calls) == 1
    assert calls[0]["plans"][0].return_columns == (requested_columns or headers)
    assert calls[0]["row_limit"] is None
    assert spools and all(spool.closed for spool in spools)
    assert list(tmp_path.iterdir()) == [tmp_path / result["files"][0]["filename"]]


def test_empty_columns_are_decided_independently_for_each_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = MetaData()
    headers = ("occurrenceID", "catalogNumber", "locality")
    toolset = GaiaToolset(
        {},
        {key: make_table(metadata, key, headers) for key in ("birds", "fish")},
        {"Collection": {"Birds": "birds", "Fish": "fish"}},
        lambda: None,
        tmp_path,
    )
    _install_stream(monkeypatch, {
        "birds": ({"occurrenceID": "bird-1", "catalogNumber": "0001", "locality": "\t"},),
        "fish": ({"occurrenceID": "fish-1", "catalogNumber": " ", "locality": "Vancouver"},),
    })
    spools = _track_spools(monkeypatch)
    query = _query("Birds", return_columns=("catalogNumber", "locality"))
    query["scope"] = {"collection": "Collection"}
    result = asyncio.run(_handler(toolset)({"layout": "single_workbook", "limit_scope": "total", "queries": [query], "limit": None}))

    path = tmp_path / result["files"][0]["filename"]
    assert worksheet_rows(path) == [
        ("occurrenceID", "catalogNumber"), ("bird-1", "0001"),
    ]
    assert worksheet_rows(path, 2) == [
        ("occurrenceID", "locality"), ("fish-1", "Vancouver"),
    ]
    assert result["record_count"] == 2
    assert result["files"][0]["datasets"] == [
        {
            "dataset_key": "birds", "sheet_name": "birds", "row_count": 1,
            "skipped_empty_columns": ["locality"],
        },
        {
            "dataset_key": "fish", "sheet_name": "fish", "row_count": 1,
            "skipped_empty_columns": ["catalogNumber"],
        },
    ]
    assert spools and all(spool.closed for spool in spools)


@pytest.mark.parametrize("explicit_columns", (False, True))
@pytest.mark.parametrize("layout", ("single_workbook", "separate_files"))
def test_zero_matches_preserves_all_requested_headers_only_for_all_mode(
    toolset: GaiaToolset,
    monkeypatch: pytest.MonkeyPatch,
    explicit_columns: bool,
    layout: str,
) -> None:
    _install_stream(monkeypatch, {"birds": ()})
    requested_columns = ("locality", "catalogNumber") if explicit_columns else ()
    result = asyncio.run(_handler(toolset)(_arguments([
        _query(
            "Birds", return_columns=requested_columns if explicit_columns else "all",
            column_mode="selected" if explicit_columns else None,
        ),
    ], layout=layout)))

    assert worksheet_rows(toolset.export_directory / result["files"][0]["filename"]) == [
        ("occurrenceID",) if explicit_columns else ("occurrenceID", "catalogNumber", "locality"),
    ]
    assert result["record_count"] == 0
    assert result["database_truncated"] is False
    assert result["unqueried_dataset_keys"] == []
    assert result["files"][0]["datasets"] == [{
        "dataset_key": "birds", "sheet_name": "birds", "row_count": 0,
        "skipped_empty_columns": list(requested_columns) if explicit_columns else [],
    }]


def test_all_mode_requires_the_literal_all_before_creating_output(toolset, tmp_path):
    destination = tmp_path / "not_created"
    configured = replace(toolset, export_directory=destination)
    with pytest.raises(ToolExecutionError) as error:
        asyncio.run(_handler(configured)(_arguments([
            _query("Birds", return_columns=(), column_mode="all"),
        ])))
    assert error.value.code == "invalid_query_plan"
    assert not destination.exists()


@pytest.mark.parametrize("limit", (0, -1, True, "200", 1.5))
def test_invalid_limits_create_no_files(toolset, limit):
    with pytest.raises(ToolExecutionError, match="limit must be"):
        asyncio.run(_handler(toolset)({"layout": "single_workbook", "limit_scope": "total", "queries": [_query("Birds", return_columns="all")], "limit": limit}))
    assert list(toolset.export_directory.iterdir()) == []


def test_overlapping_scopes_fail_before_file_creation(toolset):
    with pytest.raises(ToolExecutionError, match="more than one export query"):
        asyncio.run(_handler(toolset)({
            "layout": "single_workbook", "limit_scope": "total", "queries": [_query("Birds", return_columns="all"), _query("Birds", return_columns="all")], "limit": None,
        }))
    assert list(toolset.export_directory.iterdir()) == []


@pytest.mark.parametrize(
    "changed",
    (
        {"limit_scope": "all"}, {"limit_scope": None}, {"limit_scope": []},
        {"layout": "tabs"}, {"layout": None}, {"layout": []},
        {"unexpected": True},
    ),
)
def test_invalid_layout_or_limit_scope_prevents_file_and_database_work(
    toolset, monkeypatch, tmp_path, changed,
):
    destination = tmp_path / "not_created"
    configured = replace(toolset, export_directory=destination)
    arguments = _arguments([_query("Birds", return_columns="all")]) | changed

    monkeypatch.setattr(generation_module, "stream_query_plans", _forbidden_stream)
    with pytest.raises(ToolExecutionError) as error:
        asyncio.run(_handler(configured)(arguments))
    assert error.value.code == "invalid_query_plan"
    assert not destination.exists()


@pytest.mark.parametrize("missing", ("layout", "limit_scope"))
def test_layout_and_limit_scope_are_required(toolset, tmp_path, missing):
    destination = tmp_path / "not_created"
    configured = replace(toolset, export_directory=destination)
    arguments = _arguments([_query("Birds", return_columns="all")])
    del arguments[missing]
    with pytest.raises(ToolExecutionError, match="requires queries"):
        asyncio.run(_handler(configured)(arguments))
    assert not destination.exists()


def test_later_invalid_query_prevents_any_export(toolset, monkeypatch, tmp_path):
    destination = tmp_path / "not_created"
    configured = replace(toolset, export_directory=destination)

    monkeypatch.setattr(generation_module, "stream_query_plans", _forbidden_stream)
    with pytest.raises(ToolExecutionError) as error:
        asyncio.run(_handler(configured)(_arguments([_query("Birds", return_columns="all"), _query("Unknown")])))
    assert error.value.code == "invalid_query_plan"
    assert not destination.exists()


async def _stream_then_fail(plans, *, previous: Path, **kwargs):
    on_batch = kwargs["on_batch"]
    await on_batch("birds", ("occurrenceID", "catalogNumber", "locality"), ())
    await on_batch("birds", ("occurrenceID", "catalogNumber", "locality"), (("one", "0001", ""),))
    await kwargs["on_dataset_complete"]("birds")
    new_files = set(previous.parent.glob("*.xlsx")) - {previous}
    assert not new_files  # Database fetching completes before workbook creation.
    await on_batch("fish", ("occurrenceID", "scientificName"), ())
    await on_batch("fish", ("occurrenceID", "scientificName"), (("fish-1", "日本語"),))
    raise generation_module.QueryExecutionError("query_unavailable", "Unavailable")


async def _stream_then_cancel(plans, **kwargs):
    await kwargs["on_batch"]("birds", ("occurrenceID", "catalogNumber", "locality"), ())
    await kwargs["on_batch"](
        "birds", ("occurrenceID", "catalogNumber", "locality"), (("one", "0001", ""),),
    )
    await kwargs["on_dataset_complete"]("birds")
    await kwargs["on_batch"]("fish", ("occurrenceID", "scientificName"), ())
    await kwargs["on_batch"]("fish", ("occurrenceID", "scientificName"), (("fish-1", "日本語"),))
    raise asyncio.CancelledError


@dataclass
class _ReplayCancellation:
    cancellation: AgentCancellation
    append: Callable[..., None]
    written: int = 0


def _append_then_cancel(writer, dataset_key, values, *, counter: _ReplayCancellation) -> None:
    counter.append(writer, dataset_key, values)
    if dataset_key == "fish":
        counter.written += 1
        if counter.written == 5:
            counter.cancellation.cancel()


def _finalize_then_fail(writer, *, finalize, finalized_paths: list[Path]):
    result = finalize(writer)
    finalized_paths.append(result.path)
    if len(finalized_paths) == 2:
        raise OSError("failure after the second file was published")
    return result


class _FailingSpoolTracker(_SpoolTracker):
    def __init__(self, monkeypatch: pytest.MonkeyPatch, export_directory: Path) -> None:
        super().__init__()
        self.monkeypatch = monkeypatch
        self.export_directory = export_directory

    def fail_write(self, text: str) -> None:
        raise OSError(f"disk full at {self.export_directory / 'private-spool'}")

    def create_spool(self, *args, **kwargs) -> TextIO:
        spool = super().create_spool(*args, **kwargs)
        self.monkeypatch.setattr(spool, "write", self.fail_write)
        return spool


@pytest.mark.parametrize("layout", ("single_workbook", "separate_files"))
def test_failure_in_second_dataset_discards_new_files_and_closes_spool(
    toolset, monkeypatch, layout,
):
    spools = _track_spools(monkeypatch)
    previous = toolset.export_directory / (
        "previous.xlsx" if layout == "single_workbook" else "previous-birds.xlsx"
    )
    previous.write_bytes(b"previous session export")

    monkeypatch.setattr(generation_module, "stream_query_plans", partial(
        _stream_then_fail, previous=previous,
    ))
    with pytest.raises(ToolExecutionError, match="Unavailable") as error:
        asyncio.run(_handler(toolset)(_arguments(
            [_query("Birds", return_columns="all"), _query("Fish", return_columns="all")], layout=layout,
            filename="previous",
        )))
    assert error.value.code == "query_unavailable"
    assert spools and all(spool.closed for spool in spools)
    assert previous.read_bytes() == b"previous session export"
    assert list(toolset.export_directory.iterdir()) == [previous]


@pytest.mark.parametrize("layout", ("single_workbook", "separate_files"))
def test_cancelled_database_stream_removes_new_files_and_closes_spool(
    toolset, monkeypatch, layout,
):
    spools = _track_spools(monkeypatch)
    previous = toolset.export_directory / (
        "previous.xlsx" if layout == "single_workbook" else "previous-birds.xlsx"
    )
    previous.write_bytes(b"previous session export")

    monkeypatch.setattr(generation_module, "stream_query_plans", _stream_then_cancel)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_handler(toolset)(_arguments(
            [_query("Birds", return_columns="all"), _query("Fish", return_columns="all")], layout=layout,
            filename="previous",
        )))
    assert spools and all(spool.closed for spool in spools)
    assert previous.read_bytes() == b"previous session export"
    assert list(toolset.export_directory.iterdir()) == [previous]


@pytest.mark.parametrize("layout", ("single_workbook", "separate_files"))
def test_cancellation_during_later_spool_replay_discards_new_files(
    toolset, monkeypatch, layout,
):
    cancellation = AgentCancellation()
    previous = toolset.export_directory / "previous.xlsx"
    previous.write_bytes(b"previous session export")
    calls = _install_stream(monkeypatch, {
        "birds": ({"occurrenceID": "bird-1", "catalogNumber": "", "locality": "Vancouver"},),
        "fish": tuple(
            {"occurrenceID": str(index), "scientificName": "日本語"}
            for index in range(1_500)
        ),
    })
    spools = _track_spools(monkeypatch)
    counter = _ReplayCancellation(cancellation, generation_module.ExcelExportWriter.append)
    monkeypatch.setattr(generation_module.ExcelExportWriter, "append", partialmethod(
        _append_then_cancel, counter=counter,
    ))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_handler(toolset, cancellation)(_arguments(
            [_query("Birds", return_columns="all"), _query("Fish", return_columns="all")], layout=layout,
        )))
    assert calls[0]["cancellation"] is cancellation
    assert 5 <= counter.written < 1_500
    assert spools and all(spool.closed for spool in spools)
    assert previous.read_bytes() == b"previous session export"
    assert list(toolset.export_directory.iterdir()) == [previous]


def test_later_finalize_failure_removes_every_file_created_by_this_call(toolset, monkeypatch):
    previous = toolset.export_directory / "previous-birds.xlsx"
    previous.write_bytes(b"previous session export")
    _install_stream(monkeypatch, {
        "birds": ({"occurrenceID": "bird-1", "catalogNumber": "0001", "locality": ""},),
        "fish": ({"occurrenceID": "fish-1", "scientificName": "日本語"},),
    })
    spools = _track_spools(monkeypatch)
    finalize = generation_module.ExcelExportWriter.finalize
    finalized_paths = []

    monkeypatch.setattr(generation_module.ExcelExportWriter, "finalize", partialmethod(
        _finalize_then_fail, finalize=finalize, finalized_paths=finalized_paths,
    ))
    with pytest.raises(ToolExecutionError, match="could not be saved") as error:
        asyncio.run(_handler(toolset)(_arguments(
            [_query("Birds", return_columns="all"), _query("Fish", return_columns="all")], layout="separate_files",
            filename="previous",
        )))
    assert error.value.code == "export_unavailable"
    assert len(finalized_paths) == 2
    assert [path.name for path in finalized_paths] == ["previous-birds-2.xlsx", "previous-fish.xlsx"]
    assert all(not path.exists() for path in finalized_paths)
    assert spools and all(spool.closed for spool in spools)
    assert previous.read_bytes() == b"previous session export"
    assert list(toolset.export_directory.iterdir()) == [previous]


def test_spool_write_error_closes_handle_and_reports_export_unavailable(toolset, monkeypatch):
    _install_stream(monkeypatch, {
        "birds": ({"occurrenceID": "bird-1", "catalogNumber": "0001", "locality": ""},),
    })
    tracker = _FailingSpoolTracker(monkeypatch, toolset.export_directory)
    monkeypatch.setattr(generation_module, "TemporaryFile", tracker.create_spool)
    with pytest.raises(ToolExecutionError, match="could not be saved") as error:
        asyncio.run(_handler(toolset)({"layout": "single_workbook", "limit_scope": "total", "queries": [_query("Birds", return_columns="all")], "limit": None}))
    assert error.value.code == "export_unavailable"
    assert str(toolset.export_directory) not in str(error.value)
    assert tracker.opened and all(spool.closed for spool in tracker.opened)
    assert list(toolset.export_directory.iterdir()) == []


async def _wait_for_thread_event(event: Event) -> None:
    async with asyncio.timeout(3):
        while not event.is_set():
            await asyncio.sleep(0.001)


class _ExportThreadProbe:
    """Record file-operation ownership and pause one stage for loop checks."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
        self.monkeypatch = monkeypatch
        self.operation = operation
        self.started = Event()
        self.release = Event()
        self.owner_ids: list[int] = []
        self.temporary_file = generation_module.TemporaryFile

    def pause(self, stage: str) -> None:
        self.owner_ids.append(get_ident())
        if stage == self.operation and not self.started.is_set():
            self.started.set()
            assert self.release.wait(3), "The event loop did not release the export worker"

    def tracked_write(self, text: str, *, write: Callable[[str], int]) -> int:
        self.pause("spool")
        return write(text)

    def create_spool(self, *args, **kwargs) -> TextIO:
        spool = self.temporary_file(*args, **kwargs)
        self.monkeypatch.setattr(spool, "write", partial(self.tracked_write, write=spool.write))
        return spool


def _tracked_writer_operation(writer, *args, probe: _ExportThreadProbe, stage: str, operation):
    probe.pause(stage)
    return operation(writer, *args)


async def _run_responsive_export(toolset: GaiaToolset, probe: _ExportThreadProbe):
    task = asyncio.create_task(_handler(toolset)(_arguments([
        _query("Birds", return_columns="all"),
    ])))
    try:
        await _wait_for_thread_event(probe.started)
        for _ in range(5):
            await asyncio.sleep(0)
            assert not task.done()
    finally:
        probe.release.set()
    return await asyncio.wait_for(task, 3)


@pytest.mark.parametrize("operation", ("spool", "append", "finalize"))
def test_blocking_export_work_keeps_event_loop_responsive_and_has_one_owner(
    toolset, monkeypatch, operation,
):
    _install_stream(monkeypatch, {
        "birds": ({"occurrenceID": "bird-1", "catalogNumber": "0001", "locality": "日本語"},),
    })
    probe = _ExportThreadProbe(monkeypatch, operation)
    main_thread = get_ident()
    append = generation_module.ExcelExportWriter.append
    finalize = generation_module.ExcelExportWriter.finalize
    monkeypatch.setattr(generation_module, "TemporaryFile", probe.create_spool)
    monkeypatch.setattr(generation_module.ExcelExportWriter, "append", partialmethod(
        _tracked_writer_operation, probe=probe, stage="append", operation=append,
    ))
    monkeypatch.setattr(generation_module.ExcelExportWriter, "finalize", partialmethod(
        _tracked_writer_operation, probe=probe, stage="finalize", operation=finalize,
    ))

    result = asyncio.run(_run_responsive_export(toolset, probe))
    assert result["record_count"] == 1
    assert len(set(probe.owner_ids)) == 1
    assert main_thread not in probe.owner_ids


def _write_after_connection_released(writer, *args, engine: StreamingEngine, operation):
    assert engine.connection.events[-3:] == ["commit", "connection_closed", "disposed"]
    return operation(writer, *args)


def test_database_connection_is_released_before_any_worksheet_writing(toolset, monkeypatch):
    engine = StreamingEngine({"birds": (("bird-1", "0001", "日本語"),)})
    configured = replace(toolset, engine_factory=lambda: engine)
    add_sheet = generation_module.ExcelExportWriter.add_sheet
    finalize = generation_module.ExcelExportWriter.finalize
    monkeypatch.setattr(generation_module.ExcelExportWriter, "add_sheet", partialmethod(
        _write_after_connection_released, engine=engine, operation=add_sheet,
    ))
    monkeypatch.setattr(generation_module.ExcelExportWriter, "finalize", partialmethod(
        _write_after_connection_released, engine=engine, operation=finalize,
    ))
    result = asyncio.run(_handler(configured)(_arguments([
        _query("Birds", return_columns="all"),
    ])))
    assert result["record_count"] == 1


def _blocked_finalization(
    writer,
    *,
    finalize,
    started: Event,
    release: Event,
    finished: Event | None = None,
):
    if not started.is_set():
        started.set()
        try:
            assert release.wait(3)
            return finalize(writer)
        finally:
            if finished is not None:
                finished.set()
    return finalize(writer)


async def _cancel_blocked_export_then_reuse(
    configured: GaiaToolset,
    arguments: dict[str, object],
    *,
    started: Event,
    release: Event,
    finished: Event,
    spools: list[TextIO],
    previous: Path,
):
    task = asyncio.create_task(_handler(configured)(arguments))
    try:
        await _wait_for_thread_event(started)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()  # Repeated cancellation must not abandon the worker.
        await asyncio.sleep(0)
        assert not task.done()
        assert not finished.is_set()
        assert not spools[0].closed
        with pytest.raises(ToolExecutionError) as busy:
            await _handler(configured)(arguments)
        assert busy.value.code == "server_busy"
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 3)
    assert finished.is_set()
    assert all(spool.closed for spool in spools)
    assert list(configured.export_directory.iterdir()) == [previous]
    assert not any(thread.name.startswith("gaia-excel") for thread in threads())
    return await _handler(configured)(arguments)


def test_cancelled_export_drains_blocked_finalization_before_cleanup_and_reuse(
    toolset, monkeypatch,
):
    configured = replace(toolset, export_workers=ExcelExportWorkers(max_workers=1, max_waiting=0))
    previous = toolset.export_directory / "previous.xlsx"
    previous.write_bytes(b"retained export")
    _install_stream(monkeypatch, {
        "birds": ({"occurrenceID": "bird-1", "catalogNumber": "0001", "locality": "日本語"},),
    })
    spools = _track_spools(monkeypatch)
    started = Event()
    release = Event()
    finished = Event()
    monkeypatch.setattr(Workbook, "close", partialmethod(
        _blocked_finalization, finalize=Workbook.close,
        started=started, release=release, finished=finished,
    ))
    arguments = _arguments([_query("Birds", return_columns="all")])

    result = asyncio.run(_cancel_blocked_export_then_reuse(
        configured, arguments,
        started=started, release=release, finished=finished,
        spools=spools, previous=previous,
    ))
    assert result["record_count"] == 1
    assert previous.read_bytes() == b"retained export"
    assert not any(thread.name.startswith("gaia-excel") for thread in threads())


async def _run_exports_at_capacity(
    configured: GaiaToolset,
    arguments: dict[str, object],
    *,
    started: Event,
    release: Event,
    calls: list[dict[str, object]],
    spools: list[TextIO],
):
    first = asyncio.create_task(_handler(configured)(arguments))
    try:
        await _wait_for_thread_event(started)
        second = asyncio.create_task(_handler(configured)(arguments))
        await asyncio.sleep(0)
        with pytest.raises(ToolExecutionError) as busy:
            await _handler(configured)(arguments)
        assert busy.value.code == "server_busy"
        assert len(calls) == len(spools) == 1
        assert not second.done()
    finally:
        release.set()
    return await asyncio.wait_for(asyncio.gather(first, second), 3)


def test_export_capacity_bounds_active_and_waiting_jobs_before_file_or_database_work(
    toolset, monkeypatch,
):
    configured = replace(toolset, export_workers=ExcelExportWorkers(max_workers=1, max_waiting=1))
    calls = _install_stream(monkeypatch, {
        "birds": ({"occurrenceID": "bird-1", "catalogNumber": "0001", "locality": "日本語"},),
    })
    spools = _track_spools(monkeypatch)
    started = Event()
    release = Event()
    monkeypatch.setattr(generation_module.ExcelExportWriter, "finalize", partialmethod(
        _blocked_finalization, finalize=generation_module.ExcelExportWriter.finalize,
        started=started, release=release,
    ))
    arguments = _arguments([_query("Birds", return_columns="all")])

    results = asyncio.run(_run_exports_at_capacity(
        configured, arguments, started=started, release=release, calls=calls, spools=spools,
    ))
    assert len(calls) == len(spools) == 2
    assert all(spool.closed for spool in spools)
    assert all(result["record_count"] == 1 for result in results)
