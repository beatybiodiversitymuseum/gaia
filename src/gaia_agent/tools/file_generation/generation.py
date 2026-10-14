"""Generate local Excel workbooks from streamed occurrence queries."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryFile
from typing import TextIO

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql.schema import Table
from xlsxwriter.exceptions import FileCreateError, FileSizeError

from src.gaia_agent.runtime.contracts import AgentCancellation, ToolExecutionError
from src.gaia_agent.tools.concurrency import WorkloadLimiter
from src.gaia_agent.tools.file_generation.excel_export import (
    ExcelExportResult,
    ExcelExportWriter,
    normalize_excel_filename,
)
from src.gaia_agent.tools.file_generation.workers import ExcelExportWorker, ExcelExportWorkers
from src.gaia_agent.tools.query.columns import preferred_identifier
from src.gaia_agent.tools.query.execution import (
    ExportQueryResult,
    QueryExecutionError,
    resolve_query_datasets,
    stream_query_plans,
)
from src.gaia_agent.tools.query.plan import (
    QueryPlanError,
    ValidatedQueryPlan,
    query_plan_for_dataset,
    validate_query_plan,
)


def _validate_file_arguments(
    arguments: dict[str, object],
    *,
    table_registry: Mapping[str, Table],
    scope_registry: Mapping[str, Mapping[str, str]],
) -> tuple[tuple[ValidatedQueryPlan, ...], int | None, str, str, str | None]:
    """Resolve every export scope before opening a file or a database."""
    required = {"queries", "limit", "limit_scope", "layout"}
    if not required <= set(arguments) or set(arguments) - required - {"filename"}:
        raise QueryPlanError(
            "generate_excel requires queries, limit, limit_scope, and layout; "
            "filename is optional."
        )
    try:
        filename = normalize_excel_filename(arguments["filename"]) if "filename" in arguments else None
    except ValueError as error:
        raise QueryPlanError(str(error)) from error
    queries = arguments["queries"]
    if not isinstance(queries, list) or not queries:
        raise QueryPlanError("queries must be a non-empty array of query plans.")
    if len(queries) > len(table_registry):
        raise QueryPlanError("Use at most one export query per dataset.")
    limit = arguments["limit"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise QueryPlanError(
            "limit must be a positive integer or null for all matching records."
        )
    limit_scope = arguments["limit_scope"]
    if not isinstance(limit_scope, str) or limit_scope not in ("total", "per_dataset"):
        raise QueryPlanError("limit_scope must be total or per_dataset.")
    layout = arguments["layout"]
    if not isinstance(layout, str) or layout not in ("single_workbook", "separate_files"):
        raise QueryPlanError("layout must be single_workbook or separate_files.")

    plans: list[ValidatedQueryPlan] = []
    selected: set[str] = set()
    for query in queries:
        if not isinstance(query, dict):
            raise QueryPlanError("Each export query must be an object.")
        result = query.get("result")
        if not isinstance(result, dict) or set(result) != {"distinct_by", "order_by"}:
            raise QueryPlanError(
                "Each export query.result requires exactly distinct_by and order_by; "
                "set the export limit at the top level."
            )
        plan = validate_query_plan(
            {**query, "result": {**result, "limit": 1}},
            table_registry=table_registry,
            scope_registry=scope_registry,
        )
        for key in plan.dataset_keys:
            if key in selected:
                raise QueryPlanError(
                    f"Dataset {key!r} appears in more than one export query. "
                    "Combine its exact_any filter values in one query."
                )
            selected.add(key)
            plans.append(query_plan_for_dataset(plan, key))
    return tuple(plans), limit, limit_scope, layout, filename


def _dataset_filenames(
    plans: tuple[ValidatedQueryPlan, ...],
    *,
    table_registry: Mapping[str, Table],
    layout: str,
    filename: str | None,
) -> dict[str, str | None]:
    """Resolve all produced names before output, including collection expansion."""
    keys = tuple(
        key
        for plan in plans
        for key, _table in resolve_query_datasets(plan, table_registry=table_registry)[0]
    )
    if layout == "separate_files" and filename is not None and len(keys) > 1:
        return {
            key: normalize_excel_filename(f"{filename[:-5]}-{key}.xlsx")
            for key in keys
        }
    return dict.fromkeys(keys, filename)


def _file_metadata(
    result: ExcelExportResult,
    skipped_empty_columns: Mapping[str, list[str]],
) -> dict[str, object]:
    """Describe one workbook using the same shape for either export layout."""
    return {
        "file_id": result.file_id,
        "filename": result.filename,
        "record_count": sum(sheet.row_count for sheet in result.sheets),
        "datasets": [
            {
                "dataset_key": sheet.dataset_key,
                "sheet_name": sheet.sheet_name,
                "row_count": sheet.row_count,
                "skipped_empty_columns": skipped_empty_columns[sheet.dataset_key],
            }
            for sheet in result.sheets
        ],
    }


@dataclass(slots=True)
class _DatasetSpool:
    headers: tuple[str, ...]
    populated: set[int]
    start: int
    row_count: int = 0


class _ExportJob:
    """Keep every spool and writer operation on the export's owning thread."""

    def __init__(
        self,
        worker: ExcelExportWorker,
        export_directory: Path,
        *,
        all_column_datasets: set[str],
        dataset_filenames: Mapping[str, str | None],
        filename: str | None,
        username: str,
        layout: str,
    ) -> None:
        self.worker = worker
        self.export_directory = export_directory
        self.all_column_datasets = all_column_datasets
        self.dataset_filenames = dataset_filenames
        self.filename = filename
        self.username = username
        self.layout = layout
        self.contexts = ExitStack()
        self.datasets: dict[str, _DatasetSpool] = {}
        self.completed_files: list[ExcelExportResult] = []
        self.skipped_empty_columns: dict[str, list[str]] = {}
        self.spool: TextIO | None = None

    def open(self) -> None:
        self.worker.check_cancellation()
        self.export_directory.mkdir(parents=True, exist_ok=True)
        self.spool = self.contexts.enter_context(
            TemporaryFile(mode="w+t", encoding="utf-8", dir=self.export_directory)
        )

    def receive_batch(
        self,
        dataset_key: str,
        headers: tuple[str, ...],
        rows: tuple[tuple[str, ...], ...],
    ) -> None:
        self.worker.check_cancellation()
        assert self.spool is not None
        if not rows:
            identifier = preferred_identifier(headers)
            self.datasets[dataset_key] = _DatasetSpool(
                headers=headers,
                populated={
                    index for index, header in enumerate(headers)
                    if dataset_key in self.all_column_datasets or header == identifier
                },
                start=self.spool.tell(),
            )
        dataset = self.datasets[dataset_key]
        for row in rows:
            self.worker.check_cancellation()
            dataset.populated.update(
                index for index, value in enumerate(row)
                if index not in dataset.populated and value.strip()
            )
            self.spool.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            self.spool.write("\n")
        dataset.row_count += len(rows)

    def finish_dataset(self, dataset_key: str) -> None:
        self.worker.check_cancellation()
        assert self.spool is not None
        self.spool.flush()

    async def enqueue_batch(
        self,
        dataset_key: str,
        headers: tuple[str, ...],
        rows: tuple[tuple[str, ...], ...],
    ) -> None:
        await self.worker.run(self.receive_batch, dataset_key, headers, rows)

    async def enqueue_dataset_completion(self, dataset_key: str) -> None:
        await self.worker.run(self.finish_dataset, dataset_key)

    def _writer(self, filename: str | None) -> ExcelExportWriter:
        return self.contexts.enter_context(ExcelExportWriter(
            self.export_directory,
            filename=filename,
            username=self.username,
            check_cancellation=self.worker.check_cancellation,
        ))

    def build_workbooks(self) -> None:
        """Replay disk spools only after the database snapshot has been released."""
        self.worker.check_cancellation()
        assert self.spool is not None
        shared_writer = self._writer(self.filename) if self.layout == "single_workbook" else None
        for dataset_key, dataset in self.datasets.items():
            self.worker.check_cancellation()
            writer = shared_writer or self._writer(self.dataset_filenames[dataset_key])
            retained = tuple(
                index for index in range(len(dataset.headers)) if index in dataset.populated
            )
            self.skipped_empty_columns[dataset_key] = [
                header for index, header in enumerate(dataset.headers)
                if index not in dataset.populated
            ]
            writer.add_sheet(
                dataset_key, tuple(dataset.headers[index] for index in retained),
            )
            # Occupancy needs every selected row. Offsets keep the spool bounded
            # in memory while preserving one database snapshot across datasets.
            self.spool.seek(dataset.start)
            for _ in range(dataset.row_count):
                self.worker.check_cancellation()
                row = json.loads(self.spool.readline())
                writer.append(dataset_key, tuple(row[column] for column in retained))
            if shared_writer is None:
                self.worker.check_cancellation()
                self.completed_files.append(writer.finalize())
        if shared_writer is not None:
            self.worker.check_cancellation()
            self.completed_files.append(shared_writer.finalize())
        self.worker.check_cancellation()

    def close(self, succeeded: bool) -> None:
        """Close handles and remove this call's publications on every failed exit."""
        if succeeded:
            self.contexts.close()
        else:
            error = asyncio.CancelledError()
            try:
                self.contexts.__exit__(type(error), error, None)
            finally:
                # Also works when cancellation arrives during successful close.
                for result in self.completed_files:
                    result.path.unlink(missing_ok=True)


def _export_metadata(
    job: _ExportJob,
    query_result: ExportQueryResult,
    *,
    limit: int | None,
    limit_scope: str,
) -> dict[str, object]:
    return {
        "layout": job.layout,
        "files": [
            _file_metadata(result, job.skipped_empty_columns)
            for result in job.completed_files
        ],
        "record_count": sum(
            sheet.row_count
            for result in job.completed_files
            for sheet in result.sheets
        ),
        "skipped_datasets": [
            {
                "dataset_key": skip.dataset_key,
                "reason": skip.reason,
                "missing_headers": list(skip.missing_headers),
            }
            for skip in query_result.skipped_datasets
        ],
        "unqueried_dataset_keys": [
            key for key in query_result.compatible_dataset_keys
            if key not in query_result.queried_dataset_keys
        ],
        "limit": limit,
        "limit_scope": limit_scope,
        "database_truncated": query_result.database_truncated,
    }


async def _run_export_job(
    job: _ExportJob,
    plans: tuple[ValidatedQueryPlan, ...],
    *,
    table_registry: Mapping[str, Table],
    limit: int | None,
    limit_scope: str,
    engine_factory: Callable[[], AsyncEngine],
    cancellation: AgentCancellation | None,
    database_limiter: WorkloadLimiter | None,
) -> dict[str, object]:
    succeeded = False
    try:
        await job.worker.run(job.open)
        query_result = await stream_query_plans(
            plans,
            table_registry=table_registry,
            on_batch=job.enqueue_batch,
            on_dataset_complete=job.enqueue_dataset_completion,
            row_limit=limit,
            limit_scope=limit_scope,
            engine_factory=engine_factory,
            cancellation=cancellation,
            database_limiter=database_limiter,
        )
        if not query_result.compatible_dataset_keys:
            raise ToolExecutionError(
                "invalid_query_plan",
                "No selected dataset supports all requested filter and ordering fields.",
            )
        await job.worker.run(job.build_workbooks)
        job.worker.check_cancellation()
        result = _export_metadata(job, query_result, limit=limit, limit_scope=limit_scope)
        succeeded = True
        return result
    finally:
        try:
            await job.worker.run(job.close, succeeded)
            job.worker.check_cancellation()
        except BaseException:
            await job.worker.run(job.close, False)
            raise


async def generate_excel(
    arguments: dict[str, object],
    *,
    table_registry: Mapping[str, Table],
    scope_registry: Mapping[str, Mapping[str, str]],
    engine_factory: Callable[[], AsyncEngine],
    export_directory: Path,
    username: str = "gaia",
    cancellation: AgentCancellation | None = None,
    export_workers: ExcelExportWorkers | None = None,
    database_limiter: WorkloadLimiter | None = None,
) -> dict[str, object]:
    """Create workbooks with bounded admission and one worker per active export."""
    try:
        plans, limit, limit_scope, layout, filename = _validate_file_arguments(
            arguments,
            table_registry=table_registry,
            scope_registry=scope_registry,
        )
        dataset_filenames = _dataset_filenames(
            plans, table_registry=table_registry, layout=layout, filename=filename,
        )
    except (QueryPlanError, ValueError) as error:
        raise ToolExecutionError("invalid_query_plan", str(error)) from error

    workers = export_workers if export_workers is not None else ExcelExportWorkers()
    try:
        async with workers.slot(cancellation) as worker:
            job = _ExportJob(
                worker,
                export_directory,
                all_column_datasets={
                    plan.dataset_keys[0] for plan in plans if plan.column_mode == "all"
                },
                dataset_filenames=dataset_filenames,
                filename=filename,
                username=username,
                layout=layout,
            )
            return await _run_export_job(
                job,
                plans,
                table_registry=table_registry,
                limit=limit,
                limit_scope=limit_scope,
                engine_factory=engine_factory,
                cancellation=cancellation,
                database_limiter=database_limiter,
            )
    except QueryExecutionError as error:
        raise ToolExecutionError(error.code, str(error)) from error
    except ValueError as error:
        raise ToolExecutionError("invalid_export_data", str(error)) from error
    except (OSError, FileCreateError, FileSizeError) as error:
        raise ToolExecutionError(
            "export_unavailable", "The Excel export could not be saved."
        ) from error
