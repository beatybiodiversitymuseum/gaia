"""Stream validated Gaia JSON Lines records into PostgreSQL tables."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import case, delete, func, insert, select
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.schema import Table

from src.database.database import REPOSITORY_ROOT, create_database_engine
from src.database.datasets import DatasetSchema, load_dataset_catalog
from src.database.schema_profiles import (
    FIELD_AVAILABLE,
    FIELD_UNAVAILABLE,
    load_dataset_schema_profiles,
    write_dataset_schema_profiles,
)
from src.database.tablesetup import build_dataset_table_map, build_table_models
from src.logger import LoggerFactory, bounded_log_detail
from src.scripts.dataset_schema import OUTPUT_RECORD_KEYS, jsonl_relative_path


DEFAULT_BATCH_SIZE = 200
_LOG_DETAIL_LIMIT = 512
_LOG_FILE_LIMIT = 12
_POSIX_WHITESPACE_CHARACTERS = (" ", "\t", "\n", "\v", "\f", "\r")


# ---------------------------------------------------------------------------
# Import contracts
# ---------------------------------------------------------------------------


class PostgreSQLIngestionError(ValueError):
    """Indicate that an ingestion request or JSON Lines source is invalid."""


class DestinationNotEmptyError(PostgreSQLIngestionError):
    """Indicate that a safe default import found existing destination rows."""


class EmptyJsonlError(PostgreSQLIngestionError):
    """Indicate that a JSON Lines source contains no occurrence records."""


class JsonlRecordError(PostgreSQLIngestionError):
    """Indicate that one JSON Lines record violates the import contract."""


class JsonlSchemaError(JsonlRecordError):
    """Identify a record whose raw fields do not match the target table."""

    def __init__(
        self,
        message: str,
        *,
        missing_headers: tuple[str, ...],
    ) -> None:
        super().__init__(message)
        self.missing_headers = missing_headers


@dataclass(frozen=True, slots=True)
class DatasetImportResult:
    """Summarize one committed dataset transaction."""

    dataset_key: str
    table_name: str
    source_path: Path
    rows_inserted: int
    elapsed_ms: float
    header_availability: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class DatasetImportFailure:
    """Describe one rolled-back dataset transaction without record values."""

    dataset_key: str
    table_name: str
    source_path: Path
    error_type: str
    message: str
    elapsed_ms: float
    malformed_records: int = 0
    missing_headers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ImportResult:
    """Summarize all requested dataset transactions."""

    selected_datasets: tuple[str, ...]
    succeeded: tuple[DatasetImportResult, ...]
    failed: tuple[DatasetImportFailure, ...]
    elapsed_ms: float

    @property
    def rows_inserted(self) -> int:
        """Return the number of rows committed across successful datasets."""
        return sum(result.rows_inserted for result in self.succeeded)

    @property
    def status(self) -> str:
        """Return the aggregate command status."""
        return "completed_with_failures" if self.failed else "complete"

    @property
    def malformed_records(self) -> int:
        """Return the number of malformed records detected before rollback."""
        return sum(failure.malformed_records for failure in self.failed)

    @property
    def missing_target_schema_headers(self) -> int:
        """Return the number of missing target headers detected in failures."""
        return sum(len(failure.missing_headers) for failure in self.failed)


# ---------------------------------------------------------------------------
# Registry and request validation
# ---------------------------------------------------------------------------


def resolve_datasets(
    dataset_keys: Sequence[str] | None,
    *,
    datasets: Sequence[DatasetSchema],
) -> tuple[DatasetSchema, ...]:
    """Resolve requested keys in canonical registry order, defaulting to all."""
    registered = tuple(datasets)
    registry = {dataset.key: dataset for dataset in registered}
    if len(registry) != len(registered):
        raise PostgreSQLIngestionError("dataset registry contains duplicate keys")

    if dataset_keys is None:
        return registered
    requested = tuple(dataset_keys)
    if not requested:
        raise PostgreSQLIngestionError("at least one dataset must be selected")
    if len(set(requested)) != len(requested):
        raise PostgreSQLIngestionError("dataset selection contains duplicate keys")

    unknown = sorted(set(requested) - registry.keys())
    if unknown:
        raise PostgreSQLIngestionError(
            f"unknown dataset(s): {', '.join(unknown)}"
        )
    requested_set = frozenset(requested)
    return tuple(
        dataset for dataset in registered if dataset.key in requested_set
    )


def _validate_batch_size(batch_size: int) -> None:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise PostgreSQLIngestionError("batch_size must be a positive integer")


def _table_insert_contract(
    table: Table,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Return the canonical Occurrence columns and their SQL parameter keys."""
    source_headers = table.info.get("source_headers")
    if not isinstance(source_headers, tuple) or not source_headers:
        raise PostgreSQLIngestionError(
            f"{table.name}: table is missing its source_headers contract"
        )
    if any(not isinstance(header, str) or not header for header in source_headers):
        raise PostgreSQLIngestionError(
            f"{table.name}: source_headers must be non-empty strings"
        )

    columns = tuple(table.columns)
    if tuple(column.name for column in columns) != source_headers:
        raise PostgreSQLIngestionError(
            f"{table.name}: physical columns do not exactly match source_headers"
        )
    parameter_keys = tuple((column.name, column.key) for column in columns)
    return source_headers, parameter_keys


# ---------------------------------------------------------------------------
# JSON Lines validation
# ---------------------------------------------------------------------------


class _DuplicateJsonKey(ValueError):
    """Carry a duplicate JSON object key out of ``json.loads``."""


class _NonstandardJsonConstant(ValueError):
    """Carry a non-standard JSON constant out of ``json.loads``."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in pairs:
        if key in parsed:
            raise _DuplicateJsonKey(key)
        parsed[key] = value
    return parsed


def _reject_nonstandard_json_constant(value: str) -> object:
    raise _NonstandardJsonConstant(value)


def parse_jsonl_record(
    line: str,
    *,
    path: Path,
    line_number: int,
    expected_headers: tuple[str, ...],
) -> dict[str, str]:
    """Validate one JSON Lines record and return its lossless raw fields."""
    location = f"{path}:{line_number}"
    if not line.strip():
        raise JsonlRecordError(f"{location}: blank JSONL line")

    try:
        record = json.loads(
            line,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except json.JSONDecodeError as error:
        raise JsonlRecordError(
            f"{location}: invalid JSON: {error.msg}"
        ) from error
    except _DuplicateJsonKey as error:
        raise JsonlRecordError(
            f"{location}: duplicate JSON object key: {error}"
        ) from error
    except _NonstandardJsonConstant as error:
        raise JsonlRecordError(
            f"{location}: non-standard JSON constant is not allowed: {error}"
        ) from error

    if not isinstance(record, dict):
        raise JsonlRecordError(
            f"{location}: record must be a JSON object"
        )
    if tuple(record) != OUTPUT_RECORD_KEYS:
        raise JsonlRecordError(
            f"{location}: top-level keys and order must be exactly "
            + ", ".join(OUTPUT_RECORD_KEYS)
        )

    raw_record = record["raw_record"]
    if not isinstance(raw_record, dict):
        raise JsonlRecordError(
            f"{location}: raw_record must be a JSON object"
        )
    if tuple(raw_record) != expected_headers:
        missing_headers = tuple(
            header for header in expected_headers if header not in raw_record
        )
        raise JsonlSchemaError(
            f"{location}: raw_record keys/order must exactly match the "
            f"{len(expected_headers)} target headers; "
            f"missing_header_count={len(missing_headers)}",
            missing_headers=missing_headers,
        )
    for header, value in raw_record.items():
        if not isinstance(value, str):
            raise JsonlRecordError(
                f"{location}: raw_record.{header} must be a string "
                f"(got {type(value).__name__})"
            )

    for identity_header in ("occurrenceID", "catalogNumber"):
        if record[identity_header] != raw_record[identity_header]:
            raise JsonlRecordError(
                f"{location}: top-level {identity_header} does not match "
                "raw_record"
            )
    if not isinstance(record["chunks"], dict):
        raise JsonlRecordError(
            f"{location}: chunks must be a JSON object"
        )
    return raw_record


# ---------------------------------------------------------------------------
# Transactional ingestion
# ---------------------------------------------------------------------------


def _non_whitespace_expression(
    column: ColumnElement[str],
    *,
    dialect_name: str,
) -> ColumnElement[object]:
    """Return whether a text column contains a non-POSIX-whitespace value."""
    if dialect_name == "postgresql":
        return func.bool_or(column.op("~")(r"[^[:space:]]"))

    without_whitespace = column
    for whitespace in _POSIX_WHITESPACE_CHARACTERS:
        without_whitespace = func.replace(without_whitespace, whitespace, "")
    return func.max(
        case(
            (func.length(without_whitespace) > 0, FIELD_AVAILABLE),
            else_=FIELD_UNAVAILABLE,
        )
    )


def profile_dataset_headers(
    connection: Connection,
    table: Table,
) -> dict[str, int]:
    """Profile every column in one query without changing stored source text."""
    dialect_name = connection.dialect.name
    expressions = tuple(
        _non_whitespace_expression(column, dialect_name=dialect_name).label(
            f"available_{index}"
        )
        for index, column in enumerate(table.columns)
    )
    row = connection.execute(select(*expressions)).one()
    return {
        column.name: FIELD_AVAILABLE if row[index] else FIELD_UNAVAILABLE
        for index, column in enumerate(table.columns)
    }


def ingest_dataset(
    engine: Engine,
    dataset: DatasetSchema,
    table: Table,
    *,
    repository_root: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    replace: bool = False,
) -> DatasetImportResult:
    """Validate and load one canonical Occurrence JSONL transactionally."""
    _validate_batch_size(batch_size)
    expected_headers, parameter_keys = _table_insert_contract(table)
    source_path = repository_root / jsonl_relative_path(dataset)
    started = time.perf_counter()
    rows_inserted = 0
    header_availability: dict[str, int]

    with engine.begin() as connection:
        if replace:
            connection.execute(delete(table))
        elif connection.execute(
            select(1).select_from(table).limit(1)
        ).first() is not None:
            raise DestinationNotEmptyError(
                f"{table.name}: destination is not empty; use --replace to "
                "transactionally clear and reload it"
            )

        statement = insert(table)
        batch: list[dict[str, str]] = []
        try:
            source = source_path.open("r", encoding="utf-8", newline="")
        except FileNotFoundError as error:
            raise PostgreSQLIngestionError(
                f"missing JSONL source: {source_path}"
            ) from error

        with source:
            for line_number, line in enumerate(source, start=1):
                raw_record = parse_jsonl_record(
                    line,
                    path=source_path,
                    line_number=line_number,
                    expected_headers=expected_headers,
                )
                batch.append(
                    {
                        parameter_key: raw_record[source_header]
                        for source_header, parameter_key in parameter_keys
                    }
                )
                if len(batch) == batch_size:
                    connection.execute(statement, batch)
                    rows_inserted += len(batch)
                    batch.clear()

        if batch:
            connection.execute(statement, batch)
            rows_inserted += len(batch)
        if rows_inserted == 0:
            raise EmptyJsonlError(
                f"{source_path}: JSONL source contains no records"
            )
        header_availability = profile_dataset_headers(connection, table)

    return DatasetImportResult(
        dataset_key=dataset.key,
        table_name=table.name,
        source_path=source_path,
        rows_inserted=rows_inserted,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        header_availability=tuple(header_availability.items()),
    )


def _safe_failure_message(error: Exception, *, source_path: Path) -> str:
    """Describe an error without serializing SQL parameters or record values."""
    if isinstance(error, PostgreSQLIngestionError):
        return bounded_log_detail(error, limit=_LOG_DETAIL_LIMIT)
    if isinstance(error, UnicodeError):
        return f"{source_path}: source is not valid UTF-8"
    if isinstance(error, OSError):
        return f"{source_path}: file operation failed"
    if isinstance(error, SQLAlchemyError):
        return "database transaction failed; no rows were committed"
    return "unexpected ingestion failure; no rows were committed"


def import_datasets(
    engine: Engine,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    dataset_keys: Sequence[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    replace: bool = False,
    datasets: Sequence[DatasetSchema],
    table_models: Sequence[object],
    logger: logging.Logger,
    run_id: str,
    invocation: str,
    persist_schema_profiles: bool = True,
) -> ImportResult:
    """Load selected datasets independently and continue after failures."""
    _validate_batch_size(batch_size)
    selected = resolve_datasets(dataset_keys, datasets=datasets)
    tables = build_dataset_table_map(
        datasets=datasets,
        table_models=table_models,
    )
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    selected_keys = tuple(dataset.key for dataset in selected)
    logger.info(
        "ingest_postgresql started run_id=%s started_at=%s file_total=%d "
        "batch_size=%d replace=%s datasets=[%s] invocation=%s",
        run_id,
        started_at,
        len(selected),
        batch_size,
        replace,
        ",".join(selected_keys),
        bounded_log_detail(invocation, limit=_LOG_DETAIL_LIMIT),
    )

    succeeded: list[DatasetImportResult] = []
    failed: list[DatasetImportFailure] = []
    for dataset in selected:
        table = tables[dataset.key]
        source_path = repository_root / jsonl_relative_path(dataset)
        dataset_started = time.perf_counter()
        try:
            result = ingest_dataset(
                engine,
                dataset,
                table,
                repository_root=repository_root,
                batch_size=batch_size,
                replace=replace,
            )
        except Exception as error:  # Each dataset is an independent transaction.
            missing_headers = (
                error.missing_headers
                if isinstance(error, JsonlSchemaError)
                else ()
            )
            failed.append(
                DatasetImportFailure(
                    dataset_key=dataset.key,
                    table_name=table.name,
                    source_path=source_path,
                    error_type=type(error).__name__,
                    message=_safe_failure_message(error, source_path=source_path),
                    elapsed_ms=(time.perf_counter() - dataset_started) * 1000,
                    malformed_records=int(isinstance(error, JsonlRecordError)),
                    missing_headers=missing_headers,
                )
            )
            continue
        succeeded.append(result)

    if persist_schema_profiles and succeeded:
        profiles = load_dataset_schema_profiles(
            repository_root,
            datasets=datasets,
        )
        for dataset_result in succeeded:
            profiles[dataset_result.dataset_key] = dict(
                dataset_result.header_availability
            )
        write_dataset_schema_profiles(
            repository_root,
            profiles,
            datasets=datasets,
        )

    result = ImportResult(
        selected_datasets=selected_keys,
        succeeded=tuple(succeeded),
        failed=tuple(failed),
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
    _log_import_result(logger, run_id, result)
    return result


def _log_import_result(
    logger: logging.Logger, run_id: str, result: ImportResult
) -> None:
    logger.info(
        "ingest_postgresql finished run_id=%s status=%s file_total=%d "
        "file_succeeded=%d file_failed=%d rows_inserted=%d elapsed_ms=%.2f",
        run_id,
        result.status,
        len(result.selected_datasets),
        len(result.succeeded),
        len(result.failed),
        result.rows_inserted,
        result.elapsed_ms,
    )
    logger.info(
        "ingest_postgresql validation_summary run_id=%s "
        "malformed_records=%d missing_target_schema_headers=%d",
        run_id,
        result.malformed_records,
        result.missing_target_schema_headers,
    )
    for dataset_result in result.succeeded[:_LOG_FILE_LIMIT]:
        logger.info(
            "ingest_postgresql dataset_committed dataset=%s table=%s rows=%d "
            "elapsed_ms=%.2f source=%s",
            dataset_result.dataset_key,
            dataset_result.table_name,
            dataset_result.rows_inserted,
            dataset_result.elapsed_ms,
            dataset_result.source_path,
        )
    for failure in result.failed[:_LOG_FILE_LIMIT]:
        shown_headers = failure.missing_headers[:_LOG_FILE_LIMIT]
        logger.error(
            "ingest_postgresql dataset_rolled_back dataset=%s table=%s "
            "error_type=%s reason=%s malformed_records=%d "
            "missing_header_count=%d missing_headers_shown=%d "
            "missing_headers_omitted=%d missing_headers=[%s] "
            "elapsed_ms=%.2f source=%s",
            failure.dataset_key,
            failure.table_name,
            failure.error_type,
            failure.message,
            failure.malformed_records,
            len(failure.missing_headers),
            len(shown_headers),
            len(failure.missing_headers) - len(shown_headers),
            ",".join(shown_headers),
            failure.elapsed_ms,
            failure.source_path,
        )


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the JSONL-to-PostgreSQL importer command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate and stream Gaia JSONL datasets into their PostgreSQL "
            "tables. Existing rows are preserved unless --replace is supplied."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="dataset key to load; repeat for multiple datasets (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_integer,
        default=DEFAULT_BATCH_SIZE,
        help=f"records per bounded insert batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="repository containing the dataset directory",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="transactionally delete and reload each selected known table",
    )
    return parser


def _print_result(result: ImportResult) -> None:
    print(
        f"PostgreSQL ingestion: status={result.status} "
        f"selected={len(result.selected_datasets)} "
        f"succeeded={len(result.succeeded)} failed={len(result.failed)} "
        f"rows_inserted={result.rows_inserted} "
        f"malformed_records={result.malformed_records} "
        f"missing_target_schema_headers="
        f"{result.missing_target_schema_headers}"
    )
    for dataset_result in result.succeeded:
        print(
            f"  committed dataset={dataset_result.dataset_key} "
            f"table={dataset_result.table_name} "
            f"rows={dataset_result.rows_inserted}"
        )
    for failure in result.failed:
        print(
            f"  rolled_back dataset={failure.dataset_key} "
            f"table={failure.table_name} error={failure.error_type}: "
            f"{failure.message}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the JSONL-to-PostgreSQL importer."""
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    target = (
        "all" if arguments.dataset is None else "_".join(arguments.dataset)
    )
    logger_factory = LoggerFactory(
        LoggerFactory.operation_filename("gaia", "ingest_postgresql", target)
    )
    logger = logger_factory.get_logger()
    invocation_parts = (
        tuple(sys.argv)
        if argv is None
        else ("python", "-m", "src.scripts.ingest_postgresql", *argv)
    )
    invocation = shlex.join(invocation_parts)

    try:
        engine = create_database_engine()
    except Exception as error:
        logger.error(
            "ingest_postgresql failed run_id=%s status=failed "
            "error_type=%s reason=database_engine_creation_failed",
            logger_factory.run_id,
            type(error).__name__,
        )
        print(
            "PostgreSQL ingestion failed before loading: database engine "
            f"creation failed ({type(error).__name__}).",
            file=sys.stderr,
        )
        return 1

    try:
        repository_root = arguments.repository_root.resolve()
        datasets = load_dataset_catalog(repository_root)
        table_models = build_table_models(datasets)
        result = import_datasets(
            engine,
            repository_root=repository_root,
            dataset_keys=arguments.dataset,
            batch_size=arguments.batch_size,
            replace=arguments.replace,
            logger=logger,
            run_id=logger_factory.run_id,
            invocation=invocation,
            datasets=datasets,
            table_models=table_models,
        )
    except ValueError as error:
        logger.error(
            "ingest_postgresql failed run_id=%s status=failed "
            "error_type=%s reason=%s",
            logger_factory.run_id,
            type(error).__name__,
            bounded_log_detail(error, limit=_LOG_DETAIL_LIMIT),
        )
        print(f"PostgreSQL ingestion failed: {error}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    _print_result(result)
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
