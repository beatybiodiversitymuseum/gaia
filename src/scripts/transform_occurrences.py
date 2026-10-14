"""Transform compliant Darwin Core Archives into semantic JSON Lines chunks."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import posixpath
import re
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Literal, Mapping, TextIO
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from src.database.datasets import DatasetSchema, write_dataset_catalog
from src.database.darwin_core import (
    OCCURRENCE_ROW_TYPE,
    load_simple_darwin_core_profile,
    term_local_name,
)
from src.database.tablesetup import TABLE_MODELS
from src.database.tablesetup import build_table_models
from src.logger import LoggerFactory, bounded_log_detail
from src.scripts.dataset_schema import CHUNKS, jsonl_relative_path


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

_CHUNK_HEADERS = frozenset(
    field_name
    for chunk_fields in CHUNKS.values()
    for field_name in chunk_fields
)
_REQUIRED_HEADERS = frozenset(("occurrenceID",))
_SIMPLE_DARWIN_CORE_TERMS = frozenset(
    load_simple_darwin_core_profile().term_uris
)

_LOG_FILE_LIMIT = 100
_LOG_DETAIL_LIMIT = 256
_INVOCATION_VALUE_LIMIT = 160
_INVOCATION_LIMIT = 640
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

BatchStatus = Literal[
    "complete",
    "completed_with_failures",
    "completed_with_header_gaps",
    "completed_with_skipped_rows",
]


# ---------------------------------------------------------------------------
# Results and errors
# ---------------------------------------------------------------------------


class HeaderValidationError(ValueError):
    """Indicate that an input header cannot satisfy the output contract."""


@dataclass(frozen=True, slots=True)
class ArchiveField:
    """Describe one indexed or defaulted field declared in meta.xml."""

    term: str
    name: str
    index: int | None
    default: str


@dataclass(frozen=True, slots=True)
class ArchiveTable:
    """Describe the Occurrence core table declared by archive metadata."""

    row_type: str
    locations: tuple[str, ...]
    fields: tuple[ArchiveField, ...]
    encoding: str
    delimiter: str
    quote_character: str | None
    ignore_header_lines: int
    id_index: int | None = None
    core_id_index: int | None = None


@dataclass(frozen=True, slots=True)
class ArchiveMetadata:
    """Carry the authoritative core and reserved future extension metadata."""

    core: ArchiveTable
    extensions: tuple[ArchiveTable, ...]


@dataclass(frozen=True, slots=True)
class DatasetSource:
    """Identify a TSV source on disk or inside a Darwin Core archive."""

    path: Path
    archive_member: str | None = None
    archive_metadata: ArchiveMetadata | None = None

    @property
    def label(self) -> Path:
        """Return a printable path that includes an archive member when present."""
        if self.archive_member is None:
            return self.path
        return Path(f"{self.path}!{self.archive_member}")


@dataclass(frozen=True, slots=True)
class FileResult:
    """Summarize one successfully converted source file."""

    input_path: Path
    output_path: Path
    rows_written: int
    malformed_rows_skipped: int
    missing_headers: tuple[str, ...]
    unmapped_headers: tuple[str, ...]
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class FileFailure:
    """Describe one source file that could not be converted."""

    input_path: Path
    error_type: str
    message: str
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Summarize a requested batch of file conversions."""

    converted_files: tuple[FileResult, ...]
    failed_files: tuple[FileFailure, ...]
    elapsed_ms: float

    @property
    def rows_written(self) -> int:
        """Return the total number of serialized occurrence rows."""
        return sum(result.rows_written for result in self.converted_files)

    @property
    def malformed_rows_skipped(self) -> int:
        """Return the total number of malformed source rows skipped."""
        return sum(
            result.malformed_rows_skipped for result in self.converted_files
        )

    @property
    def files_with_missing_headers(self) -> tuple[FileResult, ...]:
        """Return converted files whose source omitted target-schema fields."""
        return tuple(
            result for result in self.converted_files if result.missing_headers
        )

    @property
    def missing_header_count(self) -> int:
        """Return the total number of absent target-schema columns."""
        return sum(
            len(result.missing_headers) for result in self.converted_files
        )

    @property
    def files_with_unmapped_headers(self) -> tuple[FileResult, ...]:
        """Return converted files containing fields outside the chunk map."""
        return tuple(
            result for result in self.converted_files if result.unmapped_headers
        )

    @property
    def unmapped_header_count(self) -> int:
        """Return the total number of source columns outside the chunk map."""
        return sum(
            len(result.unmapped_headers) for result in self.converted_files
        )

    @property
    def status(self) -> BatchStatus:
        """Classify the aggregate operational outcome."""
        if self.failed_files:
            return "completed_with_failures"
        if self.malformed_rows_skipped:
            return "completed_with_skipped_rows"
        if self.files_with_missing_headers or self.files_with_unmapped_headers:
            return "completed_with_header_gaps"
        return "complete"


# ---------------------------------------------------------------------------
# Record transformation
# ---------------------------------------------------------------------------


def validate_header(header: Sequence[str], input_path: Path) -> None:
    """Require unique named columns and the source identity fields."""
    if not header:
        raise HeaderValidationError(f"{input_path}: missing header row")

    blank_columns = [index for index, name in enumerate(header, start=1) if not name]
    if blank_columns:
        columns = ", ".join(str(index) for index in blank_columns)
        raise HeaderValidationError(
            f"{input_path}: blank header name at column(s): {columns}"
        )

    duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
    if duplicates:
        raise HeaderValidationError(
            f"{input_path}: duplicate header(s): {', '.join(duplicates)}"
        )

    missing = sorted(_REQUIRED_HEADERS.difference(header))
    if missing:
        names = ", ".join(missing)
        raise HeaderValidationError(
            f"{input_path}: missing required header(s): {names}"
        )


def find_missing_chunk_headers(header: Sequence[str]) -> tuple[str, ...]:
    """Return mapped fields absent from one valid source header."""
    available = frozenset(header)
    return tuple(
        field_name
        for chunk_fields in CHUNKS.values()
        for field_name in chunk_fields
        if field_name not in available
    )


def find_unmapped_headers(header: Sequence[str]) -> tuple[str, ...]:
    """Return source fields retained raw but absent from the chunk contract."""
    return tuple(
        field_name for field_name in header if field_name not in _CHUNK_HEADERS
    )


def build_occurrence(
    header: Sequence[str], values: Sequence[str]
) -> dict[str, object]:
    """Build one conditionally chunked occurrence and retain its source row."""
    raw_record = dict(zip(header, values, strict=True))

    occurrence: dict[str, object] = {
        "occurrenceID": raw_record["occurrenceID"],
        "catalogNumber": raw_record["catalogNumber"],
    }
    chunks: dict[str, dict[str, str]] = {}
    for chunk_name, chunk_fields in CHUNKS.items():
        populated_fields = {
            field_name: raw_record[field_name]
            for field_name in chunk_fields
            if raw_record.get(field_name, "") != ""
        }
        if populated_fields:
            chunks[chunk_name] = populated_fields

    occurrence["chunks"] = chunks
    occurrence["raw_record"] = raw_record
    return occurrence


# ---------------------------------------------------------------------------
# Streaming conversion
# ---------------------------------------------------------------------------


def create_tsv_reader(source: TextIO) -> Iterator[list[str]]:
    """Create a tab-only reader that treats quotes inside values as data."""
    return csv.reader(
        source,
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
        strict=True,
    )


def create_archive_reader(
    source: TextIO,
    table: ArchiveTable,
) -> Iterator[list[str]]:
    """Create a reader using the dialect declared by Darwin Core metadata."""
    options: dict[str, object] = {
        "delimiter": table.delimiter,
        "strict": True,
    }
    if table.quote_character is None:
        options["quoting"] = csv.QUOTE_NONE
    else:
        options["quotechar"] = table.quote_character
        options["quoting"] = csv.QUOTE_MINIMAL
    return csv.reader(source, **options)


def _field_values(table: ArchiveTable, values: Sequence[str]) -> dict[str, str]:
    """Apply indexed fields and defaults from a Darwin Core table declaration."""
    record: dict[str, str] = {}
    for field in table.fields:
        value = field.default
        if field.index is not None and field.index < len(values):
            value = values[field.index]
        record[field.name] = value
    return record


def read_header(reader: Iterator[list[str]], input_path: Path) -> list[str]:
    """Read and validate the first TSV record."""
    try:
        header = next(reader)
    except StopIteration as error:
        raise HeaderValidationError(f"{input_path}: file is empty") from error
    except csv.Error as error:
        raise HeaderValidationError(
            f"{input_path}: malformed header row: {error}"
        ) from error

    validate_header(header, input_path)
    return header


def stream_occurrences(
    reader: Iterator[list[str]],
    destination: TextIO,
    header: Sequence[str],
) -> tuple[int, int]:
    """Write valid source rows as JSON objects without buffering the dataset."""
    rows_written = 0
    malformed_rows_skipped = 0
    expected_width = len(header)

    while True:
        try:
            values = next(reader)
        except StopIteration:
            break
        except csv.Error:
            malformed_rows_skipped += 1
            continue

        if len(values) != expected_width:
            malformed_rows_skipped += 1
            continue

        serialized_occurrence = json.dumps(
            build_occurrence(header, values),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        destination.write(serialized_occurrence)
        destination.write("\n")
        rows_written += 1

    return rows_written, malformed_rows_skipped


def convert_file(
    input_path: Path,
    output_path: Path,
) -> FileResult:
    """Validate and stream one standalone TSV to an atomic JSONL output."""
    return convert_source(DatasetSource(input_path), output_path)


@contextmanager
def open_dataset_source(source: DatasetSource) -> Iterator[TextIO]:
    """Open one UTF-8 TSV from disk or a ZIP without extracting the archive."""
    if source.archive_member is None:
        with source.path.open("r", encoding="utf-8-sig", newline="") as stream:
            yield stream
        return

    encoding = (
        source.archive_metadata.core.encoding
        if source.archive_metadata is not None
        else "utf-8-sig"
    )
    with ZipFile(source.path) as archive:
        with archive.open(source.archive_member, "r") as binary:
            with TextIOWrapper(binary, encoding=encoding, newline="") as stream:
                yield stream


def read_source_header(source: DatasetSource) -> tuple[str, ...]:
    """Read one validated TSV header without depending on its filename."""
    with open_dataset_source(source) as stream:
        return tuple(read_header(create_tsv_reader(stream), source.label))


def _skip_rows(reader: Iterator[list[str]], count: int, source: Path) -> None:
    """Skip the metadata-declared leading rows or reject a truncated file."""
    for _index in range(count):
        try:
            next(reader)
        except StopIteration as error:
            raise HeaderValidationError(
                f"{source}: fewer rows than ignoreHeaderLines declares"
            ) from error


def _stream_archive_occurrences(
    source_spec: DatasetSource,
    destination: TextIO,
    output_header: Sequence[str],
) -> tuple[int, int, tuple[str, ...], tuple[str, ...]]:
    """Stream metadata-mapped occurrence rows across all declared core files."""
    metadata = source_spec.archive_metadata
    if metadata is None:
        raise ValueError("archive metadata is required")
    declared_names = tuple(field.name for field in metadata.core.fields)
    validate_header(declared_names, source_spec.label)
    # A compliant archive is a projection of the canonical Occurrence schema.
    # Undeclared registered terms are represented by empty strings below; they
    # are not schema gaps. Every declared core term was already checked against
    # the pinned GBIF Occurrence definition while parsing meta.xml.
    missing_headers: tuple[str, ...] = ()
    unmapped_headers: tuple[str, ...] = ()
    rows_written = 0
    malformed_rows = 0
    with ZipFile(source_spec.path) as archive:
        for member in metadata.core.locations:
            with archive.open(member, "r") as binary:
                with TextIOWrapper(
                    binary,
                    encoding=metadata.core.encoding,
                    newline="",
                ) as stream:
                    reader = create_archive_reader(stream, metadata.core)
                    _skip_rows(
                        reader,
                        metadata.core.ignore_header_lines,
                        source_spec.path,
                    )
                    for values in reader:
                        if any(
                            field.index is not None and field.index >= len(values)
                            for field in metadata.core.fields
                        ):
                            malformed_rows += 1
                            continue
                        mapped = _field_values(metadata.core, values)
                        output_values = tuple(
                            mapped.get(name, "") for name in output_header
                        )
                        destination.write(
                            json.dumps(
                                build_occurrence(output_header, output_values),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        )
                        destination.write("\n")
                        rows_written += 1
    return rows_written, malformed_rows, missing_headers, unmapped_headers


def convert_source(
    source_spec: DatasetSource,
    output_path: Path,
    output_header: Sequence[str] | None = None,
) -> FileResult:
    """Validate and stream a resolved source to an atomically finalized output."""
    started = time.perf_counter()
    temporary_path: Path | None = None

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            if source_spec.archive_metadata is not None:
                if output_header is None:
                    output_header = tuple(
                        field.name for field in source_spec.archive_metadata.core.fields
                    )
                (
                    rows_written,
                    malformed_rows_skipped,
                    missing_headers,
                    unmapped_headers,
                ) = _stream_archive_occurrences(
                    source_spec,
                    destination,
                    output_header,
                )
            else:
                with open_dataset_source(source_spec) as source:
                    reader = create_tsv_reader(source)
                    header = read_header(reader, source_spec.label)
                    missing_headers = find_missing_chunk_headers(header)
                    unmapped_headers = find_unmapped_headers(header)
                    rows_written, malformed_rows_skipped = stream_occurrences(
                        reader,
                        destination,
                        header,
                    )

        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return FileResult(
        input_path=source_spec.label,
        output_path=output_path,
        rows_written=rows_written,
        malformed_rows_skipped=malformed_rows_skipped,
        missing_headers=missing_headers,
        unmapped_headers=unmapped_headers,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


# ---------------------------------------------------------------------------
# Semantic logging
# ---------------------------------------------------------------------------


def _bounded_file_details(values: Sequence[str]) -> tuple[str, int]:
    """Render caller-ordered file details with an explicit omission count."""
    shown = values[:_LOG_FILE_LIMIT]
    rendered = ", ".join(
        bounded_log_detail(value, limit=_LOG_DETAIL_LIMIT) for value in shown
    )
    return rendered, len(values) - len(shown)


def _log_batch_start(
    logger: logging.Logger,
    *,
    run_id: str,
    invocation: str,
    input_paths: Sequence[Path],
) -> None:
    files, omitted = _bounded_file_details(
        tuple(str(input_path) for input_path in input_paths)
    )
    logger.info(
        "transform_occurrences started run_id=%s file_total=%d "
        "files_omitted=%d files=[%s] invocation=%s",
        run_id,
        len(input_paths),
        omitted,
        files,
        bounded_log_detail(invocation, limit=512),
    )


def _log_batch_result(
    logger: logging.Logger,
    *,
    run_id: str,
    result: BatchResult,
) -> None:
    logger.info(
        "transform_occurrences finished run_id=%s status=%s file_total=%d "
        "file_converted=%d file_failed=%d rows_written=%d "
        "malformed_rows_skipped=%d missing_header_files=%d "
        "missing_header_total=%d unmapped_header_files=%d "
        "unmapped_header_total=%d elapsed_ms=%.2f",
        run_id,
        result.status,
        len(result.converted_files) + len(result.failed_files),
        len(result.converted_files),
        len(result.failed_files),
        result.rows_written,
        result.malformed_rows_skipped,
        len(result.files_with_missing_headers),
        result.missing_header_count,
        len(result.files_with_unmapped_headers),
        result.unmapped_header_count,
        result.elapsed_ms,
    )

    _log_conversion_details(logger, result)
    _log_header_warnings(logger, result)
    _log_malformed_rows(logger, result)


def _log_conversion_details(logger: logging.Logger, result: BatchResult) -> None:
    converted_details = tuple(
        f"{file_result.input_path}->{file_result.output_path} "
        f"rows={file_result.rows_written} "
        f"malformed={file_result.malformed_rows_skipped} "
        f"elapsed_ms={file_result.elapsed_ms:.2f}"
        for file_result in result.converted_files
    )
    converted, converted_omitted = _bounded_file_details(converted_details)
    logger.info(
        "transform_occurrences converted_files count=%d shown=%d omitted=%d "
        "files=[%s]",
        len(converted_details),
        min(len(converted_details), _LOG_FILE_LIMIT),
        converted_omitted,
        converted,
    )

    failure_details = tuple(
        f"{failure.input_path} ({failure.error_type}: {failure.message}; "
        f"elapsed_ms={failure.elapsed_ms:.2f})"
        for failure in result.failed_files
    )
    if failure_details:
        failures, failures_omitted = _bounded_file_details(failure_details)
        logger.error(
            "transform_occurrences failed_files count=%d shown=%d omitted=%d "
            "files=[%s]",
            len(failure_details),
            min(len(failure_details), _LOG_FILE_LIMIT),
            failures_omitted,
            failures,
        )


def _log_header_warnings(logger: logging.Logger, result: BatchResult) -> None:
    missing_header_files = result.files_with_missing_headers
    for file_result in missing_header_files[:_LOG_FILE_LIMIT]:
        missing_headers = frozenset(file_result.missing_headers)
        for chunk_name, chunk_fields in CHUNKS.items():
            chunk_missing = tuple(
                field_name
                for field_name in chunk_fields
                if field_name in missing_headers
            )
            if not chunk_missing:
                continue
            headers, headers_omitted = _bounded_file_details(chunk_missing)
            logger.warning(
                "transform_occurrences headers_missing source=%s chunk=%s "
                "missing_count=%d shown=%d omitted=%d headers=[%s]",
                file_result.input_path,
                chunk_name,
                len(chunk_missing),
                min(len(chunk_missing), _LOG_FILE_LIMIT),
                headers_omitted,
                headers,
            )
    if len(missing_header_files) > _LOG_FILE_LIMIT:
        logger.warning(
            "transform_occurrences missing_header_files omitted=%d",
            len(missing_header_files) - _LOG_FILE_LIMIT,
        )

    unmapped_header_files = result.files_with_unmapped_headers
    for file_result in unmapped_header_files[:_LOG_FILE_LIMIT]:
        headers, headers_omitted = _bounded_file_details(
            file_result.unmapped_headers
        )
        logger.warning(
            "transform_occurrences headers_unmapped source=%s count=%d "
            "shown=%d omitted=%d headers=[%s]",
            file_result.input_path,
            len(file_result.unmapped_headers),
            min(len(file_result.unmapped_headers), _LOG_FILE_LIMIT),
            headers_omitted,
            headers,
        )
    if len(unmapped_header_files) > _LOG_FILE_LIMIT:
        logger.warning(
            "transform_occurrences unmapped_header_files omitted=%d",
            len(unmapped_header_files) - _LOG_FILE_LIMIT,
        )


def _log_malformed_rows(logger: logging.Logger, result: BatchResult) -> None:
    malformed_files = tuple(
        file_result
        for file_result in result.converted_files
        if file_result.malformed_rows_skipped
    )
    for file_result in malformed_files[:_LOG_FILE_LIMIT]:
        logger.warning(
            "transform_occurrences malformed_rows source=%s skipped=%d",
            file_result.input_path,
            file_result.malformed_rows_skipped,
        )
    if len(malformed_files) > _LOG_FILE_LIMIT:
        logger.warning(
            "transform_occurrences malformed_row_files omitted=%d",
            len(malformed_files) - _LOG_FILE_LIMIT,
        )


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


def discover_inputs(source_dir: Path) -> list[Path]:
    """Find Darwin Core ZIP archives deterministically."""
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".zip"
    )


def _xml_local_name(tag: str) -> str:
    """Return an XML element's local name without depending on its namespace."""
    return tag.rsplit("}", 1)[-1]


def _decode_meta_character(value: str, attribute: str) -> str:
    """Decode the escaped single characters used by Darwin Core metadata."""
    decoded = {"\\t": "\t", "\\n": "\n", "\\r": "\r"}.get(value, value)
    if len(decoded) != 1:
        raise HeaderValidationError(
            f"meta.xml {attribute} must describe exactly one character"
        )
    return decoded


def _resolve_archive_member(metadata_member: str, location: str) -> str:
    """Resolve a metadata-relative member while rejecting archive traversal."""
    member = posixpath.normpath(
        posixpath.join(posixpath.dirname(metadata_member), location)
    )
    if member == ".." or member.startswith("../") or member.startswith("/"):
        raise HeaderValidationError(
            "Darwin Core meta.xml contains an unsafe file location"
        )
    return member


def _term_name(term: str) -> str:
    """Return the stable local name of an absolute Darwin Core term URI."""
    try:
        return term_local_name(term)
    except ValueError as error:
        raise HeaderValidationError(
            f"meta.xml contains an invalid term URI: {term}"
        ) from error


def _parse_archive_table(
    element: ElementTree.Element,
    *,
    metadata_member: str,
    archive_members: Collection[str],
    is_core: bool,
) -> ArchiveTable:
    """Parse one Darwin Core table declaration."""
    row_type = element.attrib.get("rowType", "")
    if not row_type:
        raise HeaderValidationError("meta.xml table is missing rowType")

    locations = tuple(
        _resolve_archive_member(metadata_member, child.text.strip())
        for child in element.iter()
        if _xml_local_name(child.tag) == "location"
        and child.text
        and child.text.strip()
    )
    if not locations:
        raise HeaderValidationError("meta.xml table declares no files")
    missing = [location for location in locations if location not in archive_members]
    if missing:
        raise HeaderValidationError(
            f"Darwin Core meta.xml references a missing file: {missing[0]}"
        )

    encoding = element.attrib.get("encoding", "UTF-8")
    delimiter = _decode_meta_character(
        element.attrib.get("fieldsTerminatedBy", ","),
        "fieldsTerminatedBy",
    )
    line_terminator = _decode_meta_character(
        element.attrib.get("linesTerminatedBy", "\\n"),
        "linesTerminatedBy",
    )
    if line_terminator not in {"\n", "\r"}:
        raise HeaderValidationError(
            "meta.xml linesTerminatedBy must be a supported newline character"
        )
    quote_value = element.attrib.get("fieldsEnclosedBy", '"')
    quote_character = (
        _decode_meta_character(quote_value, "fieldsEnclosedBy")
        if quote_value
        else None
    )
    try:
        ignore_header_lines = int(element.attrib.get("ignoreHeaderLines", "0"))
    except ValueError as error:
        raise HeaderValidationError(
            "meta.xml ignoreHeaderLines must be a non-negative integer"
        ) from error
    if ignore_header_lines < 0:
        raise HeaderValidationError(
            "meta.xml ignoreHeaderLines must be a non-negative integer"
        )

    fields: list[ArchiveField] = []
    names: set[str] = set()
    indexes: set[int] = set()
    id_index: int | None = None
    core_id_index: int | None = None
    for child in element:
        local_name = _xml_local_name(child.tag)
        if local_name in {"id", "coreid"}:
            try:
                relation_index = int(child.attrib["index"])
            except (KeyError, ValueError) as error:
                raise HeaderValidationError(
                    f"meta.xml {local_name} requires an integer index"
                ) from error
            if relation_index < 0:
                raise HeaderValidationError(
                    f"meta.xml {local_name} index must be non-negative"
                )
            if local_name == "id":
                id_index = relation_index
            else:
                core_id_index = relation_index
        if local_name != "field":
            continue
        term = child.attrib.get("term", "")
        name = _term_name(term)
        index_text = child.attrib.get("index")
        index = None
        if index_text is not None:
            try:
                index = int(index_text)
            except ValueError as error:
                raise HeaderValidationError(
                    "meta.xml field index must be a non-negative integer"
                ) from error
            if index < 0 or index in indexes:
                raise HeaderValidationError(
                    "meta.xml field indexes must be unique non-negative integers"
                )
            indexes.add(index)
        if name in names:
            raise HeaderValidationError(
                f"meta.xml table declares duplicate field name: {name}"
            )
        names.add(name)
        fields.append(
            ArchiveField(term, name, index, child.attrib.get("default", ""))
        )

    if not is_core and core_id_index is None:
        raise HeaderValidationError(
            "Darwin Core extension must declare a coreid index"
        )
    return ArchiveTable(
        row_type=row_type,
        locations=locations,
        fields=tuple(fields),
        encoding=encoding,
        delimiter=delimiter,
        quote_character=quote_character,
        ignore_header_lines=ignore_header_lines,
        id_index=id_index,
        core_id_index=core_id_index,
    )


def parse_archive_metadata(
    archive: ZipFile,
    archive_path: Path,
) -> ArchiveMetadata:
    """Parse and validate authoritative Occurrence-core DwC-A metadata."""
    metadata_members = sorted(
        name
        for name in archive.namelist()
        if posixpath.basename(name).casefold() == "meta.xml"
    )
    if not metadata_members:
        raise HeaderValidationError(
            f"{archive_path}: Darwin Core archive is missing meta.xml"
        )
    if len(metadata_members) != 1:
        raise HeaderValidationError(
            f"{archive_path}: archive contains multiple meta.xml files"
        )
    metadata_member = metadata_members[0]
    try:
        with archive.open(metadata_member, "r") as stream:
            root = ElementTree.parse(stream).getroot()
    except ElementTree.ParseError as error:
        raise HeaderValidationError(
            f"{archive_path}: malformed Darwin Core meta.xml"
        ) from error

    core_elements = [
        element for element in root if _xml_local_name(element.tag) == "core"
    ]
    if len(core_elements) != 1:
        raise HeaderValidationError(
            f"{archive_path}: Darwin Core archive must declare exactly one core"
        )
    core = _parse_archive_table(
        core_elements[0],
        metadata_member=metadata_member,
        archive_members=archive.namelist(),
        is_core=True,
    )
    if core.row_type != OCCURRENCE_ROW_TYPE:
        raise HeaderValidationError(
            f"{archive_path}: unsupported core rowType: {core.row_type}"
        )
    unknown_core_terms = [
        field.term
        for field in core.fields
        if field.term not in _SIMPLE_DARWIN_CORE_TERMS
    ]
    if unknown_core_terms:
        raise HeaderValidationError(
            f"{archive_path}: Occurrence core declares unregistered term: "
            f"{unknown_core_terms[0]}"
        )

    # TODO: Parse, validate, ingest, and expose extension rows. Extensions are
    # deliberately ignored so they do not prevent ingestion of a supported
    # Occurrence core.
    return ArchiveMetadata(core=core, extensions=())


def archive_source(path: Path) -> DatasetSource:
    """Return the authoritative occurrence core from one valid DwC-A ZIP."""
    try:
        with ZipFile(path) as archive:
            metadata = parse_archive_metadata(archive, path)
    except BadZipFile as error:
        raise HeaderValidationError(f"{path}: invalid ZIP archive") from error
    return DatasetSource(path, metadata.core.locations[0], metadata)


def _registered_identities(
    datasets: Sequence[DatasetSchema],
) -> dict[str, DatasetSchema]:
    """Index explicit dataset identifiers without inferring from filenames."""
    registered: dict[str, DatasetSchema] = {}
    for dataset in datasets:
        for value in dataset.identity_values:
            if value in registered:
                raise ValueError(
                    "dataset registry contains duplicate identity value: "
                    f"{value} ({registered[value].key}, {dataset.key})"
                )
            registered[value] = dataset
    return registered


def _first_core_record(source: DatasetSource) -> dict[str, str]:
    """Read the first occurrence using the mappings declared by meta.xml."""
    metadata = source.archive_metadata
    if metadata is None:
        raise ValueError("archive metadata is required")
    with ZipFile(source.path) as archive:
        for member in metadata.core.locations:
            with archive.open(member, "r") as binary:
                with TextIOWrapper(
                    binary,
                    encoding=metadata.core.encoding,
                    newline="",
                ) as stream:
                    reader = create_archive_reader(stream, metadata.core)
                    _skip_rows(
                        reader,
                        metadata.core.ignore_header_lines,
                        source.path,
                    )
                    try:
                        values = next(reader)
                    except StopIteration:
                        continue
                    if any(
                        field.index is not None and field.index >= len(values)
                        for field in metadata.core.fields
                    ):
                        raise HeaderValidationError(
                            f"{source.path}: first core row is shorter than "
                            "meta.xml declares"
                        )
                    return _field_values(metadata.core, values)
    raise HeaderValidationError(f"{source.path}: occurrence core contains no records")


def _identify_path(
    path: Path,
    registered: Mapping[str, DatasetSchema],
) -> tuple[DatasetSource, DatasetSchema]:
    """Identify one archive from explicit datasetID or datasetName values."""
    if path.suffix.casefold() != ".zip":
        raise HeaderValidationError(
            f"{path}: input must be a standards-compliant Darwin Core ZIP archive"
        )
    source = archive_source(path)
    record = _first_core_record(source)
    identity_values = {
        value
        for name in ("datasetID", "datasetName")
        if (value := record.get(name, ""))
    }
    matches = {registered[value] for value in identity_values if value in registered}
    if len(matches) == 1:
        return source, matches.pop()
    if len(matches) > 1:
        keys = ", ".join(sorted(dataset.key for dataset in matches))
        raise HeaderValidationError(
            f"{path}: datasetID and datasetName identify different datasets: {keys}"
        )
    identity = record.get("datasetID", "") or record.get("datasetName", "")
    if not identity:
        raise HeaderValidationError(
            f"{path}: occurrence core must provide datasetID or datasetName"
        )
    label = record.get("datasetName", "") or identity
    slug = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    if not slug:
        slug = "occurrence_dataset"
    key = slug[:59]
    dataset = DatasetSchema(
        key=key,
        short_label=label,
        collection_key="darwin_core_archives",
        collection_label="Darwin Core Archives",
        relative_path=Path(
            f"dataset/darwin_core_archives/{key}/{key}.txt"
        ),
        identity_values=tuple(sorted(identity_values)),
    )
    return source, dataset


def resolve_inputs(
    input_paths: Sequence[Path],
    *,
    datasets: Sequence[DatasetSchema] = (),
    table_models: Sequence[object] = TABLE_MODELS,
) -> tuple[tuple[DatasetSource, DatasetSchema], ...]:
    """Resolve standards-compliant archives to explicit registered datasets."""
    del table_models
    registered = _registered_identities(datasets)
    resolved = tuple(
        _identify_path(path, registered)
        for path in input_paths
    )
    by_dataset: dict[str, tuple[DatasetSource, DatasetSchema]] = {}
    for source, dataset in resolved:
        previous = by_dataset.get(dataset.key)
        if previous is not None:
            print(
                f"WARNING: dataset key {dataset.key!r} is provided by both "
                f"{previous[0].label} and {source.label}; using {source.label}",
                file=sys.stderr,
            )
        by_dataset[dataset.key] = (source, dataset)
    registered_order = {dataset.key: index for index, dataset in enumerate(datasets)}
    return tuple(
        sorted(
            by_dataset.values(),
            key=lambda item: (
                registered_order.get(item[1].key, len(registered_order)),
                item[1].key,
            ),
        )
    )


def ensure_unique_output_paths(
    resolved: Sequence[tuple[DatasetSource, DatasetSchema]],
    repository_root: Path,
) -> None:
    """Reject duplicate or source-overwriting canonical destinations."""
    seen: set[Path] = set()
    for source, dataset in resolved:
        output_path = repository_root / jsonl_relative_path(dataset)
        if output_path in seen:
            raise ValueError(f"more than one input would produce {output_path}")
        if source.archive_member is None and output_path == source.path:
            raise ValueError(f"output would overwrite source file: {source.path}")
        seen.add(output_path)


def convert_files(
    resolved: Sequence[tuple[DatasetSource, DatasetSchema]],
    *,
    repository_root: Path,
    logger: logging.Logger,
    run_id: str,
    invocation: str,
    table_models: Sequence[object] = TABLE_MODELS,
) -> BatchResult:
    """Convert files to atomically finalized per-source JSON Lines outputs."""
    started = time.perf_counter()
    _log_batch_start(
        logger,
        run_id=run_id,
        invocation=invocation,
        input_paths=tuple(source.label for source, _dataset in resolved),
    )

    try:
        ensure_unique_output_paths(resolved, repository_root)
    except ValueError as error:
        logger.error(
            "transform_occurrences failed run_id=%s status=failed "
            "error_type=%s reason=%s elapsed_ms=%.2f",
            run_id,
            type(error).__name__,
            bounded_log_detail(error, limit=_LOG_DETAIL_LIMIT),
            (time.perf_counter() - started) * 1000,
        )
        raise

    converted_files: list[FileResult] = []
    failed_files: list[FileFailure] = []
    if any(
        dataset.relative_path.stem
        not in {
            getattr(model, "__table__", None).name
            for model in table_models
            if getattr(model, "__table__", None) is not None
        }
        for _source, dataset in resolved
    ):
        table_models = build_table_models(
            tuple(dataset for _source, dataset in resolved)
        )
    tables = {
        table.name: table
        for model in table_models
        if (table := getattr(model, "__table__", None)) is not None
    }
    missing_tables = tuple(
        dataset.relative_path.stem
        for _source, dataset in resolved
        if dataset.relative_path.stem not in tables
    )
    if missing_tables:
        raise ValueError(
            "dataset table(s) are not registered: " + ", ".join(missing_tables)
        )
    canonical_headers = {
        table.info.get("source_headers") for table in tables.values()
    }
    if len(canonical_headers) != 1:
        raise ValueError(
            "registered occurrence tables do not share one canonical schema"
        )
    output_header = canonical_headers.pop()
    if not isinstance(output_header, tuple) or not output_header:
        raise ValueError("occurrence tables are missing a canonical schema")

    for source, dataset in resolved:
        file_started = time.perf_counter()
        output_path = repository_root / jsonl_relative_path(dataset)
        try:
            result = convert_source(source, output_path, output_header)
        except (BadZipFile, HeaderValidationError, OSError, UnicodeError) as error:
            failed_files.append(
                FileFailure(
                    input_path=source.label,
                    error_type=type(error).__name__,
                    message=bounded_log_detail(
                        error,
                        limit=_LOG_DETAIL_LIMIT,
                    ),
                    elapsed_ms=(time.perf_counter() - file_started) * 1000,
                )
            )
            continue

        converted_files.append(result)

    result = BatchResult(
        converted_files=tuple(converted_files),
        failed_files=tuple(failed_files),
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
    _log_batch_result(logger, run_id=run_id, result=result)
    return result


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the Darwin Core source discovery and conversion command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "standards-compliant Darwin Core ZIP archives; defaults to ZIP "
            "files below --source-dir"
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=_REPOSITORY_ROOT / "dataset",
        help="source root when no inputs are supplied (default: %(default)s)",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=_REPOSITORY_ROOT,
        help="root for canonical generated JSONL paths (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Convert requested sources and report the batch's files and row counts."""
    arguments = build_argument_parser().parse_args(argv)
    input_paths = arguments.inputs or discover_inputs(arguments.source_dir)
    if not input_paths:
        print("ERROR: no Darwin Core .zip archives found", file=sys.stderr)
        return 1

    target = "all_collections"
    if arguments.inputs:
        target = (
            input_paths[0].stem if len(input_paths) == 1
            else f"selected_{len(input_paths)}_files"
        )
    logger_factory = LoggerFactory(
        LoggerFactory.operation_filename("gaia", "transform_occurrences", target)
    )
    logger = logger_factory.get_logger()
    print(f"Log: {logger_factory.log_path}")
    invocation = " ".join(
        f"{name}={bounded_log_detail(value, limit=_INVOCATION_VALUE_LIMIT)}"
        for name, value in vars(arguments).items()
    )

    try:
        resolved = resolve_inputs(input_paths)
        result = convert_files(
            resolved,
            repository_root=arguments.repository_root.resolve(),
            logger=logger,
            run_id=logger_factory.run_id,
            invocation=bounded_log_detail(invocation, limit=_INVOCATION_LIMIT),
        )
        if not result.failed_files:
            write_dataset_catalog(
                arguments.repository_root.resolve(),
                tuple(dataset for _source, dataset in resolved),
            )
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        logger.error(
            "transform_occurrences aborted run_id=%s status=failed "
            "error_type=%s reason=%s",
            logger_factory.run_id,
            type(error).__name__,
            bounded_log_detail(error, limit=_LOG_DETAIL_LIMIT),
        )
        print(
            f"ERROR: transform_occurrences failed. See {logger_factory.log_path}",
            file=sys.stderr,
        )
        return 1

    for file_result in result.converted_files:
        print(
            f"{file_result.input_path} -> {file_result.output_path}: "
            f"{file_result.rows_written} row(s) written; "
            f"{file_result.malformed_rows_skipped} malformed row(s) skipped"
        )
    for failure in result.failed_files:
        print(f"ERROR: {failure.message}", file=sys.stderr)
    print(
        f"Total: {result.rows_written} row(s) written; "
        f"{result.malformed_rows_skipped} malformed row(s) skipped; "
        f"{len(result.failed_files)} file(s) failed; "
        f"{len(result.files_with_missing_headers)} file(s) with missing headers; "
        f"{result.missing_header_count} missing header(s); "
        f"{len(result.files_with_unmapped_headers)} file(s) with unmapped headers; "
        f"{result.unmapped_header_count} unmapped header(s)"
    )
    return 1 if result.failed_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
