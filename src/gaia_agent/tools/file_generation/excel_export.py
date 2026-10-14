"""Atomic, bounded-memory Excel exports with one worksheet per dataset."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from uuid import uuid4

from xlsxwriter import Workbook
from xlsxwriter.format import Format
from xlsxwriter.worksheet import Worksheet

from src.logger import normalize_chat_username


EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_COLUMNS = 16_384
EXCEL_MAX_CELL_CHARACTERS = 32_767
EXCEL_MAX_SHEET_CHARACTERS = 31
EXCEL_MAX_FILENAME_BYTES = 255


@dataclass(frozen=True, slots=True)
class ExcelSheetResult:
    """Identify a worksheet and its number of exported data records."""

    dataset_key: str
    sheet_name: str
    row_count: int


@dataclass(frozen=True, slots=True)
class ExcelExportResult:
    """Describe a completed workbook that is ready for delivery."""

    file_id: str
    filename: str
    path: Path
    sheets: tuple[ExcelSheetResult, ...]


@dataclass(slots=True)
class _SheetState:
    worksheet: Worksheet
    headers: tuple[str, ...]
    row_count: int = 0


def normalize_excel_filename(filename: str) -> str:
    """Validate a local basename and give it exactly one lowercase .xlsx suffix."""
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a nonempty string.")
    if re.search(r"[/\\:\x00-\x1f\x7f-\x9f]", filename):
        raise ValueError(
            "filename must be a basename without paths, separators, or control characters."
        )
    stem = filename[:-5] if filename.lower().endswith(".xlsx") else filename
    if not stem.strip(" .") or stem.lower().endswith(".xlsx"):
        raise ValueError("filename must have a nonempty name and at most one .xlsx suffix.")
    normalized = f"{stem}.xlsx"
    if len(normalized.encode("utf-8")) > EXCEL_MAX_FILENAME_BYTES:
        raise ValueError(
            f"filename must fit within {EXCEL_MAX_FILENAME_BYTES} UTF-8 bytes, "
            "including the .xlsx suffix. Use a shorter name."
        )
    return normalized


def _validate_text(value: str, *, location: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{location} must be text; use an empty string for empty values.")
    if len(value) > EXCEL_MAX_CELL_CHARACTERS:
        raise ValueError(
            f"{location} exceeds Excel's {EXCEL_MAX_CELL_CHARACTERS:,}-character "
            "cell limit. Narrow the selected fields before exporting."
        )


def _sheet_name(dataset_key: str, used_names: set[str]) -> str:
    base = re.sub(r"[\[\]:*?/\\\x00-\x1f]", "_", dataset_key).strip(" '")
    base = base or "Dataset"
    name = base[:EXCEL_MAX_SHEET_CHARACTERS].rstrip("'")
    suffix_number = 2
    while name.casefold() in used_names or name.casefold() == "history":
        suffix = f"_{suffix_number}"
        name = base[: EXCEL_MAX_SHEET_CHARACTERS - len(suffix)] + suffix
        suffix_number += 1
    return name


def _write_text(
    worksheet: Worksheet,
    row_index: int,
    values: Sequence[str],
    cell_format: Format | None = None,
) -> None:
    for column_index, value in enumerate(values):
        if value.startswith("<r>") and value.endswith("</r>"):
            # XlsxWriter's streaming mode recognizes this shape as rich XML.
            # Explicit text fragments keep source markup literal in the cell.
            fragments: list[str | Format] = [value[:1], value[1:2], value[2:]]
            if cell_format is not None:
                fragments.append(cell_format)
            status = worksheet.write_rich_string(row_index, column_index, *fragments)
        else:
            status = worksheet.write_string(row_index, column_index, value, cell_format)
        if status != 0:
            raise ValueError(
                f"Could not write Excel cell at row {row_index + 1}, "
                f"column {column_index + 1} (writer status {status})."
            )


class ExcelExportWriter:
    """Write source text sequentially without retaining previously written rows.

    Use as a context manager and call ``finalize`` once all rows are appended.
    Each selected dataset must be added even when it has no matching records.
    Any failed operation aborts the export; a partial workbook is never published.
    """

    def __init__(
        self,
        output_directory: Path,
        *,
        filename: str | None = None,
        username: str = "gaia",
        check_cancellation: Callable[[], None] | None = None,
    ) -> None:
        self._filename = normalize_excel_filename(filename) if filename is not None else None
        self._username = normalize_chat_username(username)
        self._check_cancellation = check_cancellation
        output_directory = output_directory.resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        self._output_directory = output_directory
        self._file_id = uuid4().hex
        self._temporary_directory = TemporaryDirectory(
            prefix=".gaia-export-", dir=output_directory
        )
        self._temporary_path = Path(self._temporary_directory.name) / "workbook.xlsx"
        self._workbook = Workbook(
            self._temporary_path,
            {
                "constant_memory": True,
                "tmpdir": self._temporary_directory.name,
                "strings_to_formulas": False,
                "strings_to_urls": False,
                "strings_to_numbers": False,
                "use_zip64": True,
            },
        )
        self._header_format = self._workbook.add_format({"bold": True})
        self._sheets: dict[str, _SheetState] = {}
        self._sheet_names: set[str] = set()
        self._closed = False
        self._result: ExcelExportResult | None = None

    def __enter__(self) -> ExcelExportWriter:
        self._require_open()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.abort()
        if exception_type is not None and self._result is not None:
            self._result.path.unlink(missing_ok=True)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("This Excel export is already finalized or aborted.")

    def add_sheet(self, dataset_key: str, headers: Sequence[str]) -> None:
        """Add an ordered dataset worksheet and write its exact source headers."""
        self._require_open()
        try:
            if not isinstance(dataset_key, str) or not dataset_key:
                raise ValueError("A nonempty dataset key is required for each worksheet.")
            if dataset_key in self._sheets:
                raise ValueError(f"Dataset {dataset_key!r} already has an Excel worksheet.")
            if isinstance(headers, (str, bytes)) or not headers:
                raise ValueError("Excel worksheet headers must be a nonempty sequence.")
            if len(headers) > EXCEL_MAX_COLUMNS:
                raise ValueError(
                    f"Dataset {dataset_key!r} exceeds Excel's "
                    f"{EXCEL_MAX_COLUMNS:,}-column worksheet limit."
                )
            for header in headers:
                _validate_text(header, location=f"Header in dataset {dataset_key!r}")

            name = _sheet_name(dataset_key, self._sheet_names)
            worksheet = self._workbook.add_worksheet(name)
            worksheet.freeze_panes(1, 0)
            worksheet.set_column(0, len(headers) - 1, 24)
            _write_text(worksheet, 0, headers, self._header_format)
            self._sheets[dataset_key] = _SheetState(worksheet, tuple(headers))
            self._sheet_names.add(name.casefold())
        except BaseException:
            self.abort()
            raise

    def append(self, dataset_key: str, values: Sequence[str]) -> None:
        """Append one complete text record in the worksheet's header order."""
        self._require_open()
        try:
            sheet = self._sheets.get(dataset_key)
            if sheet is None:
                raise ValueError(f"Add dataset {dataset_key!r} before appending records.")
            if isinstance(values, (str, bytes)) or len(values) != len(sheet.headers):
                raise ValueError(
                    f"Dataset {dataset_key!r} requires {len(sheet.headers)} "
                    "values per record in header order."
                )
            if sheet.row_count >= EXCEL_MAX_ROWS - 1:
                raise ValueError(
                    f"Dataset {dataset_key!r} exceeds Excel's "
                    f"{EXCEL_MAX_ROWS - 1:,}-record worksheet limit (plus its header). "
                    "Narrow the export filters."
                )
            for header, value in zip(sheet.headers, values, strict=True):
                _validate_text(
                    value,
                    location=(
                        f"Dataset {dataset_key!r}, record {sheet.row_count + 1}, "
                        f"field {header!r}"
                    ),
                )
            _write_text(sheet.worksheet, sheet.row_count + 1, values)
            sheet.row_count += 1
        except BaseException:
            self.abort()
            raise

    def finalize(self) -> ExcelExportResult:
        """Atomically publish a complete workbook without replacing existing files."""
        self._require_open()
        try:
            if not self._sheets:
                raise ValueError("An Excel export requires at least one dataset worksheet.")
            for sheet in self._sheets.values():
                sheet.worksheet.autofilter(0, 0, sheet.row_count, len(sheet.headers) - 1)
            if self._check_cancellation is not None:
                self._check_cancellation()
            self._workbook.close()
            if self._check_cancellation is not None:
                self._check_cancellation()
            sheets = tuple(
                ExcelSheetResult(dataset_key, sheet.worksheet.name, sheet.row_count)
                for dataset_key, sheet in self._sheets.items()
            )
            suffix_number = 1
            if self._filename is None:
                pattern = re.compile(rf"{re.escape(self._username)}_([1-9][0-9]*)\.xlsx")
                suffix_number = max(
                    (
                        int(match[1])
                        for path in self._output_directory.iterdir()
                        if (match := pattern.fullmatch(path.name)) is not None
                    ),
                    default=0,
                ) + 1
            while True:
                if self._check_cancellation is not None:
                    self._check_cancellation()
                if self._filename is None:
                    filename = f"{self._username}_{suffix_number}.xlsx"
                elif suffix_number == 1:
                    filename = self._filename
                else:
                    filename = f"{self._filename[:-5]}-{suffix_number}.xlsx"
                candidate = self._output_directory / normalize_excel_filename(filename)
                result = ExcelExportResult(
                    file_id=self._file_id,
                    filename=candidate.name,
                    path=candidate,
                    sheets=sheets,
                )
                try:
                    # Both paths share a filesystem. Linking publishes the
                    # completed workbook atomically and fails if the name exists.
                    os.link(self._temporary_path, candidate)
                except FileExistsError:
                    suffix_number += 1
                else:
                    self._result = result
                    break
            self._temporary_directory.cleanup()
            self._closed = True
            return result
        except BaseException:
            try:
                if self._result is not None:
                    self._result.path.unlink(missing_ok=True)
                    self._result = None
            finally:
                self.abort()
            raise

    def abort(self) -> None:
        """Close and discard an unfinished export, including all workbook temp files."""
        if self._closed:
            return
        try:
            # XlsxWriter has no abort API. Its streaming worksheets own these
            # row buffers; close them without packaging a discarded workbook.
            # This also releases handles when Workbook.close itself fails.
            for worksheet in self._workbook.worksheets():
                with suppress(OSError):
                    worksheet.row_data_fh.close()
        finally:
            self._closed = True
            self._temporary_directory.cleanup()
