"""Round-trip, size-limit, and cleanup contracts for streamed Excel files."""

from __future__ import annotations

import asyncio
import re
import tracemalloc
import xml.etree.ElementTree as ET
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import Barrier
from zipfile import ZipFile

import pytest

import src.gaia_agent.tools.file_generation.excel_export as excel_export_module
from src.gaia_agent.tools.file_generation.excel_export import ExcelExportResult, ExcelExportWriter


XML_NAMESPACE = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def worksheet_rows(path: Path, sheet_number: int = 1) -> list[tuple[str, ...]]:
    """Read inline text through the independent standard-library XML parser."""
    with ZipFile(path) as archive:
        worksheet = ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet_number}.xml"))
    assert worksheet.findall(".//s:f", XML_NAMESPACE) == []
    assert worksheet.findall(".//s:hyperlink", XML_NAMESPACE) == []
    rows = []
    for row in worksheet.findall("s:sheetData/s:row", XML_NAMESPACE):
        cells = row.findall("s:c", XML_NAMESPACE)
        assert all(cell.attrib["t"] == "inlineStr" for cell in cells)
        rows.append(
            tuple(
                "".join(text.text or "" for text in cell.findall(".//s:t", XML_NAMESPACE))
                for cell in cells
            )
        )
    return rows


def _create_concurrent_workbook(
    index: int,
    *,
    directory: Path,
    filename: str | None,
    ready: Barrier,
) -> ExcelExportResult:
    with ExcelExportWriter(directory, filename=filename, username="atlas") as writer:
        writer.add_sheet("birds", ("occurrenceID",))
        writer.append("birds", (f"bird-{index}",))
        ready.wait(timeout=10)
        return writer.finalize()


def _publish_competing_export(
    source: Path,
    destination: Path,
    *,
    collision: Path,
    link: Callable[[Path, Path], None],
) -> None:
    if destination == collision:
        collision.write_bytes(b"competing export")
    link(source, destination)


def _fail_workbook_close() -> None:
    raise OSError("disk full")


def _fail_link(source: Path, destination: Path) -> None:
    raise OSError("disk failure")


def test_workbook_round_trips_multiple_datasets_and_more_than_200_records(
    tmp_path: Path,
) -> None:
    headers = ("occurrenceID", "catalogNumber", "scientificName", "locality", "remarks")
    special_record = (
        "000012345678901234567890",
        "000007",
        "Élan 日本語 🐦",
        "",
        '=HYPERLINK("https://example.org/", "specimen")',
    )
    with ExcelExportWriter(tmp_path) as writer:
        writer.add_sheet("birds", headers)
        writer.add_sheet("fish", ("occurrenceID", "verbatimLocality"))
        writer.add_sheet("mammals", ("occurrenceID", "catalogNumber"))
        writer.append("birds", special_record)
        writer.append("fish", ("fish-001", "https://example.org/specimens"))
        for index in range(250):
            writer.append("birds", (str(index), "", "", "Vancouver", ""))
        assert list(tmp_path.glob("*.xlsx")) == []
        result = writer.finalize()

    assert re.fullmatch(r"[a-f0-9]{32}", result.file_id)
    assert result.filename == "gaia_1.xlsx"
    assert result.path == tmp_path / result.filename
    assert [(sheet.dataset_key, sheet.sheet_name, sheet.row_count) for sheet in result.sheets] == [
        ("birds", "birds", 251),
        ("fish", "fish", 1),
        ("mammals", "mammals", 0),
    ]
    birds = worksheet_rows(result.path)
    assert birds[0] == headers
    assert birds[1] == special_record
    assert len(birds) == 252
    assert birds[-1] == ("249", "", "", "Vancouver", "")
    assert worksheet_rows(result.path, 2) == [
        ("occurrenceID", "verbatimLocality"),
        ("fish-001", "https://example.org/specimens"),
    ]
    assert worksheet_rows(result.path, 3) == [("occurrenceID", "catalogNumber")]
    assert list(tmp_path.iterdir()) == [result.path]


@pytest.mark.parametrize(
    ("requested", "expected"),
    (
        ("helloXXX", "helloXXX.xlsx"),
        ("helloXXX.xlsx", "helloXXX.xlsx"),
        ("helloXXX.XLSX", "helloXXX.xlsx"),
        ("Oiseaux 日本語", "Oiseaux 日本語.xlsx"),
        ("é" * 125, "é" * 125 + ".xlsx"),
    ),
)
def test_requested_filename_is_normalized_without_changing_workbook_data(
    tmp_path: Path, requested: str, expected: str,
) -> None:
    with ExcelExportWriter(tmp_path, filename=requested) as writer:
        writer.add_sheet("birds", ("occurrenceID",))
        writer.append("birds", ("00001",))
        result = writer.finalize()
    assert result.filename == expected
    assert result.path == tmp_path / expected
    assert worksheet_rows(result.path) == [("occurrenceID",), ("00001",)]
    assert list(tmp_path.iterdir()) == [result.path]


@pytest.mark.parametrize(
    "filename",
    (
        "", "   ", ".", "..", ".xlsx", "../escape", "/tmp/escape.xlsx",
        "nested/file.xlsx", r"nested\file.xlsx", r"C:\exports\file.xlsx",
        "bad\x00name", "bad\nname", "bad\x85name", "hello.xlsx.xlsx",
        "é" * 126, 42,
    ),
)
def test_invalid_filename_is_rejected_before_creating_an_output_directory(
    tmp_path: Path, filename: object,
) -> None:
    destination = tmp_path / "not_created"
    with pytest.raises(ValueError, match="filename"):
        ExcelExportWriter(destination, filename=filename)
    assert not destination.exists()


def test_default_names_continue_the_session_sequence_and_normalize_the_username(
    tmp_path: Path,
) -> None:
    existing = {
        "atlas_smith_1.xlsx": b"first workbook",
        "atlas_smith_4.xlsx": b"fourth workbook",
        "atlas_smith_4-2.xlsx": b"custom collision name",
        "atlas_smith_05.xlsx": b"custom padded number",
        "other_100.xlsx": b"another username",
        "custom.xlsx": b"custom workbook",
    }
    for filename, content in existing.items():
        (tmp_path / filename).write_bytes(content)

    for number in (5, 6):
        with ExcelExportWriter(tmp_path, username=" Atlas Smith ") as writer:
            writer.add_sheet("birds", ("occurrenceID",))
            writer.append("birds", (f"bird-{number}",))
            result = writer.finalize()
        assert result.filename == f"atlas_smith_{number}.xlsx"
        assert worksheet_rows(result.path) == [("occurrenceID",), (f"bird-{number}",)]
    for filename, content in existing.items():
        assert (tmp_path / filename).read_bytes() == content


@pytest.mark.parametrize("username", ("", "...", "a" * 65, None))
def test_invalid_username_is_rejected_before_creating_an_output_directory(
    tmp_path: Path, username: object,
) -> None:
    destination = tmp_path / "not_created"
    with pytest.raises((TypeError, ValueError), match="username"):
        ExcelExportWriter(destination, username=username)
    assert not destination.exists()


@pytest.mark.parametrize("filename", (None, "helloXXX"))
def test_concurrent_publication_uses_unique_names_and_preserves_existing_exports(
    tmp_path: Path, filename: str | None,
) -> None:
    previous = tmp_path / ("atlas_1.xlsx" if filename is None else "helloXXX.xlsx")
    previous.write_bytes(b"existing workbook")
    ready = Barrier(5)

    create_workbook = partial(
        _create_concurrent_workbook, directory=tmp_path, filename=filename, ready=ready,
    )

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = tuple(pool.map(create_workbook, range(5)))

    assert {result.filename for result in results} == {
        f"atlas_{index}.xlsx" if filename is None else f"helloXXX-{index}.xlsx"
        for index in range(2, 7)
    }
    assert len({result.file_id for result in results}) == 5
    for index, result in enumerate(results):
        assert worksheet_rows(result.path) == [("occurrenceID",), (f"bird-{index}",)]
    assert previous.read_bytes() == b"existing workbook"
    assert set(tmp_path.iterdir()) == {previous, *(result.path for result in results)}


def test_default_name_collision_advances_the_file_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    link = excel_export_module.os.link
    collision = tmp_path / "atlas_1.xlsx"

    monkeypatch.setattr(excel_export_module.os, "link", partial(
        _publish_competing_export, collision=collision, link=link,
    ))
    with ExcelExportWriter(tmp_path, username="atlas") as writer:
        writer.add_sheet("birds", ("occurrenceID",))
        result = writer.finalize()

    assert result.filename == "atlas_2.xlsx"
    assert collision.read_bytes() == b"competing export"
    assert set(tmp_path.iterdir()) == {collision, result.path}


def test_failed_export_leaves_the_next_default_number_available(tmp_path: Path) -> None:
    previous = tmp_path / "atlas_1.xlsx"
    previous.write_bytes(b"existing workbook")
    with pytest.raises(RuntimeError, match="later failure"):
        with ExcelExportWriter(tmp_path, username="atlas") as writer:
            writer.add_sheet("birds", ("occurrenceID",))
            result = writer.finalize()
            assert result.filename == "atlas_2.xlsx"
            raise RuntimeError("later failure")
    with ExcelExportWriter(tmp_path, username="atlas") as writer:
        writer.add_sheet("birds", ("occurrenceID",))
        result = writer.finalize()
    assert result.filename == "atlas_2.xlsx"
    assert previous.read_bytes() == b"existing workbook"
    assert set(tmp_path.iterdir()) == {previous, result.path}


def test_context_failure_after_named_publication_preserves_the_existing_name(
    tmp_path: Path,
) -> None:
    previous = tmp_path / "helloXXX.xlsx"
    previous.write_bytes(b"existing workbook")
    with pytest.raises(RuntimeError, match="later failure"):
        with ExcelExportWriter(tmp_path, filename="helloXXX.xlsx") as writer:
            writer.add_sheet("birds", ("occurrenceID",))
            result = writer.finalize()
            assert result.filename == "helloXXX-2.xlsx"
            raise RuntimeError("later failure")
    assert previous.read_bytes() == b"existing workbook"
    assert list(tmp_path.iterdir()) == [previous]


def test_source_markup_and_formula_like_values_stay_literal(tmp_path: Path) -> None:
    values = (
        "=1+2",
        "+1+2",
        "-0012",
        "@SUM(A1:A2)",
        "<r><t>literal XML & text</t></r>",
        " leading and trailing \t\n ",
    )
    with ExcelExportWriter(tmp_path) as writer:
        writer.add_sheet("birds", ("occurrenceID", "remarks"))
        for index, value in enumerate(values):
            writer.append("birds", (str(index), value))
        result = writer.finalize()
    assert [row[1] for row in worksheet_rows(result.path)[1:]] == list(values)


def test_sheet_names_are_valid_and_unique_ignoring_case(tmp_path: Path) -> None:
    dataset_keys = (
        "Birds",
        "birds",
        "long_dataset_" * 4,
        "LONG_DATASET_" * 4,
        "'fish/[marine]*?:'",
        "'" * 3,
        "History",
    )
    with ExcelExportWriter(tmp_path) as writer:
        for key in dataset_keys:
            writer.add_sheet(key, ("occurrenceID",))
        result = writer.finalize()
    names = [sheet.sheet_name for sheet in result.sheets]
    assert len(set(name.casefold() for name in names)) == len(dataset_keys)
    assert names[:2] == ["Birds", "birds_2"]
    assert names[-2:] == ["Dataset", "History_2"]
    for name in names:
        assert 1 <= len(name) <= 31
        assert not re.search(r"[\[\]:*?/\\]", name)
        assert not name.startswith("'") and not name.endswith("'")
    with ZipFile(result.path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    assert [sheet.attrib["name"] for sheet in workbook.findall("s:sheets/s:sheet", XML_NAMESPACE)] == names


@pytest.mark.parametrize("value", [None, 17, "x" * 32_768])
def test_invalid_cell_aborts_without_publishing_or_leaving_temp_files(
    tmp_path: Path, value: object
) -> None:
    with ExcelExportWriter(tmp_path) as writer:
        writer.add_sheet("birds", ("occurrenceID", "remarks"))
        writer.append("birds", ("valid-1", ""))
        with pytest.raises(ValueError, match="must be text|cell limit"):
            writer.append("birds", ("invalid-2", value))
        with pytest.raises(RuntimeError, match="finalized or aborted"):
            writer.finalize()
    assert list(tmp_path.iterdir()) == []


def test_excel_maximum_cell_size_is_preserved(tmp_path: Path) -> None:
    value = "é" * 32_767
    with ExcelExportWriter(tmp_path) as writer:
        writer.add_sheet("birds", ("occurrenceID", "remarks"))
        writer.append("birds", ("bird-1", value))
        result = writer.finalize()
    assert worksheet_rows(result.path)[1] == ("bird-1", value)


def test_row_limit_includes_header_and_rejects_entire_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(excel_export_module, "EXCEL_MAX_ROWS", 3)
    with ExcelExportWriter(tmp_path) as writer:
        writer.add_sheet("birds", ("occurrenceID",))
        writer.append("birds", ("one",))
        writer.append("birds", ("two",))
        with pytest.raises(ValueError, match="2-record worksheet limit"):
            writer.append("birds", ("three",))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("headers", "message"),
    [
        ((), "nonempty sequence"),
        ("occurrenceID", "nonempty sequence"),
        (("field",) * 16_385, "column worksheet limit"),
        (("x" * 32_768,), "cell limit"),
    ],
)
def test_invalid_headers_leave_no_artifact(tmp_path: Path, headers: object, message: str) -> None:
    with ExcelExportWriter(tmp_path) as writer:
        with pytest.raises(ValueError, match=message):
            writer.add_sheet("birds", headers)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("values", [("one",), ("one", "two", "three"), "ab"])
def test_malformed_width_rows_abort(tmp_path: Path, values: object) -> None:
    with ExcelExportWriter(tmp_path) as writer:
        writer.add_sheet("birds", ("occurrenceID", "country"))
        with pytest.raises(ValueError, match="requires 2 values"):
            writer.append("birds", values)
    assert list(tmp_path.iterdir()) == []


def test_context_exception_or_cancellation_removes_unfinished_files(tmp_path: Path) -> None:
    for error in (RuntimeError("database failure"), asyncio.CancelledError()):
        with pytest.raises(type(error)):
            with ExcelExportWriter(tmp_path) as writer:
                writer.add_sheet("birds", ("occurrenceID",))
                writer.append("birds", ("one",))
                raise error
        assert list(tmp_path.iterdir()) == []


def test_context_without_finalize_discards_workbook(tmp_path: Path) -> None:
    with ExcelExportWriter(tmp_path) as writer:
        writer.add_sheet("birds", ("occurrenceID",))
        writer.append("birds", ("one",))
    assert list(tmp_path.iterdir()) == []


def test_failed_workbook_close_releases_row_buffers_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with ExcelExportWriter(tmp_path) as writer:
        writer.add_sheet("birds", ("occurrenceID",))
        writer.append("birds", ("one",))
        row_buffer = writer._sheets["birds"].worksheet.row_data_fh

        monkeypatch.setattr(writer._workbook, "close", _fail_workbook_close)
        with pytest.raises(OSError, match="disk full"):
            writer.finalize()
    assert row_buffer.closed
    assert list(tmp_path.iterdir()) == []


def test_publish_failure_preserves_other_exports_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = tmp_path / "existing.xlsx"
    previous.write_bytes(b"previous workbook")

    monkeypatch.setattr(excel_export_module.os, "link", _fail_link)
    with ExcelExportWriter(tmp_path) as writer:
        writer.add_sheet("birds", ("occurrenceID",))
        with pytest.raises(OSError, match="disk failure"):
            writer.finalize()
    assert previous.read_bytes() == b"previous workbook"
    assert list(tmp_path.iterdir()) == [previous]


def test_streaming_memory_does_not_grow_with_record_count(tmp_path: Path) -> None:
    tracemalloc.start()
    try:
        with ExcelExportWriter(tmp_path) as writer:
            writer.add_sheet("birds", ("occurrenceID", "remarks"))
            for index in range(1_000):
                writer.append("birds", (str(index), f"{index:06d}" + "x" * 200))
            early_memory = tracemalloc.get_traced_memory()[0]
            for index in range(1_000, 15_000):
                writer.append("birds", (str(index), f"{index:06d}" + "x" * 200))
            later_memory = tracemalloc.get_traced_memory()[0]
            result = writer.finalize()
    finally:
        tracemalloc.stop()
    assert later_memory - early_memory < 1_000_000
    assert result.sheets[0].row_count == 15_000
    assert len(worksheet_rows(result.path)) == 15_001
