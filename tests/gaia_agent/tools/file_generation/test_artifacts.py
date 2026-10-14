"""Delivery metadata and immutable session-file access contracts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.gaia_agent.runtime.contracts import ToolCall, ToolResult
from src.gaia_agent.tools.file_generation.artifacts import (
    EXCEL_MEDIA_TYPE,
    collect_generated_files,
    open_generated_file,
)


def _metadata(directory: Path, filename: str, file_id: str) -> dict[str, str]:
    path = directory / filename
    path.write_bytes(f"workbook-{file_id}".encode())
    return {"file_id": file_id, "filename": filename}


def _result(name: str, payload: object, *, ok: bool = True) -> ToolResult:
    return ToolResult(call=ToolCall("call", name, {}), ok=ok, result=payload)


def test_collects_generated_files_in_call_order(tmp_path: Path) -> None:
    original = _metadata(tmp_path, "Échantillons 鳥.xlsx", "original")
    other = _metadata(tmp_path, "mammals.xlsx", "second")
    later = _metadata(tmp_path, "fish.xlsx", "third")
    results = (
        _result("generate_excel", {"files": [original, other]}),
        _result("generate_excel", {"files": [later]}),
        _result("generate_excel", {"files": [original]}),
        _result("generate_excel", {"files": [other]}, ok=False),
    )

    files = collect_generated_files(results, export_directory=tmp_path)

    assert [file.file_id for file in files] == ["original", "second", "third"]
    assert [file.filename for file in files] == [
        original["filename"], other["filename"], later["filename"]
    ]
    for file in files:
        assert file.media_type == EXCEL_MEDIA_TYPE
        with open_generated_file(file, export_directory=tmp_path) as source:
            assert source.read() == f"workbook-{file.file_id}".encode()
            assert file.size_bytes == source.tell()


def test_ignores_unestablished_or_foreign_files(tmp_path: Path) -> None:
    directory = tmp_path / "exports"
    directory.mkdir()
    metadata = _metadata(directory, "birds.xlsx", "birds")
    foreign = _metadata(tmp_path, "foreign.xlsx", "foreign")
    candidates = [
        foreign,
        {**foreign, "filename": "../foreign.xlsx"},
        {**foreign, "filename": str(tmp_path / "foreign.xlsx")},
        {**foreign, "filename": r"..\foreign.xlsx"},
        {**metadata, "file_id": "bad/id"},
        {**metadata, "filename": "birds\n.xlsx"},
        {**metadata, "filename": "missing.xlsx"},
        {"filename": "birds.xlsx"},
        None,
    ]
    results = (
        _result("generate_excel", {"files": candidates}),
        _result("generate_excel", {"files": metadata}),
        _result("unrelated_tool", {"files": [metadata]}),
        _result("generate_excel", {"files": [metadata]}, ok=False),
    )

    assert collect_generated_files(results, export_directory=directory) == ()


def test_metadata_cannot_redirect_downloads_with_a_filesystem_path(tmp_path: Path) -> None:
    directory = tmp_path / "exports"
    directory.mkdir()
    metadata = _metadata(directory, "birds.xlsx", "birds")
    foreign = tmp_path / "private.xlsx"
    foreign.write_bytes(b"private file")
    result = _result("generate_excel", {
        "files": [{**metadata, "path": str(foreign)}],
    })

    file, = collect_generated_files((result,), export_directory=directory)

    assert file.path == directory / "birds.xlsx"
    with open_generated_file(file, export_directory=directory) as source:
        assert source.read() == b"workbook-birds"


@pytest.mark.parametrize(
    "replacement",
    ["missing", "symlink", "directory", "fifo", "changed", "same_size"],
)
def test_refuses_missing_or_replaced_files(tmp_path: Path, replacement: str) -> None:
    metadata = _metadata(tmp_path, "birds.xlsx", "birds")
    result = _result("generate_excel", {"files": [metadata]})
    file, = collect_generated_files((result,), export_directory=tmp_path)
    file.path.unlink()
    if replacement == "symlink":
        target = tmp_path / "other.xlsx"
        target.write_bytes(b"private")
        file.path.symlink_to(target)
    elif replacement == "directory":
        file.path.mkdir()
    elif replacement == "fifo":
        os.mkfifo(file.path)
    elif replacement == "changed":
        file.path.write_bytes(b"changed")
    elif replacement == "same_size":
        file.path.write_bytes(b"X" * file.size_bytes)

    with pytest.raises((OSError, ValueError)):
        open_generated_file(file, export_directory=tmp_path)
    if replacement not in {"changed", "same_size"}:
        assert collect_generated_files((result,), export_directory=tmp_path) == ()


def test_refuses_another_session_and_replaced_export_directory(tmp_path: Path) -> None:
    directory = tmp_path / "exports"
    directory.mkdir()
    metadata = _metadata(directory, "birds.xlsx", "birds")
    result = _result("generate_excel", {"files": [metadata]})
    file, = collect_generated_files((result,), export_directory=directory)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    _metadata(foreign, "birds.xlsx", "birds")

    with pytest.raises(ValueError, match="does not belong"):
        open_generated_file(file, export_directory=foreign)

    directory.rename(tmp_path / "original")
    directory.symlink_to(foreign, target_is_directory=True)
    assert collect_generated_files((result,), export_directory=directory) == ()
    with pytest.raises((OSError, ValueError)):
        open_generated_file(file, export_directory=directory)
