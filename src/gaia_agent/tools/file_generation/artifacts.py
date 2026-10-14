"""Identify and open completed files within one chat's export directory."""

from __future__ import annotations

from hashlib import sha256
import os
import re
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from src.gaia_agent.runtime.contracts import ToolResult


EXCEL_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_FILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_UNSAFE_FILENAME_PATTERN = re.compile(r"[/\\:\x00-\x1f\x7f-\x9f]")


@dataclass(frozen=True, slots=True)
class GeneratedFile:
    """Describe a completed export independently of its delivery transport."""

    file_id: str
    filename: str
    path: Path
    size_bytes: int
    sha256_digest: str
    media_type: str = EXCEL_MEDIA_TYPE


def collect_generated_files(
    tool_results: Iterable[ToolResult], *, export_directory: Path
) -> tuple[GeneratedFile, ...]:
    """Collect existing generated workbooks in tool and file order.

    Successful file tools supply only identifiers and basenames. Files are
    resolved against the server-owned session directory; result metadata cannot
    select a different directory. Repeated results retain their first place.
    """
    files: list[GeneratedFile] = []
    seen_ids: set[str] = set()
    for result in tool_results:
        if (
            not result.ok
            or result.call.name != "generate_excel"
            or not isinstance(result.result, Mapping)
        ):
            continue
        candidates = result.result.get("files")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            file = _generated_file(candidate, export_directory)
            if file is not None and file.file_id not in seen_ids:
                files.append(file)
                seen_ids.add(file.file_id)
    return tuple(files)


def open_generated_file(
    file: GeneratedFile, *, export_directory: Path
) -> BinaryIO:
    """Open a registered file without following a replaced file/directory link.

    The caller owns the returned stream. Missing, replaced, or unsafe files
    raise OSError or ValueError; each delivered file retains its original bytes.
    """
    expected_path = export_directory.resolve(strict=True) / file.filename
    if file.path != expected_path:
        raise ValueError("The file does not belong to this export directory.")
    source = open_session_export(file.filename, export_directory=export_directory)
    if (
        os.fstat(source.fileno()).st_size != file.size_bytes
        or _sha256_digest(source) != file.sha256_digest
    ):
        source.close()
        raise ValueError("The exported file has changed since it was generated.")
    return source


def _generated_file(
    metadata: object, export_directory: Path
) -> GeneratedFile | None:
    if not isinstance(metadata, Mapping):
        return None
    file_id = metadata.get("file_id")
    filename = metadata.get("filename")
    if (
        not isinstance(file_id, str)
        or _FILE_ID_PATTERN.fullmatch(file_id) is None
        or not isinstance(filename, str)
    ):
        return None
    try:
        expected_path = export_directory.resolve(strict=True) / filename
        with open_session_export(filename, export_directory=export_directory) as source:
            size_bytes = os.fstat(source.fileno()).st_size
            sha256_digest = _sha256_digest(source)
    except (OSError, ValueError):
        return None
    return GeneratedFile(file_id, filename, expected_path, size_bytes, sha256_digest)


def _sha256_digest(source: BinaryIO) -> str:
    """Hash a regular file and restore its position for subsequent delivery."""
    digest = sha256()
    while chunk := source.read(1024 * 1024):
        digest.update(chunk)
    source.seek(0)
    return digest.hexdigest()


def _validate_filename(filename: str) -> None:
    if (
        not isinstance(filename, str)
        or not filename.endswith(".xlsx")
        or not filename[:-5].strip(" .")
        or _UNSAFE_FILENAME_PATTERN.search(filename)
    ):
        raise ValueError("An export filename must be an Excel basename.")


def open_session_export(filename: str, *, export_directory: Path) -> BinaryIO:
    """Open one Excel basename for byte transfer within the session directory.

    The caller owns the returned stream. Links and nonregular files are rejected.
    """
    _validate_filename(filename)
    # Open relative to the held directory so a concurrent rename cannot redirect
    # the file lookup. Nonblocking avoids hanging if a file is replaced by a FIFO.
    directory = os.open(
        export_directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        descriptor = os.open(
            filename,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory,
        )
    finally:
        os.close(directory)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("The export is not a regular file.")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
