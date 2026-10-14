"""File-backed logger construction for Gaia runtime observability."""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


_MAX_CHAT_USERNAME_LENGTH = 64


def bounded_log_detail(value: object, *, limit: int) -> str:
    """Render one printable, single-line log value within a fixed limit."""
    rendered = "".join(
        character if character.isprintable() else f"\\x{ord(character):02x}"
        for character in str(value)
    )
    rendered = " ".join(rendered.split()) or "<empty>"
    if len(rendered) <= limit:
        return rendered
    return f"{rendered[: limit - 3]}..."


def normalize_chat_username(value: object) -> str:
    """Normalize the username shared by CLI and web chat trace identity."""
    if not isinstance(value, str):
        raise TypeError("username must be text")
    username = LoggerFactory._filename_component(value, field="username")
    if len(username) > _MAX_CHAT_USERNAME_LENGTH:
        raise ValueError(
            "normalized username must be at most "
            f"{_MAX_CHAT_USERNAME_LENGTH} characters"
        )
    return username


def create_chat_directory(trace_root: Path, username: str) -> Path:
    """Create a private UTC-stamped directory for a normalized chat username."""
    trace_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = trace_root / f"gaia_chat_{username}_{timestamp}"
    directory.mkdir(mode=0o700, exist_ok=False)
    return directory


class LoggerFactory:
    """Create one UTF-8 file logger for a semantic Gaia operation."""

    def __init__(self, filename: str, level: int = logging.INFO) -> None:
        if not isinstance(filename, str):
            raise TypeError(
                f"filename must be a str (got {type(filename).__name__})"
            )

        base_name = Path(filename).name
        if filename != base_name:
            raise ValueError("filename must not contain directory components")
        if not base_name or base_name in {".", ".."}:
            raise ValueError("filename must be a non-empty file name")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        process_id = os.getpid()
        stem = Path(base_name).stem

        self.run_id = f"{timestamp}-{process_id}-{uuid4().hex[:8]}"
        self.level = level
        self.log_dir = Path(__file__).resolve().parents[2] / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{stem}_{timestamp}-{process_id}.log"
        self.logger_name = f"{stem}.{self.run_id}"
        self._logger: logging.Logger | None = None

    @staticmethod
    def _filename_component(value: str, *, field: str) -> str:
        """Normalize one semantic filename component to lowercase snake case."""
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a str (got {type(value).__name__})")

        component = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
        if not component:
            raise ValueError(f"{field} must contain at least one letter or number")
        return component

    @classmethod
    def operation_filename(cls, namespace: str, operation: str, target: str) -> str:
        """Build the canonical namespace-operation-target filename stem."""
        return "_".join(
            (
                cls._filename_component(namespace, field="namespace"),
                cls._filename_component(operation, field="operation"),
                cls._filename_component(target, field="target"),
            )
        )

    def get_logger(self) -> logging.Logger:
        """Return a non-propagating logger backed by this run's log file."""
        if self._logger is not None:
            return self._logger
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.level)
        logger.propagate = False

        file_handler = logging.FileHandler(
            self.log_path,
            mode="a",
            encoding="utf-8",
            delay=True,
        )
        formatter = logging.Formatter(
            "%(asctime)sZ [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        formatter.converter = time.gmtime
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self._logger = logger
        return logger
