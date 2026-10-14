"""Logger factory lifecycle contracts."""

from __future__ import annotations

import logging
from pathlib import Path

from src.logger.logger_factory import LoggerFactory


def test_get_logger_reuses_its_handler(tmp_path: Path) -> None:
    factory = LoggerFactory("publication-test.log")
    factory.log_path = tmp_path / "publication-test.log"

    first = factory.get_logger()
    second = factory.get_logger()
    try:
        assert first is second
        assert len(first.handlers) == 1
        first.info("written once")
        first.handlers[0].flush()
        assert factory.log_path.read_text(encoding="utf-8").count("written once") == 1
    finally:
        for handler in tuple(first.handlers):
            first.removeHandler(handler)
            handler.close()
        logging.Logger.manager.loggerDict.pop(factory.logger_name, None)
