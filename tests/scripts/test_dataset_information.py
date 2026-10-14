import json
import os
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.database.datasets import DATASETS, DatasetSchema
from src.scripts import dataset_information
from src.scripts.dataset_schema import jsonl_relative_path


REPORT = "# Dataset Information\n\nGenerated report — 雪.\n"
README = "# Gaia — diversité\r\n\r\nHandwritten notes 🐦.\r\n".encode("utf-8")


class _FailedWriter:
    def __init__(self, stream) -> None:
        self.stream = stream

    def write(self, content: str) -> None:
        self.stream.write(content[:8])
        raise OSError("simulated output failure")


@contextmanager
def _failed_fdopen(original_fdopen, *args, **kwargs):
    with original_fdopen(*args, **kwargs) as stream:
        yield _FailedWriter(stream)


@pytest.mark.parametrize("readme_exists", [False, True])
def test_default_report_creation_preserves_readme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, readme_exists: bool
) -> None:
    readme = tmp_path / "README.md"
    if readme_exists:
        readme.write_bytes(README)
    monkeypatch.setattr(dataset_information, "build_document", lambda root: REPORT)
    report = tmp_path / "reports" / "dataset_information.md"

    assert dataset_information.generate_dataset_information(tmp_path) == report
    assert report.read_bytes() == REPORT.encode("utf-8")
    assert readme.exists() is readme_exists
    if readme_exists:
        assert readme.read_bytes() == README

    dataset_information.generate_dataset_information(tmp_path)
    assert report.read_bytes() == REPORT.encode("utf-8")
    assert not tuple(report.parent.glob(".*.tmp"))


def test_explicit_destination_replaces_only_that_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readme = tmp_path / "README.md"
    readme.write_bytes(README)
    default = tmp_path / "reports" / "dataset_information.md"
    default.parent.mkdir()
    default.write_text("Existing default report.\n", encoding="utf-8")
    target = tmp_path / "custom.md"
    target.write_text("Previous report.\n", encoding="utf-8")
    build_report = Mock(return_value=REPORT)
    monkeypatch.setattr(dataset_information, "build_document", build_report)

    assert dataset_information.generate_dataset_information(tmp_path, target) == target
    assert target.read_bytes() == REPORT.encode("utf-8")
    assert default.read_text(encoding="utf-8") == "Existing default report.\n"
    assert readme.read_bytes() == README
    build_report.assert_called_once_with(tmp_path)


def test_generation_failure_preserves_existing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report.md"
    original = "Previous report — 雪.\n".encode("utf-8")
    report.write_bytes(original)

    failed_build = Mock(side_effect=ValueError("invalid source data"))

    monkeypatch.setattr(dataset_information, "build_document", failed_build)

    with pytest.raises(ValueError, match="invalid source data"):
        dataset_information.generate_dataset_information(tmp_path, report)

    assert report.read_bytes() == original
    assert not tuple(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("failure_stage", ["write", "replace"])
def test_output_failure_preserves_existing_report_and_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    report = tmp_path / "report.md"
    original = "Previous report — 雪.\n".encode("utf-8")
    report.write_bytes(original)
    monkeypatch.setattr(dataset_information, "build_document", lambda root: REPORT)

    if failure_stage == "write":

        monkeypatch.setattr(dataset_information.os, "fdopen", partial(_failed_fdopen, os.fdopen))
    else:
        failed_replace = Mock(side_effect=OSError("simulated output failure"))

        monkeypatch.setattr(Path, "replace", failed_replace)

    with pytest.raises(OSError, match="simulated output failure"):
        dataset_information.generate_dataset_information(tmp_path, report)

    assert report.read_bytes() == original
    assert not tuple(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "destination_kind",
    ["direct", "lowercase", "nested", "symlink", "hardlink", "readme_target"],
)
def test_readme_destinations_are_rejected_before_profiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination_kind: str
) -> None:
    readme = tmp_path / "README.md"
    readme.write_bytes(README)
    if destination_kind == "direct":
        target = readme
    elif destination_kind == "lowercase":
        target = tmp_path / "readme.md"
    elif destination_kind == "nested":
        target = tmp_path / "docs" / "README.md"
    elif destination_kind == "symlink":
        target = tmp_path / "alias.md"
        target.symlink_to(readme)
    elif destination_kind == "hardlink":
        target = tmp_path / "alias.md"
        target.hardlink_to(readme)
    else:
        target = tmp_path / "manuscript.md"
        readme.rename(target)
        readme.symlink_to(target)

    unexpected_build = Mock(
        side_effect=AssertionError("README destinations must be rejected before data profiling")
    )

    monkeypatch.setattr(dataset_information, "build_document", unexpected_build)

    with pytest.raises(ValueError, match="standalone document.*README.md"):
        dataset_information.generate_dataset_information(tmp_path, target)

    assert readme.read_bytes() == README
    assert not tuple(tmp_path.rglob(".*.tmp"))


def _write_source(
    root: Path,
    dataset: DatasetSchema,
    records: tuple[dict[str, str], ...],
) -> Path:
    source = root / dataset.relative_path
    source.parent.mkdir(parents=True)
    source.write_text("\t".join(records[0]) + "\n", encoding="utf-8")
    jsonl = root / jsonl_relative_path(dataset)
    with jsonl.open("w", encoding="utf-8") as output:
        for raw_record in records:
            record = {
                "occurrenceID": raw_record["occurrenceID"],
                "catalogNumber": raw_record["catalogNumber"],
                "chunks": {},
                "raw_record": raw_record,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return jsonl


def test_analysis_counts_records_and_exact_headers_across_sources(
    tmp_path: Path,
) -> None:
    datasets = DATASETS[:2]
    first_header = ("id", "occurrenceID", "catalogNumber", "scientificName")
    _write_source(
        tmp_path,
        datasets[0],
        (
            dict(zip(first_header, ("雪1", "雪1", "001", "苔"))),
            dict(zip(first_header, ("雪2", "雪2", "002", ""))),
        ),
    )
    _write_source(
        tmp_path,
        datasets[1],
        ({"id": "3", "occurrenceID": "3", "catalogNumber": "003", "eventDate": ""},),
    )

    analysis = dataset_information.analyze_dataset_information(tmp_path, datasets)

    assert analysis.total_records == 3
    assert [source.record_count for source in analysis.sources] == [2, 1]
    assert [source.dataset for source in analysis.sources] == list(datasets)
    assert analysis.sources[0].header == first_header
    assert set(analysis.headers) == {*first_header, "eventDate"}
    assert len(analysis.headers) == 5


@pytest.mark.parametrize(
    ("corruption", "expected_error"),
    [
        ("schema", "raw_record schema drift"),
        ("identifier_copy", "occurrenceID value mismatch"),
        ("catalog_copy", "catalogNumber value mismatch"),
        ("null_value", "raw_record values must all be strings"),
        ("invalid_json", "invalid JSON"),
        ("empty_source", "JSONL source contains no records"),
        ("unknown_header", "source header.*have no semantic chunk"),
    ],
)
def test_analysis_rejects_invalid_sources(
    tmp_path: Path, corruption: str, expected_error: str
) -> None:
    datasets = DATASETS[:1]
    jsonl = _write_source(
        tmp_path,
        datasets[0],
        ({"id": "1", "occurrenceID": "1", "catalogNumber": "001", "scientificName": "苔"},),
    )
    record = json.loads(jsonl.read_text(encoding="utf-8"))
    if corruption == "schema":
        del record["raw_record"]["scientificName"]
    elif corruption == "identifier_copy":
        record["occurrenceID"] = "other"
    elif corruption == "catalog_copy":
        record["catalogNumber"] = "other"
    elif corruption == "null_value":
        record["raw_record"]["scientificName"] = None
    elif corruption == "unknown_header":
        source = tmp_path / datasets[0].relative_path
        source.write_text(
            source.read_text(encoding="utf-8").replace("scientificName", "unknownField"),
            encoding="utf-8",
        )
    if corruption == "invalid_json":
        jsonl.write_text("{invalid}\n", encoding="utf-8")
    elif corruption == "empty_source":
        jsonl.write_text("", encoding="utf-8")
    else:
        jsonl.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        dataset_information.analyze_dataset_information(tmp_path, datasets)
