"""Generate a concise, data-derived overview of Gaia datasets."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.database.datasets import DATASETS, DatasetSchema
from src.scripts.dataset_schema import (
    CHUNKS,
    OUTPUT_RECORD_KEYS,
    jsonl_relative_path,
    read_tsv_header,
)


# ---------------------------------------------------------------------------
# Report contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceProfile:
    """Describe one validated JSON Lines source."""

    dataset: DatasetSchema
    header: tuple[str, ...]
    record_count: int


@dataclass(frozen=True, slots=True)
class HeaderAnalysis:
    """Hold the validated counts and exact field names used by the report."""

    sources: tuple[SourceProfile, ...]
    headers: tuple[str, ...]
    total_records: int


# ---------------------------------------------------------------------------
# JSON Lines validation and profiling
# ---------------------------------------------------------------------------


def _validate_jsonl_inventory(
    repository_root: Path, datasets: tuple[DatasetSchema, ...]
) -> dict[str, Path]:
    """Require the discovered JSON Lines inventory to match the registry."""
    dataset_root = repository_root / "dataset"
    discovered = {
        path.relative_to(repository_root)
        for path in dataset_root.rglob("*.jsonl")
    }
    expected = {jsonl_relative_path(dataset) for dataset in datasets}

    missing = sorted(path.as_posix() for path in expected - discovered)
    unexpected = sorted(path.as_posix() for path in discovered - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing JSONL file(s): {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected JSONL file(s): {', '.join(unexpected)}")
        raise ValueError("; ".join(details))

    return {
        dataset.key: repository_root / jsonl_relative_path(dataset)
        for dataset in datasets
    }


def _parse_raw_record(
    line: str,
    *,
    path: Path,
    line_number: int,
    expected_header: tuple[str, ...],
) -> dict[str, str]:
    """Parse and validate the reporting surface from one JSON Lines record."""
    if not line.strip():
        raise ValueError(f"{path}:{line_number}: blank JSONL line")
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path}:{line_number}: invalid JSON: {error.msg}"
        ) from error

    if not isinstance(record, dict):
        raise ValueError(f"{path}:{line_number}: record must be a JSON object")
    if tuple(record) != OUTPUT_RECORD_KEYS:
        raise ValueError(
            f"{path}:{line_number}: unexpected top-level record schema"
        )

    raw_record = record["raw_record"]
    if not isinstance(raw_record, dict):
        raise ValueError(f"{path}:{line_number}: raw_record must be an object")
    if tuple(raw_record) != expected_header:
        raise ValueError(f"{path}:{line_number}: raw_record schema drift")
    if any(not isinstance(value, str) for value in raw_record.values()):
        raise ValueError(
            f"{path}:{line_number}: raw_record values must all be strings"
        )
    for identity_header in ("occurrenceID", "catalogNumber"):
        if record[identity_header] != raw_record[identity_header]:
            raise ValueError(
                f"{path}:{line_number}: {identity_header} value mismatch"
            )

    chunks = record["chunks"]
    if not isinstance(chunks, dict):
        raise ValueError(f"{path}:{line_number}: chunks must be an object")

    return raw_record


def analyze_dataset_information(
    repository_root: Path,
    datasets: tuple[DatasetSchema, ...] = DATASETS,
) -> HeaderAnalysis:
    """Validate each registered JSONL while counting records in bounded memory."""
    jsonl_paths = _validate_jsonl_inventory(repository_root, datasets)
    header_order = tuple(name for fields in CHUNKS.values() for name in fields)
    expected_headers = {
        dataset.key: read_tsv_header(repository_root / dataset.relative_path)
        for dataset in datasets
    }
    observed_headers = {
        name for header in expected_headers.values() for name in header
    }
    unknown_headers = sorted(observed_headers - set(header_order))
    if unknown_headers:
        raise ValueError(
            "JSONL source header(s) have no semantic chunk: "
            + ", ".join(unknown_headers)
        )

    source_profiles: list[SourceProfile] = []
    total_records = 0
    for dataset in datasets:
        path = jsonl_paths[dataset.key]
        header = expected_headers[dataset.key]
        record_count = 0
        with path.open("r", encoding="utf-8", newline="") as source:
            for line_number, line in enumerate(source, start=1):
                _parse_raw_record(
                    line,
                    path=path,
                    line_number=line_number,
                    expected_header=header,
                )
                record_count += 1

        if record_count == 0:
            raise ValueError(f"{path}: JSONL source contains no records")
        source_profiles.append(
            SourceProfile(dataset=dataset, header=header, record_count=record_count)
        )
        total_records += record_count

    return HeaderAnalysis(
        sources=tuple(source_profiles),
        headers=tuple(name for name in header_order if name in observed_headers),
        total_records=total_records,
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


_FIELD_GROUP_LABELS = {
    "record_identity_admin": "Record identity and administration",
    "occurrence_organism": "Occurrence and organism",
    "event_sampling": "Event and sampling",
    "location_georeferencing": "Location and georeferencing",
    "geological_context": "Geological context",
    "identification": "Identification",
    "taxonomy_nomenclature": "Taxonomy and nomenclature",
    "associations_media": "Associations and media",
}


def render_document(analysis: HeaderAnalysis) -> str:
    """Render a standalone dataset-information Markdown report."""
    collection_count = len(
        {source.dataset.collection_key for source in analysis.sources}
    )
    observed_headers = set(analysis.headers)
    lines = [
        "# Dataset Information",
        "",
        f"The corpus contains **{len(analysis.sources)} datasets** across "
        f"**{collection_count} collections**,",
        f"with **{analysis.total_records:,} records** and "
        f"**{len(analysis.headers)} exact field names**.",
        "",
        "## Collections",
        "",
        "| Collection | Dataset | Records | Fields |",
        "| --- | --- | ---: | ---: |",
    ]
    for source in analysis.sources:
        lines.append(
            f"| {source.dataset.collection_label} | "
            f"{source.dataset.short_label} | {source.record_count:,} | "
            f"{len(source.header)} |"
        )

    lines.extend(
        (
            "",
            "## Scientific fields",
            "",
            f"Fields are organized into **{len(CHUNKS)} semantic groups**.",
            "Counts reflect exact field names present in at least one dataset.",
            "",
            "| Field group | Fields |",
            "| --- | ---: |",
        )
    )
    for group_name, field_names in CHUNKS.items():
        field_count = sum(name in observed_headers for name in field_names)
        lines.append(f"| {_FIELD_GROUP_LABELS[group_name]} | {field_count} |")

    lines.append("")
    return "\n".join(lines)


def build_document(repository_root: Path) -> str:
    """Build the JSONL-backed dataset-information report."""
    return render_document(analyze_dataset_information(repository_root))


def generate_dataset_information(
    repository_root: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Atomically write a standalone Markdown report, preserving README files."""
    resolved_root = repository_root or Path(__file__).resolve().parents[2]
    resolved_output = output_path or resolved_root / "reports" / "dataset_information.md"
    readme_path = resolved_root / "README.md"
    if (
        resolved_output.name.casefold() == "readme.md"
        or resolved_output.resolve() == readme_path.resolve()
        or (
            resolved_output.exists()
            and readme_path.exists()
            and resolved_output.samefile(readme_path)
        )
    ):
        raise ValueError(
            f"{resolved_output}: report output must be a standalone document, "
            "not README.md"
        )

    content = build_document(resolved_root)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=resolved_output.parent,
        prefix=f".{resolved_output.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
            target.write(content)
        temporary_path.replace(resolved_output)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return resolved_output


def main() -> None:
    """Generate the standalone dataset-information Markdown report."""
    output_path = generate_dataset_information()
    print(f"Dataset information generated successfully: {output_path}")


if __name__ == "__main__":
    main()
