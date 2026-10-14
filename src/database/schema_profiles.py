"""Persist per-dataset occurrence field availability for schema retrieval."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from src.database.darwin_core import load_simple_darwin_core_profile
from src.database.datasets import DatasetSchema


DATASET_SCHEMA_PROFILE_RELATIVE_PATH: Final = Path(
    "dataset/.gaia/rag_schema.json"
)
FIELD_UNAVAILABLE: Final = 0
FIELD_AVAILABLE: Final = 1
FIELD_AVAILABILITY_VALUES: Final = frozenset(
    (FIELD_UNAVAILABLE, FIELD_AVAILABLE)
)
OCCURRENCE_HEADERS: Final = load_simple_darwin_core_profile().headers

DatasetSchemaProfiles = dict[str, dict[str, int]]


def _validate_profiles(
    profiles: Mapping[str, Mapping[str, int]],
    *,
    datasets: Sequence[DatasetSchema],
) -> DatasetSchemaProfiles:
    """Return a canonical copy after validating the persisted contract."""
    known_keys = {dataset.key for dataset in datasets}
    unknown_keys = sorted(set(profiles) - known_keys)
    if unknown_keys:
        raise ValueError(
            "schema profiles contain unknown dataset(s): "
            + ", ".join(unknown_keys)
        )

    canonical: DatasetSchemaProfiles = {}
    for dataset in datasets:
        if dataset.key not in profiles:
            continue
        availability = profiles[dataset.key]
        if not isinstance(availability, Mapping):
            raise ValueError(
                f"schema profile for {dataset.key} must be an object"
            )
        if tuple(availability) != OCCURRENCE_HEADERS:
            raise ValueError(
                f"schema profile for {dataset.key} must contain every "
                "canonical occurrence header in canonical order"
            )
        invalid_headers = tuple(
            header
            for header, value in availability.items()
            if type(value) is not int or value not in FIELD_AVAILABILITY_VALUES
        )
        if invalid_headers:
            raise ValueError(
                f"schema profile for {dataset.key} has non-binary values"
            )
        canonical[dataset.key] = {
            header: availability[header] for header in OCCURRENCE_HEADERS
        }
    return canonical


def load_dataset_schema_profiles(
    repository_root: Path,
    *,
    datasets: Sequence[DatasetSchema],
) -> DatasetSchemaProfiles:
    """Load and validate schema profiles, returning empty state if absent."""
    path = repository_root / DATASET_SCHEMA_PROFILE_RELATIVE_PATH
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid schema profile JSON: {path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"schema profile document must be an object: {path}")
    return _validate_profiles(document, datasets=datasets)


def write_dataset_schema_profiles(
    repository_root: Path,
    profiles: Mapping[str, Mapping[str, int]],
    *,
    datasets: Sequence[DatasetSchema],
) -> Path:
    """Validate and atomically replace the schema profile document."""
    canonical = _validate_profiles(profiles, datasets=datasets)
    path = repository_root / DATASET_SCHEMA_PROFILE_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(canonical, ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return path
