import json
from pathlib import Path

import pytest

from src.database.darwin_core import load_simple_darwin_core_profile
from src.database.datasets import DatasetSchema
from src.database.schema_profiles import (
    DATASET_SCHEMA_PROFILE_RELATIVE_PATH,
    load_dataset_schema_profiles,
    write_dataset_schema_profiles,
)


OCCURRENCE_HEADERS = load_simple_darwin_core_profile().headers


def make_dataset(key: str) -> DatasetSchema:
    return DatasetSchema(
        key=key,
        short_label=key,
        collection_key="test",
        collection_label="Test",
        relative_path=Path(f"dataset/test/{key}.txt"),
    )


def make_availability(value: int = 0) -> dict[str, int]:
    return {header: value for header in OCCURRENCE_HEADERS}


def test_profiles_round_trip_in_dataset_and_header_order(tmp_path: Path) -> None:
    datasets = (make_dataset("birds"), make_dataset("fish"))
    profiles = {
        "fish": make_availability(),
        "birds": make_availability(1),
    }

    path = write_dataset_schema_profiles(
        tmp_path,
        profiles,
        datasets=datasets,
    )

    assert path == tmp_path / DATASET_SCHEMA_PROFILE_RELATIVE_PATH
    document = json.loads(path.read_text(encoding="utf-8"))
    assert tuple(document) == ("birds", "fish")
    assert tuple(document["birds"]) == OCCURRENCE_HEADERS
    assert load_dataset_schema_profiles(
        tmp_path,
        datasets=datasets,
    ) == document


def test_missing_profile_document_is_empty(tmp_path: Path) -> None:
    assert load_dataset_schema_profiles(
        tmp_path,
        datasets=(make_dataset("birds"),),
    ) == {}


@pytest.mark.parametrize("invalid_value", (True, 2, "1"))
def test_profile_values_must_be_integer_zero_or_one(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    dataset = make_dataset("birds")
    availability = make_availability()
    availability["occurrenceID"] = invalid_value  # type: ignore[assignment]

    with pytest.raises(ValueError, match="non-binary"):
        write_dataset_schema_profiles(
            tmp_path,
            {"birds": availability},
            datasets=(dataset,),
        )
