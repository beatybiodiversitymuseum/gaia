"""Canonical Darwin Core table registry tests."""

from pathlib import Path

from sqlalchemy import Text

from src.database.datasets import DATASETS, DatasetSchema
from src.database.darwin_core import load_simple_darwin_core_profile
from src.database.tablesetup import (
    CtcBirds,
    FcFish,
    TABLE_MODELS,
    build_dataset_table_map,
    build_table_models,
)


OCCURRENCE_PROFILE = load_simple_darwin_core_profile()
OCCURRENCE_HEADERS = OCCURRENCE_PROFILE.headers


def test_mapping_is_derived_from_source_stems() -> None:
    datasets_by_key = {dataset.key: dataset for dataset in DATASETS}

    mapping = build_dataset_table_map(
        datasets=(datasets_by_key["birds"], datasets_by_key["fish"]),
        table_models=(FcFish, CtcBirds),
    )

    assert mapping == {
        "birds": CtcBirds.__table__,
        "fish": FcFish.__table__,
    }


def test_table_model_is_built_for_a_dynamically_discovered_dataset() -> None:
    dataset = DatasetSchema(
        key="some_other_dataset_62a26a992e",
        short_label="Some Other Dataset",
        collection_key="darwin_core_archives",
        collection_label="Darwin Core Archives",
        relative_path=Path(
            "dataset/darwin_core_archives/some_other_dataset_62a26a992e/"
            "some_other_dataset_62a26a992e.txt"
        ),
        identity_values=("https://example.org/datasets/specimens-2026",),
    )

    models = build_table_models((dataset,))
    mapping = build_dataset_table_map(datasets=(dataset,), table_models=models)

    assert len(models) == 1
    assert mapping[dataset.key] is models[0].__table__
    assert models[0].__table__.name == dataset.relative_path.stem
    assert models[0].__table__.info["source_headers"] == OCCURRENCE_HEADERS


def test_occurrence_headers_are_derived_from_simple_darwin_core() -> None:
    assert OCCURRENCE_HEADERS == OCCURRENCE_PROFILE.headers
    assert len(OCCURRENCE_HEADERS) == 206
    assert len(set(OCCURRENCE_HEADERS)) == len(OCCURRENCE_HEADERS)


def test_every_dataset_uses_the_complete_canonical_occurrence_schema() -> None:
    assert len(TABLE_MODELS) == len(DATASETS)

    for model in TABLE_MODELS:
        table = model.__table__
        assert table.info["source_headers"] == OCCURRENCE_HEADERS
        assert tuple(column.name for column in table.columns) == OCCURRENCE_HEADERS
        assert all(isinstance(column.type, Text) for column in table.columns)
        assert all(not column.nullable for column in table.columns)
        assert all(column.server_default is not None for column in table.columns)
        assert all(column.server_default.arg.text == "''" for column in table.columns)


def test_occurrence_identity_and_catalog_index_contract() -> None:
    for model in TABLE_MODELS:
        table = model.__table__
        assert tuple(column.name for column in table.primary_key.columns) == (
            "occurrenceID",
        )
        assert any(
            tuple(column.name for column in index.columns) == ("catalogNumber",)
            for index in table.indexes
        )
