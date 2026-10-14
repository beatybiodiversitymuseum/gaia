"""Load and select deterministic metadata-only schema context."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import cast

from src.database.datasets import DatasetSchema
from src.database.schema_profiles import load_dataset_schema_profiles


DEFAULT_SCHEMA_TEMPLATE_PATH = Path(__file__).with_name("plan_templates.json")


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate keys instead of allowing JSON to overwrite them."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"rag schema contains a duplicate JSON key: {key}")
        result[key] = value
    return result


def _freeze(value: object) -> object:
    """Recursively protect the one process-wide schema from mutation."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _materialize(value: object) -> object:
    """Return a detached JSON-compatible value from the frozen cache."""
    if isinstance(value, Mapping):
        return {key: _materialize(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_materialize(item) for item in value]
    return value


def _apply_dataset_profiles(
    template: dict[str, object],
    datasets: tuple[DatasetSchema, ...],
    profiles: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    """Combine static plan metadata with persisted per-dataset availability."""
    expected_keys = tuple(dataset.key for dataset in datasets)
    expected_key_set = set(expected_keys)
    missing_keys = tuple(key for key in expected_keys if key not in profiles)
    extra_keys = tuple(key for key in profiles if key not in expected_key_set)
    if missing_keys or extra_keys:
        details = []
        if missing_keys:
            details.append(f"missing dataset profiles: {', '.join(missing_keys)}")
        if extra_keys:
            details.append(f"unknown dataset profiles: {', '.join(extra_keys)}")
        raise ValueError("; ".join(details))

    schema = {
        "unique identifiers": ["occurrenceID"],
        "query plan structure": template["query plan structure"],
        "statistics plan structure": template["statistics plan structure"],
    }
    for dataset_schema in datasets:
        collection = dataset_schema.collection_label
        dataset = dataset_schema.short_label
        collection_data = cast(
            dict[str, object], schema.setdefault(collection, {})
        )
        dataset_data = cast(
            dict[str, object], collection_data.setdefault(dataset, {})
        )
        dataset_data["headers"] = dict(profiles[dataset_schema.key])

    ordered_schema = {
        collection: schema[collection]
        for collection in dict.fromkeys(
            dataset.collection_label for dataset in datasets
        )
    }
    ordered_schema.update(
        {
            "unique identifiers": schema["unique identifiers"],
            "query plan structure": schema["query plan structure"],
            "statistics plan structure": schema["statistics plan structure"],
        }
    )
    return ordered_schema


@cache
def _load_template() -> Mapping[str, object]:
    """Parse and freeze the colocated static RAG template once per process."""
    try:
        schema = json.loads(
            DEFAULT_SCHEMA_TEMPLATE_PATH.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "could not load schema plan templates from "
            f"{DEFAULT_SCHEMA_TEMPLATE_PATH}: {error}"
        ) from error
    if not isinstance(schema, dict):
        raise ValueError("rag schema must be a JSON object")
    return cast(Mapping[str, object], _freeze(schema))


def load_schema(
    repository_root: Path,
    datasets: tuple[DatasetSchema, ...],
) -> Mapping[str, object]:
    """Load persisted availability and combine it with the static RAG template."""
    profiles = load_dataset_schema_profiles(repository_root, datasets=datasets)
    template = cast(dict[str, object], _materialize(_load_template()))
    schema = _apply_dataset_profiles(template, datasets, profiles)
    return cast(Mapping[str, object], _freeze(schema))


def _dataset_result(
    schema: Mapping[str, object],
    key: str,
    dataset_registry: Mapping[str, tuple[str, str]],
) -> dict[str, object]:
    collection, dataset = dataset_registry[key]
    collection_data = cast(Mapping[str, object], schema[collection])
    dataset_data = cast(Mapping[str, object], collection_data[dataset])
    headers = cast(Mapping[str, int], dataset_data["headers"])
    return {
        "key": key,
        "collection": collection,
        "dataset": dataset,
        "headers": dict(headers),
    }


def retrieve_schema_for_registry(
    schema: Mapping[str, object],
    *,
    datasets: Sequence[str] = (),
    collections: Sequence[str] = (),
    dataset_registry: Mapping[str, tuple[str, str]],
) -> dict[str, object]:
    """Return plan templates with the requested header structures."""
    dataset_selectors = {
        selector: key
        for key, (_collection, dataset) in dataset_registry.items()
        for selector in (key, dataset)
    }
    collection_selectors = {
        collection: collection
        for collection, _dataset in dataset_registry.values()
    }
    requested = (
        ("datasets", datasets, dataset_selectors),
        ("collections", collections, collection_selectors),
    )
    selections: dict[str, tuple[str, ...]] = {}
    for argument, values, options in requested:
        selected: list[str] = []
        for value in values:
            try:
                canonical = options[value]
            except (KeyError, TypeError) as error:
                raise ValueError(f"unknown {argument}: {value}") from error
            if canonical not in selected:
                selected.append(cast(str, canonical))
        selections[argument] = tuple(selected)

    selected_datasets = selections["datasets"]
    selected_collections = selections["collections"]
    if not selected_datasets and not selected_collections:
        raise ValueError("at least one of datasets or collections is required")

    result: dict[str, object] = {
        "query plan structure": _materialize(schema["query plan structure"]),
        "statistics plan structure": _materialize(schema["statistics plan structure"]),
    }
    if selected_datasets:
        result["datasets"] = [
            _dataset_result(schema, key, dataset_registry) for key in selected_datasets
        ]
    if selected_collections:
        result["collections"] = [
            {
                "collection": collection,
                "datasets": [
                    {
                        "key": key,
                        "dataset": dataset,
                        "headers": _dataset_result(
                            schema, key, dataset_registry
                        )["headers"],
                    }
                    for key, (
                        registered_collection,
                        dataset,
                    ) in dataset_registry.items()
                    if registered_collection == collection
                ],
            }
            for collection in selected_collections
        ]
    return result
