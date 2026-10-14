from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import Mock

import pytest

import src.gaia_agent.rag.schema_retrieval as schema_retrieval
from src.gaia_agent.rag.schema_retrieval import (
    DEFAULT_SCHEMA_TEMPLATE_PATH,
    _json_object,
    load_schema,
    retrieve_schema_for_registry,
)
from src.database.datasets import DATASETS
from src.database.darwin_core import load_simple_darwin_core_profile
from src.database.schema_profiles import write_dataset_schema_profiles
from src.scripts.dataset_schema import read_tsv_header


OCCURRENCE_HEADERS = load_simple_darwin_core_profile().headers


REPOSITORY_ROOT = Path(__file__).parents[3]
COLLECTIONS = (
    "Cowan Tetrapod Collection",
    "Fish Collection",
    "Herbarium",
    "Marine Invertebrates",
    "Paleontology",
    "Spencer Entomology Collection",
)
EXPECTED_QUERY_PLAN_STRUCTURE = {
    "scope": {
        "collection": "<collection>",
        "dataset": "<dataset>",
    },
    "query": {"filters": {}, "column_mode": "default", "return_columns": []},
    "result": {"distinct_by": [], "order_by": [], "limit": 0},
}
DATASET_REGISTRY = {
    dataset.key: (dataset.collection_label, dataset.short_label)
    for dataset in DATASETS
}


def retrieve_schema(
    schema: Mapping[str, object],
    **arguments: object,
) -> dict[str, object]:
    """Retrieve schema against the explicit example-data test fixture."""
    return retrieve_schema_for_registry(
        schema,
        dataset_registry=DATASET_REGISTRY,
        **arguments,  # type: ignore[arg-type]
    )


def _profiles(
    *, unavailable: Mapping[str, set[str]] | None = None,
) -> dict[str, dict[str, int]]:
    unavailable = unavailable or {}
    return {
        dataset.key: {
            header: int(header not in unavailable.get(dataset.key, set()))
            for header in OCCURRENCE_HEADERS
        }
        for dataset in DATASETS
    }


@pytest.fixture
def schema(tmp_path: Path) -> Mapping[str, object]:
    write_dataset_schema_profiles(tmp_path, _profiles(), datasets=DATASETS)
    return load_schema(tmp_path, DATASETS)


def test_json_schema_is_colocated_complete_and_ordered(
    schema: Mapping[str, object],
) -> None:
    assert (
        DEFAULT_SCHEMA_TEMPLATE_PATH
        == REPOSITORY_ROOT / "src/gaia_agent/rag/plan_templates.json"
    )
    assert tuple(schema) == COLLECTIONS + (
        "unique identifiers",
        "query plan structure",
        "statistics plan structure",
    )

    actual_counts = {}
    for collection in COLLECTIONS:
        collection_schema = schema[collection]
        assert isinstance(collection_schema, Mapping)
        actual_counts.update(
            {
                dataset: len(dataset_schema["headers"])
                for dataset, dataset_schema in collection_schema.items()
            }
        )
    assert set(actual_counts.values()) == {len(OCCURRENCE_HEADERS)}
    assert tuple(schema["Cowan Tetrapod Collection"]) == (
        "Birds",
        "Herpetology",
        "Mammals",
    )
    birds_schema = schema["Cowan Tetrapod Collection"]["Birds"]  # type: ignore[index]
    assert tuple(birds_schema) == ("headers",)
    birds = birds_schema["headers"]
    assert tuple(birds) == OCCURRENCE_HEADERS
    assert birds["occurrenceID"] == 1
    assert birds["modified"] == 1


def test_header_availability_is_binary_and_identifiers_are_available(
    schema: Mapping[str, object],
) -> None:
    assert schema["unique identifiers"] == ("occurrenceID",)

    header_groups = [
        dataset["headers"]
        for collection in COLLECTIONS
        for dataset in schema[collection].values()
    ]
    for group in header_groups:
        assert all(type(value) is int and value in (0, 1) for value in group.values())
        for identifier in schema["unique identifiers"]:
            if identifier in group:
                assert group[identifier] == 1
def test_all_dataset_headers_match_registered_occurrence_schema(
    schema: Mapping[str, object],
) -> None:
    for dataset in DATASETS:
        dataset_schema = schema[dataset.collection_label][  # type: ignore[index]
            dataset.short_label
        ]
        assert tuple(dataset_schema["headers"]) == OCCURRENCE_HEADERS
        assert set(dataset_schema["headers"].values()) == {1}


def test_dataset_headers_use_persisted_availability(tmp_path: Path) -> None:
    profiles = _profiles(
        unavailable={"birds": {"behavior", "organismID"}},
    )
    write_dataset_schema_profiles(tmp_path, profiles, datasets=DATASETS)

    schema = load_schema(tmp_path, DATASETS)
    birds = schema["Cowan Tetrapod Collection"]["Birds"]["headers"]  # type: ignore[index]
    mammals = schema["Cowan Tetrapod Collection"]["Mammals"]["headers"]  # type: ignore[index]

    assert birds["behavior"] == 0
    assert birds["organismID"] == 0
    assert birds["scientificName"] == 1
    assert mammals["behavior"] == 1


def test_loader_rejects_incomplete_active_dataset_profiles(tmp_path: Path) -> None:
    profiles = _profiles()
    profiles.pop("birds")
    write_dataset_schema_profiles(tmp_path, profiles, datasets=DATASETS)

    with pytest.raises(ValueError, match="missing dataset profiles: birds"):
        load_schema(tmp_path, DATASETS)


def test_loader_parses_static_template_once(monkeypatch, tmp_path: Path) -> None:
    schema_retrieval._load_template.cache_clear()
    counted_loads = Mock(wraps=schema_retrieval.json.loads)
    monkeypatch.setattr(schema_retrieval.json, "loads", counted_loads)
    write_dataset_schema_profiles(tmp_path, _profiles(), datasets=DATASETS)
    first = load_schema(tmp_path, DATASETS)
    second = load_schema(tmp_path, DATASETS)

    assert first == second
    assert first is not second
    assert counted_loads.call_count == 3


def test_cached_schema_is_recursively_immutable(
    schema: Mapping[str, object],
) -> None:
    with pytest.raises(TypeError):
        schema["changed"] = {}  # type: ignore[index]

    identifiers = schema["unique identifiers"]
    assert isinstance(identifiers, tuple)
    with pytest.raises(AttributeError):
        identifiers.append("changed")  # type: ignore[attr-defined]

    collection = schema["Cowan Tetrapod Collection"]
    assert isinstance(collection, Mapping)
    birds = collection["Birds"]
    assert isinstance(birds, Mapping)
    headers = birds["headers"]
    assert isinstance(headers, Mapping)
    with pytest.raises(TypeError):
        headers["changed"] = 1  # type: ignore[index]


def test_json_object_rejects_duplicate_keys_during_startup_parse() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key: same"):
        _json_object([("same", 1), ("same", 2)])


def test_query_plan_structure_is_cached_and_retrieved_as_a_fresh_json_value(
    schema: Mapping[str, object],
) -> None:
    cached = schema["query plan structure"]
    assert isinstance(cached, Mapping)
    assert tuple(cached) == ("scope", "query", "result")
    cached_scope = cached["scope"]
    cached_query = cached["query"]
    cached_result = cached["result"]
    assert isinstance(cached_scope, Mapping)
    assert cached_scope == {
        "collection": "<collection>",
        "dataset": "<dataset>",
    }
    assert isinstance(cached_query, Mapping)
    assert isinstance(cached_query["filters"], Mapping)
    assert cached_query["column_mode"] == "default"
    assert cached_query["return_columns"] == ()
    assert isinstance(cached_result, Mapping)
    assert cached_result["distinct_by"] == ()
    assert cached_result["order_by"] == ()
    assert cached_result["limit"] == 0

    first = retrieve_schema(schema, datasets=("birds",))
    second = retrieve_schema(schema, datasets=("birds",))
    assert tuple(first) == (
        "query plan structure", "statistics plan structure", "datasets",
    )
    assert [entry["key"] for entry in first["datasets"]] == ["birds"]
    first_structure = first["query plan structure"]
    second_structure = second["query plan structure"]
    assert isinstance(first_structure, dict)
    assert isinstance(second_structure, dict)
    assert tuple(first_structure) == ("scope", "query", "result")
    assert first_structure == EXPECTED_QUERY_PLAN_STRUCTURE
    assert first_structure == second_structure
    assert first_structure is not second_structure

    scope = first_structure["scope"]
    query = first_structure["query"]
    result = first_structure["result"]
    assert isinstance(scope, dict)
    assert isinstance(query, dict)
    assert isinstance(result, dict)
    scope["collection"] = "mutated"
    query["column_mode"] = "selected"
    query["return_columns"].append("mutated")  # type: ignore[union-attr]
    result["order_by"].append(  # type: ignore[union-attr]
        {"header": "mutated", "direction": "asc"}
    )
    assert second_structure == EXPECTED_QUERY_PLAN_STRUCTURE
    assert cached_scope["collection"] == "<collection>"
    assert cached_query["column_mode"] == "default"
    assert cached_query["return_columns"] == ()
    assert cached_result["order_by"] == ()


def test_statistics_plan_structure_is_detached_from_cached_schema(
    schema: Mapping[str, object],
) -> None:
    first = retrieve_schema(schema, datasets=("birds",))
    second = retrieve_schema(schema, collections=("Herbarium",))
    first["statistics plan structure"]["query"]["metrics"][0]["operation"] = "mutated"
    assert second["statistics plan structure"]["query"]["metrics"][0]["operation"] == "count"
    assert schema["statistics plan structure"]["query"]["metrics"][0]["operation"] == "count"


def test_retrieves_multiple_datasets_with_canonical_keys(
    schema: Mapping[str, object],
) -> None:
    result = retrieve_schema(
        schema,
        datasets=("Vascular", "birds", "vascular_plants"),
    )

    assert tuple(result) == (
        "query plan structure", "statistics plan structure", "datasets",
    )
    selected = result["datasets"]
    assert isinstance(selected, list)
    assert [item["key"] for item in selected] == ["vascular_plants", "birds"]
    assert selected[0] == {
        "key": "vascular_plants",
        "collection": "Herbarium",
        "dataset": "Vascular",
        "headers": schema["Herbarium"]["Vascular"]["headers"],  # type: ignore[index]
    }
    cowan = schema["Cowan Tetrapod Collection"]
    assert selected[1]["headers"] == cowan["Birds"]["headers"]  # type: ignore[index]


def test_retrieves_multiple_collections_and_all_member_datasets(
    schema: Mapping[str, object],
) -> None:
    result = retrieve_schema(
        schema,
        collections=("Cowan Tetrapod Collection", "Herbarium"),
    )

    assert tuple(result) == (
        "query plan structure", "statistics plan structure", "collections",
    )
    selected = result["collections"]
    assert isinstance(selected, list)
    assert [item["collection"] for item in selected] == [
        "Cowan Tetrapod Collection",
        "Herbarium",
    ]
    assert [dataset["key"] for dataset in selected[0]["datasets"]] == [
        "birds",
        "herpetology",
        "mammals",
    ]
    assert [dataset["key"] for dataset in selected[1]["datasets"]] == [
        "algae",
        "bryophytes",
        "fungi",
        "lichen",
        "vascular_plants",
    ]


def test_combines_dataset_and_collection_request_categories(
    schema: Mapping[str, object],
) -> None:
    result = retrieve_schema(
        schema,
        datasets=("marine_invertebrates",),
        collections=("Paleontology",),
    )

    assert tuple(result) == (
        "query plan structure",
        "statistics plan structure",
        "datasets",
        "collections",
    )
    assert result["datasets"][0]["key"] == "marine_invertebrates"  # type: ignore[index]
    selected_collection = result["collections"][0]  # type: ignore[index]
    assert selected_collection["datasets"][0]["key"] == "fossils"


def test_retrieval_returns_detached_header_mappings(
    schema: Mapping[str, object],
) -> None:
    result = retrieve_schema(schema, datasets=("birds",))
    headers = result["datasets"][0]["headers"]  # type: ignore[index]
    headers["changed"] = 1

    cowan = schema["Cowan Tetrapod Collection"]
    source_headers = cowan["Birds"]["headers"]  # type: ignore[index]
    assert "changed" not in source_headers


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        ({}, "at least one of datasets or collections"),
        ({"datasets": ("unknown",)}, "unknown datasets: unknown"),
        ({"collections": ("unknown",)}, "unknown collections: unknown"),
    ),
)
def test_retrieval_rejects_invalid_requests(
    schema: Mapping[str, object],
    arguments: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        retrieve_schema(schema, **arguments)  # type: ignore[arg-type]


def test_public_selector_api_has_only_the_supported_arguments() -> None:
    assert tuple(inspect.signature(retrieve_schema_for_registry).parameters) == (
        "schema",
        "datasets",
        "collections",
        "dataset_registry",
    )


def test_source_header_reader_accepts_utf8_bom(tmp_path: Path) -> None:
    source = tmp_path / "occurrence.txt"
    source.write_bytes("\ufeffoccurrenceID\tcatalogNumber\n".encode("utf-8"))

    assert read_tsv_header(source) == ("occurrenceID", "catalogNumber")
