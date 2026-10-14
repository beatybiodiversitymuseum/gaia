"""Tests for the pinned GBIF production extension registry."""

from __future__ import annotations

import json

import pytest

from src.scripts.gbif_extension_registry import OCCURRENCE_ROW_TYPE
from src.scripts.gbif_extension_registry import OCCURRENCE_TERMS
from src.scripts.gbif_extension_registry import REGISTERED_CORES
from src.scripts.gbif_extension_registry import REGISTERED_EXTENSIONS
from src.scripts.gbif_extension_registry import allowed_terms
from src.scripts.gbif_extension_registry import load_registered_definitions


def test_snapshot_separates_occurrence_core_from_extensions() -> None:
    assert OCCURRENCE_ROW_TYPE in REGISTERED_CORES
    assert OCCURRENCE_ROW_TYPE not in REGISTERED_EXTENSIONS
    assert "http://rs.tdwg.org/dwc/terms/occurrenceID" in OCCURRENCE_TERMS
    assert "http://rs.tdwg.org/dwc/terms/catalogNumber" in OCCURRENCE_TERMS


def test_registered_extension_terms_are_available_by_exact_row_type() -> None:
    row_type = "http://rs.tdwg.org/dwc/terms/MeasurementOrFact"
    terms = allowed_terms(row_type)

    assert terms == frozenset(REGISTERED_EXTENSIONS[row_type])
    assert "http://rs.tdwg.org/dwc/terms/measurementType" in terms
    assert allowed_terms("https://example.org/CustomExtension") is None


def test_registry_mappings_are_immutable() -> None:
    with pytest.raises(TypeError):
        REGISTERED_EXTENSIONS["https://example.org/Extension"] = frozenset()  # type: ignore[index]


def test_loader_rejects_duplicate_row_types(tmp_path) -> None:
    definition = {
        "kind": "core",
        "row_type": OCCURRENCE_ROW_TYPE,
        "terms": ["http://rs.tdwg.org/dwc/terms/occurrenceID"],
    }
    snapshot_path = tmp_path / "registry.json"
    snapshot_path.write_text(
        json.dumps({"definitions": [definition, definition]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_registered_definitions(snapshot_path)


def test_snapshot_has_only_nonempty_absolute_term_uris() -> None:
    all_definitions = (*REGISTERED_CORES.values(), *REGISTERED_EXTENSIONS.values())

    assert len(REGISTERED_CORES) == 3
    assert len(REGISTERED_EXTENSIONS) == 35
    assert all(definition for definition in all_definitions)
    assert all(
        term.startswith(("http://", "https://"))
        for definition in all_definitions
        for term in definition
    )
