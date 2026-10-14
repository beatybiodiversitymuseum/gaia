"""Tests for the installed Darwin Core semantic snapshot."""

from __future__ import annotations

from src.database.darwin_core import (
    darwin_core_classes,
    get_darwin_core_term,
    load_darwin_core_terms,
    load_simple_darwin_core_profile,
)
from src.database.gbif_registry import OCCURRENCE_TERM_URIS


def test_snapshot_exposes_current_term_semantics() -> None:
    occurrence_id = get_darwin_core_term("occurrenceID")

    assert occurrence_id is not None
    assert occurrence_id.iri == "http://rs.tdwg.org/dwc/terms/occurrenceID"
    assert occurrence_id.status == "recommended"
    assert occurrence_id.definition
    assert occurrence_id.organized_in == (
        "http://rs.tdwg.org/dwc/terms/Occurrence",
    )


def test_occurrence_definition_terms_resolve_in_standard_snapshot() -> None:
    standard_iris = {term.iri for term in load_darwin_core_terms()}

    assert set(OCCURRENCE_TERM_URIS) <= standard_iris


def test_class_listing_contains_occurrence() -> None:
    class_iris = {term.iri for term in darwin_core_classes()}

    assert "http://rs.tdwg.org/dwc/terms/Occurrence" in class_iris


def test_simple_darwin_core_profile_is_complete_and_ordered() -> None:
    profile = load_simple_darwin_core_profile()

    assert len(profile.term_uris) == len(profile.headers) == 206
    assert profile.headers[:3] == ("type", "modified", "language")
    assert "eventCategory" in profile.headers
    assert "fundingAttribution" not in profile.headers
    assert profile.term_uris[profile.headers.index("language")] == (
        "http://purl.org/dc/elements/1.1/language"
    )
