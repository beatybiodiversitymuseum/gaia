"""Tests for model-visible Darwin Core term lookup."""

from __future__ import annotations

import asyncio

import pytest

from src.database.darwin_core import DarwinCoreTerm
from src.gaia_agent.runtime.contracts import ToolExecutionError
from src.gaia_agent.tools.darwin_core import (
    get_darwin_core_terms,
    search_darwin_core_terms,
)


TERMS = (
    DarwinCoreTerm(
        iri="http://rs.tdwg.org/dwc/terms/Occurrence",
        local_name="Occurrence",
        label="Occurrence",
        definition="An existence of an organism at a place and time.",
        comments="The class of occurrence records.",
        examples="A collected specimen",
        organized_in=(),
        issued="2009-01-01",
        status="recommended",
        replaces=(),
        rdf_type="http://www.w3.org/2000/01/rdf-schema#Class",
        version_iri="http://rs.tdwg.org/dwc/terms/version/Occurrence-2009-01-01",
    ),
    DarwinCoreTerm(
        iri="http://rs.tdwg.org/dwc/terms/occurrenceID",
        local_name="occurrenceID",
        label="Occurrence ID",
        definition="An identifier for the Occurrence.",
        comments="Use a persistent identifier.",
        examples="urn:catalog:UBC:1",
        organized_in=("http://rs.tdwg.org/dwc/terms/Occurrence",),
        issued="2009-01-01",
        status="recommended",
        replaces=("http://rs.tdwg.org/dwc/terms/oldOccurrenceID",),
        rdf_type="http://www.w3.org/1999/02/22-rdf-syntax-ns#Property",
        version_iri="http://rs.tdwg.org/dwc/terms/version/occurrenceID-2009-01-01",
    ),
)


def test_searches_semantics_and_filters_by_class() -> None:
    result = asyncio.run(
        search_darwin_core_terms(
            {"query": "persistent", "organized_in": "Occurrence"},
            terms=TERMS,
        )
    )

    assert result["match_count"] == 1
    assert result["truncated"] is False
    assert result["terms"][0]["local_name"] == "occurrenceID"
    assert result["terms"][0]["examples"] == "urn:catalog:UBC:1"
    assert result["terms"][0]["version_iri"].endswith("occurrenceID-2009-01-01")


def test_empty_search_lists_filtered_terms_with_bounded_results() -> None:
    result = asyncio.run(
        search_darwin_core_terms(
            {"query": "", "term_kind": "property", "limit": 1},
            terms=TERMS,
        )
    )

    assert result["returned_count"] == 1
    assert result["match_count"] == 1
    assert result["terms"][0]["local_name"] == "occurrenceID"


def test_get_preserves_request_order_and_reports_unknowns() -> None:
    result = asyncio.run(
        get_darwin_core_terms(
            {
                "identifiers": [
                    "occurrenceID",
                    "http://rs.tdwg.org/dwc/terms/Occurrence",
                    "notATerm",
                ],
            },
            terms=TERMS,
        )
    )

    assert [term["local_name"] for term in result["terms"]] == [
        "occurrenceID",
        "Occurrence",
    ]
    assert result["missing"] == ["notATerm"]
    assert result["ambiguous"] == []


def test_get_rejects_duplicate_identifiers() -> None:
    with pytest.raises(ToolExecutionError) as captured:
        asyncio.run(
            get_darwin_core_terms(
                {"identifiers": ["occurrenceID", "occurrenceID"]},
                terms=TERMS,
            )
        )

    assert captured.value.code == "invalid_arguments"
