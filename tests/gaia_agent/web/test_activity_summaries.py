"""Tests for Gaia's public tool activity summaries."""

import pytest

from src.gaia_agent.web.activity_summaries import summarize_gaia_tool_result
from src.gaia_agent.runtime.contracts import ToolCall, ToolResult


def test_schema_activity_does_not_publish_schema_contract_details() -> None:
    summary = summarize_gaia_tool_result(
        ToolResult(
            call=ToolCall(
                "private-schema-call-id",
                "retrieve_schema",
                {"datasets": ["Birds"]},
            ),
            ok=True,
            result={
                "datasets": [
                    {
                        "key": "birds",
                        "dataset": "Birds",
                        "headers": {"catalogNumber": 1, "scientificName": 1},
                    }
                ]
            },
        )
    )

    assert summary == {
        "kind": "catalogue",
        "status": "completed",
        "label": "Reviewed catalogue fields",
        "detail": "Checked the fields available for the requested records.",
    }
    assert "private-schema-call-id" not in repr(summary)


def test_query_activity_uses_safe_arguments_and_observation_metadata_only() -> None:
    private_filter_value = "PRIVATE-FILTER-VALUE"
    private_record_value = "PRIVATE-RECORD-VALUE"
    summary = summarize_gaia_tool_result(
        ToolResult(
            call=ToolCall(
                "private-query-call-id",
                "query_occurrences",
                {
                    "scope": {
                        "collection": "Cowan Tetrapod Collection",
                        "dataset": "Birds",
                    },
                    "query": {
                        "filters": {
                            "country": {"exact_any": [private_filter_value]},
                        },
                        "return_columns": [
                            "catalogNumber",
                            "genus",
                            "specificEpithet",
                            "eventDate",
                            "locality",
                        ],
                    },
                    "result": {
                        "distinct_by": [],
                        "order_by": [],
                        "limit": 10,
                    },
                },
            ),
            ok=True,
            result={
                "records": [{"scientificName": private_record_value}],
                "metadata": {
                    "queried_dataset_keys": ["birds"],
                    "skipped_datasets": [{"dataset_key": "mammals"}],
                    "database_records_returned": 10,
                    "database_truncated": True,
                },
            },
        )
    )

    assert summary == {
        "kind": "query",
        "status": "completed",
        "label": "Searched Birds occurrence records",
        "detail": (
            "Requested up to 10; returned 10 from 1 dataset. Fields: "
            "catalogNumber, genus, specificEpithet, and 2 more. Filters: country. "
            "Skipped 1 incompatible dataset. Additional matching records exist."
        ),
    }
    rendered = repr(summary)
    assert private_filter_value not in rendered
    assert private_record_value not in rendered
    assert "private-query-call-id" not in rendered


def test_failed_activity_uses_a_fixed_public_error_description() -> None:
    private_error = "PRIVATE-DATABASE-AND-TRACE-DETAIL"
    summary = summarize_gaia_tool_result(
        ToolResult(
            call=ToolCall("private-call-id", "query_occurrences", {}),
            ok=False,
            error_code="tool_execution_error",
            error_message=private_error,
        )
    )

    assert summary == {
        "kind": "query",
        "status": "failed",
        "label": "Could not search occurrence records",
        "detail": "This step could not be completed.",
    }
    assert private_error not in repr(summary)
    assert "private-call-id" not in repr(summary)


def test_statistics_activity_reports_structure_without_group_values_or_metric_aliases() -> None:
    summary = summarize_gaia_tool_result(
        ToolResult(
            call=ToolCall(
                "private-statistics-call-id",
                "query_statistics",
                {
                    "scope": {"dataset": "Birds"},
                    "query": {
                        "filters": {"country": {"exact_any": ["PRIVATE-FILTER"]}},
                        "group_by": ["scientificName"],
                        "metrics": [{"operation": "count", "alias": "PRIVATE_ALIAS"}],
                    },
                    "result": {"order_by": [], "limit": 4},
                },
            ),
            ok=True,
            result={
                "rows": [{"scientificName": "PRIVATE-SPECIES", "PRIVATE_ALIAS": 999}],
                "metadata": {
                    "groups_returned": 4,
                    "queried_dataset_keys": ["birds"],
                    "skipped_datasets": [{"dataset_key": "mammals", "reason": "PRIVATE-REASON"}],
                    "group_by": ["scientificName"],
                    "metrics": [{"operation": "count", "alias": "PRIVATE_ALIAS"}],
                    "database_truncated": True,
                },
            },
        )
    )

    assert summary == {
        "kind": "query",
        "status": "completed",
        "label": "Summarized Birds occurrence records",
        "detail": (
            "Calculated 1 metric; returned 4 summary rows from 1 dataset. "
            "Grouped by: scientificName. Skipped 1 incompatible dataset. "
            "Additional summary groups exist."
        ),
    }
    assert "PRIVATE" not in repr(summary)
    assert "private-statistics-call-id" not in repr(summary)
    assert all(len(value) <= 240 for value in summary.values())


def test_export_activity_excludes_workbook_metadata() -> None:
    summary = summarize_gaia_tool_result(ToolResult(
        call=ToolCall("private-export-call-id", "generate_excel", {}),
        ok=True,
        result={"files": [{"filename": "PRIVATE.xlsx", "path": "/PRIVATE/export.xlsx"}]},
    ))

    assert summary == {
        "kind": "export",
        "status": "completed",
        "label": "Prepared an Excel workbook",
        "detail": "The workbook is ready to download.",
    }


def test_darwin_core_search_activity_reports_search_and_coverage() -> None:
    summary = summarize_gaia_tool_result(ToolResult(
        call=ToolCall(
            "private-dwc-search-id",
            "search_darwin_core_terms",
            {
                "query": "geological age",
                "term_kind": "property",
                "organized_in": "GeologicalContext",
                "limit": 5,
            },
        ),
        ok=True,
        result={
            "terms": [{"local_name": "earliestAgeOrLowestStage"}],
            "match_count": 8,
            "returned_count": 5,
            "truncated": True,
        },
    ))

    assert summary == {
        "kind": "catalogue",
        "status": "completed",
        "label": "Searched the Darwin Core standard",
        "detail": (
            "Matched 8 terms; returned 5. Search text: geological age. Filters: "
            "term_kind: property, organized_in: GeologicalContext. Additional "
            "matching terms exist."
        ),
    }
    assert "private-dwc-search-id" not in repr(summary)


def test_darwin_core_get_activity_reports_resolution_outcomes() -> None:
    summary = summarize_gaia_tool_result(ToolResult(
        call=ToolCall(
            "private-dwc-get-id",
            "get_darwin_core_terms",
            {
                "identifiers": [
                    "occurrenceID",
                    "scientificName",
                    "missingTerm",
                    "ambiguousTerm",
                ]
            },
        ),
        ok=True,
        result={
            "terms": [{"local_name": "occurrenceID"}, {"local_name": "scientificName"}],
            "missing": ["missingTerm"],
            "ambiguous": [{"identifier": "ambiguousTerm"}],
        },
    ))

    assert summary == {
        "kind": "catalogue",
        "status": "completed",
        "label": "Reviewed Darwin Core term definitions",
        "detail": (
            "Requested 4 terms; found 2. Terms: occurrenceID, scientificName, "
            "missingTerm, and 1 more. Missing: 1. Ambiguous: 1."
        ),
    }
    assert "private-dwc-get-id" not in repr(summary)


@pytest.mark.parametrize(
    ("tool_name", "kind", "label", "error_code", "detail"),
    [
        (
            "query_statistics", "query", "Could not summarize occurrence records",
            "query_unavailable", "The biodiversity catalogue was unavailable.",
        ),
        (
            "generate_excel", "export", "Could not prepare the Excel workbook",
            "tool_execution_error", "This step could not be completed.",
        ),
        (
            "search_darwin_core_terms", "catalogue",
            "Could not search the Darwin Core standard",
            "invalid_arguments",
            "The requested catalogue operation could not be completed.",
        ),
        (
            "get_darwin_core_terms", "catalogue",
            "Could not review Darwin Core term definitions",
            "tool_execution_error", "This step could not be completed.",
        ),
    ],
)
def test_tool_failures_use_fixed_public_descriptions(
    tool_name: str,
    kind: str,
    label: str,
    error_code: str,
    detail: str,
) -> None:
    summary = summarize_gaia_tool_result(
        ToolResult(
            call=ToolCall("private-call-id", tool_name, {}),
            ok=False,
            error_code=error_code,
            error_message="PRIVATE-WORKBOOK-PATH-OR-DATABASE-DETAIL",
        )
    )

    assert summary == {"kind": kind, "status": "failed", "label": label, "detail": detail}
    assert "PRIVATE" not in repr(summary)
