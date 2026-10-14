from __future__ import annotations

from src.gaia_agent.runtime.contracts import ToolCall, ToolResult
from src.gaia_agent.web.verification import (
    MAX_VERIFICATION_ITEMS,
    project_verification_steps,
)


def test_projects_exact_arguments_and_bounded_record_evidence() -> None:
    arguments = {
        "scope": {"dataset": "birds"},
        "query": {"filters": {}, "return_columns": ["scientificName"]},
    }
    records = [
        {"occurrenceID": f"bird-{index}", "scientificName": "Corvus corax"}
        for index in range(MAX_VERIFICATION_ITEMS + 2)
    ]
    result = ToolResult(
        ToolCall(
            "call-1",
            "query_occurrences",
            arguments,
            purpose="Retrieve records that answer the user's species request.",
        ),
        True,
        {"records": records, "metadata": {"database_records_returned": len(records)}},
    )

    steps = project_verification_steps((result,))

    assert steps[0]["sequence"] == 1
    assert steps[0]["capability"] == "query_occurrences"
    assert steps[0]["purpose"] == (
        "Retrieve records that answer the user's species request."
    )
    assert steps[0]["arguments"] == arguments
    assert steps[0]["status"] == "completed"
    evidence = steps[0]["evidence"]
    assert evidence["metadata"] == {"database_records_returned": len(records)}
    assert evidence["records"] == {
        "items": records[:MAX_VERIFICATION_ITEMS],
        "items_returned": len(records),
        "items_shown": MAX_VERIFICATION_ITEMS,
        "items_omitted": 2,
    }


def test_projects_failed_call_without_private_runtime_state() -> None:
    result = ToolResult(
        ToolCall(
            "call-2",
            "retrieve_schema",
            {"datasets": ["unknown"]},
            purpose="Check whether the requested dataset is available.",
        ),
        False,
        error_code="invalid_arguments",
        error_message="unknown datasets: unknown",
    )

    steps = project_verification_steps((result,))

    assert steps == ({
        "sequence": 1,
        "capability": "retrieve_schema",
        "purpose": "Check whether the requested dataset is available.",
        "arguments": {"datasets": ["unknown"]},
        "status": "failed",
        "evidence": {
            "error": {
                "code": "invalid_arguments",
                "message": "unknown datasets: unknown",
            }
        },
    },)
