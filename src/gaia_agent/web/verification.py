"""Project executed tool calls into bounded public verification steps."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final

from src.gaia_agent.runtime.contracts import ToolResult


MAX_VERIFICATION_ITEMS: Final = 10


def _bounded_list(value: object) -> object:
    """Bound one result list while stating how much evidence was omitted."""
    if not isinstance(value, list) or len(value) <= MAX_VERIFICATION_ITEMS:
        return value
    return {
        "items": value[:MAX_VERIFICATION_ITEMS],
        "items_returned": len(value),
        "items_shown": MAX_VERIFICATION_ITEMS,
        "items_omitted": len(value) - MAX_VERIFICATION_ITEMS,
    }


def _bounded_evidence(result: ToolResult) -> object:
    """Keep result metadata and a bounded sample of potentially large rows."""
    if not result.ok:
        return {
            "error": {
                "code": result.error_code,
                "message": result.error_message,
            }
        }
    if not isinstance(result.result, Mapping):
        return {"value": result.result}

    evidence = dict(result.result)
    for key in ("records", "rows", "terms", "found", "missing", "ambiguous"):
        if key in evidence:
            evidence[key] = _bounded_list(evidence[key])
    return evidence


def project_verification_steps(
    tool_results: Iterable[ToolResult],
) -> tuple[dict[str, object], ...]:
    """Return exact call arguments with bounded, already-projected evidence."""
    return tuple(
        {
            "sequence": sequence,
            "capability": result.call.name,
            "purpose": result.call.purpose,
            "arguments": dict(result.call.arguments),
            "status": "completed" if result.ok else "failed",
            "evidence": _bounded_evidence(result),
        }
        for sequence, result in enumerate(tool_results, start=1)
    )
