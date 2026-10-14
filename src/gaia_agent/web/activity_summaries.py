"""Format Gaia tool results as bounded public activity messages."""

from __future__ import annotations

from typing import cast

from src.gaia_agent.runtime.contracts import ToolResult


_MAX_ACTIVITY_LIST_ITEMS = 3
_MAX_ACTIVITY_TEXT_CHARACTERS = 240


def _bounded_activity_text(value: str) -> str:
    """Keep one generated public activity string within its fixed limit."""
    if len(value) <= _MAX_ACTIVITY_TEXT_CHARACTERS:
        return value
    return f"{value[: _MAX_ACTIVITY_TEXT_CHARACTERS - 1].rstrip()}…"


def _counted_noun(count: int, singular: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {singular}{suffix}"


def _render_activity_values(value: object) -> str | None:
    """Render a bounded prefix of one trusted selector or header array."""
    if not value:
        return None
    values = cast(list[str], value)
    shown = values[:_MAX_ACTIVITY_LIST_ITEMS]
    omitted = len(values) - len(shown)
    if omitted:
        rendered = f"{', '.join(shown)}, and {omitted} more"
    elif len(shown) == 1:
        rendered = shown[0]
    elif len(shown) == 2:
        rendered = f"{shown[0]} and {shown[1]}"
    else:
        rendered = f"{', '.join(shown[:-1])}, and {shown[-1]}"
    return rendered


def _completed_query_activity(result: ToolResult) -> dict[str, str]:
    arguments = result.call.arguments
    scope = cast(dict[str, object], arguments["scope"])
    scope_label = cast(str, scope.get("dataset") or scope["collection"])
    query = cast(dict[str, object], arguments["query"])
    return_columns = _render_activity_values(query["return_columns"])
    filters = cast(dict[str, object], query["filters"])
    filter_headers = _render_activity_values(list(filters))
    requested_result = cast(dict[str, object], arguments["result"])
    requested_limit = cast(int, requested_result["limit"])

    observation = cast(dict[str, object], result.result)
    metadata = cast(dict[str, object], observation["metadata"])
    returned = cast(int, metadata["database_records_returned"])
    queried_count = len(cast(list[str], metadata["queried_dataset_keys"]))
    skipped_count = len(
        cast(list[dict[str, object]], metadata["skipped_datasets"])
    )

    label = f"Searched {scope_label} occurrence records"
    detail = (
        f"Requested up to {requested_limit}; returned {returned} from "
        f"{_counted_noun(queried_count, 'dataset')}."
    )
    if return_columns:
        detail += f" Fields: {return_columns}."
    if filter_headers:
        detail += f" Filters: {filter_headers}."
    if skipped_count:
        detail += f" Skipped {_counted_noun(skipped_count, 'incompatible dataset')}."
    if metadata.get("database_truncated") is True:
        detail += " Additional matching records exist."

    return {
        "kind": "query",
        "status": "completed",
        "label": _bounded_activity_text(label),
        "detail": _bounded_activity_text(detail),
    }


def _completed_statistics_activity(result: ToolResult) -> dict[str, str]:
    scope = cast(dict[str, object], result.call.arguments["scope"])
    scope_label = cast(str, scope.get("dataset") or scope["collection"])
    observation = cast(dict[str, object], result.result)
    metadata = cast(dict[str, object], observation["metadata"])
    returned = cast(int, metadata["groups_returned"])
    queried_count = len(cast(list[str], metadata["queried_dataset_keys"]))
    skipped_count = len(
        cast(list[dict[str, object]], metadata["skipped_datasets"])
    )
    group_by = _render_activity_values(metadata["group_by"])
    metric_count = len(cast(list[dict[str, object]], metadata["metrics"]))

    detail = (
        f"Calculated {_counted_noun(metric_count, 'metric')}; returned "
        f"{_counted_noun(returned, 'summary row')} from "
        f"{_counted_noun(queried_count, 'dataset')}."
    )
    if group_by:
        detail += f" Grouped by: {group_by}."
    if skipped_count:
        detail += f" Skipped {_counted_noun(skipped_count, 'incompatible dataset')}."
    if metadata.get("database_truncated") is True:
        detail += " Additional summary groups exist."
    return {
        "kind": "query",
        "status": "completed",
        "label": _bounded_activity_text(f"Summarized {scope_label} occurrence records"),
        "detail": _bounded_activity_text(detail),
    }


def _completed_darwin_core_search_activity(result: ToolResult) -> dict[str, str]:
    arguments = result.call.arguments
    observation = cast(dict[str, object], result.result)
    match_count = cast(int, observation["match_count"])
    returned_count = cast(int, observation["returned_count"])
    query = cast(str, arguments.get("query", "")).strip()

    detail = (
        f"Matched {_counted_noun(match_count, 'term')}; returned "
        f"{returned_count}."
    )
    if query:
        detail += f" Search text: {query}."
    filters = [
        f"{name}: {arguments[name]}"
        for name in ("term_kind", "organized_in", "status")
        if arguments.get(name)
    ]
    if filters:
        detail += f" Filters: {', '.join(filters)}."
    if observation.get("truncated") is True:
        detail += " Additional matching terms exist."
    return {
        "kind": "catalogue",
        "status": "completed",
        "label": "Searched the Darwin Core standard",
        "detail": _bounded_activity_text(detail),
    }


def _completed_darwin_core_get_activity(result: ToolResult) -> dict[str, str]:
    identifiers = cast(list[str], result.call.arguments["identifiers"])
    observation = cast(dict[str, object], result.result)
    found = cast(list[object], observation["terms"])
    missing = cast(list[str], observation["missing"])
    ambiguous = cast(list[object], observation["ambiguous"])

    detail = (
        f"Requested {_counted_noun(len(identifiers), 'term')}; found {len(found)}."
    )
    requested = _render_activity_values(identifiers)
    if requested:
        detail += f" Terms: {requested}."
    if missing:
        detail += f" Missing: {len(missing)}."
    if ambiguous:
        detail += f" Ambiguous: {len(ambiguous)}."
    return {
        "kind": "catalogue",
        "status": "completed",
        "label": "Reviewed Darwin Core term definitions",
        "detail": _bounded_activity_text(detail),
    }


def _failed_tool_activity(result: ToolResult) -> dict[str, str]:
    if result.call.name == "retrieve_schema":
        kind = "catalogue"
        label = "Could not review catalogue fields"
    elif result.call.name == "query_occurrences":
        kind = "query"
        label = "Could not search occurrence records"
    elif result.call.name == "query_statistics":
        kind = "query"
        label = "Could not summarize occurrence records"
    elif result.call.name == "generate_excel":
        kind = "export"
        label = "Could not prepare the Excel workbook"
    elif result.call.name == "search_darwin_core_terms":
        kind = "catalogue"
        label = "Could not search the Darwin Core standard"
    elif result.call.name == "get_darwin_core_terms":
        kind = "catalogue"
        label = "Could not review Darwin Core term definitions"
    else:
        kind = "catalogue"
        label = "Could not complete a catalogue request"

    details = {
        "invalid_arguments": "The requested catalogue operation could not be completed.",
        "invalid_query_plan": "The requested search could not be completed.",
        "query_unavailable": "The biodiversity catalogue was unavailable.",
    }
    return {
        "kind": kind,
        "status": "failed",
        "label": label,
        "detail": details.get(
            result.error_code,
            "This step could not be completed.",
        ),
    }


def summarize_gaia_tool_result(result: ToolResult) -> dict[str, str]:
    """Return one bounded factual activity without records or private details."""
    if not isinstance(result, ToolResult):
        raise TypeError("Gaia activity summaries require a ToolResult")
    if not result.ok:
        return _failed_tool_activity(result)
    if result.call.name == "retrieve_schema":
        return {
            "kind": "catalogue",
            "status": "completed",
            "label": "Reviewed catalogue fields",
            "detail": "Checked the fields available for the requested records.",
        }
    if result.call.name == "query_occurrences":
        return _completed_query_activity(result)
    if result.call.name == "query_statistics":
        return _completed_statistics_activity(result)
    if result.call.name == "search_darwin_core_terms":
        return _completed_darwin_core_search_activity(result)
    if result.call.name == "get_darwin_core_terms":
        return _completed_darwin_core_get_activity(result)
    if result.call.name == "generate_excel":
        return {
            "kind": "export",
            "status": "completed",
            "label": "Prepared an Excel workbook",
            "detail": "The workbook is ready to download.",
        }
    return {
        "kind": "catalogue",
        "status": "completed",
        "label": "Completed a catalogue request",
        "detail": "The requested catalogue operation completed successfully.",
    }
