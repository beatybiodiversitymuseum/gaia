"""Shared Darwin Core defaults and deterministic physical-column selection."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Literal


ColumnMode = Literal["default", "all", "selected"]

DEFAULT_RETURN_COLUMNS = (
    "scientificName",
    "vernacularName",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
)


def preferred_identifier(headers: Collection[str]) -> str:
    """Choose the physical identity column used in chat and file output."""
    if "occurrenceID" in headers:
        return "occurrenceID"
    if "id" in headers:
        return "id"
    raise ValueError("A dataset must contain an 'occurrenceID' or 'id' column")


def resolve_return_columns(
    *,
    physical_headers: Sequence[str],
    column_mode: ColumnMode,
    requested_columns: Sequence[str],
    filter_headers: Sequence[str],
    distinct_headers: Sequence[str],
    order_headers: Sequence[str],
) -> tuple[str, ...]:
    """Resolve a validated selection against one dataset's physical headers."""
    if column_mode == "all":
        candidates = physical_headers
    elif column_mode == "selected":
        candidates = requested_columns
    else:
        candidates = (
            *DEFAULT_RETURN_COLUMNS,
            *requested_columns,
            *filter_headers,
            *distinct_headers,
            *order_headers,
        )
    available = frozenset(physical_headers)
    identifier = preferred_identifier(available)
    resolved = (
        identifier if header in ("occurrenceID", "id") else header
        for header in candidates
    )
    return tuple(dict.fromkeys(header for header in resolved if header in available))
