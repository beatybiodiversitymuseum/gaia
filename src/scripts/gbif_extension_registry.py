"""Compatibility exports for the database-owned GBIF schema registry."""

from src.database.gbif_registry import (
    OCCURRENCE_HEADERS,
    OCCURRENCE_ROW_TYPE,
    OCCURRENCE_TERMS,
    OCCURRENCE_TERM_URIS,
    REGISTERED_CORES,
    REGISTERED_EXTENSIONS,
    REGISTRY_PATH,
    allowed_terms,
    load_registered_definitions,
    local_names,
    term_local_name,
)


__all__ = [
    "OCCURRENCE_HEADERS",
    "OCCURRENCE_ROW_TYPE",
    "OCCURRENCE_TERMS",
    "OCCURRENCE_TERM_URIS",
    "REGISTERED_CORES",
    "REGISTERED_EXTENSIONS",
    "REGISTRY_PATH",
    "allowed_terms",
    "load_registered_definitions",
    "local_names",
    "term_local_name",
]
