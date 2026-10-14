"""Offline access to Gaia's pinned GBIF schema definitions."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping
from urllib.parse import urlsplit


OCCURRENCE_ROW_TYPE: Final = "http://rs.tdwg.org/dwc/terms/Occurrence"
REGISTRY_PATH: Final = Path(__file__).parent / "data" / "gbif_extension_registry.json"


def load_registered_definitions(path: Path = REGISTRY_PATH) -> tuple[
    Mapping[str, tuple[str, ...]], Mapping[str, tuple[str, ...]]
]:
    """Load registered cores and extensions from the pinned snapshot."""
    document = json.loads(path.read_text(encoding="utf-8"))
    definitions = document.get("definitions")
    if not isinstance(definitions, list):
        raise ValueError("GBIF extension registry snapshot has no definitions list")

    cores: dict[str, tuple[str, ...]] = {}
    extensions: dict[str, tuple[str, ...]] = {}
    for definition in definitions:
        row_type = definition.get("row_type")
        terms = definition.get("terms")
        kind = definition.get("kind")
        if (
            not isinstance(row_type, str)
            or not isinstance(terms, list)
            or kind not in {"core", "extension"}
        ):
            raise ValueError(
                "GBIF extension registry snapshot contains an invalid definition"
            )
        if row_type in cores or row_type in extensions:
            raise ValueError(f"duplicate GBIF extension rowType: {row_type}")
        if not terms or not all(isinstance(term, str) for term in terms):
            raise ValueError(f"GBIF extension {row_type!r} has invalid terms")
        if len(set(terms)) != len(terms):
            raise ValueError(f"GBIF extension {row_type!r} has duplicate terms")
        target = cores if kind == "core" else extensions
        target[row_type] = tuple(terms)

    if OCCURRENCE_ROW_TYPE not in cores:
        raise ValueError("GBIF extension registry snapshot has no Occurrence core")
    return MappingProxyType(cores), MappingProxyType(extensions)


def term_local_name(term_uri: str) -> str:
    """Return the canonical local name from an absolute term URI."""
    parsed = urlsplit(term_uri)
    local_name = parsed.fragment or parsed.path.rstrip("/").rsplit("/", 1)[-1]
    if not parsed.scheme or not local_name:
        raise ValueError(f"invalid registered term URI: {term_uri!r}")
    return local_name


def local_names(term_uris: tuple[str, ...], *, row_type: str) -> tuple[str, ...]:
    """Map registered term URIs to unique canonical local names."""
    names = tuple(term_local_name(term_uri) for term_uri in term_uris)
    if len(set(names)) != len(names):
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        raise ValueError(
            f"GBIF extension {row_type!r} has colliding local names: "
            f"{', '.join(duplicates)}"
        )
    return names


REGISTERED_CORES, REGISTERED_EXTENSIONS = load_registered_definitions()
OCCURRENCE_TERM_URIS: Final = REGISTERED_CORES[OCCURRENCE_ROW_TYPE]
OCCURRENCE_TERMS: Final = frozenset(OCCURRENCE_TERM_URIS)
OCCURRENCE_HEADERS: Final = local_names(
    OCCURRENCE_TERM_URIS,
    row_type=OCCURRENCE_ROW_TYPE,
)


def allowed_terms(row_type: str) -> frozenset[str] | None:
    """Return allowed terms for a registered extension, or None when unsupported."""
    terms = REGISTERED_EXTENSIONS.get(row_type)
    return frozenset(terms) if terms is not None else None
