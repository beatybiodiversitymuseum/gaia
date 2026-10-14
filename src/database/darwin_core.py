"""Database-owned access to the pinned Darwin Core term snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit


DARWIN_CORE_TERMS_PATH = Path(__file__).with_name("data") / "darwin_core_terms.json"
OCCURRENCE_ROW_TYPE: Final = "http://rs.tdwg.org/dwc/terms/Occurrence"


def term_local_name(term_uri: str) -> str:
    """Return the local name from an absolute standard term IRI."""
    parsed = urlsplit(term_uri)
    local_name = parsed.fragment or parsed.path.rstrip("/").rsplit("/", 1)[-1]
    if not parsed.scheme or not local_name:
        raise ValueError(f"invalid Darwin Core term IRI: {term_uri!r}")
    return local_name


@dataclass(frozen=True)
class DarwinCoreTerm:
    """One normalized current Darwin Core term."""

    iri: str
    local_name: str
    label: str
    definition: str
    comments: str
    examples: str
    organized_in: tuple[str, ...]
    issued: str
    status: str
    replaces: tuple[str, ...]
    rdf_type: str
    version_iri: str


@dataclass(frozen=True)
class SimpleDarwinCoreProfile:
    """Ordered standard properties used by Simple Darwin Core records."""

    term_uris: tuple[str, ...]
    headers: tuple[str, ...]


@lru_cache(maxsize=1)
def load_simple_darwin_core_profile() -> SimpleDarwinCoreProfile:
    """Load and validate the installed Simple Darwin Core profile."""
    document = json.loads(DARWIN_CORE_TERMS_PATH.read_text(encoding="utf-8"))
    raw_profile = document.get("simple_darwin_core")
    if not isinstance(raw_profile, list) or not raw_profile:
        raise ValueError("Darwin Core snapshot has no Simple Darwin Core profile")
    try:
        term_uris = tuple(item["iri"] for item in raw_profile)
        headers = tuple(item["local_name"] for item in raw_profile)
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Darwin Core snapshot has an invalid Simple Darwin Core profile"
        ) from error
    if (
        not all(isinstance(value, str) and value for value in (*term_uris, *headers))
        or len(set(term_uris)) != len(term_uris)
        or len(set(headers)) != len(headers)
        or len(term_uris) != len(headers)
    ):
        raise ValueError(
            "Darwin Core snapshot has an invalid Simple Darwin Core profile"
        )
    known_iris = {term.iri for term in load_darwin_core_terms()}
    if not set(term_uris) <= known_iris:
        raise ValueError("Simple Darwin Core profile contains an unknown term")
    return SimpleDarwinCoreProfile(term_uris, headers)


@lru_cache(maxsize=1)
def load_darwin_core_terms() -> tuple[DarwinCoreTerm, ...]:
    """Load all normalized terms without any network dependency."""
    document = json.loads(DARWIN_CORE_TERMS_PATH.read_text(encoding="utf-8"))
    return tuple(
        DarwinCoreTerm(
            iri=term["iri"],
            local_name=term["local_name"],
            label=term["label"],
            definition=term["definition"],
            comments=term["comments"],
            examples=term["examples"],
            organized_in=tuple(term["organized_in"]),
            issued=term["issued"],
            status=term["status"],
            replaces=tuple(term["replaces"]),
            rdf_type=term["rdf_type"],
            version_iri=term["version_iri"],
        )
        for term in document["terms"]
    )


def get_darwin_core_term(identifier: str) -> DarwinCoreTerm | None:
    """Find a term by exact IRI or case-insensitive local name."""
    folded_identifier = identifier.casefold()
    for term in load_darwin_core_terms():
        if term.iri == identifier or term.local_name.casefold() == folded_identifier:
            return term
    return None


def darwin_core_classes() -> tuple[DarwinCoreTerm, ...]:
    """Return current Darwin Core class terms in snapshot order."""
    return tuple(
        term
        for term in load_darwin_core_terms()
        if term.rdf_type == "http://www.w3.org/2000/01/rdf-schema#Class"
    )
