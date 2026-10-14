"""Search the installed Darwin Core term snapshot."""

from __future__ import annotations

from collections.abc import Sequence

from src.database.darwin_core import DarwinCoreTerm
from src.gaia_agent.runtime.contracts import ToolExecutionError


DEFAULT_SEARCH_LIMIT = 20
MAX_SEARCH_LIMIT = 50
MAX_EXACT_TERMS = 50


def _term_result(term: DarwinCoreTerm) -> dict[str, object]:
    """Render the complete normalized metadata for one Darwin Core term."""
    return {
        "iri": term.iri,
        "local_name": term.local_name,
        "label": term.label,
        "definition": term.definition,
        "comments": term.comments,
        "examples": term.examples,
        "organized_in": list(term.organized_in),
        "issued": term.issued,
        "status": term.status,
        "replaces": list(term.replaces),
        "rdf_type": term.rdf_type,
        "version_iri": term.version_iri,
    }


def _term_kind(term: DarwinCoreTerm) -> str:
    rdf_type = term.rdf_type.rsplit("/", 1)[-1].rsplit("#", 1)[-1].casefold()
    if rdf_type in {"class", "rdfs:class"}:
        return "class"
    if rdf_type in {"property", "rdf:property"}:
        return "property"
    return "other"


def _search_text(term: DarwinCoreTerm) -> str:
    values = (
        term.iri,
        term.local_name,
        term.label,
        term.definition,
        term.comments,
        term.examples,
    )
    return "\n".join(value for value in values if value).casefold()


async def search_darwin_core_terms(
    arguments: dict[str, object],
    *,
    terms: Sequence[DarwinCoreTerm],
) -> dict[str, object]:
    """Search term semantics with optional standard-native filters."""
    allowed = frozenset({"query", "term_kind", "organized_in", "status", "limit"})
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        raise ToolExecutionError(
            "invalid_arguments", f"Unknown argument(s): {', '.join(unknown)}.",
        )

    query = arguments.get("query", "")
    if not isinstance(query, str):
        raise ToolExecutionError("invalid_arguments", "query must be a string.")
    term_kind = arguments.get("term_kind")
    if term_kind not in {None, "class", "property", "other"}:
        raise ToolExecutionError(
            "invalid_arguments", "term_kind must be class, property, or other.",
        )
    organized_in = arguments.get("organized_in")
    if organized_in is not None and (
        not isinstance(organized_in, str) or not organized_in.strip()
    ):
        raise ToolExecutionError(
            "invalid_arguments", "organized_in must be a non-empty string.",
        )
    status = arguments.get("status")
    if status is not None and (not isinstance(status, str) or not status.strip()):
        raise ToolExecutionError(
            "invalid_arguments", "status must be a non-empty string.",
        )
    limit = arguments.get("limit", DEFAULT_SEARCH_LIMIT)
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_SEARCH_LIMIT
    ):
        raise ToolExecutionError(
            "invalid_arguments",
            f"limit must be an integer from 1 through {MAX_SEARCH_LIMIT}.",
        )

    needle = query.casefold()
    organized_in_key = (
        organized_in.casefold() if isinstance(organized_in, str) else None
    )
    status_key = status.casefold() if isinstance(status, str) else None
    matches = [
        term
        for term in terms
        if (not needle or needle in _search_text(term))
        and (term_kind is None or _term_kind(term) == term_kind)
        and (
            organized_in_key is None
            or any(
                organized_in_key
                in {value.casefold(), value.rsplit("/", 1)[-1].casefold()}
                for value in term.organized_in
            )
        )
        and (status_key is None or term.status.casefold() == status_key)
    ]
    matches.sort(key=lambda term: (term.local_name.casefold(), term.iri))
    return {
        "terms": [_term_result(term) for term in matches[:limit]],
        "match_count": len(matches),
        "returned_count": min(len(matches), limit),
        "truncated": len(matches) > limit,
    }


async def get_darwin_core_terms(
    arguments: dict[str, object],
    *,
    terms: Sequence[DarwinCoreTerm],
) -> dict[str, object]:
    """Retrieve terms by exact IRI or unambiguous local name."""
    unknown = sorted(set(arguments) - {"identifiers"})
    if unknown:
        raise ToolExecutionError(
            "invalid_arguments", f"Unknown argument(s): {', '.join(unknown)}.",
        )
    identifiers = arguments.get("identifiers")
    if (
        not isinstance(identifiers, list)
        or not identifiers
        or len(identifiers) > MAX_EXACT_TERMS
        or any(
            not isinstance(value, str) or not value.strip()
            for value in identifiers
        )
    ):
        raise ToolExecutionError(
            "invalid_arguments",
            f"identifiers must contain 1 through {MAX_EXACT_TERMS} non-empty strings.",
        )
    if len(set(identifiers)) != len(identifiers):
        raise ToolExecutionError(
            "invalid_arguments", "identifiers must not contain duplicates.",
        )

    by_iri = {term.iri: term for term in terms}
    by_name: dict[str, list[DarwinCoreTerm]] = {}
    for term in terms:
        by_name.setdefault(term.local_name, []).append(term)

    found = []
    missing = []
    ambiguous = []
    for identifier in identifiers:
        term = by_iri.get(identifier)
        if term is not None:
            found.append(_term_result(term))
            continue
        named = by_name.get(identifier, [])
        if len(named) == 1:
            found.append(_term_result(named[0]))
        elif len(named) > 1:
            ambiguous.append({
                "identifier": identifier,
                "matching_iris": [candidate.iri for candidate in named],
            })
        else:
            missing.append(identifier)
    return {"terms": found, "missing": missing, "ambiguous": ambiguous}
