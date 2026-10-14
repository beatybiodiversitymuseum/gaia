"""Check model output for direct disclosure of private runtime context.

This is a supplemental egress check, not a prompt-injection classifier. It
cannot recognize arbitrary paraphrases, encodings, or facts inferred by a model.
Sensitive operational data should be kept out of model observations at source.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable, Mapping
from urllib.parse import unquote

from src.gaia_agent.runtime.contracts import ToolDefinition


_PASSAGE_WORDS = 12
_WORDS = re.compile(r"\w+", re.UNICODE)


def _normalized_text(value: str) -> str:
    """Normalize common transport/formatting disguises without changing data."""
    decoded = unicodedata.normalize("NFKC", html.unescape(unquote(value)))
    decoded = re.sub(r"\\([_`*])", r"\1", decoded).replace("`", "").replace("*", "")
    return "".join(
        character for character in decoded if unicodedata.category(character) != "Cf"
    ).casefold()


def _passages(value: str) -> Iterable[tuple[str, ...]]:
    words = _WORDS.findall(value)
    return (
        tuple(words[index : index + _PASSAGE_WORDS])
        for index in range(len(words) - _PASSAGE_WORDS + 1)
    )


def _argument_identifiers(schema: Mapping[str, object]) -> Iterable[str]:
    """Select API argument identifiers, excluding ordinary words and headers."""
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for name in properties:
            if isinstance(name, str) and "_" in name:
                yield name
    for value in schema.values():
        if isinstance(value, Mapping):
            yield from _argument_identifiers(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    yield from _argument_identifiers(item)


class ResponseDisclosureGuard:
    """Hold private identifiers, prompt passages, and configured path roots.

    Values echoed from a user or database receive the same checks; exempting
    those values would let an injection whitelist the very text it requests.
    Biodiversity headers and ordinary source values are not private identifiers.
    """

    def __init__(
        self,
        *,
        system_prompt: str,
        tools: Iterable[ToolDefinition],
        private_locations: Iterable[str] = (),
    ) -> None:
        definitions = tuple(tools)
        identifiers = {
            _normalized_text(name)
            for definition in definitions
            for name in _argument_identifiers(definition.parameters)
        }
        identifiers.update(_normalized_text(definition.name) for definition in definitions)
        patterns = [re.escape(identifier) for identifier in sorted(identifiers)]
        self._identifiers = (
            re.compile(r"(?<!\w)(?:" + "|".join(patterns) + r")(?!\w)")
            if patterns else None
        )
        self._private_passages = frozenset(_passages(_normalized_text(system_prompt)))
        self._locations = tuple(
            normalized.rstrip("/\\")
            for location in private_locations
            if (normalized := _normalized_text(location)).rstrip("/\\")
        )

    def allows(self, content: str) -> bool:
        """Reject direct disclosure before the response becomes user-visible."""
        normalized = _normalized_text(content)
        if self._identifiers is not None and self._identifiers.search(normalized):
            return False
        return self._allows_private_content(normalized)

    def allows_compaction(self, content: str) -> bool:
        """Allow tool history while withholding private prompt passages and paths."""
        return self._allows_private_content(_normalized_text(content))

    def _allows_private_content(self, normalized: str) -> bool:
        if any(passage in self._private_passages for passage in _passages(normalized)):
            return False
        for location in self._locations:
            # Protect paths even inside links, while allowing longer source
            # identifiers that merely share a prefix with the configured root.
            pattern = re.escape(location) + r"(?=$|[/\\\s\"'`<>\])}.,:;!?])"
            if re.search(pattern, normalized):
                return False
        return True
