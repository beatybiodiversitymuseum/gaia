"""Tests for deterministic GBIF registry snapshot generation."""

from __future__ import annotations

import json

from src.scripts import refresh_gbif_extension_registry


DEFINITION_XML = b"""\
<extension xmlns="http://rs.gbif.org/extension/"
  rowType="https://example.org/Example">
  <property qualName="https://example.org/term/z"/>
  <property qualName="https://example.org/term/a"/>
</extension>
"""


def test_parse_definition_sorts_and_deduplicates_terms() -> None:
    xml = DEFINITION_XML.replace(
        b"</extension>",
        b'<property qualName="https://example.org/term/a"/></extension>',
    )

    terms = refresh_gbif_extension_registry.parse_definition(
        xml, "https://example.org/Example"
    )

    assert terms == (
        "https://example.org/term/a",
        "https://example.org/term/z",
    )


def test_build_snapshot_selects_only_latest_production_definitions(monkeypatch) -> None:
    registry = {
        "extensions": [
            {
                "identifier": "https://example.org/Old",
                "url": "http://rs.gbif.org/extension/old.xml",
                "title": "Old",
                "issued": "2020-01-01",
                "isLatest": False,
            },
            {
                "identifier": "https://example.org/Example",
                "url": "http://rs.gbif.org/extension/example.xml",
                "title": "Example",
                "issued": "2026-01-01",
                "isLatest": True,
            },
        ]
    }
    monkeypatch.setattr(
        refresh_gbif_extension_registry,
        "fetch_bytes",
        lambda _url: DEFINITION_XML,
    )

    snapshot = refresh_gbif_extension_registry.build_snapshot(
        json.dumps(registry).encode(), "2026-10-07"
    )

    assert snapshot["provenance"]["snapshot_date"] == "2026-10-07"
    assert [item["row_type"] for item in snapshot["definitions"]] == [
        "https://example.org/Example"
    ]
    assert snapshot["definitions"][0]["kind"] == "extension"
