"""Refresh the pinned GBIF production extension registry snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from src.database.gbif_registry import REGISTRY_PATH


REGISTRY_URL = "https://rs.gbif.org/extensions.json"
DEFAULT_OUTPUT_PATH = REGISTRY_PATH
USER_AGENT = "Gaia-GBIF-registry-refresh/1.0"


def fetch_bytes(url: str) -> bytes:
    """Fetch one registry resource using HTTPS."""
    secure_url = url.replace("http://rs.gbif.org/", "https://rs.gbif.org/", 1)
    request = Request(secure_url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read()


def local_name(tag: str) -> str:
    """Return an XML tag without its namespace."""
    return tag.rsplit("}", 1)[-1]


def parse_definition(xml_bytes: bytes, expected_row_type: str) -> tuple[str, ...]:
    """Extract the allowed property term URIs from a GBIF definition."""
    root = ElementTree.fromstring(xml_bytes)
    row_type = root.attrib.get("rowType")
    if row_type != expected_row_type:
        raise ValueError(
            f"definition rowType {row_type!r} does not match {expected_row_type!r}"
        )

    terms: set[str] = set()
    for element in root.iter():
        if local_name(element.tag) != "property":
            continue
        term = element.attrib.get("qualName")
        if not term:
            raise ValueError(f"property in {expected_row_type!r} has no qualName")
        terms.add(term)
    if not terms:
        raise ValueError(f"definition for {expected_row_type!r} has no properties")
    return tuple(sorted(terms))


def build_snapshot(registry_bytes: bytes, snapshot_date: str) -> dict[str, object]:
    """Build a deterministic snapshot from GBIF's production registry."""
    registry = json.loads(registry_bytes)
    latest = [
        entry for entry in registry["extensions"] if entry.get("isLatest") is True
    ]
    definitions: list[dict[str, object]] = []
    seen_row_types: set[str] = set()

    for entry in sorted(latest, key=lambda item: item["identifier"]):
        row_type = entry["identifier"]
        if row_type in seen_row_types:
            raise ValueError(f"duplicate latest production rowType: {row_type}")
        seen_row_types.add(row_type)
        definition_bytes = fetch_bytes(entry["url"])
        definitions.append(
            {
                "issued": entry.get("issued"),
                "kind": "core" if "/core/" in entry["url"] else "extension",
                "row_type": row_type,
                "source_sha256": hashlib.sha256(definition_bytes).hexdigest(),
                "source_url": entry["url"],
                "terms": list(parse_definition(definition_bytes, row_type)),
                "title": entry["title"],
            }
        )

    return {
        "provenance": {
            "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
            "registry_url": REGISTRY_URL,
            "snapshot_date": snapshot_date,
        },
        "definitions": definitions,
    }


def write_snapshot(snapshot: dict[str, object], output_path: Path) -> None:
    """Atomically write the canonical JSON representation of a snapshot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the refresh command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-date",
        required=True,
        help="Review date recorded in the snapshot, in YYYY-MM-DD form.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> int:
    """Refresh the checked-in production registry snapshot."""
    arguments = build_argument_parser().parse_args()
    registry_bytes = fetch_bytes(REGISTRY_URL)
    snapshot = build_snapshot(registry_bytes, arguments.snapshot_date)
    write_snapshot(snapshot, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
