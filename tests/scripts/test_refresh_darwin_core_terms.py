"""Tests for deterministic Darwin Core term snapshot generation."""

from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

from src.scripts import refresh_darwin_core_terms


TERM_HEADER = ",".join(refresh_darwin_core_terms.SNAPSHOT_FIELDS)
TERM_ROWS = "\n".join(
    (
        TERM_HEADER,
        "https://example.org/version/sample-1,sample,Old label,Old definition,,,,2020-01-01,superseded,,https://example.org/Property,https://example.org/sample",
        "https://example.org/version/sample-2,sample,Current label,Current definition,Usage note,Example,https://example.org/Class,2024-01-01,recommended,https://example.org/version/sample-1,https://example.org/Property,https://example.org/sample",
    )
) + "\n"
RS_HEADER = "version,term_localName\n"
RS_CURRENT_HEADER = (
    "term_localName,term_isDefinedBy,term_deprecated,rdf_type,"
    "tdwgutility_usageScope\n"
)


def make_archive(files: dict[str, str]) -> bytes:
    """Build a small in-memory GitHub-shaped tar archive."""
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for path, contents in files.items():
            data = contents.encode()
            member = tarfile.TarInfo(f"repo-commit/{path}")
            member.size = len(data)
            member.mtime = 0
            archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


def source_pins(dwc_archive: bytes, rs_archive: bytes) -> dict[str, object]:
    """Return pins matching the fixture archive bytes."""
    return {
        "schema_version": 1,
        "sources": [
            {
                "repository": "tdwg/dwc",
                "commit": "a" * 40,
                "archive_sha256": hashlib.sha256(dwc_archive).hexdigest(),
            },
            {
                "repository": "tdwg/rs.tdwg.org",
                "commit": "b" * 40,
                "archive_sha256": hashlib.sha256(rs_archive).hexdigest(),
            },
        ],
    }


def fixture_archives() -> tuple[bytes, bytes]:
    """Build matching compiled and source archive fixtures."""
    dwc_archive = make_archive(
        {
            refresh_darwin_core_terms.TERM_VERSIONS_PATH: TERM_ROWS,
            refresh_darwin_core_terms.QRG_LIST_PATH: (
                "recommended_term_iri\nhttps://example.org/sample\n"
            ),
            refresh_darwin_core_terms.SIMPLE_DWC_PATH: "sample\n",
        }
    )
    source_rows = (
        RS_HEADER
        + "https://example.org/version/sample-1,sample\n"
        + "https://example.org/version/sample-2,sample\n"
    )
    rs_files = {
        f"{name}-versions/{name}-versions.csv": (
            source_rows if name == "terms" else RS_HEADER
        )
        for name in refresh_darwin_core_terms.RS_TERM_LISTS
    }
    rs_files.update({
        f"{name}/{name}.csv": (
            RS_CURRENT_HEADER
            + (
                "sample,https://example.org/,,"
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property,simple\n"
                if name == "terms"
                else ""
            )
        )
        for name in refresh_darwin_core_terms.SIMPLE_DWC_TERM_LISTS
    })
    return dwc_archive, make_archive(rs_files)


def test_build_snapshot_uses_recommended_version_and_both_sources() -> None:
    dwc_archive, rs_archive = fixture_archives()

    snapshot = refresh_darwin_core_terms.build_snapshot(
        dwc_archive, rs_archive, source_pins(dwc_archive, rs_archive)
    )

    assert snapshot["terms"] == [
        {
            "comments": "Usage note",
            "definition": "Current definition",
            "examples": "Example",
            "iri": "https://example.org/sample",
            "issued": "2024-01-01",
            "label": "Current label",
            "local_name": "sample",
            "organized_in": ["https://example.org/Class"],
            "rdf_type": "https://example.org/Property",
            "replaces": ["https://example.org/version/sample-1"],
            "status": "recommended",
            "version_iri": "https://example.org/version/sample-2",
        }
    ]
    assert snapshot["simple_darwin_core"] == [
        {"iri": "https://example.org/sample", "local_name": "sample"}
    ]
    sources = snapshot["provenance"]["sources"]
    assert {source["repository"] for source in sources} == {
        "tdwg/dwc",
        "tdwg/rs.tdwg.org",
    }
    assert len(sources[1]["input_files"]) == (
        len(refresh_darwin_core_terms.RS_TERM_LISTS)
        + len(refresh_darwin_core_terms.SIMPLE_DWC_TERM_LISTS)
    )


def test_build_snapshot_rejects_unpinned_archive() -> None:
    dwc_archive, rs_archive = fixture_archives()
    pins = source_pins(dwc_archive, rs_archive)
    pins["sources"][0]["archive_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="archive hash"):
        refresh_darwin_core_terms.build_snapshot(dwc_archive, rs_archive, pins)


def test_build_snapshot_rejects_version_missing_from_source_tables() -> None:
    dwc_archive, valid_rs_archive = fixture_archives()
    empty_version_files = {
        f"{name}-versions/{name}-versions.csv": RS_HEADER
        for name in refresh_darwin_core_terms.RS_TERM_LISTS
    }
    with tarfile.open(fileobj=io.BytesIO(valid_rs_archive), mode="r:gz") as source:
        current_files = {
            member.name.split("/", 1)[-1]: source.extractfile(member).read().decode()
            for member in source.getmembers()
            if member.isfile() and "-versions/" not in member.name
        }
    rs_archive = make_archive({**empty_version_files, **current_files})

    with pytest.raises(ValueError, match="absent from pinned rs.tdwg.org"):
        refresh_darwin_core_terms.build_snapshot(
            dwc_archive, rs_archive, source_pins(dwc_archive, rs_archive)
        )
