"""Build the local Darwin Core term snapshot from pinned official sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.request import Request, urlopen


DATA_DIRECTORY = Path(__file__).parents[1] / "database" / "data"
DEFAULT_PIN_PATH = DATA_DIRECTORY / "darwin_core_sources.json"
DEFAULT_OUTPUT_PATH = DATA_DIRECTORY / "darwin_core_terms.json"
TERM_VERSIONS_PATH = "vocabulary/term_versions.csv"
QRG_LIST_PATH = "build/qrg-template/qrg-list.csv"
SIMPLE_DWC_PATH = "dist/simple_dwc_vertical.csv"
USER_AGENT = "Gaia-Darwin-Core-refresh/1.0"
RS_TERM_LISTS = (
    "terms",
    "iri",
    "dc-for-dwc",
    "dcterms-for-dwc",
    "ac-for-dwc",
    "curatorial",
    "dwcore",
    "dwctype",
    "geospatial",
    "dwc-obsolete",
)
SIMPLE_DWC_TERM_LISTS = (
    "terms",
    "iri",
    "dc-for-dwc",
    "dcterms-for-dwc",
    "ac-for-dwc",
)
SNAPSHOT_FIELDS = (
    "iri",
    "term_localName",
    "label",
    "definition",
    "comments",
    "examples",
    "organized_in",
    "issued",
    "status",
    "replaces",
    "rdf_type",
    "term_iri",
)


def fetch_bytes(url: str) -> bytes:
    """Fetch one immutable source archive."""
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response:
        return response.read()


def archive_url(repository: str, commit: str) -> str:
    """Return GitHub's archive URL for an immutable commit."""
    return f"https://github.com/{repository}/archive/{commit}.tar.gz"


def read_archive_file(archive: bytes, relative_path: str) -> bytes:
    """Read a repository-relative file from a GitHub source archive."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        matches = [
            member
            for member in source.getmembers()
            if member.isfile() and member.name.split("/", 1)[-1] == relative_path
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one {relative_path!r} in archive, found {len(matches)}"
            )
        extracted = source.extractfile(matches[0])
        if extracted is None:
            raise ValueError(f"could not read {relative_path!r} from archive")
        return extracted.read()


def parse_csv(csv_bytes: bytes) -> list[dict[str, str]]:
    """Parse a UTF-8 CSV while preserving empty strings."""
    text = csv_bytes.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text, newline="")))


def validate_pin(pin: dict[str, str], archive: bytes) -> None:
    """Reject a source archive whose bytes do not match its checked-in pin."""
    actual_hash = hashlib.sha256(archive).hexdigest()
    expected_hash = pin["archive_sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            f"archive hash for {pin['repository']} is {actual_hash}, "
            f"expected {expected_hash}"
        )


def source_versions(rs_archive: bytes) -> tuple[set[str], list[dict[str, str]]]:
    """Read the official source tables used to compile term_versions.csv."""
    version_iris: set[str] = set()
    inputs: list[dict[str, str]] = []
    for term_list in RS_TERM_LISTS:
        path = f"{term_list}-versions/{term_list}-versions.csv"
        csv_bytes = read_archive_file(rs_archive, path)
        inputs.append({"path": path, "sha256": hashlib.sha256(csv_bytes).hexdigest()})
        for row in parse_csv(csv_bytes):
            version_iri = row.get("version", "")
            if version_iri:
                version_iris.add(version_iri)
    return version_iris, inputs


def simple_darwin_core_profile(
    dwc_archive: bytes,
    rs_archive: bytes,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Reproduce the official ordered Simple Darwin Core property profile."""
    simple_iris: set[str] = set()
    rs_inputs: list[dict[str, str]] = []
    for term_list in SIMPLE_DWC_TERM_LISTS:
        path = f"{term_list}/{term_list}.csv"
        csv_bytes = read_archive_file(rs_archive, path)
        rs_inputs.append(
            {"path": path, "sha256": hashlib.sha256(csv_bytes).hexdigest()}
        )
        for row in parse_csv(csv_bytes):
            if (
                row.get("rdf_type")
                != "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"
                or row.get("tdwgutility_usageScope") != "simple"
                or row.get("term_deprecated")
            ):
                continue
            simple_iris.add(row["term_isDefinedBy"] + row["term_localName"])

    qrg_bytes = read_archive_file(dwc_archive, QRG_LIST_PATH)
    ordered_iris = [
        row["recommended_term_iri"]
        for row in parse_csv(qrg_bytes)
        if row["recommended_term_iri"] in simple_iris
    ]
    if set(ordered_iris) != simple_iris or len(ordered_iris) != len(simple_iris):
        raise ValueError("Simple Darwin Core properties do not match the QRG order")

    simple_bytes = read_archive_file(dwc_archive, SIMPLE_DWC_PATH)
    local_names = simple_bytes.decode("utf-8-sig").splitlines()
    profile = [
        {"iri": iri, "local_name": iri.rstrip("/").rsplit("/", 1)[-1]}
        for iri in ordered_iris
    ]
    if [term["local_name"] for term in profile] != local_names:
        raise ValueError(
            "derived Simple Darwin Core properties do not match distribution"
        )
    dwc_inputs = [
        {"path": QRG_LIST_PATH, "sha256": hashlib.sha256(qrg_bytes).hexdigest()},
        {"path": SIMPLE_DWC_PATH, "sha256": hashlib.sha256(simple_bytes).hexdigest()},
    ]
    return profile, dwc_inputs, rs_inputs


def current_term_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Select one current representation for every Darwin Core term IRI."""
    by_term: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        missing = [field for field in SNAPSHOT_FIELDS if field not in row]
        if missing:
            raise ValueError(f"term_versions.csv lacks columns: {', '.join(missing)}")
        by_term.setdefault(row["term_iri"], []).append(row)

    terms: list[dict[str, object]] = []
    for term_iri, versions in by_term.items():
        recommended = [row for row in versions if row["status"] == "recommended"]
        if len(recommended) > 1:
            raise ValueError(f"multiple recommended versions for {term_iri}")
        candidates = recommended or versions
        selected = max(candidates, key=lambda row: (row["issued"], row["iri"]))
        terms.append(
            {
                "comments": selected["comments"],
                "definition": selected["definition"],
                "examples": selected["examples"],
                "iri": term_iri,
                "issued": selected["issued"],
                "label": selected["label"],
                "local_name": selected["term_localName"],
                "organized_in": split_iris(selected["organized_in"]),
                "rdf_type": selected["rdf_type"],
                "replaces": split_iris(selected["replaces"]),
                "status": selected["status"],
                "version_iri": selected["iri"],
            }
        )
    return sorted(terms, key=lambda term: (str(term["local_name"]).casefold(), str(term["iri"])))


def split_iris(value: str) -> list[str]:
    """Split the pipe-delimited relationship fields used by TDWG."""
    return [item for item in value.split("|") if item]


def build_snapshot(
    dwc_archive: bytes,
    rs_archive: bytes,
    pins: dict[str, object],
) -> dict[str, object]:
    """Build and cross-check a deterministic term snapshot."""
    source_pins = {
        source["repository"]: source
        for source in pins["sources"]  # type: ignore[index]
    }
    expected_repositories = {"tdwg/dwc", "tdwg/rs.tdwg.org"}
    if set(source_pins) != expected_repositories:
        raise ValueError(f"pins must contain exactly {sorted(expected_repositories)}")
    validate_pin(source_pins["tdwg/dwc"], dwc_archive)
    validate_pin(source_pins["tdwg/rs.tdwg.org"], rs_archive)

    term_csv = read_archive_file(dwc_archive, TERM_VERSIONS_PATH)
    rows = parse_csv(term_csv)
    source_version_iris, rs_inputs = source_versions(rs_archive)
    simple_profile, simple_dwc_inputs, simple_rs_inputs = simple_darwin_core_profile(
        dwc_archive,
        rs_archive,
    )
    missing_versions = sorted(
        row["iri"] for row in rows if row["iri"] not in source_version_iris
    )
    if missing_versions:
        raise ValueError(
            "term_versions.csv contains versions absent from pinned rs.tdwg.org: "
            + ", ".join(missing_versions[:5])
        )

    sources = []
    for repository in sorted(source_pins):
        pin = source_pins[repository]
        source = {
            "archive_sha256": pin["archive_sha256"],
            "commit": pin["commit"],
            "repository": repository,
        }
        if repository == "tdwg/dwc":
            source["input_files"] = [
                {
                    "path": TERM_VERSIONS_PATH,
                    "sha256": hashlib.sha256(term_csv).hexdigest(),
                },
                *simple_dwc_inputs,
            ]
        else:
            source["input_files"] = [*rs_inputs, *simple_rs_inputs]
        sources.append(source)
    return {
        "provenance": {"sources": sources},
        "schema_version": 1,
        "simple_darwin_core": simple_profile,
        "terms": current_term_rows(rows),
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


def load_pins(path: Path) -> dict[str, object]:
    """Load the checked-in immutable source pins."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the refresh command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=Path, default=DEFAULT_PIN_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--dwc-archive", type=Path)
    parser.add_argument("--rs-archive", type=Path)
    return parser


def source_archive(path: Path | None, pin: dict[str, str]) -> bytes:
    """Read a supplied archive or download its immutable pinned commit."""
    if path is not None:
        return path.read_bytes()
    return fetch_bytes(archive_url(pin["repository"], pin["commit"]))


def main() -> int:
    """Refresh the checked-in Darwin Core term snapshot."""
    arguments = build_argument_parser().parse_args()
    pins = load_pins(arguments.pins)
    source_pins = {source["repository"]: source for source in pins["sources"]}
    dwc_archive = source_archive(arguments.dwc_archive, source_pins["tdwg/dwc"])
    rs_archive = source_archive(
        arguments.rs_archive, source_pins["tdwg/rs.tdwg.org"]
    )
    write_snapshot(
        build_snapshot(dwc_archive, rs_archive, pins),
        arguments.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
