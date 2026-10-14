import json
import logging
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from sqlalchemy import Column, MetaData, Table, Text

from src.database.darwin_core import load_simple_darwin_core_profile
from src.database.datasets import DatasetSchema
from src.scripts.dataset_schema import jsonl_relative_path
from src.scripts.transform_occurrences import (
    HeaderValidationError,
    convert_files,
    discover_inputs,
    resolve_inputs,
)


HEADERS = (
    "occurrenceID",
    "catalogNumber",
    "scientificName",
    "datasetName",
    "locality",
)
DWC = "http://rs.tdwg.org/dwc/terms/"
DATASET_NAME = "Test compliant archive"


def make_registry() -> tuple[DatasetSchema, SimpleNamespace]:
    dataset = DatasetSchema(
        key="birds",
        short_label="Birds",
        collection_key="test_collection",
        collection_label="Test Collection",
        relative_path=Path("dataset/test_collection/birds/canonical_birds.txt"),
        identity_values=(DATASET_NAME,),
    )
    table = Table(
        "canonical_birds",
        MetaData(),
        *(Column(header, Text) for header in HEADERS),
        info={"source_headers": HEADERS},
    )
    return dataset, SimpleNamespace(__table__=table)


def core_xml(*, files: tuple[str, ...] = ("occurrence.txt",), id_tag: str = "") -> str:
    locations = "".join(f"<location>{path}</location>" for path in files)
    fields = "".join(
        (
            f'<field index="2" term="{DWC}occurrenceID"/>',
            f'<field index="0" term="{DWC}catalogNumber"/>',
            f'<field index="3" term="{DWC}scientificName"/>',
            f'<field index="1" term="{DWC}datasetName"/>',
        )
    )
    return (
        '<core rowType="http://rs.tdwg.org/dwc/terms/Occurrence" '
        'encoding="UTF-8" fieldsTerminatedBy="," fieldsEnclosedBy="&quot;" '
        f'ignoreHeaderLines="1"><files>{locations}</files>{id_tag}{fields}</core>'
    )


def write_archive(
    path: Path,
    *,
    core: str | None = None,
    extension: str = "",
    rows: tuple[str, ...] = (
        'B-1,"Test compliant archive",occ-1,"Mésange — 雪"\n',
    ),
    files: tuple[str, ...] = ("occurrence.txt",),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<archive xmlns="http://rs.tdwg.org/dwc/text/">'
        f"{core or core_xml(files=files)}{extension}</archive>"
    )
    with ZipFile(path, "w") as archive:
        archive.writestr("meta.xml", metadata)
        for member, row in zip(files, rows, strict=True):
            archive.writestr(
                member,
                "catalog,dataset,occurrence,scientific\n" + row,
            )


def resolve_test_inputs(paths: tuple[Path, ...]):
    dataset, table_model = make_registry()
    resolved = resolve_inputs(paths, datasets=(dataset,))
    return dataset, table_model, resolved


def test_meta_xml_controls_mapping_dialect_and_canonical_output(tmp_path: Path) -> None:
    source = tmp_path / "arbitrary-provider-name.zip"
    write_archive(source)
    dataset, table_model, resolved = resolve_test_inputs((source,))

    result = convert_files(
        resolved,
        repository_root=tmp_path,
        logger=logging.getLogger("test_transform_occurrences.dwca"),
        run_id="test-run",
        invocation="test",
        table_models=(table_model,),
    )

    output_path = tmp_path / jsonl_relative_path(dataset)
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert result.converted_files[0].output_path == output_path
    assert record["occurrenceID"] == "occ-1"
    assert record["catalogNumber"] == "B-1"
    assert record["raw_record"] == {
        "occurrenceID": "occ-1",
        "catalogNumber": "B-1",
        "scientificName": "Mésange — 雪",
        "datasetName": DATASET_NAME,
        "locality": "",
    }


def test_meta_xml_uses_standard_csv_defaults(tmp_path: Path) -> None:
    source = tmp_path / "default-dialect.zip"
    core = core_xml().replace(
        ' encoding="UTF-8" fieldsTerminatedBy="," fieldsEnclosedBy="&quot;"',
        "",
    )
    write_archive(source, core=core)

    _dataset, table_model, resolved = resolve_test_inputs((source,))
    result = convert_files(
        resolved,
        repository_root=tmp_path,
        logger=logging.getLogger("test_transform_occurrences.defaults"),
        run_id="test-run",
        invocation="test",
        table_models=(table_model,),
    )

    assert result.rows_written == 1


def test_all_declared_core_files_are_streamed(tmp_path: Path) -> None:
    source = tmp_path / "multi.zip"
    write_archive(
        source,
        files=("one.csv", "two.csv"),
        rows=(
            'B-1,"Test compliant archive",occ-1,Raven\n',
            'B-2,"Test compliant archive",occ-2,Crow\n',
        ),
    )
    _dataset, table_model, resolved = resolve_test_inputs((source,))
    result = convert_files(
        resolved,
        repository_root=tmp_path,
        logger=logging.getLogger("test_transform_occurrences.multifile"),
        run_id="test-run",
        invocation="test",
        table_models=(table_model,),
    )
    assert result.rows_written == 2


def unsupported_extension_xml() -> str:
    return (
        '<extension rowType="https://example.org/custom" fieldsTerminatedBy="\\t">'
        '<files><location>extension.txt</location></files>'
        '<coreid index="0"/>'
        '<field index="1" term="https://example.org/custom/value"/>'
        '</extension>'
    )


def test_unsupported_extension_is_ignored_while_occurrence_core_is_resolved(
    tmp_path: Path,
) -> None:
    extension = unsupported_extension_xml()
    source = tmp_path / "extension.zip"
    write_archive(source, core=core_xml(id_tag='<id index="2"/>'), extension=extension)
    with ZipFile(source, "a") as archive:
        archive.writestr("extension.txt", "occ-1\tvalue\n")

    _dataset, _table_model, resolved = resolve_test_inputs((source,))

    assert resolved[0][0].archive_metadata is not None
    assert resolved[0][0].archive_metadata.extensions == ()


def test_later_duplicate_source_replaces_earlier_with_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    write_archive(first)
    write_archive(
        second,
        rows=('B-2,"Test compliant archive",occ-2,Crow\n',),
    )

    _dataset, _table_model, resolved = resolve_test_inputs((first, second))

    assert len(resolved) == 1
    assert resolved[0][0].path == second
    warning = capsys.readouterr().err
    assert "warning" in warning.casefold()
    assert "birds" in warning
    assert str(first) in warning
    assert str(second) in warning


def test_unknown_dataset_is_assigned_deterministic_metadata(tmp_path: Path) -> None:
    source = tmp_path / "unknown.zip"
    dataset_name = "Some Other Dataset"
    dataset_id = "https://example.org/datasets/specimens-2026"
    fields = core_xml().replace(
        f'<field index="1" term="{DWC}datasetName"/>',
        (
            f'<field index="1" term="{DWC}datasetName"/>'
            f'<field index="4" term="{DWC}datasetID"/>'
        ),
    )
    write_archive(
        source,
        core=fields,
        rows=(f'B-1,"{dataset_name}",occ-1,Raven,{dataset_id}\n',),
    )

    first = resolve_inputs((source,), datasets=())
    second = resolve_inputs((source,), datasets=())

    assert first == second
    assert len(first) == 1
    resolved_source, dataset = first[0]
    expected_key = "some_other_dataset"
    assert resolved_source.path == source
    assert dataset.key == expected_key
    assert dataset.short_label == dataset_name
    assert dataset.collection_key == "darwin_core_archives"
    assert dataset.collection_label == "Darwin Core Archives"
    assert dataset.identity_values == (dataset_name, dataset_id)
    assert dataset.relative_path == Path(
        f"dataset/darwin_core_archives/{expected_key}/{expected_key}.txt"
    )


def test_current_simple_darwin_core_term_is_accepted(tmp_path: Path) -> None:
    source = tmp_path / "current-standard.zip"
    core = core_xml().replace(
        "</core>",
        f'<field index="4" term="{DWC}eventCategory"/></core>',
    )
    write_archive(
        source,
        core=core,
        rows=('B-1,"Test compliant archive",occ-1,Raven,survey\n',),
    )

    _dataset, _table_model, resolved = resolve_test_inputs((source,))

    assert resolved[0][0].archive_metadata is not None
    assert resolved[0][0].archive_metadata.core.fields[-1].name == "eventCategory"

    dataset = resolved[0][1]
    headers = load_simple_darwin_core_profile().headers
    table = Table(
        dataset.relative_path.stem,
        MetaData(),
        *(Column(header, Text) for header in headers),
        info={"source_headers": headers},
    )
    result = convert_files(
        resolved,
        repository_root=tmp_path,
        logger=logging.getLogger("test_transform_occurrences.current_standard"),
        run_id="test-run",
        invocation="test",
        table_models=(SimpleNamespace(__table__=table),),
    )
    record = json.loads(result.converted_files[0].output_path.read_text())
    assert record["raw_record"]["eventCategory"] == "survey"
    assert tuple(record["raw_record"]) == headers


def test_gbif_only_occurrence_field_is_not_in_standard_profile(tmp_path: Path) -> None:
    source = tmp_path / "gbif-only.zip"
    core = core_xml().replace(
        "</core>",
        f'<field index="4" term="{DWC}fundingAttribution"/></core>',
    )
    write_archive(
        source,
        core=core,
        rows=('B-1,"Test compliant archive",occ-1,Raven,grant\n',),
    )

    with pytest.raises(HeaderValidationError, match="unregistered term"):
        resolve_test_inputs((source,))


def test_archive_without_meta_xml_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "missing-meta.zip"
    with ZipFile(source, "w") as archive:
        archive.writestr("occurrence.txt", "occurrenceID\nocc-1\n")
    with pytest.raises(HeaderValidationError, match="missing meta.xml"):
        resolve_test_inputs((source,))


def test_discovery_accepts_only_zip_archives(tmp_path: Path) -> None:
    archives = [tmp_path / "a.ZIP", tmp_path / "nested" / "b.zip"]
    for path in archives:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "nested" / "occurrence.txt").touch()
    (tmp_path / "nested" / "generated.jsonl").touch()
    assert discover_inputs(tmp_path) == sorted(archives)
