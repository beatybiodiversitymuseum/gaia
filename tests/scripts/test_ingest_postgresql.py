import io
import json
import logging
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import (
    Column,
    MetaData,
    Table,
    Text,
    create_engine,
    event,
    insert,
    select,
)
from sqlalchemy.engine import Engine

from src.database.base import orm_attribute_name
from src.database.datasets import DatasetSchema
from src.scripts.dataset_schema import jsonl_relative_path
from src.scripts.ingest_postgresql import (
    DEFAULT_BATCH_SIZE,
    PostgreSQLIngestionError,
    import_datasets,
    parse_jsonl_record,
)


HEADERS = ("occurrenceID", "catalogNumber", "locality", "remarks")


def make_dataset(key: str, table_name: str) -> DatasetSchema:
    return DatasetSchema(
        key=key,
        short_label=key,
        collection_key="test_collection",
        collection_label="Test Collection",
        relative_path=Path(f"dataset/test_collection/{key}/{table_name}.txt"),
    )


def make_table(metadata: MetaData, table_name: str) -> Table:
    return Table(
        table_name,
        metadata,
        *(
            Column(
                header,
                Text,
                key=orm_attribute_name(header),
                nullable=False,
                primary_key=header == "occurrenceID",
            )
            for header in HEADERS
        ),
        info={"source_headers": HEADERS},
    )


def make_database() -> tuple[Engine, DatasetSchema, Table]:
    metadata = MetaData()
    dataset = make_dataset("birds", "ctc_birds")
    table = make_table(metadata, "ctc_birds")
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
    return engine, dataset, table


def make_logger() -> tuple[logging.Logger, io.StringIO]:
    output = io.StringIO()
    logger = logging.Logger("test_ingest_postgresql", level=logging.INFO)
    logger.propagate = False
    logger.addHandler(logging.StreamHandler(output))
    return logger, output


def make_record(
    occurrence_id: str,
    catalog_number: str,
    *,
    locality: str = "",
    remarks: str = "",
    chunks: object | None = None,
) -> dict[str, object]:
    raw_record = {
        "occurrenceID": occurrence_id,
        "catalogNumber": catalog_number,
        "locality": locality,
        "remarks": remarks,
    }
    return {
        "occurrenceID": occurrence_id,
        "catalogNumber": catalog_number,
        "chunks": {} if chunks is None else chunks,
        "raw_record": raw_record,
    }


def raw_record(record: dict[str, object]) -> dict[str, object]:
    value = record["raw_record"]
    assert isinstance(value, dict)
    return value


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def run_import(
    engine: Engine,
    repository_root: Path,
    datasets: tuple[DatasetSchema, ...],
    tables: tuple[Table, ...],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    replace: bool = False,
):
    logger, output = make_logger()
    result = import_datasets(
        engine,
        repository_root=repository_root,
        batch_size=batch_size,
        replace=replace,
        datasets=datasets,
        table_models=tuple(
            SimpleNamespace(__table__=table) for table in tables
        ),
        logger=logger,
        run_id="test-run",
        invocation="python -m src.scripts.ingest_postgresql",
        persist_schema_profiles=False,
    )
    return result, output.getvalue()


def row_values(engine: Engine, table: Table) -> list[tuple[str, ...]]:
    with engine.connect() as connection:
        rows = connection.execute(select(table).order_by(table.c.occurrence_id))
        return [tuple(row) for row in rows]


def insert_record(
    engine: Engine,
    table: Table,
    record: dict[str, object],
) -> None:
    source = raw_record(record)
    parameters = {
        column.key: source[column.name] for column in table.columns
    }
    with engine.begin() as connection:
        connection.execute(insert(table), parameters)


def test_successfully_streams_lossless_unicode_and_empty_strings(
    tmp_path: Path,
) -> None:
    engine, dataset, table = make_database()
    path = tmp_path / jsonl_relative_path(dataset)
    write_jsonl(
        path,
        [
            make_record(
                "occ-α",
                "CAT-1",
                locality="Haida Gwaii — Île",
                remarks="",
                chunks={"location_georeferencing": {"locality": "ignored"}},
            ),
            make_record("occ-β", "CAT-2", locality="", remarks="雪"),
        ],
    )

    result, log_output = run_import(engine, tmp_path, (dataset,), (table,))

    assert result.status == "complete"
    assert result.rows_inserted == 2
    assert row_values(engine, table) == [
        ("occ-α", "CAT-1", "Haida Gwaii — Île", ""),
        ("occ-β", "CAT-2", "", "雪"),
    ]
    assert "chunks" not in table.c
    assert dict(result.succeeded[0].header_availability) == {
        "occurrenceID": 1,
        "catalogNumber": 1,
        "locality": 1,
        "remarks": 1,
    }
    assert "started_at=" in log_output
    assert "file_succeeded=1 file_failed=0 rows_inserted=2" in log_output
    assert (
        "malformed_records=0 missing_target_schema_headers=0" in log_output
    )


def invalid_record(case: str) -> str:
    record = make_record("occ-1", "CAT-1", locality="Vancouver")
    if case == "top_level_order":
        record = {
            "catalogNumber": record["catalogNumber"],
            "occurrenceID": record["occurrenceID"],
            "chunks": record["chunks"],
            "raw_record": record["raw_record"],
        }
    elif case == "raw_order":
        raw = raw_record(record)
        record["raw_record"] = {
            "catalogNumber": raw["catalogNumber"],
            "occurrenceID": raw["occurrenceID"],
            "locality": raw["locality"],
            "remarks": raw["remarks"],
        }
    elif case == "raw_value_type":
        raw_record(record)["locality"] = 42
    elif case == "identity_mismatch":
        record["occurrenceID"] = "different"
    elif case == "chunks_type":
        record["chunks"] = []
    else:
        raise AssertionError(case)
    return json.dumps(record)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("top_level_order", "top-level keys and order"),
        ("raw_order", "raw_record keys/order"),
        ("raw_value_type", "raw_record.locality must be a string"),
        ("identity_mismatch", "occurrenceID does not match"),
        ("chunks_type", "chunks must be a JSON object"),
    ),
)
def test_jsonl_validation_is_exact(case: str, message: str) -> None:
    with pytest.raises(PostgreSQLIngestionError, match=message):
        parse_jsonl_record(
            invalid_record(case),
            path=Path("sample.jsonl"),
            line_number=7,
            expected_headers=HEADERS,
        )


def test_mid_file_validation_failure_rolls_back_prior_batches(
    tmp_path: Path,
) -> None:
    engine, dataset, table = make_database()
    path = tmp_path / jsonl_relative_path(dataset)
    valid = make_record("occ-1", "CAT-1")
    invalid = make_record("occ-2", "CAT-2")
    raw_record(invalid)["remarks"] = None
    write_jsonl(path, [valid, invalid])

    result, _ = run_import(
        engine,
        tmp_path,
        (dataset,),
        (table,),
        batch_size=1,
    )

    assert result.status == "completed_with_failures"
    assert result.rows_inserted == 0
    assert result.failed[0].error_type == "JsonlRecordError"
    assert result.malformed_records == 1
    assert row_values(engine, table) == []


def test_missing_target_headers_are_classified_and_bounded(
    tmp_path: Path,
) -> None:
    engine, dataset, table = make_database()
    record = make_record("occ-1", "CAT-1")
    del raw_record(record)["remarks"]
    write_jsonl(tmp_path / jsonl_relative_path(dataset), [record])

    result, log_output = run_import(engine, tmp_path, (dataset,), (table,))

    assert result.failed[0].error_type == "JsonlSchemaError"
    assert result.failed[0].missing_headers == ("remarks",)
    assert result.malformed_records == 1
    assert result.missing_target_schema_headers == 1
    assert "missing_header_count=1" in log_output
    assert "missing_headers=[remarks]" in log_output
    assert row_values(engine, table) == []


def test_nonempty_destination_is_refused_without_replace(tmp_path: Path) -> None:
    engine, dataset, table = make_database()
    insert_record(engine, table, make_record("old-occ", "OLD"))
    write_jsonl(
        tmp_path / jsonl_relative_path(dataset),
        [make_record("new-occ", "NEW")],
    )

    result, _ = run_import(engine, tmp_path, (dataset,), (table,))

    assert result.failed[0].error_type == "DestinationNotEmptyError"
    assert row_values(engine, table) == [("old-occ", "OLD", "", "")]


def test_dataset_update_is_atomic_on_failure_and_replaces_all_rows_on_success(
    tmp_path: Path,
) -> None:
    engine, dataset, table = make_database()
    insert_record(engine, table, make_record("old-occ", "OLD"))
    path = tmp_path / jsonl_relative_path(dataset)
    valid = make_record("new-occ", "NEW")
    invalid = make_record("bad-occ", "BAD")
    raw_record(invalid)["locality"] = None
    write_jsonl(path, [valid, invalid])

    failed, _ = run_import(
        engine,
        tmp_path,
        (dataset,),
        (table,),
        batch_size=1,
        replace=True,
    )
    assert failed.failed
    assert row_values(engine, table) == [("old-occ", "OLD", "", "")]

    replacement = make_record("replacement-occ", "REPLACEMENT")
    write_jsonl(path, [valid, replacement])
    succeeded, _ = run_import(
        engine,
        tmp_path,
        (dataset,),
        (table,),
        replace=True,
    )
    assert succeeded.rows_inserted == 2
    assert row_values(engine, table) == [
        ("new-occ", "NEW", "", ""),
        ("replacement-occ", "REPLACEMENT", "", ""),
    ]


def test_empty_source_does_not_commit_an_empty_replacement(tmp_path: Path) -> None:
    engine, dataset, table = make_database()
    insert_record(engine, table, make_record("old-occ", "OLD"))
    path = tmp_path / jsonl_relative_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")

    result, _ = run_import(
        engine,
        tmp_path,
        (dataset,),
        (table,),
        replace=True,
    )

    assert result.failed[0].error_type == "EmptyJsonlError"
    assert row_values(engine, table) == [("old-occ", "OLD", "", "")]


def _record_insert_batches(
    batch_lengths, connection, cursor, statement, parameters, context, executemany
) -> None:
    if context.compiled is not None and context.compiled.isinsert:
        batch_lengths.append(len(parameters) if executemany else 1)


def test_ingestion_batching_is_bounded_across_datasets(
    tmp_path: Path,
) -> None:
    metadata = MetaData()
    birds = make_dataset("birds", "ctc_birds")
    fish = make_dataset("fish", "fc_fish")
    birds_table = make_table(metadata, "ctc_birds")
    fish_table = make_table(metadata, "fc_fish")
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
    write_jsonl(
        tmp_path / jsonl_relative_path(birds),
        [make_record(f"bird-{index}", f"B-{index}") for index in range(5)],
    )
    write_jsonl(
        tmp_path / jsonl_relative_path(fish),
        [make_record("fish-1", "F-1")],
    )
    batch_lengths: list[int] = []

    event.listen(
        engine, "before_cursor_execute", partial(_record_insert_batches, batch_lengths)
    )

    result, _ = run_import(
        engine,
        tmp_path,
        (birds, fish),
        (fish_table, birds_table),
        batch_size=2,
    )

    assert result.rows_inserted == 6
    assert tuple(item.dataset_key for item in result.succeeded) == (
        "birds",
        "fish",
    )
    assert batch_lengths == [2, 2, 1, 1]


def test_constraint_failure_rolls_back_and_never_logs_record_values(
    tmp_path: Path,
) -> None:
    engine, dataset, table = make_database()
    secret_value = "DO-NOT-LOG-THIS-RAW-VALUE"
    write_jsonl(
        tmp_path / jsonl_relative_path(dataset),
        [
            make_record("duplicate", "CAT-1", locality=secret_value),
            make_record("duplicate", "CAT-2", locality=secret_value),
        ],
    )

    result, log_output = run_import(
        engine,
        tmp_path,
        (dataset,),
        (table,),
        batch_size=2,
    )

    assert result.failed[0].error_type == "IntegrityError"
    assert result.failed[0].message == (
        "database transaction failed; no rows were committed"
    )
    assert row_values(engine, table) == []
    assert secret_value not in log_output
    assert secret_value not in result.failed[0].message


def test_header_profile_ignores_only_posix_whitespace_and_preserves_it(
    tmp_path: Path,
) -> None:
    engine, dataset, table = make_database()
    path = tmp_path / jsonl_relative_path(dataset)
    whitespace = " \t\n\v\f\r"
    write_jsonl(
        path,
        [make_record("occ-1", "CAT-1", locality=whitespace, remarks=" x ")],
    )

    result, _ = run_import(engine, tmp_path, (dataset,), (table,))

    assert dict(result.succeeded[0].header_availability) == {
        "occurrenceID": 1,
        "catalogNumber": 1,
        "locality": 0,
        "remarks": 1,
    }
    assert row_values(engine, table) == [
        ("occ-1", "CAT-1", whitespace, " x ")
    ]
