"""Create every dataset-specific occurrence table in PostgreSQL."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.engine import Engine

from src.database.base import OrmBase
from src.database.database import create_database_engine
from src.database.datasets import DATASETS, load_dataset_catalog
from src.database.tablesetup import TABLE_MODELS, build_table_models


TABLE_NAMES = tuple(model.__table__.name for model in TABLE_MODELS)


def initialize_tables(
    engine: Engine,
    table_models: tuple[object, ...] = TABLE_MODELS,
) -> tuple[str, ...]:
    """Create missing dataset tables without changing populated tables."""
    tables = tuple(model.__table__ for model in table_models)
    OrmBase.metadata.create_all(bind=engine, tables=tables, checkfirst=True)
    return tuple(table.name for table in tables)


def main() -> int:
    """Connect to PostgreSQL and ensure all dataset tables exist."""
    engine = create_database_engine()
    try:
        datasets = load_dataset_catalog(Path(__file__).resolve().parents[2])
        models = TABLE_MODELS if datasets == DATASETS else build_table_models(datasets)
        table_names = initialize_tables(engine, models)
    finally:
        engine.dispose()

    print(
        f"Ensured {len(table_names)} PostgreSQL table(s): "
        + ", ".join(table_names)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
