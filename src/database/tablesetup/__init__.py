"""Canonical Darwin Core Occurrence tables for Gaia datasets."""

from collections.abc import Sequence
from functools import cache

from sqlalchemy.sql.schema import Table

from src.database.base import OrmBase, build_dataset_table
from src.database.datasets import DATASETS, DatasetSchema
from src.database.darwin_core import load_simple_darwin_core_profile


_MODEL_NAMES = {
    "birds": "CtcBirds",
    "herpetology": "CtcHerpetology",
    "mammals": "CtcMammals",
    "fish": "FcFish",
    "algae": "HAlgae",
    "bryophytes": "HBryophytes",
    "fungi": "HFungi",
    "lichen": "HLichen",
    "vascular_plants": "HVascular",
    "marine_invertebrates": "MiMarineInvertebrates",
    "fossils": "PFossils",
    "entomology": "SecEntomology",
}

OCCURRENCE_HEADERS = load_simple_darwin_core_profile().headers


def _build_model(dataset: DatasetSchema) -> type[OrmBase]:
    """Build one named ORM model using the canonical Occurrence columns."""
    model_name = _MODEL_NAMES.get(
        dataset.key,
        "".join(part.capitalize() for part in dataset.key.split("_")),
    )
    if not model_name:
        raise ValueError("dataset key must produce an ORM model name")
    table_name = dataset.relative_path.stem
    table = OrmBase.metadata.tables.get(table_name)
    if table is None:
        table = build_dataset_table(
            table_name=table_name,
            source_headers=OCCURRENCE_HEADERS,
        )
    return type(
        model_name,
        (OrmBase,),
        {
            "__doc__": f"Store canonical Occurrence records for {dataset.short_label}.",
            "__table__": table,
        },
    )


@cache
def _build_table_models(
    datasets: tuple[DatasetSchema, ...],
) -> tuple[type[OrmBase], ...]:
    """Build each distinct catalog's model tuple once per process."""
    return tuple(_build_model(dataset) for dataset in datasets)


def build_table_models(
    datasets: Sequence[DatasetSchema],
) -> tuple[type[OrmBase], ...]:
    """Build canonical Occurrence models for an arbitrary dataset catalog."""
    return _build_table_models(tuple(datasets))


TABLE_MODELS = build_table_models(DATASETS)
_MODELS_BY_KEY = dict(
    zip((dataset.key for dataset in DATASETS), TABLE_MODELS, strict=True)
)

CtcBirds = _MODELS_BY_KEY["birds"]
CtcHerpetology = _MODELS_BY_KEY["herpetology"]
CtcMammals = _MODELS_BY_KEY["mammals"]
FcFish = _MODELS_BY_KEY["fish"]
HAlgae = _MODELS_BY_KEY["algae"]
HBryophytes = _MODELS_BY_KEY["bryophytes"]
HFungi = _MODELS_BY_KEY["fungi"]
HLichen = _MODELS_BY_KEY["lichen"]
HVascular = _MODELS_BY_KEY["vascular_plants"]
MiMarineInvertebrates = _MODELS_BY_KEY["marine_invertebrates"]
PFossils = _MODELS_BY_KEY["fossils"]
SecEntomology = _MODELS_BY_KEY["entomology"]


def build_dataset_table_map(
    *,
    datasets: Sequence[DatasetSchema] = DATASETS,
    table_models: Sequence[object] = TABLE_MODELS,
) -> dict[str, Table]:
    """Map dataset keys to models by authoritative source stem/table name."""
    tables_by_name: dict[str, Table] = {}
    for model in table_models:
        table = getattr(model, "__table__", None)
        if not isinstance(table, Table):
            raise ValueError(
                "every registered table model must expose a SQLAlchemy Table"
            )
        if table.name in tables_by_name:
            raise ValueError(f"duplicate registered table model: {table.name}")
        tables_by_name[table.name] = table

    mapping: dict[str, Table] = {}
    expected_table_names: set[str] = set()
    for dataset in datasets:
        table_name = dataset.relative_path.stem
        expected_table_names.add(table_name)
        table = tables_by_name.get(table_name)
        if table is None:
            raise ValueError(f"{dataset.key}: no table model named {table_name}")
        mapping[dataset.key] = table

    unexpected = sorted(tables_by_name.keys() - expected_table_names)
    if unexpected:
        raise ValueError(
            f"table model(s) have no dataset source: {', '.join(unexpected)}"
        )
    return mapping


__all__ = [
    "CtcBirds",
    "CtcHerpetology",
    "CtcMammals",
    "FcFish",
    "HAlgae",
    "HBryophytes",
    "HFungi",
    "HLichen",
    "HVascular",
    "MiMarineInvertebrates",
    "OCCURRENCE_HEADERS",
    "PFossils",
    "SecEntomology",
    "TABLE_MODELS",
    "build_dataset_table_map",
    "build_table_models",
]
