"""Shared dataset identities and persisted discovered-dataset catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final


DATASET_CATALOG_RELATIVE_PATH: Final = Path("dataset/.gaia/datasets.json")


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    """Identify one authoritative TSV header and its collection scope."""

    key: str
    short_label: str
    collection_key: str
    collection_label: str
    relative_path: Path
    identity_values: tuple[str, ...] = ()


DATASETS = (
    DatasetSchema(
        key="birds",
        short_label="Birds",
        collection_key="cowan_tetrapod_collection",
        collection_label="Cowan Tetrapod Collection",
        relative_path=Path(
            "dataset/cowan_tetrapod_collection/birds/ctc_birds.txt"
        ),
        identity_values=(
            "https://doi.org/10.15468/aytqsu",
            "Cowan Tetrapod Collection - Birds",
        ),
    ),
    DatasetSchema(
        key="herpetology",
        short_label="Herpetology",
        collection_key="cowan_tetrapod_collection",
        collection_label="Cowan Tetrapod Collection",
        relative_path=Path(
            "dataset/cowan_tetrapod_collection/herpetology/ctc_herpetology.txt"
        ),
        identity_values=("Cowan Tetrapod Collection - Herpetology",),
    ),
    DatasetSchema(
        key="mammals",
        short_label="Mammals",
        collection_key="cowan_tetrapod_collection",
        collection_label="Cowan Tetrapod Collection",
        relative_path=Path(
            "dataset/cowan_tetrapod_collection/mammals/ctc_mammals.txt"
        ),
        identity_values=("Cowan Tetrapod Collection - Mammals",),
    ),
    DatasetSchema(
        key="fish",
        short_label="Fish",
        collection_key="fish_collection",
        collection_label="Fish Collection",
        relative_path=Path("dataset/fish_collection/fish/fc_fish.txt"),
        identity_values=(
            "https://doi.org/10.15468/yacq34",
            "Beaty Biodiversity Museum Fish Collection",
        ),
    ),
    DatasetSchema(
        key="algae",
        short_label="Algae",
        collection_key="herbarium",
        collection_label="Herbarium",
        relative_path=Path("dataset/herbarium/algae/h_algae.txt"),
        identity_values=(
            "https://doi.org/10.5886/ujwtvvs2",
            "University of British Columbia Herbarium (UBC) - Algae Collection",
        ),
    ),
    DatasetSchema(
        key="bryophytes",
        short_label="Bryophytes",
        collection_key="herbarium",
        collection_label="Herbarium",
        relative_path=Path("dataset/herbarium/bryophytes/h_bryophytes.txt"),
        identity_values=(
            "https://doi.org/10.5886/5rr9fbb6",
            "University of British Columbia Herbarium (UBC) - Bryophytes Collection",
        ),
    ),
    DatasetSchema(
        key="fungi",
        short_label="Fungi",
        collection_key="herbarium",
        collection_label="Herbarium",
        relative_path=Path("dataset/herbarium/fungi/h_fungi.txt"),
        identity_values=(
            "https://doi.org/10.5886/h4px7g4b",
            "University of British Columbia Herbarium (UBC) - Fungi Collection",
        ),
    ),
    DatasetSchema(
        key="lichen",
        short_label="Lichen",
        collection_key="herbarium",
        collection_label="Herbarium",
        relative_path=Path("dataset/herbarium/lichen/h_lichen.txt"),
        identity_values=(
            "https://doi.org/10.5886/8hh3tyfd",
            "University of British Columbia Herbarium (UBC) - Lichen Collection",
        ),
    ),
    DatasetSchema(
        key="vascular_plants",
        short_label="Vascular",
        collection_key="herbarium",
        collection_label="Herbarium",
        relative_path=Path("dataset/herbarium/vascular/h_vascular.txt"),
        identity_values=(
            "https://doi.org/10.5886/rtt57cc9",
            "University of British Columbia Herbarium (UBC) - Vascular Plant Collection",
        ),
    ),
    DatasetSchema(
        key="marine_invertebrates",
        short_label="Marine Inv.",
        collection_key="marine_invertebrates",
        collection_label="Marine Invertebrates",
        relative_path=Path(
            "dataset/marine_invertebrates/marine_invertebrates/"
            "mi_marine_invertebrates.txt"
        ),
        identity_values=(
            "Beaty Biodiversity Museum Marine Invertebrate Collection",
        ),
    ),
    DatasetSchema(
        key="fossils",
        short_label="Fossils",
        collection_key="paleontology",
        collection_label="Paleontology",
        relative_path=Path("dataset/paleontology/fossils/p_fossils.txt"),
        identity_values=(
            "https://doi.org/10.15468/vfe3yq",
            "Beaty Biodiversity Museum Fossil Collection",
        ),
    ),
    DatasetSchema(
        key="entomology",
        short_label="Entomology",
        collection_key="spencer_entomology_collection",
        collection_label="Spencer Entomology Collection",
        relative_path=Path(
            "dataset/spencer_entomology_collection/entomology/"
            "sec_entomology.txt"
        ),
        identity_values=(
            "https://doi.org/10.5886/r5d29ft9",
            "University of British Columbia - Spencer Entomological Collection",
        ),
    ),
)


def load_dataset_catalog(repository_root: Path) -> tuple[DatasetSchema, ...]:
    """Load the discovered catalog, falling back to the bundled example set."""
    path = repository_root / DATASET_CATALOG_RELATIVE_PATH
    if not path.exists():
        return DATASETS
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise ValueError(f"{path}: dataset catalog must be a JSON array")
    datasets: list[DatasetSchema] = []
    for item in document:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: dataset catalog entries must be objects")
        try:
            dataset = DatasetSchema(
                key=item["key"],
                short_label=item["short_label"],
                collection_key=item["collection_key"],
                collection_label=item["collection_label"],
                relative_path=Path(item["relative_path"]),
                identity_values=tuple(item.get("identity_values", ())),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"{path}: invalid dataset catalog entry") from error
        datasets.append(dataset)
    if not datasets or len({dataset.key for dataset in datasets}) != len(datasets):
        raise ValueError(f"{path}: dataset catalog has no entries or duplicate keys")
    return tuple(datasets)


def write_dataset_catalog(
    repository_root: Path,
    datasets: tuple[DatasetSchema, ...],
) -> Path:
    """Atomically persist the datasets discovered during archive preparation."""
    path = repository_root / DATASET_CATALOG_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    document = [
        {
            "collection_key": dataset.collection_key,
            "collection_label": dataset.collection_label,
            "identity_values": list(dataset.identity_values),
            "key": dataset.key,
            "relative_path": dataset.relative_path.as_posix(),
            "short_label": dataset.short_label,
        }
        for dataset in datasets
    ]
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path
