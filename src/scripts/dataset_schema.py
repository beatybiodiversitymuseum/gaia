"""Source-file formats and paths used by repository preparation commands."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from src.database.datasets import DatasetSchema


# ---------------------------------------------------------------------------
# Occurrence schema
# ---------------------------------------------------------------------------


CHUNKS = {
    "record_identity_admin": (
        "id",
        "modified",
        "type",
        "language",
        "license",
        "rightsHolder",
        "accessRights",
        "bibliographicCitation",
        "references",
        "institutionID",
        "collectionID",
        "datasetID",
        "institutionCode",
        "collectionCode",
        "datasetName",
        "ownerInstitutionCode",
        "basisOfRecord",
        "informationWithheld",
        "dataGeneralizations",
    ),
    "occurrence_organism": (
        "dynamicProperties",
        "occurrenceID",
        "catalogNumber",
        "recordNumber",
        "recordedBy",
        "individualCount",
        "sex",
        "lifeStage",
        "reproductiveCondition",
        "behavior",
        "vitality",
        "establishmentMeans",
        "occurrenceStatus",
        "preparations",
        "disposition",
        "otherCatalogNumbers",
        "occurrenceRemarks",
        "organismID",
    ),
    "event_sampling": (
        "eventID",
        "fieldNumber",
        "eventDate",
        "eventTime",
        "startDayOfYear",
        "endDayOfYear",
        "year",
        "month",
        "day",
        "verbatimEventDate",
        "habitat",
        "samplingProtocol",
        "samplingEffort",
        "fieldNotes",
        "eventRemarks",
    ),
    "location_georeferencing": (
        "locationID",
        "higherGeographyID",
        "higherGeography",
        "continent",
        "waterBody",
        "islandGroup",
        "island",
        "country",
        "countryCode",
        "stateProvince",
        "county",
        "municipality",
        "locality",
        "verbatimLocality",
        "minimumElevationInMeters",
        "maximumElevationInMeters",
        "verbatimElevation",
        "minimumDepthInMeters",
        "maximumDepthInMeters",
        "verbatimDepth",
        "minimumDistanceAboveSurfaceInMeters",
        "maximumDistanceAboveSurfaceInMeters",
        "locationAccordingTo",
        "locationRemarks",
        "decimalLatitude",
        "decimalLongitude",
        "geodeticDatum",
        "coordinateUncertaintyInMeters",
        "coordinatePrecision",
        "pointRadiusSpatialFit",
        "verbatimCoordinates",
        "verbatimLatitude",
        "verbatimLongitude",
        "verbatimCoordinateSystem",
        "verbatimSRS",
        "footprintWKT",
        "footprintSRS",
        "footprintSpatialFit",
        "georeferencedBy",
        "georeferencedDate",
        "georeferenceProtocol",
        "georeferenceSources",
        "georeferenceVerificationStatus",
        "georeferenceRemarks",
    ),
    "geological_context": (
        "geologicalContextID",
        "earliestEonOrLowestEonothem",
        "latestEonOrHighestEonothem",
        "earliestEraOrLowestErathem",
        "latestEraOrHighestErathem",
        "earliestPeriodOrLowestSystem",
        "latestPeriodOrHighestSystem",
        "earliestEpochOrLowestSeries",
        "latestEpochOrHighestSeries",
        "earliestAgeOrLowestStage",
        "latestAgeOrHighestStage",
        "lowestBiostratigraphicZone",
        "highestBiostratigraphicZone",
        "lithostratigraphicTerms",
        "group",
        "formation",
        "member",
        "bed",
    ),
    "identification": (
        "identificationID",
        "identificationQualifier",
        "typeStatus",
        "identifiedBy",
        "dateIdentified",
        "identificationReferences",
        "identificationVerificationStatus",
        "identificationRemarks",
        "previousIdentifications",
    ),
    "taxonomy_nomenclature": (
        "taxonID",
        "scientificNameID",
        "acceptedNameUsageID",
        "parentNameUsageID",
        "originalNameUsageID",
        "nameAccordingToID",
        "namePublishedInID",
        "taxonConceptID",
        "scientificName",
        "acceptedNameUsage",
        "parentNameUsage",
        "originalNameUsage",
        "nameAccordingTo",
        "namePublishedIn",
        "namePublishedInYear",
        "higherClassification",
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "subgenus",
        "specificEpithet",
        "infraspecificEpithet",
        "cultivarEpithet",
        "taxonRank",
        "verbatimTaxonRank",
        "scientificNameAuthorship",
        "vernacularName",
        "nomenclaturalCode",
        "taxonomicStatus",
        "nomenclaturalStatus",
        "taxonRemarks",
    ),
    "associations_media": (
        "associatedMedia",
        "associatedOccurrences",
        "associatedReferences",
        "associatedSequences",
        "associatedTaxa",
    ),
}


OUTPUT_RECORD_KEYS = (
    "occurrenceID",
    "catalogNumber",
    "chunks",
    "raw_record",
)


# ---------------------------------------------------------------------------
# Source paths and headers
# ---------------------------------------------------------------------------


def read_tsv_header(path: Path) -> tuple[str, ...]:
    """Read and validate one exact tab-delimited source header."""
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)

    if not header:
        raise ValueError(f"{path}: missing header row")

    blank_columns = [
        position for position, name in enumerate(header, start=1) if not name
    ]
    if blank_columns:
        rendered = ", ".join(str(position) for position in blank_columns)
        raise ValueError(f"{path}: blank header(s) at column(s): {rendered}")

    duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
    if duplicates:
        raise ValueError(f"{path}: duplicate header(s): {', '.join(duplicates)}")

    return tuple(header)


def jsonl_relative_path(dataset: DatasetSchema) -> Path:
    """Return the contract-derived JSON Lines path for one source dataset."""
    source = dataset.relative_path
    return source.with_name(f"{source.stem}_chunked.jsonl")
