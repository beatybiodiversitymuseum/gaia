"""Shared SQLAlchemy declarations for dataset-specific occurrence tables."""

from __future__ import annotations

import hashlib
import keyword
import re
from collections.abc import Sequence

from sqlalchemy import Column, Index, MetaData, Table, Text, text
from sqlalchemy.orm import DeclarativeBase


NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "pk": "pk_%(table_name)s",
}

_ACRONYM_BOUNDARY = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_CASE_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_NON_IDENTIFIER_CHARACTER = re.compile(r"\W+")


class OrmBase(DeclarativeBase):
    """Own the metadata shared by all Gaia PostgreSQL tables."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def orm_attribute_name(source_header: str) -> str:
    """Return a safe snake_case ORM key for an exact source header name."""

    attribute_name = _ACRONYM_BOUNDARY.sub(r"\1_\2", source_header)
    attribute_name = _CAMEL_CASE_BOUNDARY.sub(r"\1_\2", attribute_name)
    attribute_name = _NON_IDENTIFIER_CHARACTER.sub("_", attribute_name).strip("_")
    attribute_name = attribute_name.lower()

    if not attribute_name:
        raise ValueError("Source headers must contain an identifier character.")
    if attribute_name[0].isdigit():
        attribute_name = f"field_{attribute_name}"
    if keyword.iskeyword(attribute_name):
        attribute_name = f"{attribute_name}_"

    return attribute_name


def build_dataset_table(
    *,
    table_name: str,
    source_headers: Sequence[str],
) -> Table:
    """Build one ORM table with every authoritative source header as a column."""

    exact_headers = tuple(source_headers)
    if not exact_headers:
        raise ValueError(f"{table_name}: source_headers must not be empty")
    if any(not header for header in exact_headers):
        raise ValueError(f"{table_name}: source headers must not be blank")
    if len(set(exact_headers)) != len(exact_headers):
        raise ValueError(f"{table_name}: source headers must be unique")
    if "occurrenceID" not in exact_headers:
        raise ValueError(f"{table_name}: occurrenceID is required")
    if "catalogNumber" not in exact_headers:
        raise ValueError(f"{table_name}: catalogNumber is required")

    attribute_names = tuple(orm_attribute_name(header) for header in exact_headers)
    if len(set(attribute_names)) != len(attribute_names):
        raise ValueError(
            f"{table_name}: source headers collide after ORM key conversion"
        )
    source_columns = [
        Column(
            header,
            Text,
            key=attribute_name,
            nullable=False,
            primary_key=header == "occurrenceID",
            server_default=text("''"),
        )
        for header, attribute_name in zip(
            exact_headers,
            attribute_names,
            strict=True,
        )
    ]

    table = Table(
        table_name,
        OrmBase.metadata,
        *source_columns,
        info={"source_headers": exact_headers},
    )
    index_name = f"ix_{table_name}_catalog_number"
    if len(index_name) > 63:
        digest = hashlib.sha256(table_name.encode("utf-8")).hexdigest()[:8]
        index_name = f"ix_{table_name[:40]}_{digest}_catalog"
    Index(index_name, table.c.catalog_number)

    return table
