"""Build model-visible tool definitions from the active dataset catalog."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from src.gaia_agent.runtime.contracts import ToolDefinition
from src.gaia_agent.tools.query.execution import MAX_CHAT_RECORDS
from src.gaia_agent.tools.query.plan import (
    MAX_DISTINCT_COLUMNS,
    MAX_EXACT_VALUES_PER_HEADER,
    MAX_ORDER_BY_COLUMNS,
    NUMERIC_VALUE_POLICY,
)
from src.gaia_agent.tools.query.statistics_plan import (
    MAX_GROUP_BY_COLUMNS,
    MAX_METRICS,
    METRIC_ALIAS_MAX_LENGTH,
    METRIC_ALIAS_PATTERN,
)
from src.gaia_agent.tools.darwin_core import MAX_EXACT_TERMS, MAX_SEARCH_LIMIT

SCHEMA_SELECTOR_NAMES = ("datasets", "collections")
PUBLIC_PURPOSE_DESCRIPTION = (
    "Concise public explanation of why this call is needed and what evidence "
    "it is expected to establish. Do not include private reasoning or instructions."
)

# ---------------------------------------------------------------------------
# Model-visible JSON Schemas
# ---------------------------------------------------------------------------


def _require_public_purpose(definition: ToolDefinition) -> ToolDefinition:
    """Add the public call-purpose annotation to a tool's root object schema."""
    parameters = deepcopy(definition.parameters)
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        raise ValueError(f"tool {definition.name!r} requires object properties")
    properties["purpose"] = {
        "type": "string",
        "minLength": 1,
        "maxLength": 300,
        "description": PUBLIC_PURPOSE_DESCRIPTION,
    }
    required = parameters.setdefault("required", [])
    if not isinstance(required, list):
        raise ValueError(f"tool {definition.name!r} has invalid required fields")
    required.append("purpose")
    return ToolDefinition(
        name=definition.name,
        description=definition.description,
        parameters=parameters,
        strict=definition.strict,
    )


def _build_schema_parameters(
    scope_registry: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    dataset_names = ", ".join(
        f"{label} ({key})"
        for members in scope_registry.values()
        for label, key in members.items()
    )
    collection_names = ", ".join(scope_registry)
    return {
        "type": "object",
        "description": (
            "Schema selectors; at least one non-empty selector array is required."
        ),
        "properties": {
            "datasets": {
                "type": "array",
                "description": (
                    f"Exact dataset names or canonical keys: {dataset_names}. "
                    "Returns each selected dataset's ordered physical-header "
                    "availability mapping."
                ),
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "uniqueItems": True,
            },
            "collections": {
                "type": "array",
                "description": (
                    f"Exact collection names: {collection_names}. Returns member "
                    "datasets and their ordered physical-header availability mappings."
                ),
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "uniqueItems": True,
            },
        },
        "additionalProperties": False,
    }

_FILTER_SCHEMA: dict[str, object] = {
    "type": "object",
    "description": (
        "Conditions for one exact physical header; at least one operator is required. "
        "Operators on the same header and filters on different headers combine "
        "with AND. Text comparisons are case-sensitive. contains and starts_with "
        "match literal text, including % and _. "
        + NUMERIC_VALUE_POLICY
    ),
    "properties": {
        "exact_any": {
            "type": "array",
            "description": "Case-sensitive literal values combined with OR.",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": MAX_EXACT_VALUES_PER_HEADER,
            "uniqueItems": True,
        },
        "exclude_any": {
            "type": "array",
            "description": "Exact, case-sensitive literal values excluded from matches.",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": MAX_EXACT_VALUES_PER_HEADER,
            "uniqueItems": True,
        },
        "contains": {"type": "string", "minLength": 1},
        "starts_with": {"type": "string", "minLength": 1},
        "is_empty": {
            "type": "boolean",
            "description": "True matches the empty string; false excludes it. Spaces remain text.",
        },
        "gt": {"type": "string"},
        "gte": {"type": "string"},
        "lt": {"type": "string"},
        "lte": {"type": "string"},
        "value_type": {
            "type": "string",
            "enum": ["text", "numeric"],
            "description": (
                "Type for gt/gte/lt/lte only; requires one of these operators "
                "and defaults to text. Bounds remain strings; invalid numeric "
                "bounds are rejected."
            ),
        },
    },
    "minProperties": 1,
    "additionalProperties": False,
}

_QUERY_SCHEMA: dict[str, object] = {
    "type": "object",
    "description": (
        "Occurrence filters and output columns; row limits are independent of "
        "column selection. Every mode prepends one identity column: occurrenceID "
        "when present, otherwise id. When both exist, id is omitted even if "
        "selected. Filters, distinct_by and ordering retain exact physical headers."
    ),
    "properties": {
        "filters": {
            "type": "object",
            "description": (
                "Exact physical headers mapped to filter conditions. "
                "Filters on different headers combine with AND."
            ),
            "additionalProperties": _FILTER_SCHEMA,
        },
        "column_mode": {
            "type": "string",
            "enum": ["default", "all", "selected"],
            "description": (
                "default uses each dataset's configured fields, then "
                "return_columns, filter headers, distinct_by headers and order_by "
                "headers. Supported fields appear once, in that order. all "
                "includes every physical source field and requires return_columns: "
                "'all'. selected uses a non-empty return_columns array; filter "
                "and ordering fields are not added."
            ),
        },
        "return_columns": {
            "anyOf": [
                {"type": "string", "enum": ["all"]},
                {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "uniqueItems": True,
                },
            ],
            "description": (
                "Ordered exact physical headers, or 'all' with column_mode: 'all'. "
                "In default mode, array fields are additions; [] retains configured "
                "and request-related fields. In selected mode, the array is the "
                "complete selection in its given order after the preferred identifier."
            ),
        },
    },
    "required": ["filters", "column_mode", "return_columns"],
    "additionalProperties": False,
    "examples": [
        {
            "filters": {"locality": {"is_empty": False}},
            "column_mode": "default",
            "return_columns": [],
        },
        {
            "filters": {},
            "column_mode": "default",
            "return_columns": ["locality"],
        },
        {
            "filters": {},
            "column_mode": "selected",
            "return_columns": ["scientificName"],
        },
        {
            "filters": {},
            "column_mode": "all",
            "return_columns": "all",
        },
    ],
}

_QUERY_PLAN_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "scope": {
            "type": "object",
            "description": (
                "At least one exact collection or dataset label from "
                "retrieve_schema is required. When both are present, the "
                "dataset must belong to that collection."
            ),
            "properties": {
                "collection": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Optional exact collection label returned by "
                        "retrieve_schema. Without dataset, covers every member "
                        "dataset in the collection."
                    ),
                },
                "dataset": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Optional exact dataset label returned by "
                        "retrieve_schema. It may be used alone when the label "
                        "uniquely identifies a dataset, or with collection."
                    ),
                },
            },
            "additionalProperties": False,
        },
        "query": _QUERY_SCHEMA,
        "result": {
            "type": "object",
            "properties": {
                "distinct_by": {
                    "type": "array",
                    "description": (
                        "Headers whose combined values define one unique "
                        "result. Default and all column modes include these "
                        "fields automatically, applying the preferred identifier "
                        "rule; in selected mode each must "
                        "appear in query.return_columns. Other returned fields come "
                        "from one representative occurrence. A non-empty "
                        "array requires exactly one resolved dataset; an "
                        "empty array disables deduplication."
                    ),
                    "maxItems": MAX_DISTINCT_COLUMNS,
                    "items": {"type": "string", "minLength": 1},
                    "uniqueItems": True,
                },
                "order_by": {
                    "type": "array",
                    "description": (
                        "Result ordering. With distinct_by, it chooses the "
                        "representative occurrence for each unique result and "
                        "then orders those results. Collection results are ordered "
                        "within each dataset."
                    ),
                    "maxItems": MAX_ORDER_BY_COLUMNS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "header": {"type": "string", "minLength": 1},
                            "direction": {"enum": ["asc", "desc"]},
                            "value_type": {
                                "type": "string",
                                "enum": ["text", "numeric"],
                                "description": (
                                    "Ordering type; default text compares source "
                                    "strings. Numeric ordering compares converted "
                                    "decimals and places blank or non-decimal values last."
                                ),
                            },
                        },
                        "required": ["header", "direction"],
                        "additionalProperties": False,
                    },
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum returned rows. With distinct_by, counts unique "
                        "value tuples after deduplication. Occurrence queries "
                        f"share a {MAX_CHAT_RECORDS}-row allowance per interaction."
                    ),
                    "minimum": 1,
                    "maximum": MAX_CHAT_RECORDS,
                },
            },
            "required": ["distinct_by", "order_by", "limit"],
            "additionalProperties": False,
        },
    },
    "required": ["scope", "query", "result"],
    "additionalProperties": False,
}

_STATISTICS_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "scope": _QUERY_PLAN_PARAMETERS["properties"]["scope"],
        "query": {
            "type": "object",
            "properties": {
                "filters": _QUERY_SCHEMA["properties"]["filters"],
                "group_by": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "maxItems": MAX_GROUP_BY_COLUMNS,
                    "uniqueItems": True,
                    "description": (
                        "Exact physical headers produce one row per distinct "
                        "value tuple; [] produces one summary for all matching "
                        "records. Collection groups combine compatible datasets "
                        "before computing metrics."
                    ),
                },
                "metrics": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_METRICS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["count", "count_distinct", "min", "max", "sum", "avg"],
                            },
                            "alias": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": METRIC_ALIAS_MAX_LENGTH,
                                "pattern": METRIC_ALIAS_PATTERN,
                                "description": "Unique output name, different from group headers.",
                            },
                            "header": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Source header; required except for count, which does not accept it.",
                            },
                            "value_type": {
                                "type": "string",
                                "enum": ["text", "numeric"],
                                "description": (
                                    "sum/avg require numeric; min/max accept numeric "
                                    "or text and default to text. count does not "
                                    "accept value_type; count_distinct accepts text only."
                                ),
                            },
                        },
                        "required": ["operation", "alias"],
                        "additionalProperties": False,
                    },
                    "description": (
                        "Metrics over every matching row, before limiting returned "
                        "groups. count counts records; count_distinct counts distinct "
                        "exact field values including the empty string. Numeric "
                        "metrics exclude empty/malformed numeric source values. "
                        "Counts are integers; decimal results are exact strings."
                    ),
                },
            },
            "required": ["filters", "group_by", "metrics"],
            "additionalProperties": False,
        },
        "result": {
            "type": "object",
            "properties": {
                "order_by": {
                    "type": "array",
                    "maxItems": MAX_ORDER_BY_COLUMNS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "header": {"type": "string", "minLength": 1},
                            "direction": {"enum": ["asc", "desc"]},
                        },
                        "required": ["header", "direction"],
                        "additionalProperties": False,
                    },
                    "description": "Ordering by group headers or metric aliases; numeric metrics sort numerically.",
                },
                "limit": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "description": (
                        "Maximum returned groups; null returns all groups. "
                        "Metrics use all matching records. Statistics results "
                        "have no chat retrieval cap."
                    ),
                },
            },
            "required": ["order_by", "limit"],
            "additionalProperties": False,
        },
    },
    "required": ["scope", "query", "result"],
    "additionalProperties": False,
}

_EXPORT_RESULT_PROPERTIES = _QUERY_PLAN_PARAMETERS["properties"]["result"]["properties"]
_EXPORT_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "minItems": 1,
            "description": (
                "Dataset or collection queries to export using the selected "
                "layout. Each dataset may appear only once and gets its own "
                "worksheet. All mode preserves empty and whitespace-only columns, "
                "including when no records match. Default and selected modes omit "
                "columns empty or whitespace-only in every exported row, except "
                "the preferred identifier."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "scope": _QUERY_PLAN_PARAMETERS["properties"]["scope"],
                    "query": _QUERY_SCHEMA,
                    "result": {
                        "type": "object",
                        "properties": {
                            "distinct_by": _EXPORT_RESULT_PROPERTIES["distinct_by"],
                            "order_by": _EXPORT_RESULT_PROPERTIES["order_by"],
                        },
                        "required": ["distinct_by", "order_by"],
                        "additionalProperties": False,
                    },
                },
                "required": ["scope", "query", "result"],
                "additionalProperties": False,
            },
        },
        "limit": {
            "type": ["integer", "null"],
            "minimum": 1,
            "description": (
                "Maximum exported rows, allocated by limit_scope. null exports "
                "all matching records regardless of limit_scope. File generation "
                "has no chat retrieval cap."
            ),
        },
        "limit_scope": {
            "type": "string",
            "enum": ["per_dataset", "total"],
            "description": (
                "per_dataset applies a positive limit "
                "independently to each resolved dataset, including each member "
                "of a collection. total shares one count across "
                "all datasets in query order. A null limit exports all matching "
                "records with either scope. File layout does not change this "
                "allocation."
            ),
        },
        "layout": {
            "type": "string",
            "enum": ["single_workbook", "separate_files"],
            "description": (
                "single_workbook puts each resolved dataset on its own worksheet "
                "in one workbook. separate_files creates one workbook per dataset."
            ),
        },
        "filename": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Optional basename without directories or paths. The tool adds "
                ".xlsx when needed. For "
                "separate_files with multiple datasets, the tool appends each dataset key "
                "to this base. Existing names receive a numbered suffix. "
                "Omitting filename uses username_filenumber.xlsx, numbered "
                "from 1 within the chat session."
            ),
        },
    },
    "required": ["queries", "limit", "limit_scope", "layout"],
    "additionalProperties": False,
}


_OCCURRENCE_DEFINITION = ToolDefinition(
    name="query_occurrences",
    description=(
        "Return filtered, ordered occurrence rows from trusted "
        "datasets for one scope/query/result plan. Supports "
        "column selection and distinct value tuples. Results include "
        "dataset coverage and truncation metadata. Occurrence "
        f"queries share a {MAX_CHAT_RECORDS}-row allowance per interaction."
    ),
    parameters=_QUERY_PLAN_PARAMETERS,
    strict=True,
)

_STATISTICS_DEFINITION = ToolDefinition(
    name="query_statistics",
    description=(
        "Return summary statistics over every matching occurrence: "
        "counts, distinct counts, minima, maxima, sums, and averages, with "
        "optional grouping and ordering. Collection scopes combine "
        "compatible datasets before computing statistics. Optional "
        "output limits bound returned groups. Results include dataset "
        "coverage and truncation metadata; skipped datasets make "
        "coverage incomplete. If no dataset is compatible, "
        "the result has no rows. Statistics results have no chat "
        "retrieval cap."
    ),
    parameters=_STATISTICS_PARAMETERS,
    strict=True,
)

_EXPORT_DEFINITION = ToolDefinition(
    name="generate_excel",
    description=(
        "Create Excel (.xlsx) files from occurrence queries. "
        "Each dataset has its own worksheet. "
        "Supports column selection, ordering, distinct values, and "
        "per-dataset or combined row limits without the chat retrieval "
        "cap. Returns file IDs, filenames, row counts, worksheet metadata, "
        "skipped columns and datasets, and coverage; no occurrence rows."
    ),
    parameters=_EXPORT_PARAMETERS,
    strict=True,
)

_DARWIN_CORE_SEARCH_DEFINITION = ToolDefinition(
    name="search_darwin_core_terms",
    description=(
        "Search the installed Darwin Core standard snapshot by term name, IRI, "
        "label, definition, comments, or examples. Optionally filter terms by "
        "their standard metadata. Returns complete term records."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Case-insensitive text to find; empty lists filtered terms.",
            },
            "term_kind": {
                "type": "string",
                "enum": ["class", "property", "other"],
            },
            "organized_in": {
                "type": "string",
                "minLength": 1,
                "description": "Exact class IRI or local name from organized_in.",
            },
            "status": {
                "type": "string",
                "minLength": 1,
                "description": "Exact standard term status.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_SEARCH_LIMIT,
                "description": "Maximum returned terms; defaults to 20.",
            },
        },
        "additionalProperties": False,
    },
    strict=True,
)

_DARWIN_CORE_GET_DEFINITION = ToolDefinition(
    name="get_darwin_core_terms",
    description=(
        "Return complete installed Darwin Core metadata for exact term IRIs or "
        "case-sensitive local names. Reports missing and ambiguous identifiers."
    ),
    parameters={
        "type": "object",
        "properties": {
            "identifiers": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": MAX_EXACT_TERMS,
                "uniqueItems": True,
            },
        },
        "required": ["identifiers"],
        "additionalProperties": False,
    },
    strict=True,
)


def build_tool_definitions(
    scope_registry: Mapping[str, Mapping[str, str]],
) -> tuple[ToolDefinition, ...]:
    """Build definitions after the active dataset catalog has been loaded."""
    schema_definition = ToolDefinition(
        name="retrieve_schema",
        description=(
            "Return selected schema slices and occurrence and statistics plan "
            "templates. Results include exact scope labels and physical-header "
            "availability mappings: 1 means at least one record contains a "
            "non-whitespace value in that field; 0 means every record is empty "
            "or whitespace-only in that field."
        ),
        parameters=_build_schema_parameters(scope_registry),
        strict=True,
    )
    return tuple(_require_public_purpose(definition) for definition in (
        schema_definition,
        _DARWIN_CORE_SEARCH_DEFINITION,
        _DARWIN_CORE_GET_DEFINITION,
        _OCCURRENCE_DEFINITION,
        _STATISTICS_DEFINITION,
        _EXPORT_DEFINITION,
    ))
