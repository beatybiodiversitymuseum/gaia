"""Project internal query results into model-visible observations."""

from __future__ import annotations

from typing import cast

from src.gaia_agent.tools.query.columns import preferred_identifier
from src.gaia_agent.tools.query.execution import (
    DatasetSkip,
    QueryExecutionResult,
)
from src.gaia_agent.tools.query.plan import NUMERIC_VALUE_POLICY
from src.gaia_agent.tools.query.statistics_execution import StatisticsExecutionResult


def _skip_metadata(skip: DatasetSkip) -> dict[str, object]:
    """Render a skipped-dataset reason with every missing header and its count."""
    return {
        "dataset_key": skip.dataset_key,
        "reason": skip.reason,
        "missing_headers": list(skip.missing_headers),
        "missing_header_count": len(skip.missing_headers),
    }


def project_query_occurrences_observation(
    internal_result: object,
) -> dict[str, object]:
    """Project only requested occurrence fields before harness persistence."""
    query_result = cast(QueryExecutionResult, internal_result)
    allowed_headers = frozenset(("dataset_key", *query_result.return_columns))
    projected_records = []
    for row in query_result.records:
        identifier = preferred_identifier(row)
        projected_records.append({
            header: value
            for header, value in row.items()
            if header == identifier
            or (header in allowed_headers and header not in {"occurrenceID", "id"})
        })

    return {
        "records": projected_records,
        "metadata": {
            "selected_dataset_keys": list(query_result.selected_dataset_keys),
            "compatible_dataset_keys": list(query_result.compatible_dataset_keys),
            "queried_dataset_keys": list(query_result.queried_dataset_keys),
            "skipped_datasets": [
                _skip_metadata(skip) for skip in query_result.skipped_datasets
            ],
            "limit": query_result.limit,
            "database_records_returned": len(query_result.records),
            "database_truncated": query_result.database_truncated,
        },
    }


def project_query_statistics_observation(
    internal_result: object,
) -> dict[str, object]:
    """Keep typed summaries and explicit coverage separate from occurrence rows."""
    result = cast(StatisticsExecutionResult, internal_result)
    return {
        "rows": [dict(row) for row in result.rows],
        "metadata": {
            "selected_dataset_keys": list(result.selected_dataset_keys),
            "compatible_dataset_keys": list(result.compatible_dataset_keys),
            "queried_dataset_keys": list(result.queried_dataset_keys),
            "skipped_datasets": [
                _skip_metadata(skip) for skip in result.skipped_datasets
            ],
            "output_columns": list(result.output_columns),
            "group_by": list(result.group_by),
            "metrics": [
                {
                    "operation": metric.operation,
                    "alias": metric.alias,
                    "header": metric.header,
                    "value_type": metric.value_type,
                }
                for metric in result.metrics
            ],
            "limit": result.limit,
            "groups_returned": len(result.rows),
            "database_truncated": result.database_truncated,
            "numeric_value_policy": NUMERIC_VALUE_POLICY,
        },
    }
