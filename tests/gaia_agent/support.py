"""Shared deterministic builders for Gaia backend tests."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from src.gaia_agent.tools.query.execution import QueryExecutionResult
from src.gaia_agent.runtime.contracts import ChatMessage, ContextInfo, ModelTurn, ToolDefinition


class RecordingClient:
    """Return scripted turns or failures and retain each model request."""

    def __init__(self, outcomes: Iterable[ModelTurn | BaseException]) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[
            tuple[tuple[ChatMessage, ...], tuple[ToolDefinition, ...]]
        ] = []

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        max_completion_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> ModelTurn:
        self.requests.append((tuple(messages), tuple(tools)))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def context_info(
        self, *, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition],
        enable_thinking: bool | None = None,
    ) -> ContextInfo:
        return ContextInfo(prompt_tokens=100, context_limit_tokens=131_072)

    async def count_text_tokens(self, text: str) -> int:
        return len(text)


def canadian_bird_plan(
    return_columns: tuple[str, ...],
    *,
    filters: dict[str, object] | None = None,
    limit: int = 5,
) -> dict[str, object]:
    """Build the recurring trusted Canadian-bird QueryPlan."""
    return {
        "scope": {
            "collection": "Cowan Tetrapod Collection",
            "dataset": "Birds",
        },
        "query": {
            "filters": (
                {"country": {"exact_any": ["Canada"]}}
                if filters is None
                else filters
            ),
            "return_columns": list(return_columns),
        },
        "result": {"distinct_by": [], "order_by": [], "limit": limit},
    }


def query_result(
    *,
    dataset_key: str = "birds",
    records: tuple[dict[str, str], ...] = (),
    return_columns: tuple[str, ...] = ("country",),
    limit: int = 5,
    database_truncated: bool = False,
) -> QueryExecutionResult:
    """Build a result for one successfully queried dataset."""
    dataset_keys = (dataset_key,)
    return QueryExecutionResult(
        records=records,
        selected_dataset_keys=dataset_keys,
        compatible_dataset_keys=dataset_keys,
        queried_dataset_keys=dataset_keys,
        skipped_datasets=(),
        return_columns=return_columns,
        limit=limit,
        database_truncated=database_truncated,
    )
