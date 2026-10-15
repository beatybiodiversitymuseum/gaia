"""Compose Gaia's trusted occurrence, statistics, and workbook tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import MappingProxyType

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql.schema import Table

from src.database.database import REPOSITORY_ROOT
from src.database.datasets import DatasetSchema, load_dataset_catalog
from src.database.darwin_core import DarwinCoreTerm, load_darwin_core_terms
from src.database.tablesetup import build_dataset_table_map, build_table_models
from src.gaia_agent.rag.schema_retrieval import (
    load_schema,
    retrieve_schema_for_registry,
)
from src.gaia_agent.runtime.contracts import (
    AgentCancellation,
    RegisteredTool,
    ToolExecutionError,
)
from src.gaia_agent.runtime.defaults import (
    DEFAULT_MAX_ACTIVE_QUERIES,
    DEFAULT_MAX_WAITING_QUERIES,
)
from src.gaia_agent.tools.concurrency import WorkloadBusyError, WorkloadLimiter
from src.gaia_agent.tools.definitions import (
    SCHEMA_SELECTOR_NAMES,
    build_tool_definitions,
)
from src.gaia_agent.tools.file_generation.generation import generate_excel
from src.gaia_agent.tools.file_generation.workers import ExcelExportWorkers
from src.gaia_agent.tools.darwin_core import (
    get_darwin_core_terms,
    search_darwin_core_terms,
)
from src.gaia_agent.tools.query.budget import ChatQueryBudget
from src.gaia_agent.tools.observations import (
    project_query_occurrences_observation,
    project_query_statistics_observation,
)
from src.gaia_agent.tools.query.execution import (
    QueryExecutionError,
    create_query_engine,
)
from src.gaia_agent.tools.query.plan import (
    QueryPlanError,
    validate_query_plan,
)

from src.gaia_agent.tools.query.statistics_execution import (
    execute_statistics_plan,
)
from src.gaia_agent.tools.query.statistics_plan import (
    validate_statistics_plan,
)


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------


def _reject_unknown_arguments(
    arguments: dict[str, object],
    allowed: frozenset[str],
) -> None:
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        raise ToolExecutionError(
            "invalid_arguments",
            f"Unknown argument(s): {', '.join(unknown)}.",
        )


def _optional_selector_values(
    arguments: dict[str, object],
    name: str,
) -> tuple[str, ...]:
    if name not in arguments:
        return ()
    value = arguments[name]
    if not isinstance(value, list) or not value:
        raise ToolExecutionError(
            "invalid_arguments",
            f"{name} must be a non-empty array of strings.",
        )
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ToolExecutionError(
            "invalid_arguments",
            f"{name} must contain only non-empty strings.",
        )
    typed_values = tuple(value)
    if len(set(typed_values)) != len(typed_values):
        raise ToolExecutionError(
            "invalid_arguments",
            f"{name} must not contain duplicates.",
        )
    return typed_values


# ---------------------------------------------------------------------------
# Gaia tool composition
# ---------------------------------------------------------------------------


def _build_scope_registry(
    datasets: Sequence[DatasetSchema],
) -> Mapping[str, Mapping[str, str]]:
    """Map the RAG collection/dataset hierarchy to canonical table keys."""
    collections: dict[str, dict[str, str]] = {}
    for dataset in datasets:
        collections.setdefault(dataset.collection_label, {})[
            dataset.short_label
        ] = dataset.key
    return MappingProxyType(
        {
            collection: MappingProxyType(members)
            for collection, members in collections.items()
        }
    )


def _create_database_limiter() -> WorkloadLimiter:
    return WorkloadLimiter(
        DEFAULT_MAX_ACTIVE_QUERIES, DEFAULT_MAX_WAITING_QUERIES, label="Queries",
    )


@dataclass(frozen=True, slots=True)
class GaiaToolset:
    """Hold trusted Gaia metadata and construct domain-tool registrations."""

    rag_schema: Mapping[str, object]
    table_registry: Mapping[str, Table]
    scope_registry: Mapping[str, Mapping[str, str]]
    engine_factory: Callable[[], AsyncEngine] = field(
        default=create_query_engine,
        repr=False,
        compare=False,
    )
    export_directory: Path | None = None
    database_limiter: WorkloadLimiter = field(
        default_factory=_create_database_limiter,
        repr=False,
        compare=False,
    )
    export_workers: ExcelExportWorkers = field(
        default_factory=ExcelExportWorkers,
        repr=False,
        compare=False,
    )
    darwin_core_terms: tuple[DarwinCoreTerm, ...] = field(
        default=(),
        repr=False,
    )

    @classmethod
    def load(
        cls,
        *,
        repository_root: Path | None = None,
        datasets: tuple[DatasetSchema, ...] | None = None,
        table_models: Sequence[object] | None = None,
        engine_factory: Callable[[], AsyncEngine] = create_query_engine,
        export_directory: Path | None = None,
        database_limiter: WorkloadLimiter | None = None,
        export_workers: ExcelExportWorkers | None = None,
        darwin_core_terms: tuple[DarwinCoreTerm, ...] | None = None,
    ) -> GaiaToolset:
        """Load trusted metadata and the table registry without connecting."""
        if repository_root is None:
            repository_root = REPOSITORY_ROOT
        if datasets is None:
            datasets = load_dataset_catalog(repository_root)
        if table_models is None:
            table_models = build_table_models(datasets)
        rag_schema = load_schema(repository_root, datasets)
        table_registry = build_dataset_table_map(
            datasets=datasets,
            table_models=table_models,
        )
        return cls(
            rag_schema,
            MappingProxyType(table_registry),
            _build_scope_registry(datasets),
            engine_factory,
            Path(export_directory) if export_directory is not None else None,
            database_limiter if database_limiter is not None else _create_database_limiter(),
            export_workers if export_workers is not None else ExcelExportWorkers(),
            darwin_core_terms if darwin_core_terms is not None else load_darwin_core_terms(),
        )

    def registered_tools(
        self,
        *,
        cancellation: AgentCancellation | None = None,
        export_directory: Path | None = None,
        username: str = "gaia",
    ) -> tuple[RegisteredTool, ...]:
        """Create model-visible tools with a fresh interaction retrieval budget."""
        budget = ChatQueryBudget()
        export_target = (
            Path(export_directory)
            if export_directory is not None
            else self.export_directory
        )
        (
            schema_definition,
            darwin_core_search_definition,
            darwin_core_get_definition,
            occurrence_definition,
            statistics_definition,
            export_definition,
        ) = build_tool_definitions(self.scope_registry)

        return (
            RegisteredTool(
                schema_definition,
                self._retrieve_schema,
            ),
            RegisteredTool(
                darwin_core_search_definition,
                partial(search_darwin_core_terms, terms=self.darwin_core_terms),
            ),
            RegisteredTool(
                darwin_core_get_definition,
                partial(get_darwin_core_terms, terms=self.darwin_core_terms),
            ),
            RegisteredTool(
                occurrence_definition,
                partial(self._query_occurrences, budget=budget),
                observation_projector=project_query_occurrences_observation,
            ),
            RegisteredTool(
                statistics_definition,
                self._query_statistics,
                observation_projector=project_query_statistics_observation,
            ),
            RegisteredTool(
                export_definition,
                partial(
                    self._export_excel,
                    export_target=export_target,
                    username=username,
                    cancellation=cancellation,
                ),
            ),
        )

    async def _export_excel(
        self,
        arguments: dict[str, object],
        *,
        export_target: Path | None,
        username: str,
        cancellation: AgentCancellation | None,
    ) -> object:
        """Generate a workbook within this registration's export session."""
        if export_target is None:
            raise ToolExecutionError(
                "export_session_required",
                "Excel files are unavailable in this conversation.",
            )
        try:
            return await generate_excel(
                arguments,
                table_registry=self.table_registry,
                scope_registry=self.scope_registry,
                engine_factory=self.engine_factory,
                export_directory=export_target,
                username=username,
                cancellation=cancellation,
                database_limiter=self.database_limiter,
                export_workers=self.export_workers,
            )
        except WorkloadBusyError as error:
            raise ToolExecutionError("server_busy", str(error)) from error

    async def _retrieve_schema(self, arguments: dict[str, object]) -> object:
        """Return the selected schema metadata."""
        _reject_unknown_arguments(
            arguments,
            frozenset(SCHEMA_SELECTOR_NAMES),
        )
        datasets = _optional_selector_values(arguments, "datasets")
        collections = _optional_selector_values(arguments, "collections")
        if not (datasets or collections):
            collections = tuple(self.scope_registry)
        try:
            dataset_registry = {
                key: (collection, dataset)
                for collection, members in self.scope_registry.items()
                for dataset, key in members.items()
            }
            return retrieve_schema_for_registry(
                self.rag_schema,
                datasets=datasets,
                collections=collections,
                dataset_registry=dataset_registry,
            )
        except ValueError as error:
            raise ToolExecutionError("invalid_arguments", str(error)) from error

    async def _query_occurrences(
        self,
        arguments: dict[str, object],
        *,
        budget: ChatQueryBudget,
    ) -> object:
        """Validate a complete plan and return internal rows for projection."""
        try:
            plan = validate_query_plan(
                arguments,
                table_registry=self.table_registry,
                scope_registry=self.scope_registry,
            )
        except QueryPlanError as error:
            raise ToolExecutionError("invalid_query_plan", str(error)) from error

        try:
            return await budget.execute(
                plan,
                table_registry=self.table_registry,
                engine_factory=self.engine_factory,
                database_limiter=self.database_limiter,
            )
        except WorkloadBusyError as error:
            raise ToolExecutionError("server_busy", str(error)) from error
        except QueryExecutionError as error:
            raise ToolExecutionError(error.code, str(error)) from error

    async def _query_statistics(
        self,
        arguments: dict[str, object],
    ) -> object:
        """Validate and execute a summary plan directly in PostgreSQL."""
        try:
            plan = validate_statistics_plan(
                arguments,
                table_registry=self.table_registry,
                scope_registry=self.scope_registry,
            )
        except QueryPlanError as error:
            raise ToolExecutionError("invalid_query_plan", str(error)) from error
        try:
            return await execute_statistics_plan(
                plan,
                table_registry=self.table_registry,
                engine_factory=self.engine_factory,
                database_limiter=self.database_limiter,
            )
        except WorkloadBusyError as error:
            raise ToolExecutionError("server_busy", str(error)) from error
        except QueryExecutionError as error:
            raise ToolExecutionError(error.code, str(error)) from error
