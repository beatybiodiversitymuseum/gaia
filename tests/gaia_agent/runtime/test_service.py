"""Deterministic tests for Gaia's agent composition."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

import src.gaia_agent.runtime.service as service_module
import src.gaia_agent.tools.registry as registry_module
from src.gaia_agent.runtime.service import build_gaia_agent
from src.gaia_agent.runtime.loop import AgentRuntime
from src.gaia_agent.runtime.defaults import CONTEXT_CONTINUATION_PROMPT
from src.gaia_agent.tools import GaiaToolset
from src.gaia_agent.tools.query.statistics_execution import StatisticsExecutionResult
from src.gaia_agent.tools.query.statistics_plan import ValidatedStatisticsPlan
from src.gaia_agent.tools.query.execution import QueryExecutionResult
from src.gaia_agent.runtime.contracts import (
    ChatMessage,
    ContextInfo,
    ModelClientExecutionError,
    ModelTurn,
    RegisteredTool,
    ResponseDisclosureError,
    ToolAttemptsExhaustedError,
    ToolCall,
    ToolExecutionError,
)
from src.database.darwin_core import load_simple_darwin_core_profile
from src.database.datasets import DATASETS
from src.database.schema_profiles import write_dataset_schema_profiles
from src.database.tablesetup import TABLE_MODELS
from src.logger import create_chat_directory
from src.logger.gaia_logger import GaiaLoggerConfig, create_chat_trace_config

from tests.gaia_agent.support import RecordingClient, canadian_bird_plan, query_result


OCCURRENCE_HEADERS = load_simple_darwin_core_profile().headers


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BIRD_RETURN_COLUMNS = ("country", "locality", "scientificName")


@pytest.fixture(autouse=True)
def isolate_example_dataset_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Keep agent contract tests independent of the local generated catalog."""
    profile_root = tmp_path_factory.mktemp("schema-profile")
    write_dataset_schema_profiles(
        profile_root,
        {
            dataset.key: {header: 1 for header in OCCURRENCE_HEADERS}
            for dataset in DATASETS
        },
        datasets=DATASETS,
    )
    original_load = GaiaToolset.load.__func__

    def load_example(cls, **kwargs):
        kwargs["repository_root"] = profile_root
        kwargs.setdefault("datasets", DATASETS)
        kwargs.setdefault("table_models", TABLE_MODELS)
        return original_load(cls, **kwargs)

    monkeypatch.setattr(GaiaToolset, "load", classmethod(load_example))


class _ContextCountingClient(RecordingClient):
    async def context_info(self, *, messages, tools, enable_thinking=None):
        return ContextInfo(
            sum(len(message.content or "") for message in messages), 1_024,
        )


def test_compaction_is_saved_beside_chat_interaction_traces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    summary = "## Chat summary\nEarlier bird request; continue with Canada.\n\n## Tools called\nNone.\n"
    compactor = AsyncMock(return_value=summary)
    monkeypatch.setattr("src.gaia_agent.runtime.loop.compact_context", compactor)
    client = _ContextCountingClient([ModelTurn(content="The answer.")])
    agent = service_module.GaiaAgent(AgentRuntime(client), system_prompt="System.")
    chat_directory = create_chat_directory(tmp_path / "logs", "contextwindow")
    config = create_chat_trace_config(
        chat_directory=chat_directory, interaction=2,
        repository_root=tmp_path, invocation={"username": "contextwindow"},
    )
    history = (
        ChatMessage("system", "System."),
        ChatMessage("user", "Birds. " * 130),
        ChatMessage("assistant", "Previous answer."),
    )

    result = asyncio.run(agent.run("Continue with Canada.", history=history, trace_config=config))

    assert (chat_directory / "history.md").read_text(encoding="utf-8") == summary
    assert config.path.is_file()
    assert result.context == (
        history[0], ChatMessage("user", CONTEXT_CONTINUATION_PROMPT),
        ChatMessage("assistant", summary), ChatMessage("assistant", "The answer."),
    )
    assert result.transcript[:len(history)] == history
    trace = json.loads(config.path.read_text(encoding="utf-8"))
    assert trace["user_prompt"] == "Continue with Canada."
    assert trace["runtime"]["context"] == {
        "prompt_tokens": result.context_info.prompt_tokens, "limit_tokens": 1_024,
    }
    compactor.assert_awaited_once()


@pytest.mark.parametrize("failure_stage", ("model", "trace"))
def test_failed_turn_restores_history_after_successful_compaction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_stage: str,
) -> None:
    replacement = "## Chat summary\nContinue the bird request.\n\n## Tools called\nNone.\n"
    compactor = AsyncMock(return_value=replacement)
    monkeypatch.setattr("src.gaia_agent.runtime.loop.compact_context", compactor)
    outcome = RuntimeError("model failed") if failure_stage == "model" else ModelTurn(content="Answer.")
    client = _ContextCountingClient([outcome])
    agent = service_module.GaiaAgent(AgentRuntime(client), system_prompt="System.")
    config = _session_trace(tmp_path, "contextwindow", tmp_path)
    original = "## Chat summary\r\nEarlier Québec request.\r\n\r\n## Tools called\r\nNone.\r\n".encode()
    history_path = tmp_path / "history.md"
    history_path.write_bytes(original)
    if failure_stage == "trace":
        monkeypatch.setattr(service_module, "write_gaia_trace", Mock(side_effect=OSError("write failed")))
    history = (
        ChatMessage("system", "System."),
        ChatMessage("user", "Birds. " * 130),
        ChatMessage("assistant", "Previous answer."),
    )

    expected_error = ModelClientExecutionError if failure_stage == "model" else OSError
    with pytest.raises(expected_error):
        asyncio.run(agent.run("Continue.", history=history, trace_config=config))

    assert history_path.read_bytes() == original
    compactor.assert_awaited_once()


def test_compaction_keeps_chat_histories_and_files_independent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    summaries = (
        "## Chat summary\nBird request in Canada.\n\n## Tools called\nNone.\n",
        "## Chat summary\nAlgae request in Japan.\n\n## Tools called\nNone.\n",
    )
    compactor = AsyncMock(side_effect=summaries)
    monkeypatch.setattr("src.gaia_agent.runtime.loop.compact_context", compactor)
    client = _ContextCountingClient([ModelTurn(content="Answer.") for _ in range(4)])
    agent = service_module.GaiaAgent(AgentRuntime(client), system_prompt="System.")
    sessions: list[tuple[Path, tuple[ChatMessage, ...]]] = []
    for username, request in (("birds", "Birds in Canada. "), ("algae", "Algae in Japan. ")):
        directory = create_chat_directory(tmp_path / "logs", username)
        config = create_chat_trace_config(
            chat_directory=directory, interaction=1,
            repository_root=tmp_path, invocation={"username": username},
        )
        history = (
            ChatMessage("system", "System."), ChatMessage("user", request * 65),
            ChatMessage("assistant", "Previous answer."),
        )
        result = asyncio.run(agent.run("Continue.", history=history, trace_config=config))
        sessions.append((directory, result.context))

    for index, (directory, history) in enumerate(sessions):
        config = create_chat_trace_config(
            chat_directory=directory, interaction=2,
            repository_root=tmp_path, invocation={"username": "researcher"},
        )
        asyncio.run(agent.run("Continue again.", history=history, trace_config=config))
        request_messages = client.requests[index + 2][0]
        assert ChatMessage("assistant", summaries[index]) in request_messages
        assert ChatMessage("assistant", summaries[1 - index]) not in request_messages
        assert (directory / "history.md").read_text(encoding="utf-8") == summaries[index]
    assert sessions[0][0] != sessions[1][0]
    assert compactor.await_count == 2


def _install_query_handler(
    monkeypatch: pytest.MonkeyPatch,
    handler,
) -> None:
    loaded = GaiaToolset.load()
    registered = tuple(
        RegisteredTool(
            tool.definition,
            handler if tool.definition.name == "query_occurrences" else tool.handler,
            observation_projector=tool.observation_projector,
        )
        for tool in loaded.registered_tools()
    )

    loader = Mock()
    loader.load.return_value.registered_tools.return_value = registered
    monkeypatch.setattr(service_module, "GaiaToolset", loader)


async def _species_statistics(
    plan: ValidatedStatisticsPlan,
    **_arguments: object,
) -> StatisticsExecutionResult:
    assert plan.dataset_keys == ("birds",)
    assert plan.filters[0].header == "scientificName"
    assert plan.filters[0].exact_any == ("Corvus corax",)
    assert plan.group_by == ()
    assert plan.limit is None
    return StatisticsExecutionResult(
        rows=({"total_records": 1_234_567},),
        selected_dataset_keys=plan.dataset_keys,
        compatible_dataset_keys=plan.dataset_keys,
        queried_dataset_keys=plan.dataset_keys,
        skipped_datasets=(),
        output_columns=("total_records",),
        group_by=plan.group_by,
        metrics=plan.metrics,
        limit=plan.limit,
        database_truncated=False,
    )


async def _dataset_records(arguments: dict[str, object]) -> QueryExecutionResult:
    scope = arguments["scope"]
    assert isinstance(scope, dict)
    dataset = scope["dataset"]
    dataset_key = "birds" if dataset == "Birds" else "algae"
    scientific_name = (
        "Corvus caurinus" if dataset == "Birds" else "Fucus gardneri"
    )
    return query_result(
        dataset_key=dataset_key,
        records=tuple(
            {
                "dataset_key": dataset_key,
                "occurrenceID": f"{dataset_key}-{index}",
                "scientificName": scientific_name,
            }
            for index in range(1, 6)
        ),
        return_columns=("scientificName",),
    )


async def _empty_dataset_result(arguments: dict[str, object]) -> QueryExecutionResult:
    scope_keys = {
        (dataset.collection_label, dataset.short_label): dataset.key
        for dataset in DATASETS
    }
    scope = arguments["scope"]
    assert isinstance(scope, dict)
    dataset_key = scope_keys[(scope["collection"], scope["dataset"])]
    return query_result(
        dataset_key=dataset_key,
        return_columns=("scientificName",),
        limit=1,
    )


async def _generate_export_summary(
    arguments: dict[str, object], **_kwargs: object,
) -> dict[str, object]:
    return {
        "layout": arguments["layout"],
        "files": [{"filename": "example.xlsx", "record_count": 1}],
        "record_count": 1,
    }


def _session_trace(directory: Path, username: str, repository_root: Path) -> GaiaLoggerConfig:
    return create_chat_trace_config(
        chat_directory=directory,
        interaction=1,
        repository_root=repository_root,
        invocation={"username": username},
    )


def test_service_runs_schema_lookup_without_setup_or_web_modules() -> None:
    script = """
import asyncio
import importlib.abc
import sys
from pathlib import Path

blocked = ('src.scripts', 'src.gaia_agent.web')
attempts = []

def forbidden(name):
    return any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)

class ImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if forbidden(fullname):
            attempts.append(fullname)
            raise ModuleNotFoundError(f'Setup/web module unavailable: {fullname}')
        return None

sys.meta_path.insert(0, ImportBlocker())

from src.gaia_agent.runtime.contracts import ContextInfo, ModelTurn, ToolCall
from src.gaia_agent.runtime.service import build_gaia_agent

class SchemaClient:
    async def context_info(self, *, messages, tools, enable_thinking=None):
        return ContextInfo(100, 131_072)

    async def count_text_tokens(self, text):
        return len(text)

    async def complete(self, *, messages, tools, max_completion_tokens=None, enable_thinking=None):
        if messages[-1].role == 'tool':
            return ModelTurn(content='Birds schema retrieved.')
        return ModelTurn(tool_calls=(
            ToolCall(
                'schema',
                'retrieve_schema',
                {'datasets': ['Cowan Tetrapod Collection - Birds']},
            ),
        ))

agent = build_gaia_agent(Path.cwd(), SchemaClient())
result = asyncio.run(agent.run('Retrieve the Birds schema.'))
assert result.final_response == 'Birds schema retrieved.'
assert result.tool_results[0].ok
assert result.tool_results[0].result['datasets'][0]['key'] == (
    'cowan_tetrapod_collection_birds'
)
assert not attempts, attempts
assert not any(forbidden(name) for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_agent_completes_model_driven_grounded_query_in_three_model_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    internal_result = query_result(
        records=(
            {
                "dataset_key": "birds",
                "occurrenceID": "bird-1",
                "country": "Canada",
                "locality": "Vancouver",
                "scientificName": "Corvus caurinus",
            },
        ),
        return_columns=BIRD_RETURN_COLUMNS,
    )
    query_handler = AsyncMock(return_value=internal_result)

    _install_query_handler(monkeypatch, query_handler)
    client = RecordingClient(
        (
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        "schema-call",
                        "retrieve_schema",
                        {"datasets": ["birds"]},
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        "query-call",
                        "query_occurrences",
                        canadian_bird_plan(BIRD_RETURN_COLUMNS),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ModelTurn(
                content="I found one Canadian bird occurrence.",
                finish_reason="stop",
            ),
        )
    )
    agent = build_gaia_agent(REPOSITORY_ROOT, client)

    result = asyncio.run(
        agent.run(
            "Find up to five bird occurrences from Canada. "
            "Show scientific name, country, and locality."
        )
    )

    assert result.final_response == "I found one Canadian bird occurrence."
    assert result.model_turns == 3
    assert result.tool_calls == 2
    assert all(tool_result.ok for tool_result in result.tool_results)
    assert [tool.name for tool in client.requests[0][1]] == [
        "retrieve_schema",
        "search_darwin_core_terms",
        "get_darwin_core_terms",
        "query_occurrences",
        "query_statistics",
        "generate_excel",
    ]
    system_content = client.requests[0][0][0].content
    assert system_content == (
        REPOSITORY_ROOT / "src/gaia_agent/runtime/system_prompt.md"
    ).read_text(encoding="utf-8")
    schema_observation = json.loads(client.requests[1][0][-1].content)
    birds_schema = next(
        entry
        for entry in schema_observation["result"]["datasets"]
        if entry["key"] == "birds"
    )
    assert birds_schema["headers"]["occurrenceID"] == 1
    assert birds_schema["headers"]["occurrenceID"] == 1
    assert birds_schema["headers"]["modified"] == 1
    assert schema_observation["result"]["query plan structure"]["result"] == {
        "distinct_by": [],
        "order_by": [],
        "limit": 0,
    }
    model_observation = json.loads(client.requests[2][0][-1].content)
    assert model_observation["result"]["records"] == [
        {
            "dataset_key": "birds",
            "occurrenceID": "bird-1",
            "country": "Canada",
            "locality": "Vancouver",
            "scientificName": "Corvus caurinus",
        }
    ]


def test_agent_retains_exact_species_count_in_model_observation_and_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    execute_statistics = AsyncMock(wraps=_species_statistics)

    monkeypatch.setattr(registry_module, "execute_statistics_plan", execute_statistics)
    client = RecordingClient((
        ModelTurn(tool_calls=(ToolCall(
            "species-total",
            "query_statistics",
            {
                "scope": {"dataset": "Birds"},
                "query": {
                    "filters": {"scientificName": {"exact_any": ["Corvus corax"]}},
                    "group_by": [],
                    "metrics": [{"operation": "count", "alias": "total_records"}],
                },
                "result": {"order_by": [], "limit": None},
            },
        ),)),
        ModelTurn(content="There are 1,234,567 Corvus corax records."),
    ))
    trace_path = tmp_path / "species-count.json"
    agent = build_gaia_agent(
        REPOSITORY_ROOT,
        client,
        trace_config=GaiaLoggerConfig(
            path=trace_path,
            run_id="species-count",
            repository_root=REPOSITORY_ROOT,
        ),
    )

    result = asyncio.run(agent.run("How many records are there for Corvus corax?"))

    assert execute_statistics.await_count == 1
    assert result.tool_calls == 1
    assert result.tool_results[0].ok
    payload = json.loads(client.requests[1][0][-1].content)
    assert payload["result"]["rows"] == [{"total_records": 1_234_567}]
    assert isinstance(payload["result"]["rows"][0]["total_records"], int)
    assert payload["result"]["metadata"]["groups_returned"] == 1
    assert payload["result"]["metadata"]["limit"] is None
    assert payload["result"]["metadata"]["queried_dataset_keys"] == ["birds"]
    assert json.loads(result.transcript[-2].content) == payload
    assert result.tool_results[0].payload() == payload
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert [request["name"] for request in trace["tools_requested"]] == ["query_statistics"]
    assert trace["tools_requested"][0]["returned_result"] == payload
    assert trace["run"]["counts"]["registered_tools"] == 6


def test_agent_submits_independent_dataset_plans_in_one_model_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_handler = AsyncMock(wraps=_dataset_records)

    _install_query_handler(monkeypatch, query_handler)
    algae_plan = {
        "scope": {"collection": "Herbarium", "dataset": "Algae"},
        "query": {"filters": {}, "return_columns": ["scientificName"]},
        "result": {"distinct_by": [], "order_by": [], "limit": 5},
    }
    client = RecordingClient(
        (
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        "schema-call",
                        "retrieve_schema",
                        {"datasets": ["birds", "algae"]},
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        "bird-query",
                        "query_occurrences",
                        canadian_bird_plan(("scientificName",), filters={}),
                    ),
                    ToolCall("algae-query", "query_occurrences", algae_plan),
                ),
                finish_reason="tool_calls",
            ),
            ModelTurn(
                content="I found five Birds and five Algae results.",
                finish_reason="stop",
            ),
        )
    )
    result = asyncio.run(
        build_gaia_agent(REPOSITORY_ROOT, client).run(
            "Return five Birds and five Algae with their scientific names."
        )
    )

    query_arguments = [call.args[0] for call in query_handler.await_args_list]
    assert result.model_turns == 3
    assert result.tool_calls == 3
    assert {
        (arguments["scope"]["collection"], arguments["scope"]["dataset"])
        for arguments in query_arguments
    } == {
        ("Cowan Tetrapod Collection", "Birds"),
        ("Herbarium", "Algae"),
    }
    assert [arguments["result"]["limit"] for arguments in query_arguments] == [5, 5]
    observations = [
        json.loads(message.content)
        for message in client.requests[2][0][-2:]
    ]
    assert [item["result"]["records"][0]["dataset_key"] for item in observations] == [
        "birds",
        "algae",
    ]
    assert [len(item["result"]["records"]) for item in observations] == [5, 5]


def test_gaia_accepts_one_same_turn_query_plan_for_every_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_handler = AsyncMock(wraps=_empty_dataset_result)

    _install_query_handler(monkeypatch, query_handler)
    client = RecordingClient(
        (
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        "schema-call",
                        "retrieve_schema",
                        {"datasets": [dataset.key for dataset in DATASETS]},
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ModelTurn(
                tool_calls=tuple(
                    ToolCall(
                        f"query-{dataset.key}",
                        "query_occurrences",
                        {
                            "scope": {
                                "collection": dataset.collection_label,
                                "dataset": dataset.short_label,
                            },
                            "query": {
                                "filters": {},
                                "return_columns": ["scientificName"],
                            },
                            "result": {
                                "distinct_by": [],
                                "order_by": [],
                                "limit": 1,
                            },
                        },
                    )
                    for dataset in DATASETS
                ),
                finish_reason="tool_calls",
            ),
            ModelTurn(
                content="I checked all twelve datasets.",
                finish_reason="stop",
            ),
        )
    )
    result = asyncio.run(
        build_gaia_agent(REPOSITORY_ROOT, client).run(
            "Return one species from every dataset."
        )
    )

    assert result.model_turns == 3
    assert result.tool_calls == 1 + len(DATASETS)
    query_arguments = [call.args[0] for call in query_handler.await_args_list]
    assert len(query_arguments) == len(DATASETS) == 12
    assert all(arguments["result"]["limit"] == 1 for arguments in query_arguments)


def test_agent_allows_two_failed_tool_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    internal_result = query_result(return_columns=BIRD_RETURN_COLUMNS)
    query_handler = AsyncMock(side_effect=[
        ToolExecutionError("invalid_query_plan", "The QueryPlan is structurally invalid."),
        ToolExecutionError("invalid_query_plan", "The QueryPlan is structurally invalid."),
        internal_result,
    ])

    _install_query_handler(monkeypatch, query_handler)
    client = RecordingClient(
        (
            *(
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            f"query-call-{turn_number}",
                            "query_occurrences",
                            canadian_bird_plan(BIRD_RETURN_COLUMNS),
                        ),
                    ),
                    finish_reason="tool_calls",
                )
                for turn_number in range(1, 4)
            ),
            ModelTurn(
                content="The corrected query completed.",
                finish_reason="stop",
            ),
        )
    )
    result = asyncio.run(
        build_gaia_agent(REPOSITORY_ROOT, client).run(
            "Find five bird occurrences from Canada."
        )
    )

    assert result.model_turns == 4
    assert query_handler.await_count == 3
    assert [tool_result.ok for tool_result in result.tool_results] == [
        False,
        False,
        True,
    ]


def test_agent_stops_on_the_third_failure_and_preserves_its_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    query_handler = AsyncMock(side_effect=ToolExecutionError(
        "invalid_query_plan", "The QueryPlan is structurally invalid.",
    ))

    _install_query_handler(monkeypatch, query_handler)
    client = RecordingClient(
        tuple(
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        f"query-call-{turn_number}",
                        "query_occurrences",
                        canadian_bird_plan(BIRD_RETURN_COLUMNS),
                    ),
                ),
                finish_reason="tool_calls",
            )
            for turn_number in range(1, 5)
        )
    )

    trace_path = tmp_path / "exhausted-attempts.json"
    with pytest.raises(ToolAttemptsExhaustedError) as error:
        asyncio.run(
            build_gaia_agent(
                REPOSITORY_ROOT,
                client,
                trace_config=GaiaLoggerConfig(
                    path=trace_path,
                    run_id="exhausted-attempts",
                    repository_root=REPOSITORY_ROOT,
                ),
            ).run("Find five bird occurrences from Canada.")
        )

    assert query_handler.await_count == 3
    assert len(client.requests) == error.value.model_turns == 3
    assert len(error.value.tool_results) == 3
    assert all(not result.ok for result in error.value.tool_results)
    assert error.value.failed_result is error.value.tool_results[-1]
    assert error.value.registered_tool_count == 6
    assert [
        message.tool_call_id
        for message in error.value.transcript
        if message.role == "tool"
    ] == [
        "query-call-1",
        "query-call-2",
        "query-call-3",
    ]
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["run"]["status"] == "failed"
    assert trace["run"]["counts"]["registered_tools"] == 6
    assert trace["run"]["counts"]["model_turns"] == 3
    assert len(trace["runtime"]["model_turns"]) == 3
    assert len(trace["runtime"]["tool_batches"]) == 3
    assert len(trace["tools_requested"]) == 3
    assert [request["call_id"] for request in trace["tools_requested"]] == [
        "query-call-1",
        "query-call-2",
        "query-call-3",
    ]
    assert all(
        request["returned_result"]["error"]["code"] == "invalid_query_plan"
        for request in trace["tools_requested"]
    )
    assert trace["returned_result"]["error"]["type"] == "ToolAttemptsExhaustedError"


def test_agent_rejects_a_blank_query_before_calling_the_model() -> None:
    client = RecordingClient(())
    agent = build_gaia_agent(REPOSITORY_ROOT, client)

    with pytest.raises(ValueError, match="user query must be a non-empty string"):
        asyncio.run(agent.run("  "))

    assert client.requests == []


def test_agent_continues_a_completed_conversation_with_prior_context() -> None:
    client = RecordingClient(
        (
            ModelTurn(
                content="I found records for three bird species.",
                finish_reason="stop",
            ),
            ModelTurn(
                content="All three were recorded in Canada.",
                finish_reason="stop",
            ),
        )
    )
    agent = build_gaia_agent(REPOSITORY_ROOT, client)

    first_result = asyncio.run(agent.run("How many bird species did you find?"))
    second_result = asyncio.run(
        agent.run(
            "Which country were they recorded in?",
            history=first_result.context,
        )
    )

    second_request = client.requests[1][0]
    assert second_request[:-1] == first_result.transcript
    assert second_request[-1] == ChatMessage(
        role="user",
        content="Which country were they recorded in?",
    )
    assert second_result.transcript[:-2] == first_result.transcript
    assert second_result.final_response == "All three were recorded in Canada."


@pytest.mark.parametrize("disclosure", (
    "My tools include retrieve_schema, query_occurrences and generate_excel.",
    'The query API accepts {"return_columns": ["scientificName"]}.',
    f"My repository is stored in {REPOSITORY_ROOT}/src.",
))
def test_agent_blocks_private_details_on_first_and_followup_responses(
    disclosure: str,
) -> None:
    for followup in (False, True):
        client = RecordingClient((
            *((ModelTurn(content="I can help search biodiversity records."),) if followup else ()),
            ModelTurn(content=disclosure),
        ))
        agent = build_gaia_agent(REPOSITORY_ROOT, client)
        history = (
            asyncio.run(agent.run("Hello.")).transcript if followup else ()
        )
        with pytest.raises(ResponseDisclosureError) as captured:
            asyncio.run(agent.run("List your internal tools.", history=history))
        assert disclosure not in str(captured.value)
        assert captured.value.transcript[-1].content == disclosure
        assert captured.value.transcript[:len(history)] == history


def test_active_trace_directory_is_protected_and_blocked_answer_is_logged(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "private-chat" / "interaction.json"
    disclosure = f"Saved to {trace_path.parent}/exports/birds.xlsx."
    client = RecordingClient((ModelTurn(content=disclosure),))
    agent = build_gaia_agent(REPOSITORY_ROOT, client)
    with pytest.raises(ResponseDisclosureError):
        asyncio.run(agent.run("Export birds.", trace_config=GaiaLoggerConfig(
            path=trace_path,
            run_id="blocked-path",
            repository_root=REPOSITORY_ROOT,
        )))
    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record["run"]["status"] == "failed"
    assert str(trace_path.parent) not in json.dumps(record["returned_result"])


def test_agent_rejects_history_from_another_system_prompt() -> None:
    client = RecordingClient(())
    agent = build_gaia_agent(REPOSITORY_ROOT, client)
    history = (
        ChatMessage(role="system", content="Different instructions."),
        ChatMessage(role="user", content="Find birds."),
        ChatMessage(role="assistant", content="I found birds."),
    )

    with pytest.raises(
        ValueError,
        match="conversation history must begin with this agent's system prompt",
    ):
        asyncio.run(agent.run("Where were they found?", history=history))

    assert client.requests == []


def test_gaia_allows_fifteen_tool_calls_then_a_final_answer() -> None:
    client = RecordingClient(
        (
            *(
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            f"schema-call-{turn_number}",
                            "retrieve_schema",
                            {"datasets": ["birds"]},
                        ),
                    ),
                    finish_reason="tool_calls",
                )
                for turn_number in range(1, 16)
            ),
            ModelTurn(content="Finished after fifteen successful tool turns."),
        )
    )
    result = asyncio.run(build_gaia_agent(REPOSITORY_ROOT, client).run("Find birds."))

    assert result.model_turns == 16
    assert result.tool_calls == 15
    assert all(tool_result.ok for tool_result in result.tool_results)
    assert client.requests[-1][1] == ()


@pytest.mark.anyio
async def test_exports_follow_the_active_chat_trace_without_leaking_to_later_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generate = AsyncMock(wraps=_generate_export_summary)

    monkeypatch.setattr(registry_module, "generate_excel", generate)
    client = RecordingClient(tuple(
        turn
        for index in range(3)
        for turn in (
            ModelTurn(tool_calls=(ToolCall(f"file-{index}", "generate_excel", {
                "queries": [{
                    "scope": {"dataset": "Birds"},
                    "query": {"filters": {}, "return_columns": ["catalogNumber"]},
                    "result": {"distinct_by": [], "order_by": []},
                }],
                "limit": None,
                "limit_scope": "per_dataset",
                "layout": "single_workbook",
            }),)),
            ModelTurn(content="Workbook created."),
        )
    ))
    default_chat = create_chat_directory(tmp_path / "logs", "first_user")
    other_chat = create_chat_directory(tmp_path / "custom_logs", "second_user")


    agent = build_gaia_agent(
        REPOSITORY_ROOT,
        client,
        trace_config=_session_trace(default_chat, "first_user", tmp_path),
    )

    first = await agent.run("Export birds.")
    other = await agent.run(
        "Export birds.",
        trace_config=_session_trace(other_chat, "second_user", tmp_path),
    )
    resumed = await agent.run("Export more birds.")
    assert all(result.tool_results[0].ok for result in (first, other, resumed))
    export_sessions = [
        (call.kwargs["export_directory"], call.kwargs["username"])
        for call in generate.await_args_list
    ]
    assert export_sessions == [
        (default_chat / "exports", "first_user"),
        (other_chat / "exports", "second_user"),
        (default_chat / "exports", "first_user"),
    ]
    for directory, username in ((default_chat, "first_user"), (other_chat, "second_user")):
        saved = json.loads((directory / "interaction_0001.json").read_text())
        assert saved["run"]["invocation"]["username"] == username
