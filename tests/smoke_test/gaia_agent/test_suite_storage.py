"""Offline checks for smoke-run naming, grouped evidence and report storage."""

import asyncio
import hashlib
import json
import re
import shlex
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest


from tests.smoke_test.gaia_agent.scripts import (
    benchmark_common,
    engine_experiments,
    gaia_http_experiments,
    mtp_experiments,
    run_suite,
    summarize_results,
)

CAPTURE_MODEL_IDENTITY = run_suite.capture_model_identity


@pytest.fixture
def suite(monkeypatch, tmp_path):
    modules = SimpleNamespace(
        common=benchmark_common, engine=engine_experiments,
        gaia=gaia_http_experiments, mtp=mtp_experiments,
        runner=run_suite, reports=summarize_results,
    )
    monkeypatch.setattr(modules.common, "REPORT_ROOT", tmp_path / "reports/smoke_test")
    async def capture(run_dir, manifest, config):
        manifest["model_identity"] = {
            "requested_model_id": config.model,
            "resolved_inventory_entry": {"id": config.model, "max_model_len": 131072},
            "context_tokens": 131072, "revision": None,
            "revision_status": "missing_from_endpoint",
        }
        save(run_dir / "suite_manifest.json", manifest)
    monkeypatch.setattr(modules.runner, "capture_model_identity", capture)
    return modules


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def read(path):
    return json.loads(path.read_text())


def mock_clients(monkeypatch, handler):
    factory = partial(httpx.AsyncClient, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, "AsyncClient", factory)


def endpoint(request, *, complete_stream=True, gaia_terminal="response"):
    if request.url.path == "/metrics":
        return httpx.Response(200, text="vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n")
    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "default_model", "max_model_len": 131072}]})
    if request.url.path == "/api/health":
        return httpx.Response(200, json={"status": "healthy"})
    if request.url.path == "/api/chat":
        events = [{"type": "conversation", "conversation_id": "retained-chat"},
                  {"type": gaia_terminal, **({"data": {"interaction": 1, "message": "Québec"}}
                    if gaia_terminal == "response" else {"error": "Context exhausted"})}]
        return httpx.Response(200, text="\n".join(json.dumps(event) for event in events))
    if request.url.path == "/v1/chat/completions":
        events = [{"choices": [{"delta": {"reasoning": "Consider the record."}}]},
                  {"choices": [{"delta": {"content": "Québec"}, "finish_reason": "length"}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                   "metrics": {"generation_time_ms": 100, "speculative_decoding": None}}]
        body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        return httpx.Response(200, text=body + ("data: [DONE]\n\n" if complete_stream else ""))
    raise AssertionError(f"Unexpected HTTP request in storage test: {request.url}")


class _FrozenTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 14, 15, 30, 45, tzinfo=timezone.utc)


async def _isolated_engine_workload(payload, experiment):
    await experiment.isolated("warm_1_user", payload,
                              {"input_tokens": 10, "requested_output_tokens": 2})


async def _chat_storage_workload(gaia, directory, traces, client, started, results):
    await gaia.chat(client, directory, started, "user_01", "storage_user", "Québec",
                          trace_root=traces, results=results)
    return {}


async def _interrupted_phase(visited, evidence, failure_type, directory, workload, config=None):
    level = int(directory.name.removeprefix("users_"))
    visited.append(level)
    assert level in (1, 2)
    summary = phase_record(directory, requests=level)
    rows = read(directory / "results.json")
    if level == 2:
        summary.update(status="cancelled" if failure_type is asyncio.CancelledError else "failed",
                       successful=1)
        rows[-1]["status"] = "cancelled_by_client" if failure_type is asyncio.CancelledError else "failed"
        save(directory / "results.json", rows)
        save(directory / "summary.json", summary)
    for row in rows:
        save(directory / "requests" / row["request_id"] / "result.json", row)
    evidence.update({path: path.read_bytes() for path in directory.rglob("*.json")})
    if level == 2:
        raise failure_type("Interrupted after saving the second level")
    return summary


class _GaiaStages:
    """Track the order and shared seed/pool of the offline workload replacements."""

    def __init__(self, suite) -> None:
        self.suite = suite
        self.order = []
        self.seed = None
        self.pool = None

    async def ladder(self, run, identity, traces, config=None):
        self.order.append(self.suite.common.CONCURRENCY)
        summaries = [phase_record(run / self.suite.common.CONCURRENCY / f"users_{users:02d}", requests=users)
                     for users in (1, 2, 3, 5, 7, 10)]
        save(run / self.suite.common.CONCURRENCY / "summary.json", summaries)
        self.seed = {"conversation_id": "retained-ladder-chat", "username": identity}
        return self.seed

    async def seed_conversation(self, run, identity, traces, config=None):
        self.order.append(self.suite.common.CONVERSATION_SETUP)
        phase_record(run / self.suite.common.CONVERSATION_SETUP)
        self.seed = {"conversation_id": "standalone-seed-chat", "username": identity}
        return self.seed

    async def history(self, run, seed, traces, config=None):
        self.order.append(self.suite.common.CONVERSATION)
        assert seed is self.seed
        return phase_record(run / self.suite.common.CONVERSATION, requests=4, growth=2)

    async def context(self, experiment):
        self.order.append(self.suite.runner.INITIAL_CONTEXT)
        source = experiment.directory / "inputs/manifest.json"
        save(source, [{"identity": experiment.identity, "file": "user_01.json"}])
        self.pool = (source, source.read_bytes())

    async def mixed(self, experiment):
        self.order.append(self.suite.common.INTERFERENCE)

    async def retained(self, experiment, initial_directory):
        self.order.append(self.suite.runner.RETAINED_CONTEXT)
        source, original = self.pool
        assert initial_directory == source.parent.parent
        assert source.read_bytes() == original

    async def engine_phase(self, directory, identity, workload, *, phase_name, config=None):
        await workload(SimpleNamespace(directory=directory, identity=identity))
        requests = 10 if phase_name == self.suite.runner.RETAINED_CONTEXT else 22
        comparisons = [{"group": "cold_10_users", "status": "completed"}] if phase_name == self.suite.runner.INITIAL_CONTEXT else []
        return phase_record(directory, requests=requests, comparisons=comparisons)


class _GrowthReplies:
    """Supply saved-response measurements to the retained-conversation runner."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = iter(rows)
        self.prompts: list[str] = []

    async def __call__(
        self, client, output_dir, start, request_id, username, prompt, conversation_id,
        **kwargs,
    ) -> dict:
        self.prompts.append(prompt)
        row = {
            "request_id": request_id, "status": "success", "prompt": prompt,
            "classification": kwargs["classification"], **next(self.rows),
        }
        kwargs["results"].append(row)
        return row


class _SuiteHandoffs:
    """Record CLI phase transitions and supply resumable configuration answers."""

    def __init__(self, suite, evidence) -> None:
        self.suite = suite
        self.evidence = evidence
        self.seen = []
        self.prompts = []
        self.answers = iter(())

    async def gaia(self, run, trace_root, identity, target, config=None):
        assert target == "all"
        assert trace_root == self.suite.common.REPOSITORY_ROOT / "logs"
        self.seen.append(("gaia", identity))
        phase_record(run / "gaia/ordinary_concurrency/users_01")
        save(run / "gaia/summary.json", {"status": "completed", "requests": 1, "successful": 1})
        print("Gaia storage stage complete", flush=True)

    async def mtp(self, run, identity, mode, path, config=None):
        assert path == self.evidence[mode]
        self.seen.append((mode, identity))
        phase_record(run / "mtp" / mode, mtp=True)
        print(f"MTP {mode} storage stage complete", flush=True)

    def answer(self, prompt):
        self.prompts.append(prompt)
        try:
            return str(next(self.answers))
        except StopIteration:
            raise EOFError from None


async def _record_mtp_run(calls, run, identity, mode, evidence, config=None):
    calls.append((identity, mode, evidence))
    phase_record(run / "mtp" / mode, mtp=True)


def test_default_report_paths_are_repository_anchored(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    common = benchmark_common
    repository = Path(__file__).resolve().parents[3]
    assert common.REPORT_ROOT == repository / "reports/smoke_test"
    assert common.report_directory(tmp_path / "logs/campaign") == common.REPORT_ROOT / "campaign"
    with pytest.raises(ValueError):
        common.report_directory(tmp_path / "Invalid run")


def test_run_paths_are_anchored_and_new_names_do_not_collide(suite, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    repository = Path(__file__).resolve().parents[3]
    assert suite.common.REPOSITORY_ROOT == repository
    assert suite.common.LOG_ROOT == repository / "logs/smoke_test/gaia_agent"

    monkeypatch.setattr(suite.common, "datetime", _FrozenTime)
    monkeypatch.setattr(suite.common, "LOG_ROOT", tmp_path / "logs/smoke_test/gaia_agent")
    first = suite.common.new_run_directory("all")
    save(first / "record.json", {"keep": True})
    second = suite.common.new_run_directory("all")
    assert first.name == "2026-09-14_15-30-45_utc_all"
    assert second.name == first.name + "_2"
    assert suite.common.existing_run_directory(first.name) == first
    assert read(first / "record.json") == {"keep": True}
    orphan = suite.common.REPORT_ROOT / (first.name + "_3") / "findings.md"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("Retained findings — Québec.\n", encoding="utf-8")
    third = suite.common.new_run_directory("all")
    assert third.name == first.name + "_4"
    assert not (suite.common.LOG_ROOT / orphan.parent.name).exists()
    assert orphan.read_text(encoding="utf-8") == "Retained findings — Québec.\n"
    with pytest.raises(ValueError):
        suite.common.existing_run_directory("../outside")


def test_campaign_configuration_is_sanitized_hashed_and_fixed_on_resume(suite, tmp_path):
    run = tmp_path / "campaign"
    config = suite.common.BenchmarkConfig(
        "https://gaia-user:gaia-secret@example.test:9443/root/?token=hidden",
        "https://engine-user:engine-secret@models.test/vllm/?api_key=hidden",
        "cheaper-model",
    )
    manifest = suite.runner.campaign_manifest(run, None, None, "gaia", config)
    encoded = json.dumps(manifest)
    assert "gaia-secret" not in encoded and "engine-secret" not in encoded and "hidden" not in encoded
    assert manifest["configuration"]["snapshot"] == {
        "gaia_url": "https://example.test:9443/root",
        "engine_url": "https://models.test/vllm",
        "requested_model_id": "cheaper-model",
    }
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["configuration"]["sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["workload_configuration"]["sha256"])
    with pytest.raises(ValueError, match="Resume configuration differs"):
        suite.runner.campaign_manifest(run, None, None, "gaia", suite.common.BenchmarkConfig())


def test_model_inventory_identity_records_context_and_missing_revision(monkeypatch, tmp_path):
    mock_clients(monkeypatch, endpoint)
    run = tmp_path / "campaign"
    config = benchmark_common.BenchmarkConfig()
    manifest = run_suite.campaign_manifest(run, None, None, "gaia", config)
    asyncio.run(CAPTURE_MODEL_IDENTITY(run, manifest, config))
    saved = read(run / "suite_manifest.json")["model_identity"]
    assert saved["requested_model_id"] == "default_model"
    assert saved["resolved_inventory_entry"]["max_model_len"] == 131072
    assert saved["context_tokens"] == 131072
    assert saved["revision"] is None
    assert saved["revision_status"] == "missing_from_endpoint"
    assert re.fullmatch(r"[0-9a-f]{64}", saved["inventory_sha256"])
    assert read(run / "model_inventory.json")["data"][0]["id"] == "default_model"


def test_model_inventory_identity_rejects_resolved_model_drift(monkeypatch, tmp_path):
    mock_clients(monkeypatch, endpoint)
    run = tmp_path / "campaign"
    config = benchmark_common.BenchmarkConfig()
    manifest = run_suite.campaign_manifest(run, None, None, "gaia", config)
    manifest["model_identity"] = {
        "requested_model_id": "default_model",
        "resolved_inventory_entry": {"id": "default_model", "max_model_len": 65536},
    }
    with pytest.raises(ValueError, match="Live model inventory differs"):
        asyncio.run(CAPTURE_MODEL_IDENTITY(run, manifest, config))


def test_container_evidence_redaction_covers_environment_and_command_secrets(suite):
    value = {"Config": {"Env": ["SAFE=yes", "API_TOKEN=private"],
                        "Cmd": ["serve", "--api-key", "private", "--port=8000"]},
             "RegistryAuth": "private"}
    redacted = suite.mtp.redact_configuration_evidence(value)
    assert redacted["Config"]["Env"] == ["SAFE=yes", "API_TOKEN=[REDACTED]"]
    assert redacted["Config"]["Cmd"] == ["serve", "--api-key", "[REDACTED]", "--port=8000"]
    assert redacted["RegistryAuth"] == "[REDACTED]"
    assert "private" not in json.dumps(redacted)


@pytest.mark.parametrize("target", ["gaia", "mtp"])
def test_completed_section_is_reused_without_overwriting_records(suite, monkeypatch, tmp_path, target):
    run = tmp_path / "campaign"
    manifest = suite.runner.campaign_manifest(run, None, None, target)
    asyncio.run(suite.runner.capture_model_identity(run, manifest, suite.common.BenchmarkConfig()))
    directory = run / "mtp/off" if target == "mtp" else run / "gaia"
    directory.mkdir(parents=True)
    if target == "gaia":
        (run / ".gaia_started").mkdir()
    save(directory / "summary.json", {"status": "completed", "requests": 1, "successful": 1})
    save(directory / "requests/prior/result.json", {"content": "Keep this measured result"})
    before = {path: path.read_bytes() for path in run.rglob("*.json")}
    gaia, mtp = AsyncMock(), AsyncMock()
    monkeypatch.setattr(suite.runner, "run_gaia", gaia)
    monkeypatch.setattr(suite.runner, "run_mode", mtp)
    arguments = {"mtp_mode": "off", "config_evidence": tmp_path / "inspect.json"} if target == "mtp" else {}
    result = asyncio.run(suite.runner.run(run, target=target, **arguments))
    assert result["status"] == "completed"
    assert Path(result["report"]) == suite.common.REPORT_ROOT / run.name / "report.json"
    assert Path(result["report"]).is_file()
    assert suite.runner.campaign_manifest(run, None, None, target)["identity"] == manifest["identity"]
    gaia.assert_not_awaited()
    mtp.assert_not_awaited()
    assert all(path.read_bytes() == value for path, value in before.items())
    assert not (run / ".section_running").exists()


@pytest.mark.parametrize("complete_stream", [True, False])
def test_engine_request_and_partial_error_use_grouped_paths(suite, monkeypatch, tmp_path, complete_stream):
    mock_clients(monkeypatch, partial(endpoint, complete_stream=complete_stream))
    directory = tmp_path / "gaia/context_cache/initial"
    payload = {"model": "default_model", "messages": [{"role": "user", "content": "Québec"}],
               "max_completion_tokens": 2, "stream": True}

    workload = partial(_isolated_engine_workload, payload)

    operation = suite.engine.run_phase(directory, "storage", workload, phase_name="context")
    if complete_stream:
        asyncio.run(operation)
    else:
        with pytest.raises(RuntimeError, match="failed"):
            asyncio.run(operation)
    request_dir = directory / "requests/warm_1_user"
    assert read(request_dir / "request.json") == payload
    row = read(request_dir / "result.json")
    assert row["status"] == ("success" if complete_stream else "failed")
    assert row["reasoning"] == "Consider the record."
    assert row["content"] == "Québec"
    assert ("[DONE]" in (request_dir / "events.jsonl").read_text()) == complete_stream
    assert {path.name for path in (directory / "metrics").iterdir()} == {
        "before.prom", "after.prom", "warm_1_user.before.prom", "warm_1_user.after.prom", "samples.jsonl"}
    assert read(directory / "environment/models.json")["data"][0]["max_model_len"] == 131072
    assert read(directory / "results.json")[0]["status"] == row["status"]
    assert read(directory / "summary.json")["status"] == ("completed" if complete_stream else "failed")


@pytest.mark.parametrize("terminal", ["response", "error"])
def test_gaia_copies_success_and_failed_traces_into_request_directory(suite, monkeypatch, tmp_path, terminal):
    mock_clients(monkeypatch, partial(endpoint, gaia_terminal=terminal))
    traces = tmp_path / "service_traces"
    source = traces / "gaia_chat_storage_user_20260914/interaction_0001.json"
    trace = {"run": {"trace_version": 5, "status": "completed" if terminal == "response" else "failed"},
             "returned_result": {"message": "Québec"},
             "runtime": {"tool_batches": [],
                         "context": {"prompt_tokens": 50, "limit_tokens": 131072},
                         "model_turns": [
                             {"turn": 1, "purpose": "compaction", "token_usage": {"prompt_tokens": 300}},
                             {"turn": 2, "token_usage": {"prompt_tokens": 10}},
                             {"turn": 3, "purpose": "compaction", "token_usage": {"prompt_tokens": 400}},
                         ]}}
    save(source, trace)
    original = source.read_bytes()
    directory = tmp_path / "gaia/ordinary_concurrency/users_01"

    workload = partial(_chat_storage_workload, suite.gaia, directory, traces)

    summary = asyncio.run(suite.gaia.phase(directory, workload))
    request_dir = directory / "requests/user_01"
    row = read(request_dir / "result.json")
    assert summary["status"] == ("completed" if terminal == "response" else "failed")
    assert read(request_dir / "request.json") == {"username": "storage_user", "message": "Québec"}
    assert row["conversation_id"] == "retained-chat"
    assert Path(row["trace_path"]) == request_dir / "trace.json"
    assert (request_dir / "trace.json").read_bytes() == original == source.read_bytes()
    assert row["trace_run"]["status"] == trace["run"]["status"]
    assert row["active_context"] == {"prompt_tokens": 50, "limit_tokens": 131072}
    assert row["first_model_input_tokens"] == 10
    if terminal == "error":
        assert row["failed_trace"] == trace
    assert (request_dir / "events.jsonl").is_file()
    assert {path.name for path in (directory / "metrics").iterdir()} == {
        "before.prom", "after.prom", "samples.jsonl"}


@pytest.mark.parametrize("model_runtime,context_metadata,expected_tokens", [
    ([{"token_usage": {"total_tokens": 100}}, {"token_usage": {"total_tokens": 250}}], {}, None),
    ([], {}, None),
    ([{"token_usage": {"total_tokens": 100}}, {}], {}, None),
    ([{"token_usage": {"total_tokens": 100}},
      {"purpose": "compaction", "token_usage": {"total_tokens": 250}}], {}, None),
    ([{"purpose": "compaction", "token_usage": {"total_tokens": 250}}], {}, None),
    ([{"token_usage": {"total_tokens": 100}},
      {"purpose": "compaction", "token_usage": {"total_tokens": 250}}],
     {"active_context": {"prompt_tokens": 30, "limit_tokens": 131072}}, 30),
    ([{"token_usage": {"total_tokens": 100}}],
     {"active_context": {"prompt_tokens": True, "limit_tokens": 131072}}, None),
])
def test_gaia_context_measurements_require_active_context_from_trace(
        suite, monkeypatch, tmp_path, model_runtime, context_metadata, expected_tokens):
    run = tmp_path / "campaign"
    directory = run / suite.common.CONVERSATION
    row = {"request_id": "turn_002", "status": "success", "elapsed_seconds": 2,
           "username": "storage_user", "conversation_id": "retained-chat", "prompt": "Québec",
           "classification": "synthetic_context_acknowledgment", "data": {"interaction": 2},
           "model_runtime": model_runtime, **context_metadata}
    seed = suite.gaia.conversation_seed(row, directory)
    assert seed["used_tokens"] == expected_tokens

    chat = AsyncMock(return_value=row)
    monkeypatch.setattr(suite.gaia, "chat", chat)
    growth = suite.gaia._ConversationGrowth(directory, {**seed, "used_tokens": 1}, tmp_path / "traces")
    asyncio.run(growth.submit(None, 0, [], row["prompt"], row["classification"]))
    assert growth.used_tokens == expected_tokens

    save(directory / "results.json", [row])
    save(directory / "summary.json", {"status": "completed", "wall_seconds": 2})
    analysis = suite.reports.summarize(run)
    assert analysis[suite.common.CONVERSATION]["observed_context_total_tokens"] == [
        {"request_id": row["request_id"], "tokens": expected_tokens}]
    assert analysis[suite.common.CONVERSATION]["active_context_from_traces"] == (
        [{"request_id": row["request_id"], **context_metadata["active_context"]}]
        if context_metadata else []
    )

    if expected_tokens is None:
        with pytest.raises(RuntimeError, match="Active context measurement is required"):
            asyncio.run(growth.run(None, 0, []))
        chat.assert_awaited_once()


def test_conversation_growth_stops_when_final_traced_usage_is_missing(suite, monkeypatch, tmp_path):
    row = {"request_id": "turn_002", "status": "success",
           "model_runtime": [{"token_usage": {"total_tokens": 100}}, {}]}
    chat = AsyncMock(return_value=row)
    monkeypatch.setattr(suite.gaia, "chat", chat)
    seed = {"interaction": 1, "used_tokens": 50,
            "username": "storage_user", "conversation_id": "retained-chat"}
    growth = suite.gaia._ConversationGrowth(tmp_path, seed, tmp_path / "traces")

    with pytest.raises(RuntimeError, match="Reported context stopped growing"):
        asyncio.run(growth.run(None, 0, []))

    assert growth.used_tokens is None
    chat.assert_awaited_once()


@pytest.mark.parametrize("compacted_tokens", (3000, 71000))
def test_conversation_growth_stops_at_compaction_and_runs_followups(
    suite, monkeypatch, tmp_path, compacted_tokens,
):
    events = [
        {"elapsed_seconds": 1, "event": {"type": "progress", "phase": "context_compacting"}},
        {"elapsed_seconds": 2, "event": {"type": "progress", "phase": "context_compacted"}},
    ]
    trace_path = str(tmp_path / "requests/turn_003/trace.json")
    replies = _GrowthReplies([
        {"active_context": {"prompt_tokens": 70000, "limit_tokens": 131072}},
        {"active_context": {"prompt_tokens": compacted_tokens, "limit_tokens": 131072},
         "model_runtime": [{"purpose": "compaction", "token_usage": {"total_tokens": 130000}}],
         "events": events, "trace_path": trace_path},
        {"active_context": {"prompt_tokens": 72000, "limit_tokens": 131072}},
        {"active_context": {"prompt_tokens": 73000, "limit_tokens": 131072}},
    ])
    monkeypatch.setattr(suite.gaia, "chat", replies)
    seed = {"interaction": 1, "used_tokens": 50,
            "username": "storage_user", "conversation_id": "retained-chat"}
    growth = suite.gaia._ConversationGrowth(tmp_path, seed, tmp_path / "traces")
    rows = []
    details = asyncio.run(growth.run(None, 0, rows))

    assert details["growth_stop_reason"] == "context_compacted"
    assert details["growth_turns"] == 2
    assert details["followups"] == 2
    assert details["last_observed_total_context_tokens"] == 73000
    assert [row["classification"] for row in rows] == [
        "synthetic_context_acknowledgment", "synthetic_context_acknowledgment",
        "owned_observation_reuse", "new_dataset_query",
    ]
    assert "same five Fungi occurrence records" in replies.prompts[2]
    assert replies.prompts[3] == suite.gaia.ordinary_prompt(0)
    assert [checkpoint["target_total_context_tokens"] for checkpoint in details["checkpoints"]] == [8192, 32768, 65536]
    expected = [{"request_id": "turn_003", "previous_context_tokens": 70000,
                 "active_context_tokens": compacted_tokens,
                 "events": [events[1]], "trace_path": trace_path}]
    assert details["compactions"] == read(tmp_path / "growth_compactions.json") == expected
    assert read(tmp_path / "growth_checkpoints.json") == details["checkpoints"]


def test_conversation_growth_still_stops_at_final_checkpoint(suite, monkeypatch, tmp_path):
    replies = _GrowthReplies([
        {"active_context": {"prompt_tokens": tokens, "limit_tokens": 131072}}
        for tokens in (123000, 123100, 123200)
    ])
    monkeypatch.setattr(suite.gaia, "chat", replies)
    seed = {"interaction": 1, "used_tokens": 50,
            "username": "storage_user", "conversation_id": "retained-chat"}
    growth = suite.gaia._ConversationGrowth(tmp_path, seed, tmp_path / "traces")
    details = asyncio.run(growth.run(None, 0, []))
    assert details["growth_stop_reason"] == "checkpoint_reached"
    assert details["growth_turns"] == 1
    assert details["compactions"] == []
    assert details["followups"] == 2


@pytest.mark.parametrize("context_tokens", (50, 30))
def test_conversation_growth_rejects_stalled_or_reduced_usage_without_compaction(
    suite, monkeypatch, tmp_path, context_tokens,
):
    replies = _GrowthReplies([{
        "active_context": {"prompt_tokens": context_tokens, "limit_tokens": 131072},
        "events": [{"event": {"type": "progress", "phase": "context_compacting"}}],
    }])
    monkeypatch.setattr(suite.gaia, "chat", replies)
    seed = {"interaction": 1, "used_tokens": 50,
            "username": "storage_user", "conversation_id": "retained-chat"}
    growth = suite.gaia._ConversationGrowth(tmp_path, seed, tmp_path / "traces")
    with pytest.raises(RuntimeError, match="Reported context stopped growing"):
        asyncio.run(growth.run(None, 0, []))
    assert len(replies.prompts) == 1


def test_conversation_growth_has_bounded_requests(suite, monkeypatch, tmp_path):
    replies = _GrowthReplies([
        {"active_context": {"prompt_tokens": tokens, "limit_tokens": 131072}}
        for tokens in (51, 52)
    ])
    monkeypatch.setattr(suite.gaia, "chat", replies)
    monkeypatch.setattr(suite.gaia, "MAX_GROWTH_TURNS", 2)
    seed = {"interaction": 1, "used_tokens": 50,
            "username": "storage_user", "conversation_id": "retained-chat"}
    growth = suite.gaia._ConversationGrowth(tmp_path, seed, tmp_path / "traces")
    with pytest.raises(RuntimeError, match="request limit without compaction"):
        asyncio.run(growth.run(None, 0, []))
    assert len(replies.prompts) == 2


def test_compaction_evidence_and_model_purposes_are_retained_in_analysis(suite, tmp_path):
    run = tmp_path / "campaign"
    directory = run / suite.common.CONVERSATION
    event = {"elapsed_seconds": 2, "event": {"type": "progress", "phase": "context_compacted"}}
    row = {"request_id": "turn_003", "status": "success", "elapsed_seconds": 3,
           "classification": "synthetic_context_acknowledgment", "events": [event],
           "active_context": {"prompt_tokens": 3000, "limit_tokens": 131072},
           "model_runtime": [{"turn": 1, "token_usage": {"total_tokens": 120000}},
                             {"turn": 2, "purpose": "compaction", "token_usage": {"total_tokens": 130000}}]}
    save(directory / "results.json", [row])
    save(directory / "summary.json", {"status": "completed", "wall_seconds": 3})
    analysis = suite.reports.summarize(run)[suite.common.CONVERSATION]
    assert analysis["context_compactions"] == [
        {"request_id": "turn_003", "events": [event], "trace_path": None}]
    assert [turn["purpose"] for turn in analysis["gaia_provider_metrics_seconds"]] == ["response", "compaction"]
    assert analysis["observed_context_total_tokens"] == [{"request_id": "turn_003", "tokens": 3000}]


def test_retention_reads_original_pool_from_inputs_directory(suite, monkeypatch, tmp_path):
    initial = tmp_path / "gaia/context_cache/initial"
    manifest, expected = [], []
    for index in range(1, 11):
        payload = {"messages": [{"role": "user", "content": f"Original Québec {index}"}]}
        details = {"input_tokens": 130000, "requested_output_tokens": 256}
        name = f"user_{index:02d}.json"
        save(initial / "inputs" / name, payload)
        manifest.append({"file": name, "details": details})
        expected.append((payload, details, None))
    save(initial / "inputs/manifest.json", manifest)
    assert suite.engine.load_pool(initial) == expected
    replay = AsyncMock(return_value=[])
    monkeypatch.setattr(suite.engine, "simultaneous", replay)
    experiment = suite.engine.EngineExperiment(None, tmp_path / "gaia/context_cache/retention", "storage")
    asyncio.run(suite.engine.retain_pool(experiment, initial))
    replay.assert_awaited_once_with(experiment, "retention_10_users", expected,
                                   condition="original_pool_after_growth_and_interference")
    source = read(experiment.inputs_dir / "pool_source.json")
    assert source["initial_directory"] == str(initial.resolve())
    assert source["manifest_sha256"] == hashlib.sha256((initial / "inputs/manifest.json").read_bytes()).hexdigest()


def phase_record(directory, *, mtp=False, requests=1, growth=0, comparisons=None):
    rows = [{"request_id": f"user_{index + 1:02d}", "status": "success", "elapsed_seconds": 2,
             "start_offset_seconds": 0,
             **({"classification": "synthetic_context_acknowledgment"} if index < growth else {})}
            for index in range(requests)]
    if mtp:
        rows = [{**rows[0], "request_id": f"{role}_users_01_user_01", "role": role,
                 "group": f"{role}_users_01", "measurement_status": "verified",
                 "usage": {"completion_tokens": tokens},
                 "provider_metrics_raw": {"generation_time_ms": 10000, "speculative_decoding": None}}
                for role, tokens in [("warmup", 32), ("measured", 512)]]
        comparisons = [{"group": "measured_users_01", "status": "completed", "wall_seconds": 2}]
    save(directory / "results.json", rows)
    summary = {"status": "completed", "requests": len(rows), "successful": len(rows),
               "wall_seconds": 2, "comparisons": comparisons or [], "growth_requests": growth}
    save(directory / "summary.json", summary)
    return summary


@pytest.mark.parametrize("failure_type", [
    pytest.param(RuntimeError, id="failure"),
    pytest.param(asyncio.CancelledError, id="cancellation"),
])
def test_interrupted_ladder_counts_saved_partial_level(suite, monkeypatch, tmp_path, failure_type):
    run = tmp_path / "campaign"
    manifest = suite.runner.campaign_manifest(run, None, None, "concurrency")
    visited, evidence = [], {}

    partial_phase = partial(_interrupted_phase, visited, evidence, failure_type)

    monkeypatch.setattr(suite.gaia, "phase", partial_phase)
    operation = suite.runner.run_gaia(run, tmp_path / "traces", manifest["identity"], "concurrency")
    if failure_type is asyncio.CancelledError:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(operation)
    else:
        assert asyncio.run(operation)["status"] == "incomplete"
    summary = read(run / "gaia/summary.json")
    analysis = suite.reports.summarize(run)
    assert visited == [1, 2]
    assert summary["requests"] == sum(value["recorded_requests"] for value in analysis.values()) == 3
    assert summary["successful"] == sum(value["successful"] for value in analysis.values()) == 2
    assert summary["status"] == ("cancelled" if failure_type is asyncio.CancelledError else "incomplete")
    assert all(path.read_bytes() == original for path, original in evidence.items())


@pytest.fixture
def gaia_stages(suite, monkeypatch):
    """Replace workload boundaries while exercising the real category orchestrator."""
    state = _GaiaStages(suite)
    for name, replacement in {
        "ladder": state.ladder, "seed_conversation": state.seed_conversation,
        "gaia_history": state.history, "context_pool": state.context,
        "mixed_workload": state.mixed, "retain_pool": state.retained,
        "run_phase": state.engine_phase,
    }.items():
        monkeypatch.setattr(suite.runner, name, replacement)
    return state


@pytest.mark.parametrize("target,phases,fixed,growth", [
    ("all", ["ordinary_concurrency", "context_cache/initial", "growing_conversation",
             "prefill_interference", "context_cache/retention"], 84, 2),
    ("gaia", ["ordinary_concurrency", "context_cache/initial", "growing_conversation",
              "prefill_interference", "context_cache/retention"], 84, 2),
    ("concurrency", ["ordinary_concurrency"], 28, 0),
    ("context-cache", ["context_cache/initial"], 22, 0),
    ("conversation-growth", ["conversation_setup", "growing_conversation"], 3, 2),
    ("prefill-interference", ["prefill_interference"], 22, 0),
])
def test_category_runs_selected_phases_and_reuses_full_run_preparation(
        suite, gaia_stages, tmp_path, target, phases, fixed, growth):
    run = tmp_path / "campaign"
    manifest = suite.runner.campaign_manifest(run, None, None, target)
    result = asyncio.run(suite.runner.run_gaia(run, tmp_path / "traces", manifest["identity"], target))
    expected = ["gaia/" + name for name in phases]
    assert gaia_stages.order == manifest["gaia_phases"] == expected
    assert result["status"] == "completed"
    assert result["requests"] == result["successful"] == fixed + growth
    assert result["growth_requests"] == growth
    assert manifest["gaia_fixed_requests"] == fixed
    assert manifest["planned_total_fixed_requests"] == fixed + (208 if target == "all" else 0)
    assert manifest["planned_warmup_requests"] == (104 if target == "all" else 0)
    assert set(result["stages"]) == set(expected)
    assert (gaia_stages.pool is not None) == ("context_cache/initial" in phases)
    summary = suite.runner.update_report(run, manifest)
    assert summary["status"] == ("pending" if target == "all" else "completed")
    assert summary["requests"] == fixed + growth
    report = read(suite.common.REPORT_ROOT / run.name / "report.json")
    completion = report["sections"][0]["tables"][0]["rows"]
    assert all(row[2] == "completed" for row in completion)
    if target != "all":
        assert report["sections"][1]["status"] == "Not selected."


@pytest.mark.parametrize("custom_output", [False, True])
def test_new_phase_discovery_and_combined_reports_exclude_warmups(suite, tmp_path, custom_output):
    run = tmp_path / "logs/smoke_test/gaia_agent/campaign"
    readme = tmp_path / "README.md"
    original_readme = "# Gaia — Québec\r\n\r\nHandwritten documentation.\r\n".encode("utf-8")
    readme.write_bytes(original_readme)
    manifest = suite.runner.campaign_manifest(run, None, None)
    phases = ["gaia/ordinary_concurrency/users_01", "gaia/context_cache/initial",
              "gaia/context_cache/retention", "gaia/growing_conversation", "gaia/prefill_interference", "mtp/off"]
    for name in phases:
        phase_record(run / name, mtp=name.startswith("mtp/"))
    phase_record(run / "unrelated")
    evidence = {path: path.read_bytes() for path in run.rglob("*.json")}
    output = tmp_path / "custom_reports" if custom_output else None
    destination = output or suite.common.REPORT_ROOT / run.name
    analysis = suite.reports.summarize(run, output_dir=output)
    assert set(analysis) == set(phases)
    cells = analysis["mtp/off"]["mtp_cells"]
    assert len(cells) == 1 and cells[0]["requests"] == 1
    assert cells[0]["decode_tps"] == pytest.approx(51.1)
    assert cells[0]["batch_output_tps"] == 256
    summary = suite.runner.campaign_summary(run, manifest, analysis)
    findings = suite.reports.write_suite_report(run, analysis, summary, output_dir=output)
    assert findings == destination / "findings.md"
    assert {path.name for path in destination.iterdir()} == {
        "analysis.json", "comparison.csv", "report.json", "findings.md"}
    report = read(destination / "report.json")
    assert len(report["sections"]) == 2
    assert "1/1 measured requests succeeded" in report["sections"][1]["status"]
    for section in report["sections"]:
        assert all(Path(path).is_file() for path in section["sources"])
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", findings.read_text())
    assert links and all(not Path(link).is_absolute() for link in links)
    assert {(findings.parent / link).resolve() for link in links} == {
        (destination / "analysis.json").resolve(), (run / "mtp/off/summary.json").resolve()}
    assert all(path.read_bytes() == original for path, original in evidence.items())
    assert readme.read_bytes() == original_readme
    assert not (run / "reports").exists()
    assert not (run / "analysis.json").exists()
    if custom_output:
        assert not (suite.common.REPORT_ROOT / run.name).exists()


def test_failed_report_replacement_preserves_prior_report(suite, monkeypatch):
    destination = suite.common.REPORT_ROOT / "campaign/findings.md"
    destination.parent.mkdir(parents=True)
    original = "# Retained findings — Québec\n".encode("utf-8")
    destination.write_bytes(original)

    failed_replace = Mock(side_effect=OSError("simulated report replacement failure"))

    monkeypatch.setattr(Path, "replace", failed_replace)
    with pytest.raises(OSError, match="simulated report replacement failure"):
        suite.reports.write_report_text(destination, "# Replacement findings\n")

    assert destination.read_bytes() == original
    assert list(destination.parent.iterdir()) == [destination]


def test_cli_all_handoffs_resume_and_append_visible_log(suite, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(suite.common, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.chdir(tmp_path)
    evidence = {mode: tmp_path / f"{mode}_inspect.json" for mode in ("off", "k1", "k2", "k3")}
    for mode, path in evidence.items():
        save(path, {"saved_mode": mode})

    handoffs = _SuiteHandoffs(suite, evidence)
    seen, prompts = handoffs.seen, handoffs.prompts
    handoffs.answers = iter([evidence["off"]])

    monkeypatch.setattr(suite.runner, "run_gaia", handoffs.gaia)
    monkeypatch.setattr(suite.runner, "run_mode", handoffs.mtp)
    monkeypatch.setattr("builtins.input", handoffs.answer)
    monkeypatch.setattr(sys, "argv", ["run_suite.py", "all"])
    with pytest.raises(SystemExit) as first:
        suite.runner.main()
    assert first.value.code == 130
    visible = capsys.readouterr().out
    run, = (tmp_path / "logs").iterdir()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_utc_all", run.name)
    initial_log = (run / "runner.log").read_text()
    assert visible == initial_log and "Gaia storage stage complete" in visible
    interrupted = json.loads(visible.splitlines()[-1])
    assert shlex.split(interrupted["resume"])[:6] == [
        sys.executable, "-m", "tests.smoke_test.gaia_agent.scripts.run_suite",
        "all", "--resume", run.name,
    ]
    identity = read(run / "suite_manifest.json")["identity"]
    summary = read(run / "suite_summary.json")
    assert summary["status"] == "pending"
    assert summary["sections"]["mtp/off"]["status"] == "completed"
    assert all(summary["sections"][f"mtp/{mode}"]["status"] == "pending" for mode in ("k1", "k2", "k3"))
    assert not (run / ".section_running").exists()
    before = {path: path.read_bytes() for path in (run / "gaia").rglob("*.json")}
    before.update({path: path.read_bytes() for path in (run / "mtp/off").rglob("*.json")})
    handoffs.answers = iter(evidence[mode] for mode in ("k1", "k2", "k3"))
    monkeypatch.setattr(sys, "argv", ["run_suite.py", "all", "--resume", run.name])
    with pytest.raises(SystemExit) as second:
        suite.runner.main()
    assert second.value.code == 0
    visible = capsys.readouterr().out
    assert "MTP k3 storage stage complete" in visible
    finished = json.loads(visible.splitlines()[-1])
    assert finished["event"] == "invocation_finished"
    assert Path(finished["report"]) == suite.common.REPORT_ROOT / run.name / "report.json"
    assert Path(finished["summary"]) == run / "suite_summary.json"
    assert (run / "runner.log").read_text() == initial_log + visible
    assert seen == [(section, identity) for section in ("gaia", "off", "k1", "k2", "k3")]
    assert len(prompts) == 5
    assert all(f"for {mode}:" in prompt for mode, prompt in zip(("off", "k1", "k1", "k2", "k3"), prompts))
    assert all(path.read_bytes() == value for path, value in before.items())
    assert list((tmp_path / "logs").iterdir()) == [run]
    assert read(run / "suite_summary.json")["status"] == "completed"
    assert read(Path(finished["report"]))["sections"][1]["status"].startswith("4/4")
    assert not (run / "reports").exists()


def test_single_mtp_mode_plans_comparison_and_keeps_resume_identity(suite, monkeypatch, tmp_path):
    run = tmp_path / "campaign"
    calls = []

    mtp = partial(_record_mtp_run, calls)

    monkeypatch.setattr(suite.runner, "run_mode", mtp)
    for mode in ("off", "k1"):
        result = asyncio.run(suite.runner.run(run, target="mtp", mtp_mode=mode,
                                             config_evidence=tmp_path / f"{mode}.json"))
        assert result["status"] == "completed"
        assert result["campaign_status"] == "pending"
    manifest = read(run / "suite_manifest.json")
    assert manifest["sections"] == ["mtp/off", "mtp/k1", "mtp/k2", "mtp/k3"]
    assert manifest["gaia_phases"] == []
    assert manifest["planned_total_fixed_requests"] == 208
    assert manifest["planned_measured_fixed_requests"] == manifest["planned_warmup_requests"] == 104
    assert calls == [(manifest["identity"], mode, tmp_path / f"{mode}.json") for mode in ("off", "k1")]
    before = (run / "suite_manifest.json").read_bytes()
    with pytest.raises(ValueError, match="targets mtp"):
        asyncio.run(suite.runner.run(run, target="concurrency"))
    assert (run / "suite_manifest.json").read_bytes() == before
    assert not (run / "gaia").exists()
