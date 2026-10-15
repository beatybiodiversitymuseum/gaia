"""Run one smoke-test category, all Gaia tests, the MTP sweep, or the entire suite."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shlex
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from pathlib import Path

import httpx

from tests.smoke_test.gaia_agent.scripts.benchmark_common import (
    CONCURRENCY,
    CONTEXT_CACHE,
    CONVERSATION,
    CONVERSATION_SETUP,
    INTERFERENCE,
    REPOSITORY_ROOT,
    BenchmarkConfig,
    configuration_identity,
    existing_run_directory,
    new_run_directory,
    report_directory,
    sanitized_url,
    utc_now,
    write_json,
)
from tests.smoke_test.gaia_agent.scripts.engine_experiments import (
    context_pool,
    mixed_workload,
    retain_pool,
    run_phase,
)
from tests.smoke_test.gaia_agent.scripts.gaia_http_experiments import (
    gaia_history,
    ladder,
    seed_conversation,
)
from tests.smoke_test.gaia_agent.scripts.mtp_experiments import MODES, run_mode
from tests.smoke_test.gaia_agent.scripts.summarize_results import summarize, write_suite_report

GAIA_CATEGORIES = ("concurrency", "context-cache", "conversation-growth", "prefill-interference")
TARGETS = ("all", "gaia", "mtp", *GAIA_CATEGORIES)
FIXED_GAIA_REQUESTS = {"all": 84, "gaia": 84, "mtp": 0, "concurrency": 28,
                       "context-cache": 22, "conversation-growth": 3, "prefill-interference": 22}
INITIAL_CONTEXT = CONTEXT_CACHE + "/initial"
RETAINED_CONTEXT = CONTEXT_CACHE + "/retention"


def gaia_phases(target: str) -> list[str]:
    if target in {"all", "gaia"}:
        return [CONCURRENCY, INITIAL_CONTEXT, CONVERSATION, INTERFERENCE, RETAINED_CONTEXT]
    return {"concurrency": [CONCURRENCY], "context-cache": [INITIAL_CONTEXT],
            "conversation-growth": [CONVERSATION_SETUP, CONVERSATION],
            "prefill-interference": [INTERFERENCE], "mtp": []}[target]


async def _run_stage(
    run_dir: Path,
    gaia_dir: Path,
    stages: dict[str, dict],
    name: str,
    workload: Callable[[], Awaitable[object]],
    *,
    missing_dependency: str | None = None,
) -> object:

    directory = run_dir / name
    if missing_dependency:
        stages[name] = {"status": "skipped", "reason": missing_dependency, "directory": str(directory)}
        write_json(gaia_dir / "progress.json", stages)
        return None
    record = stages[name] = {"status": "running", "started_at": utc_now(), "directory": str(directory)}
    write_json(gaia_dir / "progress.json", stages)
    try:
        return await workload()
    except asyncio.CancelledError:
        record["status"] = "cancelled"
        raise
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        source = directory / "summary.json"
        if source.exists():
            summaries = json.loads(source.read_text())
            summaries = summaries if isinstance(summaries, list) else [summaries]
            statuses = {item["status"] for item in summaries}
            if record["status"] == "running":
                record["status"] = ("completed" if statuses == {"completed"} else
                                    "incomplete" if statuses <= {"completed", "incomplete"} else "failed")
            record.update(requests=sum(item["requests"] for item in summaries),
                          successful=sum(item["successful"] for item in summaries),
                          growth_requests=sum(item.get("growth_requests", 0) for item in summaries))
        elif record["status"] == "running":
            record.update(status="failed", error="The phase produced no summary")
        record["ended_at"] = utc_now()
        write_json(gaia_dir / "progress.json", stages)


async def run_gaia(run_dir: Path, trace_root: Path, identity: str, target: str = "gaia",
                   config: BenchmarkConfig | None = None) -> dict:
    config = config or BenchmarkConfig()
    (run_dir / ".gaia_started").mkdir()
    gaia_dir = run_dir / "gaia"
    gaia_dir.mkdir()
    started = time.perf_counter()
    phases = gaia_phases(target)
    manifest = {"started_at": utc_now(), "identity": identity, "target": target,
                "gaia_url": configuration_identity(config)["snapshot"]["gaia_url"],
                "engine_url": configuration_identity(config)["snapshot"]["engine_url"],
                "requested_model_id": config.model, "trace_root": str(trace_root),
                "family_order": phases, "planned_fixed_requests": FIXED_GAIA_REQUESTS[target],
                "variable_requests": "Conversation growth turns" if CONVERSATION in phases else None,
                "maximum_offered_requests": 10}
    write_json(gaia_dir / "manifest.json", manifest)
    stages = {}

    stage = partial(_run_stage, run_dir, gaia_dir, stages)

    cancelled, seed = False, None
    try:
        for name in phases:
            if name == CONCURRENCY:
                seed = await stage(name, partial(ladder, run_dir, identity, trace_root, config))
            elif name == CONVERSATION_SETUP:
                seed = await stage(name, partial(seed_conversation, run_dir, identity, trace_root, config))
            elif name == INITIAL_CONTEXT:
                await stage(name, partial(run_phase, run_dir / name, identity, context_pool,
                                          phase_name=name, config=config))
            elif name == CONVERSATION:
                await stage(name, partial(gaia_history, run_dir, seed, trace_root, config),
                            missing_dependency="No successful initial Fungi conversation" if seed is None else None)
            elif name == INTERFERENCE:
                await stage(name, partial(run_phase, run_dir / name, identity, mixed_workload,
                                          phase_name=name, config=config))
            elif name == RETAINED_CONTEXT:
                initial = run_dir / INITIAL_CONTEXT
                source = initial / "summary.json"
                initial_summary = json.loads(source.read_text()) if source.exists() else {}
                pool_ready = any(item.get("group") == "cold_10_users" and item.get("status") == "completed"
                                 for item in initial_summary.get("comparisons", []))
                await stage(name, partial(run_phase, run_dir / name, identity,
                            partial(retain_pool, initial_directory=initial), phase_name=name, config=config),
                            missing_dependency=None if pool_ready else "Initial ten-input cold pool did not complete")
    except asyncio.CancelledError:
        cancelled = True
        raise
    finally:
        for name in phases:
            stages.setdefault(name, {"status": "skipped", "reason": "Cancelled before this phase"})
        summary = {"started_at": manifest["started_at"], "ended_at": utc_now(),
                   "wall_seconds": time.perf_counter() - started,
                   "status": "cancelled" if cancelled else "completed" if all(
                       item["status"] == "completed" for item in stages.values()) else "incomplete",
                   "planned_fixed_requests": FIXED_GAIA_REQUESTS[target], "stages": stages,
                   **{key: sum(item.get(key, 0) for item in stages.values())
                      for key in ("requests", "successful", "growth_requests")}}
        write_json(gaia_dir / "summary.json", summary)
    return summary


def campaign_manifest(run_dir: Path, trace_root: Path | None, evidence_path: Path | None,
                      target: str = "all", config: BenchmarkConfig | None = None) -> dict:
    config = config or BenchmarkConfig()
    config_record = configuration_identity(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "suite_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get("suite") != "gaia_agent_smoke":
            raise ValueError("Existing manifest belongs to a different suite; start a new run")
        if target != manifest["target"] and not (manifest["target"] == "all" and target in {"gaia", "mtp"}):
            raise ValueError(f"This run targets {manifest['target']}; start a new run for {target}")
        if manifest.get("configuration", {}).get("sha256") != config_record["sha256"]:
            raise ValueError("Resume configuration differs from the campaign manifest; use the original endpoints and model")
        return manifest
    inputs = [path for path in (trace_root, evidence_path) if path is not None]
    for existing in run_dir.iterdir():
        if existing.name != "runner.log" and not any(
                existing == supplied or existing in supplied.parents for supplied in inputs):
            raise FileExistsError(f"Measurement evidence already exists in {run_dir}")
    phases = gaia_phases(target)
    mtp = target in {"all", "mtp"}
    workload_snapshot = {"target": target, "gaia_phases": phases, "mtp_modes": list(MODES) if mtp else [],
                         "gaia_fixed_requests": FIXED_GAIA_REQUESTS[target], "maximum_offered_requests": 10}
    workload_hash = hashlib.sha256(json.dumps(
        workload_snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {"suite": "gaia_agent_smoke", "target": target, "started_at": utc_now(),
                "identity": "suite_" + uuid.uuid4().hex[:12],
                "gaia_url": config_record["snapshot"]["gaia_url"],
                "engine_url": config_record["snapshot"]["engine_url"],
                "requested_model_id": config.model, "configuration": config_record,
                "workload_configuration": {"snapshot": workload_snapshot, "sha256": workload_hash},
                "sections": (["gaia"] if phases else []) + ([f"mtp/{mode}" for mode in MODES] if mtp else []),
                "gaia_phases": phases, "gaia_fixed_requests": FIXED_GAIA_REQUESTS[target],
                "planned_measured_fixed_requests": FIXED_GAIA_REQUESTS[target] + (104 if mtp else 0),
                "planned_warmup_requests": 104 if mtp else 0,
                "planned_total_fixed_requests": FIXED_GAIA_REQUESTS[target] + (208 if mtp else 0),
                "variable_requests": "Conversation growth turns" if CONVERSATION in phases else None,
                "maximum_offered_requests": 10}
    with path.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2))
    return manifest


async def capture_model_identity(run_dir: Path, manifest: dict, config: BenchmarkConfig) -> None:
    """Resolve the requested served model once without inventing unavailable revision data."""
    async with httpx.AsyncClient(trust_env=False) as client:
        response = await client.get(config.engine_url + "/v1/models", timeout=15.0)
        response.raise_for_status()
    inventory = response.json()
    data = inventory.get("data")
    if not isinstance(data, list):
        raise ValueError("Model inventory response must contain a data list")
    resolved = next((item for item in data if isinstance(item, dict) and item.get("id") == config.model), None)
    if resolved is None:
        raise ValueError(f"Requested model {config.model!r} is absent from the model inventory")
    prior_identity = manifest.get("model_identity")
    if prior_identity is not None:
        if prior_identity.get("resolved_inventory_entry") != resolved:
            raise ValueError(
                "Live model inventory differs from the campaign model identity; "
                "start a new run"
            )
        return
    inventory_path = run_dir / "model_inventory.json"
    write_json(inventory_path, inventory)
    revision = next((resolved.get(key) for key in ("revision", "model_revision", "commit_hash")
                     if resolved.get(key)), None)
    manifest["model_identity"] = {
        "requested_model_id": config.model, "resolved_inventory_entry": resolved,
        "context_tokens": resolved.get("max_model_len"), "revision": revision,
        "revision_status": "reported" if revision is not None else "missing_from_endpoint",
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": hashlib.sha256(json.dumps(
            inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    write_json(run_dir / "suite_manifest.json", manifest)


def campaign_summary(run_dir: Path, manifest: dict, analysis: dict) -> dict:
    sections = {}
    for name in manifest["sections"]:
        path = run_dir / name / "summary.json"
        if path.exists():
            source = json.loads(path.read_text())
            sections[name] = {key: source.get(key) for key in (
                "status", "requests", "successful", "wall_seconds", "error")}
            sections[name]["source"] = str(path.resolve())
        else:
            sections[name] = {"status": "pending"}
    statuses = {item["status"] for item in sections.values()}
    return {"started_at": manifest["started_at"], "updated_at": utc_now(),
            "status": "completed" if statuses == {"completed"} else "pending" if statuses <= {"completed", "pending"} else "incomplete",
            "sections": sections, "target": manifest["target"], "gaia_phases": manifest["gaia_phases"],
            "requests": sum(item["recorded_requests"] for item in analysis.values()),
            "successful": sum(item["successful"] for item in analysis.values()),
            "growth_requests": sum(item["growth_requests"] for item in analysis.values()),
            **{key: manifest[key] for key in ("planned_measured_fixed_requests", "planned_warmup_requests", "planned_total_fixed_requests")}}


def update_report(run_dir: Path, manifest: dict) -> dict:
    analysis = summarize(run_dir)
    summary = campaign_summary(run_dir, manifest, analysis)
    write_json(run_dir / "suite_summary.json", summary)
    write_suite_report(run_dir, analysis, summary)
    return summary


async def run_mtp_mode(run_dir: Path, identity: str, mode: str, evidence_path: Path | None,
                       config: BenchmarkConfig | None = None) -> dict:
    config = config or BenchmarkConfig()
    while True:
        if evidence_path is None:
            print(f"\nMTP {mode}: configure vLLM in the GX10 terminal, wait until ready, "
                  "then save a fresh docker inspect JSON locally.", flush=True)
            path = Path(input(f"Local configuration JSON for {mode}: ").strip()).expanduser().resolve()
        else:
            path = evidence_path
        try:
            return await run_mode(run_dir, identity, mode, path, config)
        except (ValueError, KeyError, OSError) as error:
            # Local evidence is checked before the protected measurement folder is created.
            if evidence_path is not None or (run_dir / "mtp" / mode).exists():
                raise
            print(f"Configuration file rejected: {error}. Supply the corrected file.", flush=True)


async def run(run_dir: Path, *, target: str, trace_root: Path | None = None,
              mtp_mode: str | None = None, config_evidence: Path | None = None,
              config: BenchmarkConfig | None = None) -> dict:
    config = config or BenchmarkConfig()
    manifest = campaign_manifest(run_dir, trace_root, config_evidence, target, config)
    await capture_model_identity(run_dir, manifest, config)
    running = run_dir / ".section_running"
    running.mkdir()
    selected = (["gaia"] if target != "mtp" else [])
    if target in {"all", "mtp"}:
        selected.extend(f"mtp/{mode}" for mode in ((mtp_mode,) if mtp_mode else MODES))
    summary = None
    try:
        for name in selected:
            path = run_dir / name / "summary.json"
            if path.exists() and json.loads(path.read_text())["status"] == "completed":
                print(f"Already completed: {name}; reusing its saved measurements.", flush=True)
                continue
            try:
                if name == "gaia":
                    await run_gaia(run_dir, trace_root or REPOSITORY_ROOT / "logs", manifest["identity"], target, config)
                else:
                    await run_mtp_mode(run_dir, manifest["identity"], name.split("/")[1], config_evidence, config)
            finally:
                summary = update_report(run_dir, manifest)
        if summary is None:
            summary = update_report(run_dir, manifest)
    finally:
        running.rmdir()
    status = "completed" if all(summary["sections"][name]["status"] == "completed" for name in selected) else "incomplete"
    return {"target": target, "status": status, "campaign_status": summary["status"],
            "summary": str(run_dir / "suite_summary.json"),
            "report": str(report_directory(run_dir) / "report.json")}


class ConsoleLog:
    """Keep progress visible while appending the same output to this run's log."""

    def __init__(self, console, log) -> None:
        self.console = console
        self.log = log

    def write(self, text: str) -> int:
        self.console.write(text)
        return self.log.write(text)

    def flush(self) -> None:
        self.console.flush()
        self.log.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=TARGETS, help="Category to run; all includes Gaia and the four-mode MTP sweep")
    parser.add_argument("--resume", help="Existing run folder to continue; completed sections are reused")
    parser.add_argument("--trace-root", type=Path, default=REPOSITORY_ROOT / "logs", help="Gaia service trace directory (default: repository logs/)")
    parser.add_argument("--mtp-mode", choices=MODES, help="mtp only: run one already-configured mode instead of the prompted sweep")
    parser.add_argument("--config-evidence", type=Path, help="Saved Docker inspect JSON; required with --mtp-mode")
    parser.add_argument("--gaia-url", default=BenchmarkConfig().gaia_url, help="Gaia HTTP service base URL")
    parser.add_argument("--engine-url", default=BenchmarkConfig().engine_url, help="vLLM HTTP service base URL")
    parser.add_argument("--model", default=BenchmarkConfig().model, help="Served model identifier")
    args = parser.parse_args()
    if (args.mtp_mode is None) != (args.config_evidence is None) or (args.mtp_mode and args.target != "mtp"):
        parser.error("Use --mtp-mode and --config-evidence together with the mtp category")
    try:
        run_dir = existing_run_directory(args.resume) if args.resume else new_run_directory(args.target.replace("-", "_"))
    except ValueError as error:
        parser.error(str(error))
    with (run_dir / "runner.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(ConsoleLog(sys.stdout, log)), redirect_stderr(ConsoleLog(sys.stderr, log)):
            print(json.dumps({"event": "invocation_started", "utc": utc_now(), "run": run_dir.name,
                              "target": args.target, "mtp_mode": args.mtp_mode, "output_directory": str(run_dir)}), flush=True)
            try:
                config = BenchmarkConfig(args.gaia_url.rstrip("/"), args.engine_url.rstrip("/"), args.model)
                result = asyncio.run(run(run_dir, target=args.target, trace_root=args.trace_root.expanduser().resolve(),
                                         mtp_mode=args.mtp_mode,
                                         config_evidence=args.config_evidence.expanduser().resolve() if args.config_evidence else None,
                                         config=config))
            except (KeyboardInterrupt, EOFError):
                source = run_dir / "suite_summary.json"
                state = json.loads(source.read_text()) if source.exists() else {}
                partial = [name for name, item in state.get("sections", {}).items()
                           if item["status"] != "completed" and (run_dir / name).exists()]
                event = {"event": "invocation_interrupted", "utc": utc_now()}
                if partial:
                    event.update(incomplete_sections=partial,
                                 next_step="Start a new run to repeat interrupted measurements; saved evidence is protected.")
                else:
                    command = [sys.executable, "-m", "tests.smoke_test.gaia_agent.scripts.run_suite",
                               args.target, "--resume", run_dir.name]
                    if args.target != "mtp":
                        command.extend(("--trace-root", str(args.trace_root.expanduser().resolve())))
                    if args.mtp_mode:
                        command.extend(("--mtp-mode", args.mtp_mode, "--config-evidence", str(args.config_evidence.expanduser().resolve())))
                    defaults = BenchmarkConfig()
                    if args.gaia_url.rstrip("/") != defaults.gaia_url:
                        command.extend(("--gaia-url", sanitized_url(args.gaia_url)))
                    if args.engine_url.rstrip("/") != defaults.engine_url:
                        command.extend(("--engine-url", sanitized_url(args.engine_url)))
                    if args.model != defaults.model:
                        command.extend(("--model", args.model))
                    event["resume"] = shlex.join(command)
                print(json.dumps(event), flush=True)
                raise SystemExit(130)
            except Exception as error:
                print(json.dumps({"target": args.target, "status": "failed", "utc": utc_now(),
                                  "error": f"{type(error).__name__}: {error}",
                                  "summary": str(run_dir / "suite_summary.json")}), flush=True)
                raise SystemExit(1)
            print(json.dumps({"event": "invocation_finished", "utc": utc_now(), **result}), flush=True)
            raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
