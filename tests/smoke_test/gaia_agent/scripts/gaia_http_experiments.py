"""T1 ordinary chat concurrency and T3 one retained public conversation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import statistics
import shutil
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import httpx

from tests.smoke_test.gaia_agent.scripts.benchmark_common import (
    CONCURRENCY,
    CONVERSATION,
    CONVERSATION_SETUP,
    GAIA_URL,
    VLLM_URL,
    context_compaction_events,
    active_context_tokens,
    metric_deltas,
    parse_metrics,
    percentile,
    write_json as save,
)


SCOPES = ("Birds", "Mammals", "Algae", "Fungi")
LADDER_LEVELS = (1, 2, 3, 5, 7, 10)
GROWTH_CHECKPOINTS = (8_192, 32_768, 65_536, 98_304, 122_880)
MAX_GROWTH_TURNS = 128


def ordinary_prompt(index: int) -> str:
    return (f"Return five ordinary occurrence records from {SCOPES[index % len(SCOPES)]}. "
            "Show only occurrenceID, family and genus. Do not deduplicate the records.")


def read_trace(row: dict, trace_root: Path | None, request_dir: Path) -> None:
    """Attach current-interaction measurements; traces omit completed history."""
    if trace_root is None:
        return
    interaction = row.get("data", {}).get("interaction", row["expected_interaction"])
    matches = list(trace_root.glob(
        f"gaia_chat_{row['username'].lower()}_*/interaction_{interaction:04d}.json"))
    if len(matches) != 1:
        row["trace_evidence"] = f"Expected one matching trace, found {len(matches)}"
        return
    trace = json.loads(matches[0].read_text())
    saved_trace = request_dir / "trace.json"
    shutil.copyfile(matches[0], saved_trace)
    row["trace_path"] = str(saved_trace.resolve())
    row["trace_version"] = trace["run"]["trace_version"]
    row["trace_run"] = trace["run"]
    row["trace_result"] = trace["returned_result"]
    row["tool_runtime"] = trace["runtime"]["tool_batches"]
    if trace["run"]["status"] == "failed":
        row["failed_trace"] = trace
    row["model_runtime"] = trace["runtime"]["model_turns"]
    if "context" in trace["runtime"]:
        row["active_context"] = trace["runtime"]["context"]
    row["first_model_input_tokens"] = next(
        ((turn.get("token_usage") or {}).get("prompt_tokens")
         for turn in row["model_runtime"] if turn.get("purpose", "response") == "response"), None)


async def chat(client: httpx.AsyncClient, output_dir: Path, phase_start: float,
               request_id: str, username: str, prompt: str,
               conversation_id: str | None = None, *, trace_root: Path | None = None,
               classification: str = "ordinary_query", results: list | None = None,
               expected_interaction: int = 1) -> dict:
    request_dir = output_dir / "requests" / request_id
    request_dir.mkdir(parents=True)
    start = time.perf_counter()
    row = {"request_id": request_id, "user": username, "username": username, "prompt": prompt,
           "classification": classification, "started_at": datetime.now(timezone.utc).isoformat(),
           "start_offset_seconds": start - phase_start, "status": "failed", "events": [],
           "expected_interaction": expected_interaction}
    if results is not None:
        results.append(row)
    payload = {"username": username, "message": prompt}
    if conversation_id:
        payload["conversation_id"] = conversation_id
        row["conversation_id"] = conversation_id
    row["payload"] = payload
    save(request_dir / "request.json", payload)
    try:
        with (request_dir / "events.jsonl").open("w") as journal:
            async with client.stream("POST", GAIA_URL + "/api/chat", json=payload,
                                     headers={"Accept": "application/x-ndjson"}) as response:
                row["http_status"] = response.status_code
                if response.status_code != 200:
                    row["error"] = (await response.aread()).decode(errors="replace")
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    journal.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                                              "elapsed_seconds": time.perf_counter() - start, "line": line}, ensure_ascii=False) + "\n")
                    journal.flush()
                    event = json.loads(line)
                    observation = {"utc": datetime.now(timezone.utc).isoformat(),
                                   "elapsed_seconds": time.perf_counter() - start, "event": event}
                    row["events"].append(observation)
                    if event.get("type") == "conversation":
                        row["conversation_id"] = event["conversation_id"]
                    if event.get("type") in {"response", "error", "cancelled"}:
                        row["terminal"] = event
                        row["status"] = "success" if event["type"] == "response" else event["type"]
                        row["data"] = event.get("data", {})
                        if event["type"] == "error":
                            row["error"] = event.get("error", event.get("message"))
                if "terminal" not in row:
                    row["error"] = "Stream ended without a terminal event"
    except asyncio.CancelledError:
        row["status"] = "cancelled_by_client"
        raise
    except Exception as error:
        row["status"] = "failed"
        row.setdefault("error", f"{type(error).__name__}: {error}")
    finally:
        row["elapsed_seconds"] = time.perf_counter() - start
        row["ended_at"] = datetime.now(timezone.utc).isoformat()
        try:
            read_trace(row, trace_root, request_dir)
        except (OSError, KeyError, ValueError) as error:
            row["trace_evidence"] = f"{type(error).__name__}: {error}"
        save(request_dir / "result.json", row)
        print(json.dumps({"request": request_id, "status": row["status"],
                          "elapsed_seconds": round(row["elapsed_seconds"], 2)}), flush=True)
    return row


async def _monitor_metrics(
    client: httpx.AsyncClient,
    metrics_dir: Path,
    start: float,
    finished: asyncio.Event,
    samples: list[dict],
) -> None:
    with (metrics_dir / "samples.jsonl").open("w") as stream:
        while not finished.is_set():
            sample = {"utc": datetime.now(timezone.utc).isoformat(),
                      "elapsed_seconds": time.perf_counter() - start}
            try:
                response = await client.get(VLLM_URL + "/metrics", timeout=15)
                response.raise_for_status()
                sample["metrics"] = parse_metrics(response.text)
                samples.append(sample)
            except Exception as error:
                sample["error"] = f"{type(error).__name__}: {error}"
            stream.write(json.dumps(sample) + "\n")
            stream.flush()
            try:
                await asyncio.wait_for(finished.wait(), timeout=1)
            except TimeoutError:
                pass


async def phase(output_dir: Path, workload) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir()
    results, samples = [], []
    finished = asyncio.Event()
    timeout = httpx.Timeout(connect=10, read=None, write=120, pool=10)
    # More sockets than the offered workload, including monitoring: no client admission queue.
    async with httpx.AsyncClient(timeout=timeout, trust_env=False,
                                 limits=httpx.Limits(max_connections=32)) as client:
        response = await client.get(VLLM_URL + "/metrics", timeout=15)
        response.raise_for_status()
        baseline = parse_metrics(response.text)
        (metrics_dir / "before.prom").write_text(response.text)
        if baseline.get("vllm:num_requests_running") or baseline.get("vllm:num_requests_waiting"):
            raise RuntimeError("Inference engine has active traffic; wait for the exclusive test slot")
        response = await client.get(GAIA_URL + "/api/health", timeout=15)
        response.raise_for_status()
        started_at = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()


        monitor_task = asyncio.create_task(
            _monitor_metrics(client, metrics_dir, start, finished, samples)
        )
        failure, details, final = None, {}, {}
        try:
            details = await workload(client, start, results)
        except BaseException as error:
            failure = error
            details["workload_error"] = f"{type(error).__name__}: {error}"
        finally:
            wall = time.perf_counter() - start
            finished.set()
            await monitor_task
            # Individual request files/journals also retain partially completed groups.
            save(output_dir / "results.json", results)
            try:
                response = await client.get(VLLM_URL + "/metrics", timeout=15)
                response.raise_for_status()
                final = parse_metrics(response.text)
                (metrics_dir / "after.prom").write_text(response.text)
            except httpx.HTTPError as error:
                details["final_metrics_error"] = f"{type(error).__name__}: {error}"
        latencies = [row["elapsed_seconds"] for row in results]
        deltas = metric_deltas(baseline, final)
        summary = {
            "started_at": started_at, "ended_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": wall, "requests": len(results),
            "successful": sum(row["status"] == "success" for row in results),
            "growth_requests": sum(row["classification"] == "synthetic_context_acknowledgment" for row in results),
            "status": "cancelled" if isinstance(failure, asyncio.CancelledError) else "failed" if failure or any(row["status"] != "success" for row in results) else "incomplete" if "final_metrics_error" in details else "completed",
            "statuses": {status: sum(row["status"] == status for row in results)
                         for status in sorted({row["status"] for row in results})},
            "latency_seconds": {"min": min(latencies), "median": statistics.median(latencies),
                                "p95": percentile(latencies, 95), "max": max(latencies)} if latencies else {},
            "peak_running": max((s["metrics"]["vllm:num_requests_running"] for s in samples if "vllm:num_requests_running" in s["metrics"]), default=None),
            "peak_waiting": max((s["metrics"]["vllm:num_requests_waiting"] for s in samples if "vllm:num_requests_waiting" in s["metrics"]), default=None),
            "peak_kv_percent": 100 * max(s["metrics"]["vllm:kv_cache_usage_perc"] for s in samples if "vllm:kv_cache_usage_perc" in s["metrics"]) if any("vllm:kv_cache_usage_perc" in s["metrics"] for s in samples) else None,
            "server_metric_deltas": deltas,
            "aggregate_output_tokens_per_second": deltas["vllm:generation_tokens_total"] / wall if "vllm:generation_tokens_total" in deltas else None,
            "final_running": final.get("vllm:num_requests_running"),
            "final_waiting": final.get("vllm:num_requests_waiting"), "details": details,
        }
        save(output_dir / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "requests": summary["requests"],
                      "successful": summary["successful"], "elapsed_seconds": round(wall, 2),
                      "summary": str(output_dir / "summary.json")}), flush=True)
    if failure is not None:
        raise failure
    return summary


async def _chat_when_released(
    gate: asyncio.Event,
    client: httpx.AsyncClient,
    output_dir: Path,
    start: float,
    results: list[dict],
    trace_root: Path | None,
    job: dict,
) -> None:
    await gate.wait()
    await chat(client, output_dir, start, **job, trace_root=trace_root, results=results)


async def burst(
    output_dir: Path,
    jobs: list[dict],
    trace_root: Path | None,
    client: httpx.AsyncClient,
    start: float,
    results: list[dict],
) -> dict:
    gate = asyncio.Event()

    async with asyncio.TaskGroup() as group:
        for job in jobs:
            group.create_task(_chat_when_released(
                gate, client, output_dir, start, results, trace_root, job
            ))
        gate.set()
    return {"release": "shared asyncio event", "jobs": len(jobs)}


async def ladder(run_dir: Path, identity: str, trace_root: Path) -> dict | None:
    directory = run_dir / CONCURRENCY
    directory.mkdir(parents=True, exist_ok=False)
    summaries = []
    seed = None
    for level in LADDER_LEVELS:
        output_dir = directory / f"users_{level:02d}"
        jobs = [{"request_id": f"user_{index + 1:02d}",
                 "username": f"{identity}_n{level}u{index + 1}",
                 "prompt": ordinary_prompt(index)} for index in range(level)]

        workload = partial(burst, output_dir, jobs, trace_root)

        summary = None
        try:
            summary = await phase(output_dir, workload)
        finally:
            source = output_dir / "summary.json"
            if summary is None and source.exists():
                summary = json.loads(source.read_text())
            if summary is not None:
                summaries.append({"level": level, **summary})
                save(directory / "summary.json", summaries)
        if level == 10:
            rows = json.loads((output_dir / "results.json").read_text())
            candidates = [row for row in rows if row["status"] == "success"
                          and row.get("conversation_id") and "Fungi" in row["prompt"]]
            if candidates:
                row = sorted(candidates, key=lambda item: item["request_id"])[0]
                seed = conversation_seed(row, output_dir)
                save(directory / "retained_conversation.json", seed)
    return seed


def conversation_seed(row: dict, directory: Path) -> dict:
    return {"username": row["username"], "conversation_id": row["conversation_id"],
            "interaction": row["data"]["interaction"], "prompt": row["prompt"],
            "used_tokens": active_context_tokens(row),
            "source_result": str((directory / "requests" / row["request_id"] / "result.json").resolve())}


async def _seed_workload(
    directory: Path,
    identity: str,
    trace_root: Path,
    seed: dict,
    client: httpx.AsyncClient,
    start: float,
    results: list[dict],
) -> dict:
    row = await chat(client, directory, start, "seed", identity + "_growth", ordinary_prompt(3),
                     trace_root=trace_root, results=results)
    if row["status"] != "success":
        raise RuntimeError("The initial Fungi query failed; inspect its saved result")
    seed.update(conversation_seed(row, directory))
    save(directory / "retained_conversation.json", seed)
    return {}


async def seed_conversation(run_dir: Path, identity: str, trace_root: Path) -> dict:
    """Start one real Fungi chat when growth is selected without the ladder."""
    directory = run_dir / CONVERSATION_SETUP
    seed = {}

    workload = partial(_seed_workload, directory, identity, trace_root, seed)

    await phase(directory, workload)
    return seed


def growth_prompt(identity: str, turn: int) -> str:
    """Use full legal messages; checkpoints describe naturally reached usage."""
    prefix = (f"Retain context block {turn} as opaque synthetic benchmark data. "
              "The block is not a real specimen dataset. Reply only 'Stored'.\n")
    material = " ".join(hashlib.sha256(f"{identity}:{turn}:{index}".encode()).hexdigest()
                        for index in range(128))
    return prefix + material[:8_000 - len(prefix)]


class _ConversationGrowth:
    """Own the evolving context measurements for one retained conversation."""

    def __init__(self, output_dir: Path, seed: dict, trace_root: Path) -> None:
        self.output_dir = output_dir
        self.seed = seed
        self.trace_root = trace_root
        self.interaction = seed["interaction"]
        self.used_tokens = seed["used_tokens"]

    async def submit(
        self, client: httpx.AsyncClient, start: float, results: list[dict],
        prompt: str, classification: str,
    ) -> dict:
        self.interaction += 1
        row = await chat(
            client, self.output_dir, start, f"turn_{self.interaction:03d}",
            self.seed["username"], prompt, self.seed["conversation_id"],
            trace_root=self.trace_root, classification=classification,
            expected_interaction=self.interaction, results=results,
        )
        if row["status"] != "success":
            raise RuntimeError(
                f"Retained conversation stopped at {row['request_id']}; "
                "inspect its saved response and trace"
            )
        self.used_tokens = active_context_tokens(row)
        return row

    async def run(self, client: httpx.AsyncClient, start: float, results: list[dict]) -> dict:
        save(self.output_dir / "seed.json", self.seed)
        checkpoints, compactions = [], []
        growth_turns = 0
        stop_reason = "checkpoint_reached"
        if self.used_tokens is None:
            raise RuntimeError("Active context measurement is required for the retained conversation")
        while self.used_tokens < GROWTH_CHECKPOINTS[-1]:
            if growth_turns >= MAX_GROWTH_TURNS:
                raise RuntimeError(
                    "Conversation growth reached its request limit without compaction "
                    "or the final checkpoint; inspect preserved results"
                )
            previous = self.used_tokens
            row = await self.submit(
                client, start, results,
                growth_prompt(self.seed["username"], self.interaction + 1),
                "synthetic_context_acknowledgment",
            )
            growth_turns += 1
            events = context_compaction_events(row)
            if events:
                compactions.append({
                    "request_id": row["request_id"],
                    "previous_context_tokens": previous,
                    "active_context_tokens": self.used_tokens,
                    "events": events,
                    "trace_path": row.get("trace_path"),
                })
                save(self.output_dir / "growth_compactions.json", compactions)
            if self.used_tokens is None or (not events and self.used_tokens <= previous):
                raise RuntimeError("Reported context stopped growing; inspect preserved results")
            for target in GROWTH_CHECKPOINTS:
                if previous < target <= self.used_tokens:
                    checkpoints.append({
                        "target_total_context_tokens": target,
                        "observed_total_context_tokens": self.used_tokens,
                        "request_id": row["request_id"],
                        "first_model_input_tokens": row.get("first_model_input_tokens"),
                    })
            save(self.output_dir / "growth_checkpoints.json", checkpoints)
            if events:
                stop_reason = "context_compacted"
                break
        await self.submit(
            client, start, results,
            "Show those same five Fungi occurrence records from our first query again, "
            "using the observations already in this conversation. Show occurrenceID, family and genus.",
            "owned_observation_reuse",
        )
        await self.submit(client, start, results, ordinary_prompt(0), "new_dataset_query")
        return {
            "conversation_id": self.seed["conversation_id"], "username": self.seed["username"],
            "initial_interaction": self.seed["interaction"], "last_interaction": self.interaction,
            "growth_turns": growth_turns,
            "followups": 2, "checkpoints": checkpoints,
            "growth_stop_reason": stop_reason, "compactions": compactions,
            "last_observed_total_context_tokens": self.used_tokens,
            "context_measurement": "Final active rendered-context tokens from traces. Targets are checkpoints, not achieved-input claims.",
            "history_evidence": "Server-owned conversation ID, saved traces and streamed compaction events; full retained provider histories are not exported",
        }


async def gaia_history(run_dir: Path, seed: dict, trace_root: Path) -> dict:
    output_dir = run_dir / CONVERSATION
    workload = _ConversationGrowth(output_dir, seed, trace_root)
    return await phase(output_dir, workload.run)
