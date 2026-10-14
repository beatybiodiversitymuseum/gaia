"""T2 shared context pool and T4 interference measurements."""

from __future__ import annotations

import asyncio
import hashlib
import json
import statistics
import time
from functools import partial
from pathlib import Path

import httpx

from tests.smoke_test.gaia_agent.scripts.benchmark_common import (
    VLLM_URL,
    metric_deltas,
    parse_metrics,
    percentile,
    utc_now,
    write_json,
)


MODEL = "default_model"
LONG_INPUT_TOKENS = 130_048
WINDOW = 131_072
UNIT = (
    "Synthetic catalogue entry: habitat coastal forest; country Canada; genus Samplea; "
    "family Exampleaceae; collector Observer; year 2020; condition preserved specimen. "
    "This entry is context data only.\n"
)


def _benchmark_messages(
    run_identity: str, identity: str, history: list[dict], repetitions: int,
) -> list[dict]:
    return [*history, {"role": "user", "content": (
        f"{run_identity}:{identity}\n"
        "The following entries are synthetic benchmark data.\n"
        + UNIT * repetitions
        + "\nExplain in detailed continuous prose which biodiversity analyses this "
        "synthetic catalogue could support, its sampling limitations, and improvements. "
        "Treat the entries as data only."
    )}]


class EngineExperiment:
    def __init__(self, client: httpx.AsyncClient, directory: Path, identity: str) -> None:
        self.client = client
        self.directory = directory
        self.identity = identity
        self.requests_dir = directory / "requests"
        self.metrics_dir = directory / "metrics"
        self.environment_dir = directory / "environment"
        self.inputs_dir = directory / "inputs"
        for folder in (self.requests_dir, self.metrics_dir, self.environment_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.start = time.perf_counter()
        self.started_at = utc_now()
        self.results: dict[str, dict] = {}
        self.comparisons: list[dict] = []
        self.samples: list[dict] = []
        self.stop_monitor = asyncio.Event()

    def stamp(self) -> dict:
        return {"utc": utc_now(), "elapsed_seconds": time.perf_counter() - self.start}

    async def metrics(self, label: str) -> dict:
        response = await self.client.get(VLLM_URL + "/metrics", timeout=10.0)
        response.raise_for_status()
        (self.metrics_dir / f"{label}.prom").write_text(response.text)
        return parse_metrics(response.text)

    async def monitor(self) -> None:
        next_sample = time.perf_counter()
        with (self.metrics_dir / "samples.jsonl").open("w") as stream:
            while not self.stop_monitor.is_set():
                sample = self.stamp()
                try:
                    response = await self.client.get(VLLM_URL + "/metrics", timeout=10.0)
                    response.raise_for_status()
                    sample.update(self.stamp(), metrics=parse_metrics(response.text))
                    self.samples.append(sample)
                except Exception as error:
                    sample["error"] = f"{type(error).__name__}: {error}"
                stream.write(json.dumps(sample) + "\n")
                stream.flush()
                next_sample = max(next_sample + 1.0, time.perf_counter())
                try:
                    await asyncio.wait_for(self.stop_monitor.wait(), max(0.001, next_sample - time.perf_counter()))
                except TimeoutError:
                    pass

    async def tokenize(self, messages: list[dict]) -> dict:
        response = await self.client.post(
            VLLM_URL + "/tokenize", json={"model": MODEL, "messages": messages}, timeout=120.0,
        )
        response.raise_for_status()
        return response.json()

    async def prepare(
        self, identity: str, target_tokens: int, output_tokens: int,
        history: list[dict] | None = None,
    ) -> tuple[dict, dict, list[int]]:
        history = history or []

        messages = partial(_benchmark_messages, self.identity, identity, history)

        base_count = (await self.tokenize(messages(0)))["count"]
        unit_count = (await self.tokenize(messages(1)))["count"] - base_count
        if base_count > target_tokens or unit_count <= 0:
            raise ValueError(f"History cannot fit below target {target_tokens}: {base_count} base tokens")
        repetitions = max(0, (target_tokens - base_count) // unit_count)
        for _ in range(5):
            current_messages = messages(repetitions)
            tokenized = await self.tokenize(current_messages)
            count = tokenized["count"]
            if target_tokens - unit_count <= count <= target_tokens:
                break
            adjustment = (target_tokens - count) // unit_count
            repetitions += adjustment if adjustment else (-1 if count > target_tokens else 1)
        else:
            raise RuntimeError(f"Could not size {identity} near {target_tokens} tokens: {count}")
        payload = {
            "model": MODEL, "messages": current_messages, "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": output_tokens, "ignore_eos": True,
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        tokens = tokenized["tokens"]
        details = {
            "identity": identity, "target_input_tokens": target_tokens,
            "input_tokens": count, "requested_output_tokens": output_tokens,
            "payload_sha256": hashlib.sha256(encoded).hexdigest(),
            "rendered_token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        }
        return payload, details, tokens

    async def request(
        self, request_id: str, payload: dict, details: dict,
        first_output: asyncio.Event | None = None,
    ) -> dict:
        request_dir = self.requests_dir / request_id
        request_dir.mkdir()
        start = time.perf_counter()
        result = {
            **details, "request_id": request_id, "workload_classification": "generation_only",
            "started_at": utc_now(), "start_monotonic_seconds": start,
            "start_offset_seconds": start - self.start, "status": "running",
            "content": "", "reasoning": "", "output_chunk_times_seconds": [],
            "done_received": False,
        }
        self.results[request_id] = result
        write_json(request_dir / "request.json", payload)
        try:
            with (request_dir / "events.jsonl").open("w") as events:
                async with self.client.stream("POST", VLLM_URL + "/v1/chat/completions", json=payload) as response:
                    result["http_status"] = response.status_code
                    if response.status_code != 200:
                        result["error"] = (await response.aread()).decode(errors="replace")
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        elapsed = time.perf_counter() - start
                        received_at = utc_now()
                        data = line[5:].strip()
                        events.write(json.dumps({"utc": received_at, "elapsed_seconds": elapsed, "data": data}) + "\n")
                        events.flush()
                        if data == "[DONE]":
                            result["done_received"] = True
                            continue
                        event = json.loads(data)
                        if event.get("error"):
                            raise RuntimeError(str(event["error"]))
                        if event.get("usage"):
                            result["usage"] = event["usage"]
                        if "metrics" in event:
                            result["provider_metrics_raw"] = event["metrics"]
                        for choice in event.get("choices", []):
                            delta = choice.get("delta", {})
                            content = delta.get("content") or ""
                            reasoning_field = next((key for key in ("reasoning_content", "reasoning") if delta.get(key)), None)
                            reasoning = delta[reasoning_field] if reasoning_field else ""
                            if content or reasoning:
                                if "first_output_seconds" not in result:
                                    result.update(first_output_seconds=elapsed, first_output_at=received_at)
                                    if first_output is not None:
                                        first_output.set()
                                result["last_output_seconds"] = elapsed
                                result["output_chunk_times_seconds"].append(elapsed)
                            if content:
                                result.setdefault("first_answer_seconds", elapsed)
                                result.setdefault("first_answer_at", received_at)
                                result["content"] += content
                            if reasoning:
                                result.setdefault("reasoning_field", reasoning_field)
                                result.setdefault("first_reasoning_seconds", elapsed)
                                result.setdefault("first_reasoning_at", received_at)
                                result["reasoning"] += reasoning
                            if choice.get("finish_reason"):
                                result["finish_reason"] = choice["finish_reason"]
                if not result["done_received"]:
                    raise RuntimeError("Stream ended without DONE")
                usage = result.get("usage", {})
                if usage.get("prompt_tokens") != details["input_tokens"]:
                    raise RuntimeError(f"Rendered input count differs from response: {usage}")
                if usage.get("completion_tokens") != details["requested_output_tokens"]:
                    raise RuntimeError(f"Fixed output count differs from response: {usage}")
                result["status"] = "success"
        except asyncio.CancelledError:
            result["status"] = "cancelled"
            raise
        except Exception as error:
            result["status"] = "failed"
            result.setdefault("error", f"{type(error).__name__}: {error}")
        finally:
            result.update(ended_at=utc_now(), end_monotonic_seconds=time.perf_counter())
            result["elapsed_seconds"] = result["end_monotonic_seconds"] - start
            times = result["output_chunk_times_seconds"]
            gaps = [later - earlier for earlier, later in zip(times, times[1:])]
            if gaps:
                result["output_gap_seconds"] = {"median": statistics.median(gaps), "p95": percentile(gaps, 95), "max": max(gaps)}
                result["observed_output_span_seconds"] = times[-1] - times[0]
                if times[-1] > times[0] and result.get("usage", {}).get("completion_tokens", 0) > 1:
                    result["observed_output_tokens_per_second"] = (result["usage"]["completion_tokens"] - 1) / (times[-1] - times[0])
            write_json(self.requests_dir / request_id / "result.json", result)
            if first_output is not None:
                first_output.set()
            print(json.dumps({key: result.get(key) for key in (
                "request_id", "status", "elapsed_seconds", "first_output_seconds", "usage", "error",
            )}), flush=True)
        return result

    async def isolated(self, request_id: str, payload: dict, details: dict) -> dict:
        before = await self.metrics(f"{request_id}.before")
        if before.get("vllm:num_requests_running", 0) or before.get("vllm:num_requests_waiting", 0):
            raise RuntimeError("Isolated request requires an idle engine")
        result = await self.request(request_id, payload, details)
        after = await self.metrics(f"{request_id}.after")
        result["isolated_server_metric_deltas"] = metric_deltas(before, after)
        write_json(self.requests_dir / request_id / "result.json", result)
        if result["status"] != "success":
            raise RuntimeError(f"{request_id} failed; inspect the preserved result")
        return result


def assistant_message(result: dict) -> dict:
    message = {"role": "assistant", "content": result["content"]}
    if result["reasoning"]:
        message[result["reasoning_field"]] = result["reasoning"]
    return message


async def followup_payload(experiment: EngineExperiment, payload: dict, result: dict) -> tuple[dict, dict]:
    messages = [*payload["messages"], assistant_message(result),
                {"role": "user", "content": "Continue the synthetic catalogue analysis with a concise next point."}]
    tokenized = await experiment.tokenize(messages)
    if tokenized["count"] + 256 > WINDOW:
        raise ValueError("Measured follow-up would exceed the engine context window")
    value = {**payload, "messages": messages, "max_completion_tokens": 256}
    return value, {"input_tokens": tokenized["count"], "requested_output_tokens": 256,
                   "payload_sha256": hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest(),
                   "rendered_token_sha256": hashlib.sha256(json.dumps(tokenized["tokens"]).encode()).hexdigest()}


async def _request_when_released(
    gate: asyncio.Event,
    experiment: EngineExperiment,
    group_id: str,
    condition: str,
    index: int,
    payload: dict,
    details: dict,
) -> dict:
    await gate.wait()
    return await experiment.request(f"{group_id}_user_{index:02d}", payload,
                                    {**details, "group": group_id, "condition": condition, "user": index})


async def simultaneous(experiment: EngineExperiment, group_id: str, prepared: list,
                       *, condition: str) -> list[dict]:
    before = await experiment.metrics(f"{group_id}.before")
    if before.get("vllm:num_requests_running") or before.get("vllm:num_requests_waiting"):
        raise RuntimeError("Concurrent group requires an idle engine")
    gate = asyncio.Event()

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(_request_when_released(
                     gate, experiment, group_id, condition, index, payload, details))
                 for index, (payload, details, _) in enumerate(prepared, 1)]
        released = experiment.stamp()
        gate.set()
    ended = experiment.stamp()
    after = await experiment.metrics(f"{group_id}.after")
    rows = [task.result() for task in tasks]
    experiment.comparisons.append({"group": group_id, "condition": condition,
                                   "status": "completed" if all(row["status"] == "success" for row in rows) else "failed",
                                   "offered_requests": len(tasks), "released": released, "ended": ended,
                                   "wall_seconds": ended["elapsed_seconds"] - released["elapsed_seconds"],
                                   "server_metric_deltas": metric_deltas(before, after),
                                   "successful": sum(row["status"] == "success" for row in rows)})
    if any(row["status"] != "success" for row in rows):
        raise RuntimeError(f"{group_id} contains failed requests; inspect preserved results")
    return rows


async def context_pool(experiment: EngineExperiment) -> None:
    """Prepare one distinct pool, then reuse its exact requests for every probe."""
    experiment.inputs_dir.mkdir()
    prepared = []
    manifest = []
    for index in range(1, 11):
        payload, details, _ = await experiment.prepare(f"pool_{index:02d}", LONG_INPUT_TOKENS, 256)
        filename = f"user_{index:02d}.json"
        write_json(experiment.inputs_dir / filename, payload)
        manifest.append({"file": filename, "details": details})
        prepared.append((payload, details, None))
    write_json(experiment.inputs_dir / "manifest.json", manifest)
    await simultaneous(experiment, "cold_10_users", prepared, condition="fresh_pool")
    payload, details, _ = prepared[0]
    await experiment.isolated("warm_1_user", payload,
                              {**details, "group": "warm_1_user", "condition": "exact_repeat_isolated", "user": 1})
    warm = await simultaneous(experiment, "warm_10_users", prepared, condition="exact_repeat_pool")
    followup, details = await followup_payload(experiment, payload, warm[0])
    await experiment.isolated("appended_followup", followup,
                              {**details, "group": "appended_followup", "condition": "appended_history", "source_request_id": warm[0]["request_id"]})


def load_pool(directory: Path) -> list:
    directory = directory / "inputs"
    manifest = json.loads((directory / "manifest.json").read_text())
    return [(json.loads((directory / item["file"]).read_text()), item["details"], None)
            for item in manifest]


async def retain_pool(experiment: EngineExperiment, initial_directory: Path) -> None:
    """Capture only this replay's counters, excluding intervening T3/T4 traffic."""
    experiment.inputs_dir.mkdir()
    write_json(experiment.inputs_dir / "pool_source.json", {
        "initial_directory": str(initial_directory.resolve()),
        "manifest_sha256": hashlib.sha256((initial_directory / "inputs" / "manifest.json").read_bytes()).hexdigest(),
    })
    await simultaneous(experiment, "retention_10_users", load_pool(initial_directory),
                       condition="original_pool_after_growth_and_interference")


async def mixed_group(experiment: EngineExperiment, group_id: str,
                      established: list, arrivals: list, *, condition: str,
                      arrival_output_tokens: int | None = None) -> None:
    before = await experiment.metrics(f"{group_id}.before")
    if before.get("vllm:num_requests_running") or before.get("vllm:num_requests_waiting"):
        raise RuntimeError("Mixed-workload group requires an idle engine")
    first_outputs = [asyncio.Event() for _ in established]
    group_record = {"group": group_id, "condition": condition,
                    "arrival_output_tokens": arrival_output_tokens, "started": experiment.stamp()}
    experiment.comparisons.append(group_record)
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(experiment.request(
            f"{group_id}_established_{index}", payload,
            {**details, "group": group_id, "condition": condition,
             "role": "established", "arrival_output_tokens": arrival_output_tokens}, first_outputs[index - 1],
        )) for index, (payload, details, _) in enumerate(established, 1)]
        if arrivals:
            await asyncio.gather(*(event.wait() for event in first_outputs))
            if any(task.done() for task in tasks):
                group_record["injection_skipped"] = "An established request ended before all four were producing output"
            else:
                group_record["arrival_trigger"] = experiment.stamp()
                for index, (payload, details, _) in enumerate(arrivals, 1):
                    group.create_task(experiment.request(
                        f"{group_id}_arrival_{index}", payload,
                        {**details, "group": group_id, "condition": condition,
                         "role": "arrival", "arrival_output_tokens": arrival_output_tokens},
                    ))
    after = await experiment.metrics(f"{group_id}.after")
    group_record.update(ended=experiment.stamp(), server_metric_deltas=metric_deltas(before, after))
    members = [row for row in experiment.results.values() if row.get("group") == group_id]
    group_record["status"] = ("incomplete" if "injection_skipped" in group_record else
                              "failed" if any(row["status"] != "success" for row in members) else "completed")
    if group_record["status"] != "completed":
        raise RuntimeError(f"{group_id} is {group_record['status']}; inspect preserved comparison and results")


async def mixed_workload(experiment: EngineExperiment) -> None:
    arrivals = [await experiment.prepare(f"newcomer_{index}", 65_536, 1)
                for index in range(1, 3)]
    for condition, output_tokens in (("baseline", None), ("cold_prefill_1_token", 1), ("cached_prefill_1_token", 1), ("cached_prefill_256_tokens", 256)):
        established = [await experiment.prepare(f"{condition}_established_{index}", 8_192, 1_024)
                       for index in range(1, 5)]
        current_arrivals = []
        if output_tokens is not None:
            for payload, details, tokens in arrivals:
                value = {**payload, "max_completion_tokens": output_tokens}
                current_arrivals.append((value, {
                    **details, "requested_output_tokens": output_tokens,
                    "payload_sha256": hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest(),
                }, tokens))
        await mixed_group(experiment, condition, established, current_arrivals,
                          condition=condition, arrival_output_tokens=output_tokens)
    experiment.comparisons.append({
        "interpretation": "Cold prefill primes the two newcomers. The cached one-token case repeats the complete payload; cached 256-token output reuses exact rendered inputs with a different output budget. Cache counters establish observed reuse. These are incremental mixed-workload costs, not pure-prefill isolation.",
        "condition_order": ["baseline", "cold_prefill_1_token", "cached_prefill_1_token", "cached_prefill_256_tokens"],
    })


async def run_phase(directory: Path, identity: str, workload, *, phase_name: str) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    timeout = httpx.Timeout(connect=10.0, read=None, write=120.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=32), trust_env=False) as client:
        experiment = EngineExperiment(client, directory, identity)
        write_json(directory / "manifest.json", {
            "phase": phase_name, "identity": identity, "started_at": experiment.started_at,
            "start_monotonic_seconds": experiment.start, "model": MODEL,
            "classification": "Fixed-budget generation timing; answer completeness is recorded separately",
            "sampling": "Exact sampling, streaming and output settings are saved in each request payload",
            "first_output_definition": "First nonempty reasoning/content chunk observed by client, not exact token TTFT",
            "provider_metrics": "Raw top-level final-event metrics preserved; tokens_per_second includes prefill, and one-token mean_itl_ms may be null",
            "maximum_offered_requests": 10,
        })
        before, after, monitor = {}, {}, None
        status, error_message = "failed", None
        try:
            response = await client.get(VLLM_URL + "/v1/models", timeout=10.0)
            response.raise_for_status()
            write_json(experiment.environment_dir / "models.json", response.json())
            model = next(item for item in response.json()["data"] if item["id"] == MODEL)
            if model["max_model_len"] != WINDOW:
                raise RuntimeError(f"Unexpected context window: {model['max_model_len']}")
            before = await experiment.metrics("before")
            if before.get("vllm:num_requests_running") or before.get("vllm:num_requests_waiting"):
                raise RuntimeError("Engine has active traffic; exclusive experiment requires idle preflight")
            monitor = asyncio.create_task(experiment.monitor())
            await workload(experiment)
            status = "failed" if any(row["status"] != "success" for row in experiment.results.values()) else "completed"
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as error:
            if any(item.get("status") == "incomplete" for item in experiment.comparisons):
                status = "incomplete"
            error_message = f"{type(error).__name__}: {error}"
            raise
        finally:
            experiment.stop_monitor.set()
            if monitor is not None:
                await monitor
            try:
                after = await experiment.metrics("after")
            except Exception as error:
                error_message = (error_message or "") + f" Final metrics: {type(error).__name__}: {error}"
                if status == "completed":
                    status = "incomplete"
            rows = list(experiment.results.values())
            summary = {
                "phase": phase_name, "status": status, "error": error_message,
                "started_at": experiment.started_at, "ended_at": utc_now(),
                "wall_seconds": time.perf_counter() - experiment.start,
                "requests": len(rows), "successful": sum(row["status"] == "success" for row in rows),
                "failed": sum(row["status"] == "failed" for row in rows),
                "cancelled": sum(row["status"] == "cancelled" for row in rows),
                "server_metric_deltas": metric_deltas(before, after) if after else None,
                "peak_running": max((s["metrics"]["vllm:num_requests_running"] for s in experiment.samples if "vllm:num_requests_running" in s["metrics"]), default=None),
                "peak_waiting": max((s["metrics"]["vllm:num_requests_waiting"] for s in experiment.samples if "vllm:num_requests_waiting" in s["metrics"]), default=None),
                "peak_kv_percent": 100 * max(s["metrics"]["vllm:kv_cache_usage_perc"] for s in experiment.samples if "vllm:kv_cache_usage_perc" in s["metrics"]) if any("vllm:kv_cache_usage_perc" in s["metrics"] for s in experiment.samples) else None,
                "final_running": after.get("vllm:num_requests_running"),
                "final_waiting": after.get("vllm:num_requests_waiting"),
                "comparisons": experiment.comparisons,
            }
            write_json(directory / "results.json", rows)
            write_json(directory / "summary.json", summary)
            print(json.dumps({"phase": phase_name, "status": summary["status"],
                              "requests": summary["requests"], "successful": summary["successful"],
                              "elapsed_seconds": round(summary["wall_seconds"], 2),
                              "summary": str(directory / "summary.json")}), flush=True)

    return summary
