"""T5: one uniform MTP comparison per externally configured engine mode."""

from __future__ import annotations

import hashlib
import json
import re
from functools import partial
from pathlib import Path

from tests.smoke_test.gaia_agent.scripts.benchmark_common import VLLM_URL, write_json
from tests.smoke_test.gaia_agent.scripts.engine_experiments import (
    MODEL,
    EngineExperiment,
    run_phase,
    simultaneous,
)

MODES = ("off", "k1", "k2", "k3")
LEVELS = (1, 3, 5, 7, 10)

CONTEXT = '''You are analyzing a fictional biodiversity collection. Think carefully, then give a concrete, rigorous answer. All information below is invented benchmark data, not a real museum collection.

Tables:
occurrence(occurrence_id TEXT PRIMARY KEY, scientific_name TEXT, accepted_name TEXT, locality TEXT, event_date TEXT, decimal_latitude DOUBLE, decimal_longitude DOUBLE, individual_count INTEGER, institution_code TEXT, collection_code TEXT, catalog_number TEXT, basis_of_record TEXT).
identification(identification_id TEXT PRIMARY KEY, occurrence_id TEXT, identified_by TEXT, date_identified TEXT, identification_qualifier TEXT).
media(media_id TEXT PRIMARY KEY, occurrence_id TEXT, media_type TEXT, license TEXT).

Contracts: occurrence_id is an opaque stable identifier and must never be generated from row order. Empty text and SQL NULL remain distinct unless a field's contract explicitly maps them. Source Unicode spelling and catalog-number leading zeros must survive round trips. Identification and media are one-to-many. There are 12 million occurrences, 19 million identifications, and 8 million media items. Some occurrences have no identifications or media. Query results are paginated by occurrence_id, using a fixed snapshot. Page size is 500. Filtering by date must allow partial dates without silently inventing missing days. Geographic bounds include the antimeridian. User-facing counts must distinguish occurrences from joined child rows.

Example source observations:
'''
for i in range(1, 25):
    CONTEXT += (f'R{i:02d}: occurrence_id=demo-{i:06d}; scientific_name=' +
                ['Poa annua', 'Quercus garryana', 'Rana aurora', 'Bombus occidentalis'][i % 4] +
                f'; locality=Site {i % 7}; event_date=20{10 + i % 15}-0{1 + i % 9}; catalog_number=00{i:04d}; individual_count={1 + i % 6}; identifications={i % 4}; media={i % 3}.\n')

QUESTIONS = [
    'Design an exact SQL query and a bounded-memory processing plan for reporting distinct occurrences by accepted taxon and year with identification and media summaries. Explain join multiplication, partial dates, missing identifications, nulls, Unicode preservation, stable pagination under concurrent updates, useful indexes, and reconciliation of input/output counts. Work through at least three concrete edge cases and justify the order of operations.',
    'Design a robust reversible normalization pipeline for these records, including a precise representation for empty values, partial dates, signed coordinates, antimeridian bounding boxes, Unicode scientific names, opaque identifiers, and leading-zero catalog numbers. Provide pseudocode and worked examples, then analyze determinism, memory complexity, duplicate rows, and the invariants needed to reconcile failed, skipped, and written records.',
    'Design an inference workload scheduler for an assistant querying this collection: at most ten running requests and five waiting, token streaming, cancellations, and reusable prompt prefixes. Analyze admission races, fairness, queue accounting, cancellation during generation, disconnects, retries, and cache isolation. Explain how to measure per-request decode speed separately from prefill and end-to-end throughput, with concrete examples and failure cases.'
]


def command_option(command: list[str], name: str) -> str | None:
    values = []
    for index, item in enumerate(command):
        if item == name:
            if index + 1 == len(command) or command[index + 1].startswith("--"):
                raise ValueError(f"Missing value for {name} in configuration evidence")
            values.append(command[index + 1])
        elif item.startswith(name + "="):
            values.append(item.split("=", 1)[1])
    if len(values) > 1:
        raise ValueError(f"Repeated {name} in configuration evidence")
    return values[0] if values else None


def configuration(evidence_path: Path, mode: str) -> tuple[dict, dict]:
    raw = evidence_path.read_bytes()
    evidence = json.loads(raw)
    if isinstance(evidence, list):
        if len(evidence) != 1:
            raise ValueError("Configuration evidence must describe exactly one container")
        evidence = evidence[0]
    config = evidence["Config"]
    command = config["Cmd"]
    environment = config["Env"]
    runner = [value for value in environment if value.startswith("VLLM_USE_V2_MODEL_RUNNER=")]
    if len(runner) != 1 or runner[0].split("=", 1)[1] not in {"0", "1"}:
        raise ValueError("Configuration evidence must explicitly set VLLM_USE_V2_MODEL_RUNNER=0 or =1")
    encoded_spec = command_option(command, "--speculative-config")
    spec = json.loads(encoded_spec) if encoded_spec is not None else None
    k = 0 if mode == "off" else int(mode[1:])
    if (k == 0 and encoded_spec is not None) or (k and (
        not isinstance(spec, dict) or spec.get("method") != "mtp" or spec.get("num_speculative_tokens") != k
    )):
        raise ValueError(f"Saved speculative configuration does not match requested mode {mode}")
    target_command = []
    skip_value = False
    for item in command:
        if skip_value:
            skip_value = False
            continue
        if item in {"--speculative-config", "--per-request-spec-decode-metrics"}:
            skip_value = True
        elif not item.startswith(("--speculative-config=", "--per-request-spec-decode-metrics=")):
            target_command.append(item)
    target = {"image": evidence["Image"], "command": target_command, "environment": sorted(environment)}
    labels = config.get("Labels") or {}
    image_version = (labels.get("ai.vllm.image.tag") or labels.get("org.opencontainers.image.version")
                     or config.get("Image", ""))
    match = re.search(r"(?:^|:|/)v?(\d+\.\d+\.\d+)(?:$|[-+])", image_version)
    if match is None:
        raise ValueError("Docker image evidence lacks a version tag; retain its vLLM image labels")
    context = command_option(command, "--max-model-len")
    model = command_option(command, "--served-model-name")
    if context is None or model != MODEL:
        raise ValueError(f"Evidence must specify --max-model-len and --served-model-name {MODEL}")
    return evidence, {
        "mode": mode, "k": k, "evidence_path": str(evidence_path.resolve()),
        "evidence_sha256": hashlib.sha256(raw).hexdigest(),
        "container_id": evidence["Id"], "container_started_at": evidence["State"]["StartedAt"],
        "model_runner": runner[0], "speculative_config": spec,
        "target_configuration": target,
        "target_configuration_sha256": hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest(),
        "expected_version": match[1], "expected_context_tokens": int(context), "expected_model": model,
        "proof_limit": "Saved Docker configuration is caller-supplied; HTTP version/model and response speculation verify exposed fields, not the live container image digest",
    }


async def verify_configuration(experiment: EngineExperiment, evidence: dict, verification: dict) -> None:
    write_json(experiment.environment_dir / "container.json", evidence)
    errors = []
    for path in sorted(experiment.directory.parent.glob("*/environment/verification.json")):
        other = json.loads(path.read_text())
        if other["target_configuration_sha256"] != verification["target_configuration_sha256"]:
            errors.append(f"Target image/command/environment differs from {path.parent.parent.name}")
        other_spec = other.get("speculative_config")
        spec = verification["speculative_config"]
        if other_spec and spec and ({key: value for key, value in other_spec.items() if key != "num_speculative_tokens"}
                                   != {key: value for key, value in spec.items() if key != "num_speculative_tokens"}):
            errors.append(f"MTP draft settings differ from {path.parent.parent.name}")
    response = await experiment.client.get(VLLM_URL + "/version", timeout=10)
    response.raise_for_status()
    version = response.json()
    write_json(experiment.environment_dir / "version.json", version)
    verification["observed_version"] = version
    if version.get("version", "").split("+", 1)[0] != verification["expected_version"]:
        errors.append("HTTP version differs from saved image version")
    models = json.loads((experiment.environment_dir / "models.json").read_text())["data"]
    model = next((item for item in models if item["id"] == verification["expected_model"]), None)
    if model is None or model.get("max_model_len") != verification["expected_context_tokens"]:
        errors.append("HTTP served model/context differs from saved configuration")
    verification.update(status="verified_exposed_fields" if not errors else "incomplete", errors=errors)
    write_json(experiment.environment_dir / "verification.json", verification)
    if errors:
        experiment.comparisons.append({"check": "configuration", "status": "incomplete", "errors": errors})
        raise ValueError("; ".join(errors))


async def prepare_requests(experiment: EngineExperiment, level: int, *, warmup: bool) -> list:
    phase = "warm" if warmup else "measured"
    output_tokens = 32 if warmup else 512
    prepared = []
    for index in range(level):
        identity = f"{experiment.identity}:mtp:n{level:02d}:{phase}:user{index:02d}"
        messages = [{"role": "user", "content": identity + "\n" + CONTEXT + "\nTask: "
                     + QUESTIONS[index % len(QUESTIONS)] + f" Use case identifier B{index:02d} in your answer."}]
        template = {"enable_thinking": True, "preserve_thinking": True}
        response = await experiment.client.post(VLLM_URL + "/tokenize", json={
            "model": MODEL, "messages": messages, "chat_template_kwargs": template,
        }, timeout=120)
        response.raise_for_status()
        tokenized = response.json()
        payload = {"model": MODEL, "messages": messages, "max_completion_tokens": output_tokens,
                   "temperature": 1.0, "top_p": .95, "top_k": 20, "seed": 5100 + index,
                   "chat_template_kwargs": template, "stream": True,
                   "stream_options": {"include_usage": True}, "ignore_eos": True}
        details = {
            "identity": identity, "input_tokens": tokenized["count"], "requested_output_tokens": output_tokens,
            "payload_sha256": hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest(),
            "rendered_token_sha256": hashlib.sha256(json.dumps(tokenized["tokens"]).encode()).hexdigest(),
            "role": "warmup" if warmup else "measured", "concurrency": level,
        }
        prepared.append((payload, details, None))
    return prepared


def verify_measurements(experiment: EngineExperiment, rows: list[dict], mode: str) -> None:
    k = 0 if mode == "off" else int(mode[1:])
    failed = []
    for row in rows:
        metrics = row.get("provider_metrics_raw") or {}
        errors = []
        if "speculative_decoding" not in metrics:
            errors.append("Missing speculative_decoding field; mode is unverified")
        else:
            spec = metrics["speculative_decoding"]
            if k == 0:
                if spec is not None:
                    errors.append("Off mode returned speculative decoding")
            elif not isinstance(spec, dict) or spec.get("num_spec_tokens") != k or not (
                spec.get("num_draft_tokens", 0) > 0 and spec.get("num_spec_steps", 0) > 0
            ):
                errors.append("Returned speculative counters do not confirm the requested MTP mode")
        if not isinstance(metrics.get("generation_time_ms"), (int, float)) or metrics["generation_time_ms"] <= 0:
            errors.append("Missing positive provider generation duration")
        row.update(mode=mode, measurement_status="incomplete" if errors else "verified", measurement_errors=errors)
        write_json(experiment.requests_dir / row["request_id"] / "result.json", row)
        if errors:
            failed.append(row["request_id"])
    if failed:
        experiment.comparisons[-1]["status"] = "incomplete"
        raise RuntimeError(f"Mode/generation measurement unverified for {', '.join(failed)}")


async def _measure_mode(
    evidence: dict, verification: dict, mode: str, identity: str,
    experiment: EngineExperiment,
) -> None:
    await verify_configuration(experiment, evidence, verification)
    write_json(experiment.directory / "workload.json", {
        "mode": mode, "levels": LEVELS, "measured_requests": 26, "warmup_requests": 26,
        "measured_output_tokens": 512, "warmup_output_tokens": 32,
        "identity": identity, "matched_inputs_across_modes": True,
        "prefix_policy": "Distinct early prefix for each level, request and warmup/measured role",
        "classification": "Fixed-budget short-input generation; reasoning and answer retained separately",
        "decode_formula": "(completion_tokens - 1) * 1000 / generation_time_ms",
        "summed_decode_formula": "Sum of individual decode rates; not shared-wall-clock throughput",
    })
    for level in LEVELS:
        for warmup in (True, False):
            group = f"{'warmup' if warmup else 'measured'}_users_{level:02d}"
            prepared = await prepare_requests(experiment, level, warmup=warmup)
            for _, details, _ in prepared:
                details["mode"] = mode
            rows = await simultaneous(experiment, group, prepared,
                                      condition="warmup" if warmup else "measured")
            experiment.comparisons[-1]["mode"] = mode
            if not warmup:
                verify_measurements(experiment, rows, mode)


async def run_mode(run_dir: Path, identity: str, mode: str, evidence_path: Path) -> dict:
    evidence, verification = configuration(evidence_path, mode)

    workload = partial(_measure_mode, evidence, verification, mode, identity)

    return await run_phase(run_dir / "mtp" / mode, identity, workload, phase_name=f"mtp/{mode}")
