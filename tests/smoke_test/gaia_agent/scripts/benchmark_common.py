"""Shared measurements for the Gaia and MTP benchmark sections."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

GAIA_URL = "http://127.0.0.1:8765"
VLLM_URL = "http://127.0.0.1:8000"
MODEL = "default_model"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
LOG_ROOT = REPOSITORY_ROOT / "logs" / "smoke_test" / "gaia_agent"
REPORT_ROOT = REPOSITORY_ROOT / "reports" / "smoke_test"
CONCURRENCY = "gaia/ordinary_concurrency"
CONTEXT_CACHE = "gaia/context_cache"
CONVERSATION = "gaia/growing_conversation"
CONVERSATION_SETUP = "gaia/conversation_setup"
INTERFERENCE = "gaia/prefill_interference"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Explicit endpoints and served-model identity for one campaign."""

    gaia_url: str = GAIA_URL
    engine_url: str = VLLM_URL
    model: str = MODEL

    def snapshot(self) -> dict[str, str]:
        return asdict(self)


def sanitized_url(value: str) -> str:
    """Keep endpoint routing evidence without persisting credentials or queries."""
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))


def configuration_identity(config: BenchmarkConfig) -> dict:
    snapshot = {"gaia_url": sanitized_url(config.gaia_url),
                "engine_url": sanitized_url(config.engine_url), "requested_model_id": config.model}
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    return {"snapshot": snapshot, "sha256": hashlib.sha256(encoded).hexdigest()}


METRIC_NAMES = {
    "vllm:num_requests_running", "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
    "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
    "vllm:request_success_total",
    "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    *(f"vllm:{name}_{suffix}" for name in (
        "time_to_first_token_seconds", "request_queue_time_seconds",
        "request_prefill_time_seconds", "request_decode_time_seconds",
        "request_time_per_output_token_seconds", "e2e_request_latency_seconds",
    ) for suffix in ("sum", "count")),
}


def validate_run_name(name: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError("Run names use lowercase letters, digits, underscores or hyphens; supply a name, not a path")
    return name


def report_directory(run_dir: Path) -> Path:
    return REPORT_ROOT / validate_run_name(run_dir.name)


def new_run_directory(label: str) -> Path:
    label = validate_run_name(label)
    stem = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_utc") + "_" + label
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for attempt in count(1):
        path = LOG_ROOT / (stem if attempt == 1 else f"{stem}_{attempt}")
        if report_directory(path).exists():
            continue
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def existing_run_directory(name: str) -> Path:
    path = LOG_ROOT / validate_run_name(name)
    if not path.is_dir() or path.resolve().parent != LOG_ROOT.resolve():
        raise ValueError(f"No run named {name} under {LOG_ROOT}")
    return path


def parse_metrics(body):
    metrics = {}
    for line in body.splitlines():
        match = re.match(r"([^ {]+)(?:\{[^}]*\})? ([^ ]+)", line)
        if match and match[1] in METRIC_NAMES:
            metrics[match[1]] = metrics.get(match[1], 0.0) + float(match[2])
    return metrics


def percentile(values, percent):
    values = sorted(values)
    index = (len(values) - 1) * percent / 100
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def active_context_tokens(row: dict) -> int | None:
    """Read the final rendered-context measurement from the private trace."""
    context = row.get("active_context")
    if not isinstance(context, dict):
        return None
    count, limit = context.get("prompt_tokens"), context.get("limit_tokens")
    if type(count) is not int or count < 0 or type(limit) is not int or limit <= 0:
        return None
    return count


def context_compaction_events(row: dict) -> list[dict]:
    """Retain streamed evidence of completed compactions for one request."""
    return [observation for observation in row.get("events", [])
            if observation.get("event", {}).get("type") == "progress"
            and observation["event"].get("phase") == "context_compacted"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_deltas(before: dict, after: dict) -> dict:
    return {key: after[key] - value for key, value in before.items()
            if key in after and key.endswith(("_total", "_sum", "_count"))}
