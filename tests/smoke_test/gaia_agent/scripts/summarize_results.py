"""Build a compact Gaia/MTP report and detailed machine-readable measurements."""

import argparse
import csv
import json
import os
import statistics
import tempfile
from io import StringIO
from pathlib import Path

from tests.smoke_test.gaia_agent.scripts.benchmark_common import (
    CONCURRENCY,
    CONTEXT_CACHE,
    INTERFERENCE,
    context_compaction_events,
    existing_run_directory,
    active_context_tokens,
    percentile,
    report_directory,
)


SERVER_DURATIONS = ("time_to_first_token_seconds", "request_queue_time_seconds",
                    "request_prefill_time_seconds", "request_decode_time_seconds",
                    "e2e_request_latency_seconds", "request_time_per_output_token_seconds")


def write_report_text(path: Path, content: str) -> None:
    """Replace one generated report only after its complete contents are written."""
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def distribution(values: list[float]) -> dict | None:
    return {"min": min(values), "median": statistics.median(values),
            "p95": percentile(values, 95), "max": max(values)} if values else None


def output_overlap(rows: list[dict]) -> int:
    events = []
    for row in rows:
        if "first_output_seconds" in row and row.get("last_output_seconds", 0) > row["first_output_seconds"]:
            offset = row["start_offset_seconds"]
            events.extend([(offset + row["first_output_seconds"], 1),
                           (offset + row["last_output_seconds"], -1)])
    active = peak = 0
    for _, change in sorted(events):
        active += change
        peak = max(peak, active)
    return peak


def phase_files(run_dir: Path) -> list[tuple[Path, Path]]:
    return [(path, path.parent / "summary.json")
            for section in ("gaia", "mtp")
            for path in sorted((run_dir / section).rglob("results.json"))
            if (path.parent / "summary.json").exists()]


def server_means(counters: dict) -> dict:
    return {name: counters[f"vllm:{name}_sum"] / counters[f"vllm:{name}_count"]
            if counters.get(f"vllm:{name}_count") and f"vllm:{name}_sum" in counters else None for name in SERVER_DURATIONS}


def group_measurements(rows: list[dict]) -> dict:
    result = {}
    for group in dict.fromkeys(row["group"] for row in rows if row.get("group")):
        members = [row for row in rows if row.get("group") == group]
        selected = [row for row in members if row.get("role") == "established"] or members
        result[group] = {
            "requests": len(members), "successful": sum(row["status"] == "success" for row in members),
            "condition": members[0].get("condition"),
            "isolated_server_metric_deltas": members[0].get("isolated_server_metric_deltas") if len(members) == 1 else None,
            "arrival_output_tokens": members[0].get("arrival_output_tokens"),
            "measured_role": "established" if any(row.get("role") == "established" for row in members) else "all",
            "elapsed_seconds": distribution([row["elapsed_seconds"] for row in selected]),
            "provider_decode_tokens_per_second": distribution([rate for row in selected if (rate := engine_decode_rate(row)) is not None]),
            "client_first_output_seconds": distribution([row["first_output_seconds"] for row in selected if "first_output_seconds" in row]),
            "observed_output_tokens_per_second": distribution([row["observed_output_tokens_per_second"] for row in selected if "observed_output_tokens_per_second" in row]),
            "per_request_p95_chunk_gap_seconds": distribution([row["output_gap_seconds"]["p95"] for row in selected if row.get("output_gap_seconds")]),
            "max_overlapping_client_output_intervals": output_overlap(members),
            "arrival_start_offsets_seconds": [row["start_offset_seconds"] for row in members if row.get("role") == "arrival"],
        }
    return result


def summarize(run_dir: Path, output_dir: Path | None = None) -> dict:
    analysis, export_rows = {}, []
    for result_path, summary_path in phase_files(run_dir):
        label = str(result_path.parent.relative_to(run_dir))
        summary = json.loads(summary_path.read_text())
        rows = json.loads(result_path.read_text())
        if not isinstance(summary, dict):
            continue
        counters = summary.get("server_metric_deltas") or {}
        queries = counters.get("vllm:prefix_cache_queries_total", 0)
        model_turns = [turn for row in rows for turn in row.get("model_runtime", [])]
        derived = {
            "summary": summary, "recorded_requests": len(rows),
            "successful": sum(row["status"] == "success" for row in rows),
            "growth_requests": sum(row.get("classification") == "synthetic_context_acknowledgment" for row in rows),
            "elapsed_seconds": distribution([row["elapsed_seconds"] for row in rows]),
            "client_first_output_seconds": distribution([row["first_output_seconds"] for row in rows if "first_output_seconds" in row]),
            "model_turns_from_traces": len(model_turns),
            "model_decode_tokens_per_second": distribution([rate for turn in model_turns if (rate := gaia_decode_rate(turn)) is not None]),
            "tool_requests_from_traces": sum(row.get("trace_run", {}).get("counts", {}).get("tool_requests", 0) for row in rows),
            "trace_failures": [{"request_id": row["request_id"], "returned_result": row.get("trace_result")}
                               for row in rows if row.get("trace_run", {}).get("status") == "failed"],
            "observed_context_total_tokens": [{"request_id": row["request_id"],
                "tokens": active_context_tokens(row)}
                for row in rows if row.get("classification")],
            "active_context_from_traces": [{"request_id": row["request_id"],
                **row["active_context"]} for row in rows if isinstance(row.get("active_context"), dict)],
            "context_compactions": [{"request_id": row["request_id"], "events": events,
                "trace_path": row.get("trace_path")}
                for row in rows if (events := context_compaction_events(row))],
            "aggregate_generated_tokens_per_wall_second": counters["vllm:generation_tokens_total"] / summary["wall_seconds"] if summary.get("wall_seconds") and "vllm:generation_tokens_total" in counters else None,
            "prefix_cache_hit_fraction": counters["vllm:prefix_cache_hits_total"] / queries if queries and "vllm:prefix_cache_hits_total" in counters else None,
            "server_mean_seconds_per_model_request": server_means(counters),
            "max_overlapping_client_output_intervals": output_overlap(rows),
            "per_request_observed_output_tokens_per_second": distribution([row["observed_output_tokens_per_second"] for row in rows if "observed_output_tokens_per_second" in row]),
            "prompt_tokens_total_from_responses": sum(row.get("usage", {}).get("prompt_tokens", 0) for row in rows),
            "completion_tokens_total_from_responses": sum(row.get("usage", {}).get("completion_tokens", 0) for row in rows),
            "groups": group_measurements(rows),
            "provider_metrics_raw": {row["request_id"]: row["provider_metrics_raw"] for row in rows if "provider_metrics_raw" in row},
            "gaia_provider_metrics_seconds": [{"request_id": row["request_id"], "turn": turn.get("turn"),
                "purpose": turn.get("purpose", "response"),
                "token_usage": turn.get("token_usage"), "metrics": turn.get("provider_metrics")}
                for row in rows for turn in row.get("model_runtime", [])],
            "metric_limits": "Cache queries are server lookup counters, not submitted input totals. Provider tokens_per_second includes prefill. Client chunk intervals are not exact token/decoder intervals.",
        }
        if label.startswith("mtp/"):
            derived["mtp_cells"] = mtp_cells(rows, summary, result_path.parent.name)
        analysis[label] = derived
        for row in rows:
            export_rows.append({"phase": label, "request": row.get("request_id", row.get("user")),
                                **{key: row.get(key) for key in (
                                    "status", "http_status", "condition", "group", "role",
                                    "arrival_output_tokens", "start_offset_seconds", "elapsed_seconds",
                                    "first_output_seconds", "last_output_seconds", "input_tokens",
                                    "requested_output_tokens", "observed_output_tokens_per_second", "finish_reason", "error")}})
    destination = output_dir or report_directory(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_report_text(destination / "analysis.json", json.dumps(analysis, indent=2))
    if export_rows:
        stream = StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(export_rows[0]))
        writer.writeheader()
        writer.writerows(export_rows)
        write_report_text(destination / "comparison.csv", stream.getvalue())
    return analysis


def engine_decode_rate(row: dict) -> float | None:
    tokens = row.get("usage", {}).get("completion_tokens")
    ms = (row.get("provider_metrics_raw") or {}).get("generation_time_ms")
    return (tokens - 1) * 1000 / ms if tokens and tokens > 1 and ms and ms > 0 else None


def gaia_decode_rate(turn: dict) -> float | None:
    tokens = turn.get("token_usage", {}).get("completion_tokens")
    seconds = (turn.get("provider_metrics") or {}).get("generation_seconds")
    return (tokens - 1) / seconds if tokens and tokens > 1 and seconds and seconds > 0 else None


def mtp_cells(rows: list[dict], summary: dict, mode: str) -> list[dict]:
    cells = []
    for users in (1, 3, 5, 7, 10):
        group = f"measured_users_{users:02d}"
        members = [row for row in rows if row.get("group") == group and row.get("role") == "measured"]
        if not members:
            continue
        comparison = next((item for item in summary.get("comparisons", []) if item.get("group") == group), {})
        rates = [rate for row in members if (rate := engine_decode_rate(row)) is not None]
        wall = comparison.get("wall_seconds")
        specs = [(row.get("provider_metrics_raw") or {}).get("speculative_decoding") for row in members]
        drafted = sum(value.get("num_draft_tokens", 0) for value in specs if value)
        accepted = sum(value.get("num_accepted_draft_tokens", 0) for value in specs if value)
        counters = comparison.get("server_metric_deltas", {})
        queries, hits = counters.get("vllm:prefix_cache_queries_total"), counters.get("vllm:prefix_cache_hits_total")
        complete_usage = all("completion_tokens" in row.get("usage", {}) for row in members)
        verified = (len(members) == users and comparison.get("status") == "completed"
                    and all(row["status"] == "success" and row.get("measurement_status") == "verified" for row in members))
        complete_spec = all(isinstance(value, dict) and "num_accepted_draft_tokens" in value and "num_draft_tokens" in value for value in specs)
        cells.append({
            "mode": mode, "users": users, "requests": len(members),
            "successful": sum(row["status"] == "success" for row in members),
            "status": "verified" if verified else "incomplete",
            "decode_tps": statistics.mean(rates) if verified and len(rates) == users else None,
            "batch_output_tps": sum(row["usage"]["completion_tokens"] for row in members) / wall if verified and complete_usage and wall else None,
            "client_first_output_seconds": distribution([row["first_output_seconds"] for row in members if "first_output_seconds" in row]),
            "draft_acceptance_fraction": accepted / drafted if verified and drafted and complete_spec else None,
            "draft_tokens": drafted if complete_spec else None,
            "accepted_draft_tokens": accepted if complete_spec else None,
            "prefix_cache_hit_fraction": hits / queries if verified and queries and hits is not None else None,
        })
    return cells


def number(value, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if value is not None else "—"


def median(measurement: dict | None):
    return measurement.get("median") if measurement else None


def table(title: str, columns: list[str], rows: list[list]) -> dict:
    return {"title": title, "columns": columns, "rows": rows}


def build_suite_report(run_dir: Path, analysis: dict, suite_summary: dict) -> dict:
    gaia = {name: value for name, value in analysis.items() if not name.startswith("mtp/")}
    mtp = {name: value for name, value in analysis.items() if name.startswith("mtp/")}
    ordinary, contexts, interference, completion = [], [], [], []
    for name, value in gaia.items():
        phase = value["summary"]
        if name.startswith(CONCURRENCY + "/"):
            ordinary.append([int(Path(name).name.removeprefix("users_")), number(median(value["elapsed_seconds"])), number(median(value["model_decode_tokens_per_second"]))])
        for group, measures in value["groups"].items():
            row = [group, number(median(measures["client_first_output_seconds"])), number(median(measures["elapsed_seconds"])), number(median(measures["provider_decode_tokens_per_second"]))]
            if name.startswith(CONTEXT_CACHE + "/"):
                contexts.append(row)
            elif name == INTERFERENCE:
                interference.append([group, row[2], row[3]])
    planned_phases = suite_summary["gaia_phases"]
    families = dict.fromkeys(CONTEXT_CACHE if name.startswith(CONTEXT_CACHE + "/") else name
                             for name in planned_phases)
    for family in families:
        expected = 6 if family == CONCURRENCY else sum(
            name == family or name.startswith(family + "/") for name in planned_phases)
        values = [value for name, value in gaia.items() if name == family or name.startswith(family + "/")]
        status = "completed" if len(values) == expected and all(value["summary"]["status"] == "completed" for value in values) else "incomplete" if values else "unrecorded"
        completion.append([family, f"{sum(v['successful'] for v in values)}/{sum(v['recorded_requests'] for v in values)}", status])
    gaia_tables = [table("Completion", ["Test/group", "Succeeded", "Status"], completion)] if completion else []
    if ordinary:
        gaia_tables.append(table("Ordinary concurrency", ["Users", "Median chat s", "Median model decode tok/s"], ordinary))
    if contexts:
        gaia_tables.append(table("Near-window cache", ["Condition", "First output s", "Completion s", "Decode tok/s"], contexts))
    if interference:
        gaia_tables.append(table("Effect on established streams", ["Arrivals", "Completion s", "Decode tok/s"], interference))
    cells = [cell for value in mtp.values() for cell in value.get("mtp_cells", [])]
    modes = ("off", "k1", "k2", "k3")
    matrix, ten = [], []
    for mode in modes:
        by_level = {cell["users"]: cell for cell in cells if cell["mode"] == mode}
        matrix.append([mode, *(number(by_level.get(n, {}).get("decode_tps"), 1) for n in (1, 3, 5, 7, 10))])
        top = by_level.get(10, {})
        rate = top.get("draft_acceptance_fraction")
        ten.append([mode, number(top.get("batch_output_tps"), 1), number(rate * 100 if rate is not None else None, 1)])
    issues = []
    if suite_summary.get("status") != "completed":
        issues.append("Suite is incomplete or has pending sections; use the saved section statuses before interpreting missing measurements.")
    for name, value in analysis.items():
        phase = value["summary"]
        if phase.get("status") != "completed":
            issues.append(f"{name}: {phase.get('error') or phase.get('details', {}).get('workload_error') or phase['status']}.")
        for failure in value.get("trace_failures", []):
            issues.append(f"{name}/{failure['request_id']}: failed model turn; inspect the saved trace.")
    issues.extend(["Use cache counters to verify cold/warm conditions and inspect generation gaps when long inputs arrive.",
                   "One batch per condition is a screening result; small differences between MTP modes need repeat evidence."])
    return {
        "title": "Gaia suite findings", "basis": f"Campaign {run_dir.name}; selection: {suite_summary['target']}; status: {suite_summary['status']}. Completion covers the selected categories.",
        "sections": [
            {"title": "Gaia — application, context and interference", "configuration": "T1–T4 use the running Gaia/model configuration; engine metadata and traces are saved with each phase.",
             "status": f"{sum(v['successful'] for v in gaia.values())}/{sum(v['recorded_requests'] for v in gaia.values())} recorded submissions succeeded." if gaia else "Pending." if "gaia" in suite_summary["sections"] else "Not selected.",
             "tables": gaia_tables, "notes": ["Gaia chats include model/tool rounds. Direct context outputs are fixed at 256 tokens; interference established streams use 1,024. First output, completion and provider decode measure different intervals."],
             "sources": [str((report_directory(run_dir) / "analysis.json").resolve())]},
            {"title": "MTP — short-input decode comparison", "configuration": "Off/k1/k2/k3; matched approximately 2k inputs and 512 outputs at 1/3/5/7/10 users. Runner, startup configuration and actual k are checked per mode.",
             "status": f"{sum(cell['successful'] for cell in cells)}/{sum(cell['requests'] for cell in cells)} measured requests succeeded; warmups excluded from the matrix." if cells else "Pending." if any(name.startswith("mtp/") for name in suite_summary["sections"]) else "Not selected.",
             "tables": [table("Mean individual decode, tok/s", ["MTP", "1 user", "3", "5", "7", "10"], matrix), table("Ten-user batch", ["MTP", "Output tok/s (wall clock)", "Draft acceptance %"], ten)],
             "notes": ["One 32-token warmup at each batch size is charged to suite time but excluded from measured results. Decode rate is (output tokens − 1) / provider generation time; batch throughput uses shared wall time. These short-input results do not establish long-context MTP gains."],
             "sources": [str((run_dir / name / "summary.json").resolve()) for name in mtp]},
        ], "worth_inspecting": issues,
    }


def render_report(report: dict, destination: Path) -> None:
    """Render the self-contained report, including any retained evidence links."""
    lines = [f"# {report['title']}", "", report["basis"], ""]
    for section in report["sections"]:
        lines.extend([f"## {section['title']}", "", section["configuration"], "", section["status"], ""])
        for item in section["tables"]:
            if not item["rows"]:
                continue
            lines.extend([f"**{item['title']}**", "", "| " + " | ".join(item["columns"]) + " |", "| " + " | ".join("---" for _ in item["columns"]) + " |"])
            for row in item["rows"]:
                lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
            lines.append("")
        for note in section["notes"]:
            lines.extend([note, ""])
        links = [f"[{Path(source).stem}]({os.path.relpath(source, destination.parent)})" for source in section["sources"]]
        if links:
            lines.extend(["Evidence: " + ", ".join(links) + ".", ""])
    lines.extend(["## Worth inspecting", "", *(f"- {item}" for item in report["worth_inspecting"]), ""])
    write_report_text(destination, "\n".join(lines))


def write_suite_report(run_dir: Path, analysis: dict, suite_summary: dict, output_dir: Path | None = None) -> Path:
    report = build_suite_report(run_dir, analysis, suite_summary)
    output = output_dir or report_directory(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    report["sections"][0]["sources"] = [str((output / "analysis.json").resolve())]
    write_report_text(output / "report.json", json.dumps(report, indent=2) + "\n")
    destination = output / "findings.md"
    render_report(report, destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="Run folder name under logs/smoke_test/gaia_agent")
    source.add_argument("--report-json", type=Path, help="Render an existing report without raw measurements")
    args = parser.parse_args()
    if args.report_json:
        source_path = args.report_json.expanduser().resolve()
        output = source_path.parent
        output.mkdir(parents=True, exist_ok=True)
        destination = output / "findings.md"
        render_report(json.loads(source_path.read_text()), destination)
        print(f"Report: {destination}")
    else:
        run_dir = existing_run_directory(args.run)
        state = json.loads((run_dir / "suite_summary.json").read_text())
        analysis = summarize(run_dir)
        report = write_suite_report(run_dir, analysis, state)
        print(f"Report: {report}; {sum(value['recorded_requests'] for value in analysis.values())} recorded submissions.")
