# Gaia test suite

The suite has **two sections: Gaia and MTP**. Run the entire suite, either section,
or one Gaia category; each campaign records its selected scope in one report.
The [audited baseline findings](../../../reports/smoke_test/2026-09-14_audited_baseline/findings.md)
combine the latest measured Gaia and MTP evidence; the optimized workload is not
yet measured live. Suite code lives here in `scripts/`. Generated reports live
under `reports/smoke_test/<run>/`; raw evidence and runner state live under
`logs/smoke_test/gaia_agent/<run>/`.

## Section A — Gaia

| Test | What and why | Procedure |
|---|---|---|
| T1 — Ordinary concurrency | Measure complete chat latency with real model/database tools | One burst at **1, 2, 3, 5, 7, 10** users: **28 chats**. Retain one final-burst Fungi conversation for T3. |
| T2 — Context capacity/cache | Separate cold ingestion, cached decoding, continuation and retention | Prepare ten distinct ~130,048-token inputs once, with **256 outputs**. Run cold10 → isolated warm1 → warm10 → appended follow-up1; replay original10 after T3/T4. **32 requests**. |
| T3 — Growing conversation | Exercise server-owned history and real follow-ups near the window | Grow the retained T1 chat toward **122,880 tokens** using full legal messages. Then reuse its observations and request new data. **Growth turns + two follow-ups**. |
| T4 — Prefill interference | Separate fresh-input interference from added generation | Four 8k-input/1,024-output streams; add two 64k newcomers after all four begin output. One pass: baseline, cold/1 output, warm/1, warm/256. **22 requests**. Cold arrivals prime both warm cases. |

T2 initial and retention have separate metric boundaries. T3/T4 provide the
intervening work; no extra retention workload or timed pause is added. T3 records
naturally crossed checkpoints, actual token usage and failures. A failed turn
stops its dependent follow-ups; independent tests can continue when the engine is
healthy and idle. There is no test-only 2,048-token reserve or silent history trim.

**Gaia cost: 84 fixed submissions plus one chat's growth turns**, approximately
100–110 based on previous growth, versus 587 selected submissions previously.
Cuts include duplicate preparation, repeated ladders and nine growing histories.
This no longer directly tests ten long retained Gaia chats together; T2 still tests
ten near-window engine requests. Repeated statistics, varied query/answer cases
and the cold/256-output interference condition are also outside this lean section.

## Section B — MTP

Test **off, k1, k2 and k3** at **1, 3, 5, 7, 10 concurrent requests**. Each mode
runs one measured batch per level with the same approximately 2k input and
**512 output tokens**. Inputs, questions, seeds and sampling match across modes:
temperature 1, top-p 0.95, top-k 20, thinking enabled. Distinct early prefixes
separate every level/request and warmup from measured traffic.

At each concurrency, run a short **32-token warmup batch** before measurement.
Saved logs show compilation at different batch sizes, so warming only the largest
size would not establish that the other sizes were ready. Warmups are recorded
and charged to suite time, but excluded from comparison cells.

| MTP work | Previous matrix/source runs | Unified section |
|---|---:|---:|
| Measured requests | 212 | **104** |
| Measured output tokens | 114,688 | **53,248** |
| Warmup requests | 112 | **104** |
| Warmup output tokens | 14,336 | **3,328** |

The second measured batches, extra single-user samples, separate sanity queries,
k4/off-recheck and duplicate runner sweeps are removed from active execution.
The new output workload is **56,576 tokens including warmups**, versus 129,024
previously; this comparison excludes additional old sanity/recheck work.
Each mode needs one engine startup. Recorded startups took roughly 120–151 seconds;
startup/compilation cost remains separate from measured request latency.

**Mode verification:** supply a saved `docker inspect` JSON for the running mode.
The runner checks its speculative setting, explicit model-runner environment,
image/target settings, and HTTP version/model/context. Each measured response must
also confirm the actual speculative token count and draft activity. Off requires
an explicit null speculation field; missing telemetry is unverified. Configuration
mismatches or incomplete evidence are recorded and excluded from named-mode rates.
A supplied snapshot does not independently attest the live container image digest.

Use the same runner, model revision, image and target settings across the four
MTP modes. Existing successful MTP measurements used **V1**; **V2** for the measured
Gaia campaign is inferred from restored defaults and earlier startup evidence.
Their configuration difference stays visible in
the combined report; short-input gains are not claimed for long Gaia histories.

## Running one campaign

The Gaia service, PostgreSQL and model endpoint must already be available. Defaults
are Gaia `127.0.0.1:8765`, vLLM `127.0.0.1:8000`; the expected context is **131,072**.
Offer at most ten concurrent requests. Keep one benchmark invocation active at a time.

From the repository root, choose the category directly:

```bash
.venv/bin/python -m tests.smoke_test.gaia_agent.scripts.run_suite all
```

Replace `all` with any category below. Each command starts one new campaign.

| Category | Work selected | Submissions |
|---|---|---:|
| `all` | Full Gaia sequence, then the four MTP modes | 292 + growth |
| `gaia` | T1 → T2 initial → T3 → T4 → T2 retention | 84 + growth |
| `mtp` | Off → k1 → k2 → k3, with operator handoffs | 208 including warmups |
| `concurrency` | Ordinary Gaia ladder only | 28 |
| `context-cache` | Cold10 → warm1 → warm10 → appended follow-up1 | 22 |
| `conversation-growth` | One Fungi seed chat, its growth and two real follow-ups | 3 + growth |
| `prefill-interference` | Baseline and three arrival conditions | 22 |

The full Gaia sequence shares T1's Fungi conversation and T2's prepared context
pool. Standalone conversation growth creates just its necessary seed chat.
Standalone context/cache ends after the appended follow-up; retention after other
work belongs to the full Gaia sequence.

Run names come from the category, for example `YYYY-MM-DD_HH-MM-SS_utc_all`.
The same name identifies the raw evidence folder under `logs/smoke_test/gaia_agent/`
and the report folder under `reports/smoke_test/`. The runner prints the run name
and final report path. Collisions in either location add `_2`, then `_3`.
Both roots stay anchored to the repository regardless of the working directory.

Gaia traces default to the repository's `logs/` directory. If the running service
uses another local trace directory, supply `--trace-root /absolute/service/trace/root`.
This directory is an input: request traces are copied into the campaign beside
their results.

`mtp` guides the operator through **off, k1, k2 and k3**. Before each mode, switch
the engine configuration in the GX10 app terminal and save a fresh `docker inspect`
JSON locally; enter that file's path at the runner's prompt. `all` completes Gaia,
including context retention, before starting these same handoffs. Configuration
changes remain with the GX10 operator.

To continue an interrupted handoff, use the original category and printed run folder:

```bash
.venv/bin/python -m tests.smoke_test.gaia_agent.scripts.run_suite all --resume RUN_FOLDER
```

Completed sections and modes are skipped by resumed `all`/`mtp` sweeps. The same
campaign identity preserves matched MTP inputs and combined reporting. Ending
input at a handoff leaves unfinished modes pending. Failed measurements remain
evidence; repeating a failed measurement requires a new campaign.

For a single externally prepared MTP mode, use the advanced form:

```bash
.venv/bin/python -m tests.smoke_test.gaia_agent.scripts.run_suite mtp --mtp-mode off --config-evidence /saved/off-inspect.json
.venv/bin/python -m tests.smoke_test.gaia_agent.scripts.run_suite mtp --resume RUN_FOLDER --mtp-mode k1 --config-evidence /saved/k1-inspect.json
```

This still plans the complete four-mode MTP comparison. Fill k2 and k3 in the
same campaign, or resume `mtp` without `--mtp-mode` for the remaining handoffs.

The full plan is **188 measured fixed submissions + 104 warmups + Gaia growth**:
roughly **308–318 total submissions**, excluding tokenizer/health/metrics HTTP
calls. Its live duration is unmeasured. The Gaia section is run once, not once per
MTP mode. Cache retention finishes before any engine restart for MTP.

## Evidence and report

Each run is organized by the behavior being measured:

```text
reports/smoke_test/RUN_FOLDER/
  analysis.json
  comparison.csv
  report.json
  findings.md

logs/smoke_test/gaia_agent/RUN_FOLDER/
  suite_manifest.json
  suite_summary.json
  runner.log
  gaia/
    ordinary_concurrency/users_01/ ... users_10/
    context_cache/initial/
    context_cache/retention/
    conversation_setup/          # standalone conversation-growth only
    growing_conversation/
    prefill_interference/
  mtp/
    off/  k1/  k2/  k3/
```

In the report folder, `findings.md` is the concise combined report; `report.json`
supports regeneration, while `analysis.json` and `comparison.csv` hold detailed
measurements. Evidence links point to the corresponding raw records in `logs/`.

In the evidence folder, `suite_manifest.json` and `suite_summary.json` hold the
plan, counts and section status. `runner.log` appends progress from each invocation.

Within each measured phase, files have a consistent purpose:

- `requests/<descriptive_request_id>/`: `request.json`, `events.jsonl`,
  `result.json`, and the copied `trace.json` for Gaia requests.
- `metrics/`: `samples.jsonl` and named before/after `.prom` snapshots.
- `environment/`: model inventory and, for MTP, the saved container configuration,
  verification and version records.
- `inputs/`: reusable context-pool payloads and their manifest where applicable.

Phase summaries stay with their phase. Warmup and measured request names distinguish
their roles, and the MTP report excludes warmups from comparison cells.

Rebuild a campaign's report offline into `reports/smoke_test/RUN_FOLDER/`:

```bash
.venv/bin/python -m tests.smoke_test.gaia_agent.scripts.summarize_results --run RUN_FOLDER
```

Compare provider decode as `(output tokens − 1) / generation time`; batch throughput
uses actual batch wall time. Keep first output, completion, chunk gaps and queue
measurements distinct. Count prefix hits against queried tokens, not allocated KV.
Missing data stays missing. Fixed-budget reasoning-only generation is not an answered
user query. Single batches provide a screen, not reliable production tail estimates.

The [audited baseline report](../../../reports/smoke_test/2026-09-14_audited_baseline/report.json)
and [findings](../../../reports/smoke_test/2026-09-14_audited_baseline/findings.md)
combine the measured campaigns without relabeling them as a new unified run. Their values were
cross-checked against primary responses, traces, metrics and available server logs.
At the user's request, those local captures and duplicate reports were then removed;
`report.json` retains the recomputed aggregates, significant failures, audit coverage
and historical source identifiers. The terminal excerpts had coverage gaps; a full
remote-log audit was not possible from those excerpts.

Regenerate the retained findings without the deleted captures:

```bash
.venv/bin/python -m tests.smoke_test.gaia_agent.scripts.summarize_results --report-json reports/smoke_test/2026-09-14_audited_baseline/report.json
```

## Prompt-injection checks

The security runners are separate from the performance suite above. The direct
runner starts a temporary loopback Gaia HTTP server using the current repository
code and the live model, then exercises disclosure requests, transformed output,
conversation attacks, file boundaries, and normal search/export controls:

```bash
.venv/bin/python -m tests.smoke_test.gaia_agent.scripts.prompt_injection
```

The model endpoint defaults to `http://127.0.0.1:8000/v1` and model name to
`default_model`; both can be overridden with `--endpoint` and `--model`. The
direct runner needs the normal PostgreSQL configuration for its record/export
controls. It creates isolated chat/export directories within the campaign under
`logs/smoke_test/gaia_agent/<run>/direct/`, captures full model requests/responses
and public HTTP events, and closes its HTTP server when finished. Each turn has
a 120-second test timeout. Use `--cases NAME ...` to select cases, or `--run RUN`
to put results into an existing campaign. Repeats use `--batch direct_followup`
so the original results are preserved.

The companion indirect runner uses the live model and current runtime, prompt,
definitions, and response guard with synthetic query observations. It records
downstream action attempts using controlled handlers:

```bash
.venv/bin/python -m tests.smoke_test.gaia_agent.scripts.prompt_injection_indirect --run RUN
```

Review decoded content, instruction excerpts, stored history, tool attempts, and
public delivery independently. A delivered HTTP response is not a security pass,
and an output blocked by the runtime is distinct from a model refusal. Record
findings with evidence links in `reports/smoke_test/<run>/`. Single attempts
measure observed outcomes; they do not establish attack success probabilities.
