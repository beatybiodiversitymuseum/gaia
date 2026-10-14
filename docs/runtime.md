# Gaia runtime

This document describes Gaia's runtime contracts and the boundaries that must
remain deterministic when the system prompt or model changes.

## Request path

A browser request is authenticated by the frontend server and proxied to Gaia's
loopback HTTP bridge with the authenticated username. `GaiaChatService` owns
conversation identity, per-conversation serialization, cancellation, trace
storage, and generated-file ownership. The agent runtime sends the active
conversation and registered tool definitions to the configured OpenAI-compatible
model server.

The model may return a final response or request tools. Tool calls are validated
and executed by their registered Python handlers. Independent calls in one model
round may run concurrently, but one user turn has a single allowance of 15 tool
calls across every tool type and model round. Results are appended in the
model-emitted call order before the next model request.

## Tool boundaries

Gaia exposes six core capabilities:

- `retrieve_schema` returns deterministic schema metadata selected by exact
  dataset or collection names.
- `search_darwin_core_terms` searches the installed standard snapshot by term
  text and standard metadata.
- `get_darwin_core_terms` returns complete installed metadata for exact Darwin
  Core term names or IRIs.
- `query_occurrences` validates a structured query plan and retrieves bounded
  occurrence rows from registered PostgreSQL tables.
- `query_statistics` computes metrics across the complete compatible matching
  scope independently of the occurrence-row retrieval allowance.
- `generate_excel` validates export plans and atomically publishes workbooks in
  the active conversation's server-owned export directory.

The model never supplies SQL, database table names, or filesystem paths.
Model-visible descriptions explain inputs and output schemas; workflow and tool
selection guidance belongs in `system_prompt.md`.

Darwin Core lookup is fully offline at runtime. The installed term snapshot is
built deterministically from exact commits of the official `tdwg/dwc` and
`tdwg/rs.tdwg.org` repositories. It covers standard term semantics; archive
extensions and controlled-vocabulary values are outside this tool contract.

## Context management

Context is managed independently for each conversation using the deployed
model's tokenizer and reported context limit. Gaia reserves 5% of the model
window and starts compaction at 90% of that usable capacity.

Compaction produces one cumulative Markdown summary containing `Chat summary`
and `Tools called` sections. It is limited to 12,000 summary tokens, 4,000 tool
history tokens, and 16,000 tokens in total. Before active context is replaced,
`history.md` is written atomically beside the conversation traces. Execution
records, attachments, and the current turn's tool allowance survive compaction.

## Query behavior

Every dataset table and schema-tool response uses the ordered Simple Darwin Core
property profile from the pinned standard snapshot. Query plans support
validated literal filters, selected columns, distinct fields, ordering, and
bounded chat retrieval. Statistics use a separate execution path so totals are
not inferred from a sample of occurrence rows.

Source values remain text at the data boundary. Empty strings and SQL nulls are
not interchangeable. Gaia preserves exact field spelling, Unicode, identifiers,
and result ordering wherever the source and query permit it.

Dataset preparation accepts supported Darwin Core ZIP archives with an Occurrence
core without relying on their filenames or a source-code allowlist. It uses
`meta.xml` for the core file layout, maps registered Occurrence terms through
the pinned official Simple Darwin Core profile, derives deterministic dataset
metadata, and atomically writes intermediate JSONL to the canonical path.
Dataset keys and table names are readable normalized forms of `datasetName`,
without generated hash suffixes. If multiple archives resolve to the same
dataset name and
identity, the later source in deterministic input order is treated as an update:
preparation emits a console warning naming the replaced and replacing sources,
then converts only the later source. The archive's `meta.xml` maps physical
fields into canonical columns; registered fields omitted by an archive are
stored as empty strings.

Database updates replace the complete dataset in one transaction. Validation,
reading, deletion of existing rows, and insertion of all replacement rows either
commit together or roll back together; failed and empty updates preserve the
previous dataset contents. Distinct archives that are not updates must provide
distinct dataset identities.

Only the Occurrence core is supported. Extension parsing, validation, ingestion,
and querying are TODOs; extension rows are ignored. Broader archive-compatibility
coverage—including alternate encodings and dialects, malformed metadata indexes,
missing members, and unsafe member paths—is also a TODO.

## Files and traces

Successful exports are attachments only after their workbooks have been
published atomically. Download requests are resolved against the owning
conversation's export directory, reject links and nonregular files, and verify
that the registered content has not changed.

Interaction traces are stored under `logs/` with bounded, redacted operational
metadata. Completed public HTTP responses include a bounded verification trace
containing the executed capability name, exact validated arguments, completion
status, and already-projected result evidence in call order. Every call also
contains a concise public purpose, authored when the call is requested, stating
why it is needed and what evidence it should establish. The purpose is removed
before domain validation and execution. Result arrays show at most ten items and
report omissions. The verification trace never includes raw model reasoning,
prompts, operational paths, backend configuration, credentials, or unprojected
database fields. Live progress remains limited to user-facing activity
descriptions.

## Cancellation and failures

Cancellation is cooperative across model requests, tool execution, queued work,
stream delivery, and file generation. An interrupted or failed operation must
not establish a partial attachment or replace a previously completed artifact.
Public errors are mapped to stable user-facing categories while detailed errors
remain in local traces.

## Frontend trust boundary

The browser never chooses the trusted username forwarded to Gaia. Frontend API
routes derive it from the signed session token and overwrite any browser-supplied
value. `GAIA_UI_PASSWORD` controls access, while `GAIA_SESSION_SECRET` is an
independent high-entropy signing key. Both must be configured outside version
control.
