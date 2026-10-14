# Gaia Repository Guide

This guide applies to the repository tree. Check for more specific `AGENTS.md`
files in directories you change; their instructions take precedence on conflict
within their scope. Direct system, developer, and user instructions take precedence
over repository guides.

## Architecture and data boundaries

- `main.py` exposes only `query` and `serve`. Keep argument parsing and terminal
  presentation thin; reusable behavior belongs in independently callable
  functions.
- `src/gaia_agent/` owns agent behavior; `src/scripts/` owns setup and
  reproducibility commands. Neither may import the other. Load command-specific
  implementations only for the invoked command.
- `src/gaia_agent/runtime/` owns agent composition, the execution loop, model
  contracts and client, defaults, and the system prompt. Keep its initializer
  lightweight so importing defaults or contracts does not load the model client,
  tools, or web transport.
- Enforce the 15-tool-call allowance in the runtime across all tool types and
  model rounds for one user turn. Independent calls may run concurrently;
  each user turn owns its own allowance.
- Manage context independently per chat using the deployed tokenizer and server
  limit. Reserve 5% of the context and compact at 90% of the usable window.
  Carry forward one cumulative model-generated compaction, capped at 12,000
  summary tokens, 4,000 tool-history tokens, and 16,000 tokens in total.
  Atomically save `history.md` beside that chat's interaction traces before
  replacing active context with the system prompt and compaction. Preserve
  execution records, attachments, and per-turn tool allowances across resets.
- Preserve model-emitted tool-call sequence and argument field order, including
  nested objects, when replaying conversation history and saving traces.
- `src/gaia_agent/tools/` owns model-visible tool registration. Its `query/`
  package owns query validation, execution, and retrieval budgets;
  `file_generation/` owns export schemas, generation, serialization, naming,
  and delivery metadata. File generation may reuse query behavior. Keep agent
  orchestration in `runtime/`.
- Keep model-visible tool and argument descriptions and returned schema guides
  concise and factual about capabilities, inputs, outputs, and limits. Put
  tool-selection, workflow, and user-intent guidance in the system prompt.
  Keep API examples, argument mechanics, and tool-specific contract details
  in tool definitions or returned schema metadata, without repeating them in
  the system prompt.
- Statistics queries compute over the full compatible matching scope and run
  independently of the occurrence retrieval allowance. Statistics output groups
  have no fixed cap; an explicit positive limit bounds returned groups and null
  returns all groups. Counts describe their returned dataset coverage.
- File generation accepts requested basenames and otherwise uses the session
  username with a sequential file number and `.xlsx` extension. Publish completed
  workbooks atomically within the active session's export directory, preserving
  existing files and download links. File capabilities are limited to generation,
  naming at creation, and delivery; workbook content inspection and editing are
  outside this domain.
- Keep operational paths and backend configuration out of model observations
  and public responses. Resolve generated-file basenames against server-owned
  session directories; only successful operations establish attachments. Public
  progress contains user-facing activities, not tool identities or execution
  counters. Validate final responses before delivery and successful history
  storage. Treat disclosure checks as supplemental to capability boundaries,
  never as complete prompt-injection prevention. Render assistant content without
  automatic external resource requests or model-authored download capabilities.
- `src/gaia_agent/web/` owns HTTP transport and web conversation state.
  `query`, including its logging, must work without this package or
  `src/frontend/`. Keep shared logging outside the web package, and load web
  imports inside the `serve` path.
- `src/database/` owns the dataset registry (`datasets.py`), SQLAlchemy models,
  table mappings, and database configuration shared by runtime and setup.
  Keep its initializer lightweight: importing the registry must not load
  SQLAlchemy or database configuration.
- `src/scripts/dataset_schema.py` owns file-format definitions and source-file
  helpers. Keep collection-specific changes within the collection's subject
  directory; coordinate shared changes across the affected domains.
- Preserve exact biodiversity and Darwin Core field spelling at data
  boundaries. Store source fields as `TEXT NOT NULL`, preserving Unicode and
  empty strings. `occurrenceID` is the primary key; `catalogNumber` has a
  non-unique index. Import identifier copies must match `raw_record`; store
  its flat fields, keeping derived `chunks` as import-only projections.
- Keep `src/frontend/` in this root Git repository on `main`. A nested Git
  repository or frontend-specific branch requires an explicit user request.
  The frontend targets desktop browsers: retain the desktop sidebar and
  controls; add mobile-specific layouts or behavior only when requested.

## Implementation conventions

- Use snake_case for Python modules, functions, methods, and variables;
  PascalCase for classes; UPPER_SNAKE_CASE for module constants. Exact source
  field names at data boundaries are the exception.
- Prefer functions for stateless transformations, validation, serialization,
  and thin orchestration. Use classes for cohesive state, lifecycle,
  invariants, or interchangeable behavior.
- Prefer direct control flow and the standard library. Add dependencies or
  abstractions only when they provide a concrete benefit over simpler code.
- Remove replaced behavior and its obsolete tests; do not retain compatibility
  shims or legacy fallbacks. Investigate excess or unused code. Remove obsolete
  logic within the requested scope; ask the user when a proposed deletion raises
  an unresolved decision about intended behavior or breakage.
- Validate at data and public API boundaries. Repeat validation internally
  only when a layer owns a distinct contract.
- Use type hints for public functions and non-obvious internal contracts.
  Use descriptive names and actionable errors; comments should explain why a
  decision exists.
- Keep serialization and transformations deterministic where source ordering
  permits it. Avoid hidden global state, import-time processing, and side
  effects unrelated to the invoked behavior.
- Group related definitions contiguously when it improves navigation. Use
  this exact three-line header for a meaningful group, with a concise noun
  phrase; omit headers that merely repeat nearby code:

  ```python
  # ---------------------------------------------------------------------------
  # <Concise group name>
  # ---------------------------------------------------------------------------
  ```

- Avoid writing nested functions unless absolutely necessary. Break up large functions into smaller ones when refactoring.
- Avoid import statements inside of classes and functions. Use it at the beginning of the file.
- Don't use ... imports in importing. Always import with full path

## Delegation and integration

For non-trivial work, identify the affected domains and dependency order, then
delegate independently scoped tasks. Assign one owner per scope; do not run
duplicate active owners unless the user requests that parallelism. The root
agent may handle narrow, tightly coupled, or documentation-only changes
directly.

Each delegation must specify the task, owned files, applicable guide,
cross-domain dependencies, and required validation. Subagents stay within that
scope. Handoffs must state completed work, blockers or out-of-scope work, the
next owner when applicable, and validation run or omitted with reasons.

The root agent owns integration, conflict resolution, validation of shared
contracts, and user-facing reporting.

## Validation and completion

- Mirror the `src/` package layout under `tests/`. Place tests with the code
  that owns the behavior; keep integration tests with the orchestrating module.
- Finish the related changes before running tests. Use the narrowest
  validation that proves the changed behavior and adjacent contracts.
- Do not write test cases for `main.py` or the UI. Validate their behavior
  directly when relevant. For import-boundary changes, check the affected
  commands and logging with the excluded layers unavailable.
- Check a failing test against the intended contract before changing code or
  expectations. Fix code when the test is correct. Cover required behavior and
  real edge cases; add no production behavior solely to satisfy a test.
- For data transformations, check output shape and required identifiers,
  empty-value/null behavior, Unicode round-tripping, written/skipped/failed
  counts, malformed-width rows, and bounded memory where applicable.
- Use temporary or dedicated validation outputs. Do not modify source data to
  construct a test. Report the commands run, contracts proved, row or file
  failures, and relevant checks omitted with reasons.
- Update relevant documentation when setup, public commands, file layout,
  transformation contracts, or stable workflows change. Leave documentation
  unchanged when its contract is unaffected.
- Keep README focused on the project overview, setup, and basic usage. Put
  detailed runtime behavior and API contracts in `docs/runtime.md`. Keep setup
  instructions self-contained, with public dataset sources and the required
  local file layout. Exclude references to ignored personal notes.
- Repository report and Markdown generators write standalone `.md` reports.
  Maintain `README.md` directly; generators must never modify it. Use
  deterministic rendering and atomic report replacement, and verify that
  generation preserves existing reports on failure and leaves the README
  unchanged.
- Store smoke-test reports and findings under `reports/smoke_test/<run>/`.
  Keep raw evidence and runner state under `logs/smoke_test/gaia_agent/<run>/`,
  with report links pointing to the corresponding evidence.
- Keep agent guides limited to durable boundaries, decision rules, contracts,
  and validation obligations. Update them when those instructions change;
  keep task logs and incidental file inventories elsewhere.
