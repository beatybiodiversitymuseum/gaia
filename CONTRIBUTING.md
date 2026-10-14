# Contributing to Gaia

Gaia is a Python and PostgreSQL agent service with a TypeScript browser
interface. Changes should preserve the boundary between model guidance and
deterministically enforced behavior.

## Development setup

From the repository root:

```bash
make setup
```

Copy the required datasets into the paths documented in `README.md`, configure
the generated `.env` files, and load PostgreSQL:

```bash
make data
```

With the configured model endpoint available, start the application:

```bash
make dev
```

Run the complete local verification before submitting a change:

```bash
make check
```

## Change boundaries

- Keep `main.py` and frontend proxy routes thin.
- Put agent orchestration in `src/gaia_agent/runtime/`.
- Put model-visible tools in `src/gaia_agent/tools/` and validate their inputs.
- Preserve source Darwin Core field names and values at data boundaries.
- Never rely on the system prompt to enforce permissions, ownership, limits, or
  filesystem safety.
- Keep externally retrieved enrichment separate from museum source records and
  retain its provenance.

Tests mirror the `src/` package layout. Add the narrowest tests that prove the
new behavior and its boundary conditions.

## Pull requests

Describe the user-visible change, affected contracts, validation performed, and
any operational or data migration requirements. Do not include datasets,
credentials, generated logs, local environment files, or exported workbooks.
