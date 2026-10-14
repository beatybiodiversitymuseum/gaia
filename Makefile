SHELL := /bin/sh

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip
FRONTEND := src/frontend
SNAPSHOT_DATE ?= $(shell date +%F)

.DEFAULT_GOAL := help

.PHONY: help setup db data refresh-gbif refresh-darwin-core api web dev test lint build python-check check

help: ## Show the available development commands.
	@awk 'BEGIN {FS = ":.*## "; printf "Gaia development commands:\n\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create the Python environment and install backend and frontend dependencies.
	@test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install -r requirements-dev.txt
	@test -f .env || cp .env.example .env
	@test -f $(FRONTEND)/.env.local || cp $(FRONTEND)/.env.example $(FRONTEND)/.env.local
	@SECRET=$$(openssl rand -hex 32); \
	if grep -q '^GAIA_SESSION_SECRET=' $(FRONTEND)/.env.local; then \
		sed -i '' "s/^GAIA_SESSION_SECRET=$$/GAIA_SESSION_SECRET=$$SECRET/" $(FRONTEND)/.env.local; \
	else \
		printf '\nGAIA_SESSION_SECRET=%s\n' "$$SECRET" >> $(FRONTEND)/.env.local; \
	fi
	npm --prefix $(FRONTEND) ci

db: ## Start PostgreSQL and wait until it is healthy.
	docker compose up -d --wait db

data: db ## Initialize tables, transform source data, and ingest it into PostgreSQL.
	$(PYTHON) -m src.scripts.transform_occurrences
	$(PYTHON) -m src.scripts.initialize_tables
	$(PYTHON) -m src.scripts.ingest_postgresql --replace

refresh-gbif: ## Refresh the pinned GBIF production extension registry.
	$(PYTHON) -m src.scripts.refresh_gbif_extension_registry --snapshot-date $(SNAPSHOT_DATE)

refresh-darwin-core: ## Rebuild terms from the pinned official Darwin Core sources.
	$(PYTHON) -m src.scripts.refresh_darwin_core_terms

api: ## Run the Gaia API development server.
	$(PYTHON) main.py serve

web: ## Run the frontend development server.
	npm --prefix $(FRONTEND) run dev

dev: db ## Run the API and frontend together; Ctrl-C stops both.
	@set -eu; \
	$(PYTHON) main.py serve & \
	api_pid=$$!; \
	cleanup() { \
		trap - EXIT INT TERM; \
		kill "$$api_pid" 2>/dev/null || true; \
		wait "$$api_pid" 2>/dev/null || true; \
	}; \
	trap cleanup EXIT INT TERM; \
	npm --prefix $(FRONTEND) run dev

test: ## Run the Python test suite.
	$(PYTHON) -m pytest

lint: ## Run the frontend linter.
	npm --prefix $(FRONTEND) run lint

build: ## Build the production frontend bundle.
	npm --prefix $(FRONTEND) run build

python-check: ## Compile Python source and tests to catch syntax errors.
	$(PYTHON) -m compileall -q main.py src tests

check: python-check lint test build ## Run all automated checks.
