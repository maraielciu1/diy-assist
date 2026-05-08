PYTHON ?= python3
BACKEND_HOST ?= 127.0.0.1
BACKEND_PORT ?= 8000

.PHONY: setup backend bootstrap run-backend test ingest-ifixit ingest-sample ingest-ifixit-appliance

# Prefer root venv if present, otherwise fall back to backend/.venv (used in CI/sandbox)
VENV_DIR := $(shell if [ -d .venv ]; then echo .venv; elif [ -d backend/.venv ]; then echo backend/.venv; else echo .venv; fi)

setup: bootstrap

backend: run-backend

bootstrap:
	./scripts/bootstrap.sh

run-backend:
	. $(VENV_DIR)/bin/activate && uvicorn app.main:app --reload --reload-dir backend --reload-exclude ".venv/*" --app-dir backend --host $(BACKEND_HOST) --port $(BACKEND_PORT)

test:
	. $(VENV_DIR)/bin/activate && cd backend && pytest -q

ingest-ifixit:
	. $(VENV_DIR)/bin/activate && python scripts/ingest_ifixit.py --limit 25

ingest-sample:
	. $(VENV_DIR)/bin/activate && python scripts/ingest_ifixit.py --input-json data/raw/sample_ifixit_minimal.json

ingest-ifixit-appliance:
	. $(VENV_DIR)/bin/activate && python scripts/ingest_ifixit.py --limit 50 --category Appliance
