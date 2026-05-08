PYTHON ?= python3
BACKEND_HOST ?= 127.0.0.1
BACKEND_PORT ?= 8000

.PHONY: setup backend bootstrap run-backend test ingest-ifixit ingest-sample ingest-ifixit-appliance

setup: bootstrap

backend: run-backend

bootstrap:
	./scripts/bootstrap.sh

run-backend:
	. .venv/bin/activate && uvicorn app.main:app --reload --reload-dir backend --reload-exclude ".venv/*" --app-dir backend --host $(BACKEND_HOST) --port $(BACKEND_PORT)

test:
	. .venv/bin/activate && pytest -q

ingest-ifixit:
	. .venv/bin/activate && python scripts/ingest_ifixit.py --limit 25

ingest-sample:
	. .venv/bin/activate && python scripts/ingest_ifixit.py --input-json data/raw/sample_ifixit_minimal.json

ingest-ifixit-appliance:
	. .venv/bin/activate && python scripts/ingest_ifixit.py --limit 50 --category Appliance
