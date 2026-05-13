PYTHON ?= python3
BACKEND_HOST ?= 127.0.0.1
BACKEND_PORT ?= 8000
CONDA_ENV ?= uni
PY_RUN := conda run -n $(CONDA_ENV) python

.PHONY: setup backend bootstrap run-backend test ingest-ifixit ingest-sample reset-ingest-sample ingest-ifixit-appliance ingest-ifixit-broad frontend-install run-frontend evaluate-rag

setup: bootstrap

backend: run-backend

bootstrap:
	./scripts/bootstrap.sh

run-backend:
	$(PY_RUN) -m uvicorn app.main:app --reload --reload-dir backend --reload-exclude ".venv/*" --app-dir backend --host $(BACKEND_HOST) --port $(BACKEND_PORT)

test:
	cd backend && $(PY_RUN) -m pytest -q

frontend-install:
	cd frontend && npm install

run-frontend:
	cd frontend && npm run dev

evaluate-rag:
	$(PY_RUN) scripts/evaluate_rag.py --input data/eval/troubleshooting_queries.json --strategies naive reranked hyde

ingest-ifixit:
	$(PY_RUN) scripts/ingest_ifixit.py --limit 25

ingest-sample:
	$(PY_RUN) scripts/ingest_ifixit.py --input-json data/raw/sample_ifixit_minimal.json

reset-ingest-sample:
	rm -rf data/chroma
	$(PY_RUN) scripts/ingest_ifixit.py --input-json data/raw/sample_ifixit_minimal.json

ingest-ifixit-appliance:
	$(PY_RUN) scripts/ingest_ifixit.py --limit 50 --category Appliance

ingest-ifixit-broad:
	$(PY_RUN) scripts/ingest_ifixit.py --categories "Washing Machine,Dryer,Dishwasher,Refrigerator,Oven,Microwave" --per-category 150
