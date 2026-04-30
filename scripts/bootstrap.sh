#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 -m venv "${ROOT_DIR}/.venv"
source "${ROOT_DIR}/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/backend/requirements.txt"

echo "Bootstrap complete."
echo "Activate with: source .venv/bin/activate"
echo "Run backend: uvicorn app.main:app --reload --app-dir backend"
