#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

echo "Creating virtual environment at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"

echo "Activating virtual environment"
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip"
python -m pip install --upgrade pip

echo "Installing backend requirements"
python -m pip install -r "${ROOT_DIR}/backend/requirements.txt"

echo "Bootstrap complete."
echo "Activate env: source .venv/bin/activate"
