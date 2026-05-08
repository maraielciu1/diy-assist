#!/usr/bin/env python3
"""
Shim module for tests and backend-local execution.

The canonical ingestion script lives at repo root: `scripts/ingest_ifixit.py`.
Some tests are executed with the working directory set to `backend/`, so they
expect `scripts/ingest_ifixit.py` to exist relative to that directory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_ROOT_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ingest_ifixit.py"

spec = importlib.util.spec_from_file_location("root_ingest_ifixit", _ROOT_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load root ingestion script at {_ROOT_SCRIPT}")

_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_module)

# Re-export everything so tests can call helpers like `_normalized_guide_record`.
globals().update(_module.__dict__)

