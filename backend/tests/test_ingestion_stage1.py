import importlib.util
import json
from pathlib import Path


def test_sample_ingestion_json_exists_and_extracts_guides() -> None:
    module_path = Path("scripts/ingest_ifixit.py")
    spec = importlib.util.spec_from_file_location("ingest_ifixit", module_path)
    assert spec is not None
    assert spec.loader is not None
    ingest_ifixit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ingest_ifixit)

    sample_path = Path("data/raw/sample_ifixit_minimal.json")
    assert sample_path.exists()

    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    guides = ingest_ifixit._extract_guides(payload)
    assert len(guides) >= 1

    steps = ingest_ifixit._extract_step_texts(guides[0])
    assert steps
    assert any("Step 1:" in step for step in steps)
