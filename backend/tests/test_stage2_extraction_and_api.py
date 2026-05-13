import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services.retrieval import NaiveRetriever


def _load_ingest_module():
    module_path = Path("scripts/ingest_ifixit.py")
    spec = importlib.util.spec_from_file_location("ingest_ifixit", module_path)
    assert spec is not None
    assert spec.loader is not None
    ingest_ifixit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ingest_ifixit)
    return ingest_ifixit


def test_normalized_guide_record_extracts_stage2_metadata() -> None:
    ingest_ifixit = _load_ingest_module()
    sample_payload = json.loads(
        Path("data/raw/sample_ifixit_minimal.json").read_text(encoding="utf-8")
    )
    guide = sample_payload["guides"][0]
    record = ingest_ifixit._normalized_guide_record(guide, index_hint=1)

    for key in (
        "guide_id",
        "guide_title",
        "appliance_category",
        "brand",
        "model",
        "difficulty",
        "tools",
        "steps",
    ):
        assert key in record
    assert isinstance(record["steps"], list)
    assert record["steps"]


def test_extraction_handles_alternate_ifixit_shapes() -> None:
    ingest_ifixit = _load_ingest_module()
    guide = {
        "id": 12,
        "title": "Dryer Does Not Heat",
        "type": "Appliance",
        "subject": {"manufacturer": "Bosch", "model_number": "WTR85V"},
        "difficulty": {"title": "Moderate"},
        "tools": [{"name": "Phillips #2"}, {"title": "Multimeter"}],
        "steps": [
            {
                "title": "Check thermal fuse",
                "lines": [{"text_raw": "Disconnect power and test continuity."}],
                "tools": [{"name": "Multimeter"}],
            }
        ],
    }
    record = ingest_ifixit._normalized_guide_record(guide, index_hint=1)
    assert record["brand"] == "Bosch"
    assert record["model"] == "WTR85V"
    assert record["difficulty"] == "Moderate"
    assert "Phillips #2" in record["tools"]
    assert record["steps"][0]["step_number"] == 1
    assert not record["steps"][0]["text"].startswith("Step 1:")


def test_retriever_where_clause_supports_brand_and_model() -> None:
    where = NaiveRetriever._build_where_clause(
        appliance_category="Appliance",
        appliance_type="washer",
        brand="Whirlpool",
        model="WFW5605MC",
    )
    assert where is not None
    assert "$and" in where
    filters = where["$and"]
    assert {"appliance_category": {"$eq": "Appliance"}} in filters
    assert {"appliance_type": {"$eq": "washer"}} in filters
    assert {"brand": {"$eq": "Whirlpool"}} in filters
    assert {"model": {"$eq": "WFW5605MC"}} in filters


@dataclass
class _FakeChunk:
    text: str
    score: float
    metadata: dict


class _CapturingRetriever:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(
        self,
        query: str,
        appliance_category: str | None = None,
        appliance_type: str | None = None,
        brand: str | None = None,
        model: str | None = None,
        top_k: int = 5,
    ):
        self.calls.append(
            {
                "query": query,
                "appliance_category": appliance_category,
                "appliance_type": appliance_type,
                "brand": brand,
                "model": model,
                "top_k": top_k,
            }
        )
        return [
            _FakeChunk(
                text="Step 2: Clean the drain filter.",
                score=0.88,
                metadata={
                    "guide_id": "washer-001",
                    "guide_title": "Washer Not Draining Troubleshooting",
                    "appliance_category": "Appliance",
                    "brand": "Whirlpool",
                    "model": "WFW5605MC",
                    "difficulty": "easy",
                    "tools": "towel, tray",
                    "step_number": 2,
                    "chunk_number": 1,
                    "source_url": "https://example.com/washer",
                    "previous_steps": [
                        {"step_number": 1, "text": "Unplug washer and inspect hose."}
                    ],
                },
            )
        ]


def test_api_response_shape_includes_stage2_fields(monkeypatch) -> None:
    from app.api import routes

    retriever = _CapturingRetriever()
    monkeypatch.setattr(routes, "_retriever", retriever)
    monkeypatch.setattr(
        routes.ifixit_live_client,
        "suggest_guides",
        lambda query, appliance_category=None, limit=3: [],
    )
    monkeypatch.setattr(
        routes.slm,
        "generate_answer",
        lambda user_query, retrieved_chunks, live_ifixit_guides=None: "Mocked answer.",
    )

    client = TestClient(app)
    rag_response = client.post(
        "/api/v1/rag/naive",
        json={
            "query": "washer not draining",
            "appliance_category": "Appliance",
            "brand": "Whirlpool",
            "model": "WFW5605MC",
            "agent_mode": "pipeline",
            "top_k": 3,
        },
    )
    assert rag_response.status_code == 200
    rag_payload = rag_response.json()
    result = rag_payload["results"][0]
    assert result["guide_title"] == "Washer Not Draining Troubleshooting"
    assert result["step_number"] == 2
    assert isinstance(result["previous_steps"], list)
    assert "metadata" in result

    chat_response = client.post(
        "/api/v1/chat",
        json={
            "query": "washer not draining",
            "appliance_category": "Appliance",
            "brand": "Whirlpool",
            "model": "WFW5605MC",
            "top_k": 3,
        },
    )
    assert chat_response.status_code == 200
    chat_payload = chat_response.json()
    citation = chat_payload["citations"][0]
    assert "score" in citation
    assert "metadata" in citation
    assert citation["step_number"] == 2

    assert retriever.calls
    first = retriever.calls[0]
    assert first["brand"] == "Whirlpool"
    assert first["model"] == "WFW5605MC"
    assert first["appliance_type"] == "washer"
