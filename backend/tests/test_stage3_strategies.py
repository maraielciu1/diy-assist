from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.main import app


@dataclass
class _FakeChunk:
    text: str
    score: float
    metadata: dict


class _FakeStrategyRetriever:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = []

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
                text=f"{self.name}: Step 1: Mock fix.",
                score=0.99,
                metadata={
                    "guide_title": f"{self.name} guide",
                    "source_url": "https://example.com/mock",
                    "step_number": 1,
                    "previous_steps": [],
                },
            )
        ]


def test_reranked_rag_endpoint_keeps_shape(monkeypatch) -> None:
    from app.api import routes

    fake = _FakeStrategyRetriever("reranked")
    monkeypatch.setattr(routes, "_reranked_retriever", fake)
    client = TestClient(app)
    resp = client.post(
        "/api/v1/rag/reranked",
        json={"query": "washer not draining", "appliance_category": "Appliance", "top_k": 2},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["strategy"] == "reranked"
    assert payload["results"][0]["metadata"]["guide_title"] == "reranked guide"


def test_hyde_rag_endpoint_keeps_shape(monkeypatch) -> None:
    from app.api import routes

    fake = _FakeStrategyRetriever("hyde")
    monkeypatch.setattr(routes, "_hyde_retriever", fake)
    client = TestClient(app)
    resp = client.post(
        "/api/v1/rag/hyde",
        json={"query": "fridge warm freezer cold", "appliance_category": "Appliance", "top_k": 2},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["strategy"] == "hyde"
    assert payload["results"][0]["metadata"]["guide_title"] == "hyde guide"


def test_chat_reranked_strategy_field_present(monkeypatch) -> None:
    from app.api import routes

    monkeypatch.setattr(routes, "_reranked_retriever", _FakeStrategyRetriever("reranked"))
    monkeypatch.setattr(
        routes.ifixit_live_client,
        "suggest_guides",
        lambda query, appliance_category=None, limit=3: [],
    )
    monkeypatch.setattr(
        routes.slm,
        "generate_answer",
        lambda user_query, retrieved_chunks, live_ifixit_guides=None: "Safety: unplug.\nSources: x",
    )
    client = TestClient(app)
    resp = client.post(
        "/api/v1/chat/reranked",
        json={"query": "washer not draining", "appliance_category": "Appliance", "top_k": 2},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["strategy"] == "reranked"
    assert payload["answer"] is not None

