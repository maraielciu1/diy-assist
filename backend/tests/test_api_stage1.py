from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.main import app


@dataclass
class _FakeChunk:
    text: str
    score: float
    metadata: dict


class _FakeRetriever:
    def search(
        self,
        query: str,
        appliance_category: str | None = None,
        appliance_type: str | None = None,
        brand: str | None = None,
        model: str | None = None,
        top_k: int = 5,
    ):
        return [
            _FakeChunk(
                text="Step 2: Clean the washer drain pump filter.",
                score=0.92,
                metadata={
                    "guide_id": "washer-001",
                    "guide_title": "Washer Not Draining Troubleshooting",
                    "source_url": "https://example.com/washer",
                    "step_number": 2,
                    "previous_steps": [
                        {"step_number": 1, "text": "Unplug washer and inspect hose."}
                    ],
                },
            )
        ][:top_k]


class _BrokenRetriever:
    def search(
        self,
        query: str,
        appliance_category: str | None = None,
        appliance_type: str | None = None,
        brand: str | None = None,
        model: str | None = None,
        top_k: int = 5,
    ):
        raise RuntimeError("chroma unavailable")


def test_root_and_health() -> None:
    client = TestClient(app)
    root = client.get("/")
    assert root.status_code == 200
    payload = root.json()
    assert payload["docs"] == "/docs"
    assert payload["frontend"] == "/frontend"

    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_naive_rag_success(monkeypatch) -> None:
    from app.api import routes

    monkeypatch.setattr(routes, "_retriever", _FakeRetriever())
    client = TestClient(app)
    response = client.post(
        "/api/v1/rag/naive",
        json={
            "query": "washer not draining",
            "appliance_category": "Appliance",
            "top_k": 1,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy"] == "naive"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["metadata"]["guide_title"] == "Washer Not Draining Troubleshooting"


def test_chat_success(monkeypatch) -> None:
    from app.api import routes

    monkeypatch.setattr(routes, "_retriever", _FakeRetriever())
    monkeypatch.setattr(
        routes.ifixit_live_client,
        "suggest_guides",
        lambda query, appliance_category=None, limit=3: [],
    )
    monkeypatch.setattr(
        routes.slm,
        "generate_answer",
        lambda user_query, retrieved_chunks, live_ifixit_guides=None: "Mocked safe answer.",
    )

    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={
            "query": "my washer is not draining and makes a humming noise",
            "appliance_category": "Appliance",
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Mocked safe answer."
    assert payload["retrieval_count"] == 1
    assert len(payload["citations"]) == 1


def test_safety_blocking_chat() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={"query": "I smell gas near my dryer, what should I do?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["guardrail_blocked"] is True
    assert "hazard_detected" in payload["reason"]


def test_empty_or_unavailable_retriever(monkeypatch) -> None:
    from app.api import routes

    monkeypatch.setattr(routes, "_retriever", _BrokenRetriever())
    client = TestClient(app)

    rag_response = client.post(
        "/api/v1/rag/naive",
        json={"query": "washer not draining", "appliance_category": "Appliance", "top_k": 3},
    )
    assert rag_response.status_code == 200
    rag_payload = rag_response.json()
    assert rag_payload["results"] == []
    assert "Retriever is unavailable" in rag_payload["error"]

    chat_response = client.post(
        "/api/v1/chat",
        json={"query": "washer not draining", "appliance_category": "Appliance", "top_k": 3},
    )
    assert chat_response.status_code == 200
    chat_payload = chat_response.json()
    assert chat_payload["answer"] is None
    assert chat_payload["retrieval_count"] == 0
    assert "Retriever is unavailable" in chat_payload["error"]
