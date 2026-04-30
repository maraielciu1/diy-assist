from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.guardrails import evaluate_query_safety
from app.services.llm import SLMWrapper
from app.services.retrieval import NaiveRetriever

router = APIRouter()
retriever = NaiveRetriever()
slm = SLMWrapper()


class RAGQuery(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    appliance_category: str | None = None
    top_k: int = Field(default=settings.top_k_default, ge=1, le=20)


class ChatQuery(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    appliance_category: str | None = None
    top_k: int = Field(default=settings.top_k_default, ge=1, le=20)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@router.post("/rag/naive")
def naive_rag(payload: RAGQuery) -> dict:
    if settings.guardrails_enabled:
        decision = evaluate_query_safety(payload.query)
        if not decision.allow:
            return {
                "query": payload.query,
                "strategy": "naive",
                "guardrail_blocked": True,
                "reason": decision.reason,
                "message": decision.escalation_message,
                "results": [],
            }

    hits = retriever.search(
        query=payload.query,
        appliance_category=payload.appliance_category,
        top_k=payload.top_k,
    )
    return {
        "query": payload.query,
        "strategy": "naive",
        "results": [
            {"text": chunk.text, "score": chunk.score, "metadata": chunk.metadata}
            for chunk in hits
        ],
    }


@router.post("/chat")
def chat(payload: ChatQuery) -> dict:
    if settings.guardrails_enabled:
        decision = evaluate_query_safety(payload.query)
        if not decision.allow:
            return {
                "query": payload.query,
                "guardrail_blocked": True,
                "reason": decision.reason,
                "message": decision.escalation_message,
                "answer": None,
                "citations": [],
            }

    hits = retriever.search(
        query=payload.query,
        appliance_category=payload.appliance_category,
        top_k=payload.top_k,
    )
    retrieved = [{"text": h.text, "score": h.score, "metadata": h.metadata} for h in hits]
    answer = slm.generate_answer(payload.query, retrieved)

    citations = []
    for item in retrieved:
        meta = item.get("metadata", {})
        citations.append(
            {
                "guide_title": meta.get("guide_title"),
                "source_url": meta.get("source_url"),
                "step_number": meta.get("step_number"),
                "previous_steps": meta.get("previous_steps", []),
            }
        )

    return {
        "query": payload.query,
        "answer": answer,
        "citations": citations,
        "retrieval_count": len(retrieved),
    }
