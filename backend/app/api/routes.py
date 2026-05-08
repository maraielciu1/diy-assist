from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.guardrails import evaluate_query_safety
from app.services.ifixit_live import IFixitLiveClient
from app.services.llm import SLMWrapper
from app.services.retrieval import NaiveRetriever

router = APIRouter()
slm = SLMWrapper()
ifixit_live_client = IFixitLiveClient()
_retriever: NaiveRetriever | None = None


APPLIANCE_TYPE_HINTS = {
    "washer": "washer",
    "washing machine": "washer",
    "dryer": "dryer",
    "dishwasher": "dishwasher",
    "refrigerator": "refrigerator",
    "fridge": "refrigerator",
    "oven": "oven",
    "microwave": "microwave",
}


def _provider_name() -> str:
    return str(getattr(settings, "slm_provider", "unknown"))


def _get_retriever() -> NaiveRetriever:
    global _retriever
    if _retriever is None:
        _retriever = NaiveRetriever()
    return _retriever


def _infer_appliance_type_from_query(query: str) -> str | None:
    lowered = query.lower()
    for keyword, appliance_type in APPLIANCE_TYPE_HINTS.items():
        if keyword in lowered:
            return appliance_type
    return None


def _resolve_appliance_type(explicit: str | None, query: str) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip().lower()
    return _infer_appliance_type_from_query(query)


class RAGQuery(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    appliance_category: str | None = None
    appliance_type: str | None = None
    brand: str | None = None
    model: str | None = None
    top_k: int = Field(default=settings.top_k_default, ge=1, le=20)


class ChatQuery(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    appliance_category: str | None = None
    appliance_type: str | None = None
    brand: str | None = None
    model: str | None = None
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

    try:
        appliance_type = _resolve_appliance_type(
            explicit=payload.appliance_type,
            query=payload.query,
        )
        hits = _get_retriever().search(
            query=payload.query,
            appliance_category=payload.appliance_category,
            appliance_type=appliance_type,
            brand=payload.brand,
            model=payload.model,
            top_k=payload.top_k,
        )
    except Exception as exc:
        return {
            "query": payload.query,
            "strategy": "naive",
            "results": [],
            "error": (
                "Retriever is unavailable. If Chroma is corrupted, clear `data/chroma/` "
                "and re-run ingestion."
            ),
            "details": str(exc),
        }

    return {
        "query": payload.query,
        "strategy": "naive",
        "results": [
            {
                "text": chunk.text,
                "score": chunk.score,
                "guide_title": chunk.metadata.get("guide_title"),
                "step_number": chunk.metadata.get("step_number"),
                "previous_steps": chunk.metadata.get("previous_steps", []),
                "metadata": chunk.metadata,
            }
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

    try:
        appliance_type = _resolve_appliance_type(
            explicit=payload.appliance_type,
            query=payload.query,
        )
        hits = _get_retriever().search(
            query=payload.query,
            appliance_category=payload.appliance_category,
            appliance_type=appliance_type,
            brand=payload.brand,
            model=payload.model,
            top_k=payload.top_k,
        )
    except Exception as exc:
        return {
            "query": payload.query,
            "answer": None,
            "citations": [],
            "live_ifixit_guides": [],
            "live_ifixit_used": False,
            "slm_provider": _provider_name(),
            "retrieval_count": 0,
            "error": (
                "Retriever is unavailable. If Chroma is corrupted, clear `data/chroma/` "
                "and re-run ingestion."
            ),
            "details": str(exc),
        }

    retrieved = [{"text": h.text, "score": h.score, "metadata": h.metadata} for h in hits]
    live_ifixit_guides = []
    if settings.use_ifixit_live_lookup:
        live_ifixit_guides = ifixit_live_client.suggest_guides(
            query=payload.query,
            appliance_category=payload.appliance_category,
            limit=3,
        )
    answer = slm.generate_answer(
        payload.query,
        retrieved,
        live_ifixit_guides=live_ifixit_guides,
    )

    citations = []
    for item in retrieved:
        meta = item.get("metadata", {})
        citations.append(
            {
                "guide_title": meta.get("guide_title"),
                "source_url": meta.get("source_url"),
                "step_number": meta.get("step_number"),
                "score": item.get("score"),
                "previous_steps": meta.get("previous_steps", []),
                "metadata": meta,
            }
        )

    return {
        "query": payload.query,
        "answer": answer,
        "citations": citations,
        "live_ifixit_guides": live_ifixit_guides,
        "live_ifixit_used": bool(live_ifixit_guides),
        "slm_provider": _provider_name(),
        "retrieval_count": len(retrieved),
    }
