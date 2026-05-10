from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.guardrails import evaluate_query_safety
from app.services.ifixit_live import IFixitLiveClient
from app.services.llm import SLMWrapper
from app.services.agent.orchestrator import ChatTurnInput, run_agent_chat
from app.services.retrieval import NaiveRetriever, RerankedRetriever
from app.services.hyde import HyDERetriever, HyDERerankedRetriever

router = APIRouter()
slm = SLMWrapper()
ifixit_live_client = IFixitLiveClient()
_retriever: NaiveRetriever | None = None
_reranked_retriever: RerankedRetriever | None = None
_hyde_retriever: HyDERetriever | None = None
_hyde_reranked_retriever: HyDERerankedRetriever | None = None


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


def _get_reranked_retriever() -> RerankedRetriever:
    global _reranked_retriever
    if _reranked_retriever is None:
        _reranked_retriever = RerankedRetriever(base=_get_retriever())
    return _reranked_retriever


def _get_hyde_retriever() -> HyDERetriever:
    global _hyde_retriever
    if _hyde_retriever is None:
        _hyde_retriever = HyDERetriever(base=_get_retriever(), slm=slm)
    return _hyde_retriever


def _get_hyde_reranked_retriever() -> HyDERerankedRetriever:
    global _hyde_reranked_retriever
    if _hyde_reranked_retriever is None:
        _hyde_reranked_retriever = HyDERerankedRetriever(base=_get_retriever(), slm=slm)
    return _hyde_reranked_retriever


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
    # Stage 4: session + agent / legacy
    session_id: str | None = None
    use_legacy_chat: bool = False
    retrieval_strategy: str | None = None


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


@router.post("/rag/reranked")
def reranked_rag(payload: RAGQuery) -> dict:
    if settings.guardrails_enabled:
        decision = evaluate_query_safety(payload.query)
        if not decision.allow:
            return {
                "query": payload.query,
                "strategy": "reranked",
                "guardrail_blocked": True,
                "reason": decision.reason,
                "message": decision.escalation_message,
                "results": [],
            }

    try:
        appliance_type = _resolve_appliance_type(explicit=payload.appliance_type, query=payload.query)
        hits = _get_reranked_retriever().search(
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
            "strategy": "reranked",
            "results": [],
            "error": (
                "Retriever is unavailable. If Chroma is corrupted, clear `data/chroma/` "
                "and re-run ingestion."
            ),
            "details": str(exc),
        }

    return {
        "query": payload.query,
        "strategy": "reranked",
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


@router.post("/rag/hyde")
def hyde_rag(payload: RAGQuery) -> dict:
    if settings.guardrails_enabled:
        decision = evaluate_query_safety(payload.query)
        if not decision.allow:
            return {
                "query": payload.query,
                "strategy": "hyde",
                "guardrail_blocked": True,
                "reason": decision.reason,
                "message": decision.escalation_message,
                "results": [],
            }

    try:
        appliance_type = _resolve_appliance_type(explicit=payload.appliance_type, query=payload.query)
        hits = _get_hyde_retriever().search(
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
            "strategy": "hyde",
            "results": [],
            "error": (
                "Retriever is unavailable. If Chroma is corrupted, clear `data/chroma/` "
                "and re-run ingestion."
            ),
            "details": str(exc),
        }

    return {
        "query": payload.query,
        "strategy": "hyde",
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


@router.post("/rag/hyde_reranked")
def hyde_reranked_rag(payload: RAGQuery) -> dict:
    if settings.guardrails_enabled:
        decision = evaluate_query_safety(payload.query)
        if not decision.allow:
            return {
                "query": payload.query,
                "strategy": "hyde_reranked",
                "guardrail_blocked": True,
                "reason": decision.reason,
                "message": decision.escalation_message,
                "results": [],
            }

    try:
        appliance_type = _resolve_appliance_type(explicit=payload.appliance_type, query=payload.query)
        hits = _get_hyde_reranked_retriever().search(
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
            "strategy": "hyde_reranked",
            "results": [],
            "error": (
                "Retriever is unavailable. If Chroma is corrupted, clear `data/chroma/` "
                "and re-run ingestion."
            ),
            "details": str(exc),
        }

    return {
        "query": payload.query,
        "strategy": "hyde_reranked",
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


def _retrieve_by_strategy(strategy: str):
    def fn(**kwargs):
        if strategy == "reranked":
            return _get_reranked_retriever().search(**kwargs)
        if strategy == "hyde":
            return _get_hyde_retriever().search(**kwargs)
        if strategy == "hyde_reranked":
            return _get_hyde_reranked_retriever().search(**kwargs)
        return _get_retriever().search(**kwargs)

    return fn


def _live_ifixit_lookup(query: str, appliance_category: str | None = None) -> list:
    if not settings.use_ifixit_live_lookup:
        return []
    return ifixit_live_client.suggest_guides(
        query=query,
        appliance_category=appliance_category,
        limit=3,
    )


def _chat_legacy(payload: ChatQuery) -> dict:
    """Direct RAG + SLM without agent orchestration (Stage 1–3 compatibility)."""
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


@router.post("/chat")
def chat(payload: ChatQuery) -> dict:
    if payload.use_legacy_chat:
        return _chat_legacy(payload)

    strat = (payload.retrieval_strategy or settings.chat_retrieval_strategy).lower()
    if strat not in {"naive", "reranked", "hyde", "hyde_reranked"}:
        strat = "naive"

    appliance_type = _resolve_appliance_type(
        explicit=payload.appliance_type,
        query=payload.query,
    )
    retrieve_fn = _retrieve_by_strategy(strat)

    def retrieve_with_filters(**kwargs) -> list:
        at = kwargs.get("appliance_type") or appliance_type
        return retrieve_fn(
            query=kwargs["query"],
            appliance_category=kwargs.get("appliance_category"),
            appliance_type=at,
            brand=kwargs.get("brand"),
            model=kwargs.get("model"),
            top_k=kwargs.get("top_k"),
        )

    turn = ChatTurnInput(
        query=payload.query,
        appliance_category=payload.appliance_category,
        appliance_type=appliance_type,
        brand=payload.brand,
        model=payload.model,
        top_k=payload.top_k,
        session_id=payload.session_id,
        strategy_label=strat,
    )
    try:
        return run_agent_chat(
            turn,
            retrieve_fn=retrieve_with_filters,
            slm_generate=slm.generate_answer,
            live_ifixit_fn=_live_ifixit_lookup,
            provider_name=_provider_name(),
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
            "agent_mode": True,
            "error": (
                "Retriever is unavailable. If Chroma is corrupted, clear `data/chroma/` "
                "and re-run ingestion."
            ),
            "details": str(exc),
        }


def _chat_from_hits(payload: ChatQuery, hits, strategy: str) -> dict:
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
        "strategy": strategy,
        "answer": answer,
        "citations": citations,
        "live_ifixit_guides": live_ifixit_guides,
        "live_ifixit_used": bool(live_ifixit_guides),
        "slm_provider": _provider_name(),
        "retrieval_count": len(retrieved),
    }


@router.post("/chat/reranked")
def chat_reranked(payload: ChatQuery) -> dict:
    if settings.guardrails_enabled:
        decision = evaluate_query_safety(payload.query)
        if not decision.allow:
            return {
                "query": payload.query,
                "strategy": "reranked",
                "guardrail_blocked": True,
                "reason": decision.reason,
                "message": decision.escalation_message,
                "answer": None,
                "citations": [],
            }

    try:
        appliance_type = _resolve_appliance_type(explicit=payload.appliance_type, query=payload.query)
        hits = _get_reranked_retriever().search(
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
            "strategy": "reranked",
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

    return _chat_from_hits(payload=payload, hits=hits, strategy="reranked")


@router.post("/chat/hyde")
def chat_hyde(payload: ChatQuery) -> dict:
    if settings.guardrails_enabled:
        decision = evaluate_query_safety(payload.query)
        if not decision.allow:
            return {
                "query": payload.query,
                "strategy": "hyde",
                "guardrail_blocked": True,
                "reason": decision.reason,
                "message": decision.escalation_message,
                "answer": None,
                "citations": [],
            }

    try:
        appliance_type = _resolve_appliance_type(explicit=payload.appliance_type, query=payload.query)
        hits = _get_hyde_retriever().search(
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
            "strategy": "hyde",
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

    return _chat_from_hits(payload=payload, hits=hits, strategy="hyde")


@router.post("/chat/hyde_reranked")
def chat_hyde_reranked(payload: ChatQuery) -> dict:
    if settings.guardrails_enabled:
        decision = evaluate_query_safety(payload.query)
        if not decision.allow:
            return {
                "query": payload.query,
                "strategy": "hyde_reranked",
                "guardrail_blocked": True,
                "reason": decision.reason,
                "message": decision.escalation_message,
                "answer": None,
                "citations": [],
            }

    try:
        appliance_type = _resolve_appliance_type(explicit=payload.appliance_type, query=payload.query)
        hits = _get_hyde_reranked_retriever().search(
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
            "strategy": "hyde_reranked",
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

    return _chat_from_hits(payload=payload, hits=hits, strategy="hyde_reranked")
