"""ReAct-style orchestration for Stage 4 chat (deterministic tool sequence)."""

from __future__ import annotations

import re
from typing import Any, Callable

from app.services.agent.schemas import (
    ClarifyInput,
    FormatGuideInput,
    ManualSearchInput,
    PartIdentifyInput,
    SafetyCheckInput,
    StructuredChatPayload,
)
from app.services.agent.tools import (
    ManualSearchTool,
    PartIdentifier,
    SafetyProtocolChecker,
    StepByStepGuideFormatter,
    SymptomClarifier,
    extract_required_tools,
)
from app.services.chat_store import ChatStore, get_chat_store
from app.core.config import settings


_APPLIANCE_WORDS = {
    "washer",
    "washing machine",
    "dryer",
    "dishwasher",
    "refrigerator",
    "fridge",
    "freezer",
    "oven",
    "microwave",
    "stove",
}
_SYMPTOM_WORDS = {
    "leak",
    "leaking",
    "noise",
    "noisy",
    "hum",
    "humming",
    "drain",
    "draining",
    "spin",
    "spinning",
    "heat",
    "heating",
    "cool",
    "cooling",
    "warm",
    "cold",
    "start",
    "broken",
    "error",
    "smell",
    "smoke",
    "spark",
    "vibrate",
    "shaking",
    "latch",
    "clog",
    "filter",
    "pump",
    "vent",
}
_CAPABILITY_QUESTIONS = (
    "what can you help",
    "how can you help",
    "what do you do",
    "who are you",
    "help me with",
)


def _is_capability_question(query: str) -> bool:
    lowered = query.lower().strip()
    return any(phrase in lowered for phrase in _CAPABILITY_QUESTIONS)


def _looks_like_repair_request(query: str) -> bool:
    lowered = query.lower()
    return any(word in lowered for word in _APPLIANCE_WORDS) or any(
        word in lowered for word in _SYMPTOM_WORDS
    )


def _retrieval_confidence(snippets: list[Any]) -> float:
    scores: list[float] = []
    for snippet in snippets:
        meta = getattr(snippet, "metadata", {}) or {}
        raw_score = meta.get("baseline_score", getattr(snippet, "score", 0.0))
        try:
            scores.append(float(raw_score))
        except (TypeError, ValueError):
            continue
    return max(scores) if scores else 0.0


def _answer_summary(answer_text: str | None) -> str | None:
    if not answer_text:
        return None
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer_text.strip())
        if sentence.strip()
    ]
    if not sentences:
        return answer_text.strip()[:500]
    return " ".join(sentences[:2])[:500]


class ChatTurnInput:
    """Bag of fields from API request (avoid circular import with routes)."""

    def __init__(
        self,
        *,
        query: str,
        appliance_category: str | None,
        appliance_type: str | None,
        brand: str | None,
        model: str | None,
        top_k: int,
        session_id: str | None,
        strategy_label: str,
    ) -> None:
        self.query = query
        self.appliance_category = appliance_category
        self.appliance_type = appliance_type
        self.brand = brand
        self.model = model
        self.top_k = top_k
        self.session_id = session_id
        self.strategy_label = strategy_label


def run_agent_chat(
    turn: ChatTurnInput,
    *,
    retrieve_fn: Callable[..., Any],
    slm_generate: Callable[..., str],
    live_ifixit_fn: Callable[..., list] | None,
    provider_name: str,
    store: ChatStore | None = None,
    skip_user_append: bool = False,
) -> dict[str, Any]:
    """
    Run safety → clarify → manual search → parts → format → SLM answer.
    Returns API-ready dict including legacy `answer` / `citations` keys.
    """
    trace: list[dict[str, Any]] = []
    store = store or get_chat_store()

    safety = SafetyProtocolChecker()
    s_out = safety.check(SafetyCheckInput(user_query=turn.query))
    trace.append({"tool": "Safety_Protocol_Checker", "allow": s_out.allow_troubleshooting})

    sid = store.create_session(
        session_id=turn.session_id,
        appliance_category=turn.appliance_category,
        appliance_type=turn.appliance_type,
        brand=turn.brand,
        model=turn.model,
    )
    if not skip_user_append:
        store.append_message(sid, role="user", content=turn.query)

    if not s_out.allow_troubleshooting:
        structured = StructuredChatPayload(
            answer_summary=None,
            clarifying_question=None,
            likely_issue=None,
            steps=[],
            parts_list=[],
            retrieved_guide_snippets=[],
            tool_trace=trace,
        )
        payload = {
            "query": turn.query,
            "session_id": sid,
            "strategy": turn.strategy_label,
            "guardrail_blocked": True,
            "reason": s_out.reason,
            "message": s_out.escalation_message,
            "safety_warning": s_out.escalation_message,
            "structured": structured.model_dump(),
            "answer": None,
            "citations": [],
            "live_ifixit_guides": [],
            "live_ifixit_used": False,
            "slm_provider": provider_name,
            "retrieval_count": 0,
            "agent_mode": True,
        }
        store.append_message(
            sid,
            role="assistant",
            content=s_out.escalation_message or "",
            structured=payload["structured"],
        )
        return payload

    if _is_capability_question(turn.query) or not _looks_like_repair_request(turn.query):
        trace.append({"tool": "Intent_Router", "intent": "general_help"})
        answer = (
            "I can help troubleshoot home appliance problems using indexed repair guides. "
            "Tell me the appliance type, symptom, noises, smells, leaks, error codes, and "
            "when the problem happens. If the issue sounds hazardous, I will stop repair "
            "guidance and tell you to contact a qualified professional."
        )
        structured = StructuredChatPayload(
            answer_summary=answer,
            clarifying_question=(
                "What appliance are you working on, and what symptom are you seeing?"
            ),
            likely_issue=None,
            steps=[],
            parts_list=[],
            retrieved_guide_snippets=[],
            tool_trace=trace,
        )
        payload = {
            "query": turn.query,
            "session_id": sid,
            "strategy": turn.strategy_label,
            "guardrail_blocked": False,
            "safety_warning": None,
            "structured": structured.model_dump(),
            "answer": answer,
            "citations": [],
            "live_ifixit_guides": [],
            "live_ifixit_used": False,
            "slm_provider": provider_name,
            "retrieval_count": 0,
            "agent_mode": True,
            "intent": "general_help",
        }
        store.append_message(sid, role="assistant", content=answer, structured=payload["structured"])
        return payload

    clarify = SymptomClarifier()
    c_out = clarify.analyze(ClarifyInput(user_query=turn.query))
    trace.append(
        {"tool": "Symptom_Clarifier", "needs_clarification": c_out.needs_clarification}
    )

    if c_out.needs_clarification and c_out.clarifying_question:
        structured = StructuredChatPayload(
            answer_summary=(
                "More detail is needed before suggesting repair steps "
                "without guessing."
            ),
            clarifying_question=c_out.clarifying_question,
            likely_issue=None,
            steps=[],
            parts_list=[],
            retrieved_guide_snippets=[],
            tool_trace=trace,
        )
        ans_text = structured.answer_summary + "\n\n" + c_out.clarifying_question
        payload = {
            "query": turn.query,
            "session_id": sid,
            "strategy": turn.strategy_label,
            "guardrail_blocked": False,
            "safety_warning": None,
            "structured": structured.model_dump(),
            "answer": ans_text,
            "citations": [],
            "live_ifixit_guides": [],
            "live_ifixit_used": False,
            "slm_provider": provider_name,
            "retrieval_count": 0,
            "agent_mode": True,
            "uncertainty": True,
        }
        store.append_message(sid, role="assistant", content=ans_text, structured=payload["structured"])
        return payload

    search_tool = ManualSearchTool(retrieve_fn)
    ms_in = ManualSearchInput(
        query=turn.query,
        appliance_category=turn.appliance_category,
        appliance_type=turn.appliance_type,
        brand=turn.brand,
        model=turn.model,
        top_k=turn.top_k,
    )
    ms_out = search_tool.run(ms_in)
    if turn.appliance_type:
        ms_out.snippets = [
            snippet
            for snippet in ms_out.snippets
            if (snippet.metadata or {}).get("appliance_type") in {turn.appliance_type, None, ""}
        ]
        ms_out.retrieval_count = len(ms_out.snippets)
    trace.append({"tool": "Manual_Search_Tool", "retrieval_count": ms_out.retrieval_count})

    confidence = _retrieval_confidence(ms_out.snippets)
    min_score = float(getattr(settings, "retrieval_min_score", 0.55))
    trace.append(
        {
            "tool": "Retrieval_Confidence_Check",
            "top_score": confidence,
            "min_score": min_score,
            "passed": confidence >= min_score,
        }
    )

    if ms_out.retrieval_count == 0 or confidence < min_score:
        structured = StructuredChatPayload(
            answer_summary=(
                "I do not have enough matching repair-guide context to give reliable steps yet."
            ),
            clarifying_question=(
                "What appliance type, brand/model if known, and exact symptom should I troubleshoot?"
            ),
            likely_issue=None,
            steps=[],
            parts_list=[],
            retrieved_guide_snippets=[],
            tool_trace=trace,
        )
        txt = structured.answer_summary
        payload = {
            "query": turn.query,
            "session_id": sid,
            "strategy": turn.strategy_label,
            "guardrail_blocked": False,
            "safety_warning": None,
            "structured": structured.model_dump(),
            "answer": txt,
            "citations": [],
            "live_ifixit_guides": [],
            "live_ifixit_used": False,
            "slm_provider": provider_name,
            "retrieval_count": 0,
            "agent_mode": True,
            "uncertainty": True,
        }
        store.append_message(sid, role="assistant", content=txt, structured=payload["structured"])
        return payload

    parts_tool = PartIdentifier()
    p_out = parts_tool.identify(PartIdentifyInput(retrieved_snippets=ms_out.snippets))
    trace.append({"tool": "Part_Identifier", "parts_found": len(p_out.parts)})

    formatter = StepByStepGuideFormatter()
    g_out = formatter.format(FormatGuideInput(retrieved_snippets=ms_out.snippets))

    trace.append({"tool": "Step_By_Step_Guide_Formatter", "steps": len(g_out.steps)})

    retrieved_for_slm = [
        {"text": s.text, "score": s.score, "metadata": s.metadata} for s in ms_out.snippets
    ]
    live_guides: list = []
    if live_ifixit_fn:
        live_guides = live_ifixit_fn(
            query=turn.query,
            appliance_category=turn.appliance_category,
        )

    answer_text = slm_generate(
        turn.query,
        retrieved_for_slm,
        live_guides,
    )

    citations = []
    for item in retrieved_for_slm:
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

    snippets_payload = [
        {
            "text": s.text[:400],
            "score": s.score,
            "guide_title": (s.metadata or {}).get("guide_title"),
            "step_number": (s.metadata or {}).get("step_number"),
        }
        for s in ms_out.snippets[:5]
    ]

    structured = StructuredChatPayload(
        clarifying_question=None,
        likely_issue=(
            ms_out.snippets[0].metadata.get("guide_title") if ms_out.snippets else None
        ),
        answer_summary=_answer_summary(answer_text),
        steps=[s.instruction for s in g_out.steps],
        parts_list=[p.name for p in p_out.parts],
        tools_required=p_out.tools_required or extract_required_tools(ms_out.snippets),
        retrieved_guide_snippets=snippets_payload,
        tool_trace=trace,
    )

    payload = {
        "query": turn.query,
        "session_id": sid,
        "strategy": turn.strategy_label,
        "guardrail_blocked": False,
        "safety_warning": None,
        "structured": structured.model_dump(),
        "answer": answer_text,
        "citations": citations,
        "live_ifixit_guides": live_guides,
        "live_ifixit_used": bool(live_guides),
        "slm_provider": provider_name,
        "retrieval_count": ms_out.retrieval_count,
        "agent_mode": True,
    }
    store.append_message(
        sid,
        role="assistant",
        content=answer_text,
        structured=payload["structured"],
    )
    return payload
