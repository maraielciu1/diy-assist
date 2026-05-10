"""ReAct-style orchestration for Stage 4 chat (deterministic tool sequence)."""

from __future__ import annotations

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
)
from app.services.chat_store import ChatStore, get_chat_store


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
    trace.append({"tool": "Manual_Search_Tool", "retrieval_count": ms_out.retrieval_count})

    if ms_out.retrieval_count == 0:
        structured = StructuredChatPayload(
            answer_summary=(
                "No indexed repair guides matched closely enough. "
                "Try ingesting more guides or narrowing appliance filters."
            ),
            clarifying_question=None,
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
        answer_summary=answer_text.split("\n")[0][:500] if answer_text else None,
        clarifying_question=None,
        likely_issue=(
            ms_out.snippets[0].metadata.get("guide_title") if ms_out.snippets else None
        ),
        steps=[s.instruction for s in g_out.steps],
        parts_list=[p.name for p in p_out.parts],
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
