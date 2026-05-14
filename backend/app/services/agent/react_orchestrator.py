"""LLM-driven ReAct orchestration with real LM Studio tool calls.

The agent never synthesises responses on the LLM's behalf. If the model fails to
produce a valid tool-call chain that ends in ``finalize_answer``, the turn fails
loudly and an error payload is returned to the UI instead of a templated answer.
The only server-side passthroughs are deterministic helpers that the model
explicitly invokes (safety check, retrieval, part identifier, step formatter)
and citation metadata copied from real retrieval hits (anti-hallucination).
"""

from __future__ import annotations

import json
from typing import Any, Callable

from app.core.config import settings
from app.services.agent.orchestrator import ChatTurnInput
from app.services.agent.schemas import (
    ClarifyInput,
    FormatGuideInput,
    ManualSearchInput,
    PartIdentifyInput,
    RetrievedSnippet,
    SafetyCheckInput,
    StructuredChatPayload,
)
from app.services.agent.tool_schemas import AGENT_TOOL_SCHEMAS
from app.services.agent.tools import (
    ManualSearchTool,
    PartIdentifier,
    SafetyProtocolChecker,
    StepByStepGuideFormatter,
    SymptomClarifier,
    extract_required_tools,
)
from app.services.chat_store import ChatStore, get_chat_store

MAX_REACT_STEPS = 8


def _dump(obj: Any) -> str:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    return json.dumps(obj, ensure_ascii=True)


def _citations(snippets: list[RetrievedSnippet]) -> list[dict[str, Any]]:
    return [
        {
            "guide_title": s.metadata.get("guide_title"),
            "source_url": s.metadata.get("source_url"),
            "step_number": s.metadata.get("step_number"),
            "score": s.score,
            "previous_steps": s.metadata.get("previous_steps", []),
            "metadata": s.metadata,
        }
        for s in snippets
    ]


def _snippets_payload(snippets: list[RetrievedSnippet]) -> list[dict[str, Any]]:
    return [
        {
            "text": s.text[:400],
            "score": s.score,
            "guide_title": s.metadata.get("guide_title"),
            "step_number": s.metadata.get("step_number"),
        }
        for s in snippets[:5]
    ]


def _compact_search_tool_result(snippets: list[RetrievedSnippet]) -> str:
    """Tiny JSON returned to the model after manual_search.

    Full snippet objects (with metadata + previous_steps) are kept server-side
    in ``last_snippets`` and re-used by ``part_identifier`` / ``step_by_step_formatter``.
    The model only needs enough context to decide what to do next.
    """
    compact = [
        {
            "rank": i + 1,
            "text": (s.text or "")[:240],
            "score": round(float(s.score), 4),
            "guide_title": s.metadata.get("guide_title"),
            "step_number": s.metadata.get("step_number"),
        }
        for i, s in enumerate(snippets[:5])
    ]
    return json.dumps(
        {"snippets": compact, "retrieval_count": len(snippets)}, ensure_ascii=True
    )


def _shape_history_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Rewrite stored chat history so prior assistant turns do not look like free-form chat.

    Earlier turns persist the finalized ``summary`` as plain assistant text. If we
    replay that verbatim, smaller / less tool-call-adherent models pattern-match
    on the chat shape and reply in prose instead of calling tools. We wrap the
    stored content in an explicit marker so the model treats it as a record of a
    completed tool call rather than as a few-shot example of chat-mode output.
    """
    role = msg["role"]
    content = (msg.get("content") or "").strip()
    if role == "assistant":
        truncated = content[:600]
        return {
            "role": "assistant",
            "content": (
                "[Previous turn finalized via finalize_answer tool. "
                f"Recorded summary: {truncated}]"
            ),
        }
    return {"role": role, "content": content}


def _assistant_tool_message(content: str | None, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=True),
                },
            }
            for call in tool_calls
        ],
    }


def _system_prompt(session_context: dict[str, Any]) -> str:
    appliance_type = session_context.get("appliance_type") or "unknown"
    brand = session_context.get("brand") or "unknown"
    model = session_context.get("model") or "unknown"
    return (
        "You are DIY-Assist, a safety-first appliance troubleshooting ReAct agent.\n"
        "You MUST respond by calling tools. Never reply with plain text or an empty message.\n"
        "Every turn ends by calling finalize_answer exactly once.\n\n"
        "Required order for a new symptom:\n"
        "  1. safety_protocol_checker on the current user message.\n"
        "  2. If unsafe: finalize_answer immediately with a safety_warning and no steps.\n"
        "  3. If safe and the symptom is clear, call manual_search using the appliance context below.\n"
        "  4. If manual_search returns snippets, call step_by_step_formatter and part_identifier (no arguments needed; the server uses the latest retrieval), then finalize_answer.\n"
        "  5. If manual_search returns no snippets, finalize_answer with a clarifying_question that asks for diagnostic detail (sounds, smells, error codes, lights, when it started). Do NOT default to leak wording.\n"
        "  6. If the user pivots to a different symptom after an earlier safety escalation, treat it as a new turn and restart at step 1.\n"
        "  7. If the user asks what you can do, finalize_answer with a short capability summary and a clarifying_question.\n\n"
        "Rules:\n"
        "  - Never invent citations, parts, tools, or steps. They are taken server-side from the retrieved snippets — do not echo any retrieved JSON back inside tool arguments.\n"
        "  - Only pass natural-language fields (summary, likely_issue, clarifying_question, safety_warning) to finalize_answer.\n"
        "  - Reuse the appliance context below when the user message omits the appliance.\n"
        "  - Keep summaries concise (1-3 sentences).\n\n"
        f"Current appliance context: appliance_type={appliance_type}, brand={brand}, model={model}."
    )


def run_react_agent_chat(
    turn: ChatTurnInput,
    *,
    retrieve_fn: Callable[..., Any],
    slm: Any,
    live_ifixit_fn: Callable[..., list] | None,
    store: ChatStore | None = None,
    slm_model_name: str | None = None,
) -> dict[str, Any]:
    store = store or get_chat_store()
    sid = store.create_session(
        session_id=turn.session_id,
        appliance_category=turn.appliance_category,
        appliance_type=turn.appliance_type,
        brand=turn.brand,
        model=turn.model,
    )
    session = store.get_session(sid) or {}
    raw_history = store.get_recent_messages(sid, limit=6)
    history = [
        _shape_history_message(msg)
        for msg in raw_history
        if msg.get("role") in {"user", "assistant"} and (msg.get("content") or "").strip()
    ]
    store.append_message(sid, role="user", content=turn.query)

    session_appliance_type = turn.appliance_type or session.get("appliance_type")
    store.update_session_context(
        sid,
        appliance_category=turn.appliance_category,
        appliance_type=session_appliance_type,
        brand=turn.brand,
        model=turn.model,
    )
    session = store.get_session(sid) or {}

    trace: list[dict[str, Any]] = []
    last_snippets: list[RetrievedSnippet] = []
    last_parts: list[str] = []
    last_tools: list[str] = []
    last_steps: list[str] = []
    safety_checked = False
    safety_blocked = False
    safety_warning: str | None = None

    def guarded_retrieve(**kwargs) -> list[Any]:
        appliance_type = kwargs.get("appliance_type") or session_appliance_type
        hits = retrieve_fn(
            query=kwargs["query"],
            appliance_category=kwargs.get("appliance_category") or turn.appliance_category,
            appliance_type=appliance_type,
            brand=kwargs.get("brand") or turn.brand,
            model=kwargs.get("model") or turn.model,
            top_k=kwargs.get("top_k") or turn.top_k,
        )
        if appliance_type:
            hits = [
                hit
                for hit in hits
                if (hit.metadata or {}).get("appliance_type") in {appliance_type, None, ""}
            ]
        min_score = float(getattr(settings, "retrieval_min_score", 0.45))
        return [hit for hit in hits if float(getattr(hit, "score", 0.0)) >= min_score]

    search_tool = ManualSearchTool(guarded_retrieve)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(session)},
        *history,
        {"role": "user", "content": turn.query},
    ]

    for _ in range(MAX_REACT_STEPS):
        if slm_model_name:
            model_out = slm.chat_with_tools(
                messages=messages,
                tools=AGENT_TOOL_SCHEMAS,
                model_name=slm_model_name,
            )
        else:
            model_out = slm.chat_with_tools(messages=messages, tools=AGENT_TOOL_SCHEMAS)

        llm_error = model_out.get("error")
        tool_calls = model_out.get("tool_calls") or []
        content = model_out.get("content")

        if llm_error:
            trace.append({"tool": "llm_error", "error": str(llm_error)})
            return _error_payload(
                turn=turn,
                sid=sid,
                store=store,
                error_message=(
                    "The local model could not complete this turn (LLM error: "
                    f"{llm_error}). No deterministic fallback is used; please retry "
                    "or switch to a model with reliable tool-call support."
                ),
                trace=trace,
                safety_warning=safety_warning,
                guardrail_blocked=safety_blocked,
            )

        if not tool_calls:
            trace.append({"tool": "llm_no_tool_call", "content": (content or "")[:200]})
            return _error_payload(
                turn=turn,
                sid=sid,
                store=store,
                error_message=(
                    "The local model returned no tool call. The ReAct loop requires a "
                    "finalize_answer tool call to produce a response. Common causes: the "
                    "model emitted malformed tool-call JSON (parsed and dropped by LM "
                    "Studio), the model is too small for tool calling, or the context "
                    "window is exhausted."
                ),
                trace=trace,
                safety_warning=safety_warning,
                guardrail_blocked=safety_blocked,
            )

        messages.append(_assistant_tool_message(content, tool_calls))
        for call in tool_calls:
            name = call["name"]
            args = call.get("arguments") or {}

            if name != "safety_protocol_checker" and not safety_checked:
                forced_safety = SafetyProtocolChecker().check(SafetyCheckInput(user_query=turn.query))
                safety_checked = True
                safety_blocked = not forced_safety.allow_troubleshooting
                safety_warning = forced_safety.escalation_message
                trace.append(
                    {
                        "tool": "safety_protocol_checker",
                        "forced": True,
                        "allow": forced_safety.allow_troubleshooting,
                    }
                )
                if safety_blocked:
                    payload = _finalize_payload(
                        turn=turn,
                        sid=sid,
                        summary=None,
                        clarifying_question=None,
                        likely_issue=None,
                        steps=[],
                        parts=[],
                        tools=[],
                        citations=[],
                        snippets=[],
                        trace=trace,
                        safety_warning=safety_warning,
                        guardrail_blocked=True,
                        live_guides=[],
                    )
                    store.append_message(
                        sid,
                        role="assistant",
                        content=safety_warning or "",
                        structured=payload["structured"],
                    )
                    return payload

            trace.append({"tool": name, "arguments": args})

            if name == "safety_protocol_checker":
                out = SafetyProtocolChecker().check(
                    SafetyCheckInput(user_query=str(args.get("user_query") or turn.query))
                )
                safety_checked = True
                safety_blocked = not out.allow_troubleshooting
                safety_warning = out.escalation_message
                tool_content = _dump(out)
                if safety_blocked:
                    payload = _finalize_payload(
                        turn=turn,
                        sid=sid,
                        summary=None,
                        clarifying_question=None,
                        likely_issue=None,
                        steps=[],
                        parts=[],
                        tools=[],
                        citations=[],
                        snippets=[],
                        trace=trace,
                        safety_warning=safety_warning,
                        guardrail_blocked=True,
                        live_guides=[],
                    )
                    store.append_message(
                        sid,
                        role="assistant",
                        content=safety_warning or "",
                        structured=payload["structured"],
                    )
                    return payload

            elif name == "symptom_clarifier":
                out = SymptomClarifier().analyze(
                    ClarifyInput(user_query=str(args.get("user_query") or turn.query))
                )
                tool_content = _dump(out)

            elif name == "manual_search":
                out = search_tool.run(
                    ManualSearchInput(
                        query=str(args.get("query") or turn.query),
                        appliance_category=turn.appliance_category,
                        appliance_type=args.get("appliance_type") or session_appliance_type,
                        brand=args.get("brand") or turn.brand,
                        model=args.get("model") or turn.model,
                        top_k=int(args.get("top_k") or turn.top_k),
                    )
                )
                last_snippets = out.snippets
                trace.append(
                    {
                        "tool": "manual_search_result",
                        "retrieval_count": len(last_snippets),
                        "top_score": last_snippets[0].score if last_snippets else None,
                        "top_guide": (
                            last_snippets[0].metadata.get("guide_title")
                            if last_snippets
                            else None
                        ),
                    }
                )
                tool_content = _compact_search_tool_result(last_snippets)

            elif name == "part_identifier":
                out = PartIdentifier().identify(
                    PartIdentifyInput(retrieved_snippets=last_snippets)
                )
                last_parts = [part.name for part in out.parts]
                last_tools = out.tools_required
                tool_content = _dump(out)

            elif name == "step_by_step_formatter":
                out = StepByStepGuideFormatter().format(
                    FormatGuideInput(
                        retrieved_snippets=last_snippets,
                        max_steps=int(args.get("max_steps") or 8),
                    )
                )
                last_steps = [step.instruction for step in out.steps]
                tool_content = _dump(out)

            elif name == "finalize_answer":
                live_guides = (
                    live_ifixit_fn(query=turn.query, appliance_category=turn.appliance_category)
                    if live_ifixit_fn
                    else []
                )
                summary = args.get("summary")
                if not isinstance(summary, str) or not summary.strip():
                    return _error_payload(
                        turn=turn,
                        sid=sid,
                        store=store,
                        error_message=(
                            "The local model called finalize_answer without a summary. "
                            "No deterministic summary is synthesised; please retry."
                        ),
                        trace=trace,
                        safety_warning=safety_warning,
                        guardrail_blocked=safety_blocked,
                    )
                payload = _finalize_payload(
                    turn=turn,
                    sid=sid,
                    summary=summary,
                    clarifying_question=args.get("clarifying_question"),
                    likely_issue=args.get("likely_issue"),
                    steps=last_steps,
                    parts=last_parts,
                    tools=last_tools or extract_required_tools(last_snippets),
                    citations=_citations(last_snippets),
                    snippets=_snippets_payload(last_snippets),
                    trace=trace,
                    safety_warning=args.get("safety_warning") or safety_warning,
                    guardrail_blocked=safety_blocked,
                    live_guides=live_guides,
                )
                store.append_message(
                    sid,
                    role="assistant",
                    content=payload.get("answer") or payload.get("message") or "",
                    structured=payload["structured"],
                )
                return payload

            else:
                tool_content = _dump({"error": f"Unknown tool: {name}"})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": tool_content,
                }
            )

    return _error_payload(
        turn=turn,
        sid=sid,
        store=store,
        error_message=(
            "The ReAct loop hit the maximum step budget without the model calling "
            "finalize_answer. No deterministic fallback is used."
        ),
        trace=trace,
        safety_warning=safety_warning,
        guardrail_blocked=safety_blocked,
    )


def _finalize_payload(
    *,
    turn: ChatTurnInput,
    sid: str,
    summary: str | None,
    clarifying_question: str | None,
    likely_issue: str | None,
    steps: list[str],
    parts: list[str],
    tools: list[str],
    citations: list[dict[str, Any]],
    snippets: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    safety_warning: str | None,
    guardrail_blocked: bool,
    live_guides: list,
) -> dict[str, Any]:
    structured = StructuredChatPayload(
        answer_summary=summary,
        clarifying_question=clarifying_question,
        likely_issue=likely_issue,
        steps=steps,
        parts_list=parts,
        tools_required=tools,
        retrieved_guide_snippets=snippets,
        tool_trace=trace,
    )
    answer = summary
    if clarifying_question:
        answer = f"{summary or ''}\n\n{clarifying_question}".strip()
    return {
        "query": turn.query,
        "session_id": sid,
        "strategy": turn.strategy_label,
        "guardrail_blocked": guardrail_blocked,
        "safety_warning": safety_warning,
        "message": safety_warning if guardrail_blocked else None,
        "structured": structured.model_dump(),
        "answer": None if guardrail_blocked else answer,
        "citations": citations,
        "live_ifixit_guides": live_guides,
        "live_ifixit_used": bool(live_guides),
        "slm_provider": "lmstudio",
        "retrieval_count": len(snippets),
        "agent_mode": "react",
    }


def _error_payload(
    *,
    turn: ChatTurnInput,
    sid: str,
    store: ChatStore,
    error_message: str,
    trace: list[dict[str, Any]],
    safety_warning: str | None,
    guardrail_blocked: bool,
) -> dict[str, Any]:
    structured = StructuredChatPayload(
        answer_summary=None,
        clarifying_question=None,
        likely_issue=None,
        steps=[],
        parts_list=[],
        tools_required=[],
        retrieved_guide_snippets=[],
        tool_trace=trace,
    )
    payload = {
        "query": turn.query,
        "session_id": sid,
        "strategy": turn.strategy_label,
        "guardrail_blocked": guardrail_blocked,
        "safety_warning": safety_warning,
        "message": error_message,
        "structured": structured.model_dump(),
        "answer": None,
        "citations": [],
        "live_ifixit_guides": [],
        "live_ifixit_used": False,
        "slm_provider": "lmstudio",
        "retrieval_count": 0,
        "agent_mode": "react",
        "error": error_message,
    }
    store.append_message(
        sid,
        role="assistant",
        content=error_message,
        structured=payload["structured"],
    )
    return payload
