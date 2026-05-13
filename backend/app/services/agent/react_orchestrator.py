"""LLM-driven ReAct orchestration with real LM Studio tool calls."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from app.core.config import settings
from app.services.agent.orchestrator import ChatTurnInput, run_agent_chat
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


def _snippets_from_payload(payload: Any, fallback: list[RetrievedSnippet]) -> list[RetrievedSnippet]:
    if not isinstance(payload, list) or not payload:
        return fallback
    snippets: list[RetrievedSnippet] = []
    for item in payload:
        if isinstance(item, RetrievedSnippet):
            snippets.append(item)
        elif isinstance(item, dict):
            snippets.append(
                RetrievedSnippet(
                    text=str(item.get("text") or ""),
                    score=float(item.get("score") or 0.0),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
    return snippets or fallback


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


def _looks_like_echo(summary: Any, query: str) -> bool:
    if not isinstance(summary, str) or not summary.strip():
        return True
    clean_summary = re.sub(r"[^a-z0-9]+", " ", summary.lower()).strip()
    clean_query = re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()
    return clean_summary == clean_query or clean_query in clean_summary


def _summary_from_context(
    *,
    appliance_type: str | None,
    likely_issue: str | None,
    steps: list[str],
) -> str:
    appliance_label = appliance_type or "appliance"
    if likely_issue:
        return f"Retrieved guide context points to {likely_issue} for this {appliance_label} problem."
    if steps:
        return f"Retrieved guide context found {len(steps)} relevant troubleshooting steps for this {appliance_label} problem."
    return f"Retrieved guide context found matching information for this {appliance_label} problem."


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
        "  2. If unsafe: finalize_answer immediately with safety_warning and no steps.\n"
        "  3. If safe and the symptom is clear, call manual_search using the appliance context below.\n"
        "  4. If manual_search returns snippets, call step_by_step_formatter and part_identifier, then finalize_answer.\n"
        "  5. If manual_search returns no snippets, finalize_answer with a clarifying_question that asks for diagnostic detail (sounds, smells, error codes, lights, when it started). Do NOT default to leak wording.\n"
        "  6. If the user pivots to a different symptom after an earlier safety escalation, treat it as a new turn and restart at step 1.\n"
        "  7. If the user asks what you can do, finalize_answer with a short capability summary and a clarifying_question.\n\n"
        "Rules:\n"
        "  - Never invent citations. Citations come from manual_search results only.\n"
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
        {"role": msg["role"], "content": msg["content"]}
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
    last_search_attempted = False
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
        tool_calls = model_out.get("tool_calls") or []
        content = model_out.get("content")
        if not tool_calls:
            break

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
                last_search_attempted = True
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
                tool_content = _dump(out)

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
                if not safety_checked:
                    safety = SafetyProtocolChecker().check(SafetyCheckInput(user_query=turn.query))
                    trace.append({"tool": "safety_protocol_checker", "forced": True})
                    if not safety.allow_troubleshooting:
                        safety_blocked = True
                        safety_warning = safety.escalation_message
                if last_search_attempted and not last_snippets:
                    appliance_label = session_appliance_type or "the appliance"
                    summary = (
                        f"I could not find specific guidance for that {appliance_label} symptom "
                        "in my indexed repair guides yet."
                    )
                    question = (
                        f"Can you describe the {appliance_label} problem in more detail: any error codes, "
                        "lights, sounds, smells, leaks, when it started, and anything you have already tried?"
                    )
                    payload = _finalize_payload(
                        turn=turn,
                        sid=sid,
                        summary=summary,
                        clarifying_question=question,
                        likely_issue=None,
                        steps=[],
                        parts=[],
                        tools=[],
                        citations=[],
                        snippets=[],
                        trace=trace,
                        safety_warning=safety_warning,
                        guardrail_blocked=False,
                        live_guides=[],
                    )
                    store.append_message(
                        sid,
                        role="assistant",
                        content=payload.get("answer") or summary,
                        structured=payload["structured"],
                    )
                    return payload
                if last_snippets:
                    if not last_steps:
                        forced_steps = StepByStepGuideFormatter().format(
                            FormatGuideInput(retrieved_snippets=last_snippets, max_steps=8)
                        )
                        last_steps = [step.instruction for step in forced_steps.steps]
                        trace.append(
                            {
                                "tool": "step_by_step_formatter",
                                "forced": True,
                                "steps": len(last_steps),
                            }
                        )
                    if not last_parts and not last_tools:
                        forced_parts = PartIdentifier().identify(
                            PartIdentifyInput(retrieved_snippets=last_snippets)
                        )
                        last_parts = [part.name for part in forced_parts.parts]
                        last_tools = forced_parts.tools_required
                        trace.append(
                            {
                                "tool": "part_identifier",
                                "forced": True,
                                "parts_found": len(last_parts),
                            }
                        )
                live_guides = live_ifixit_fn(query=turn.query, appliance_category=turn.appliance_category) if live_ifixit_fn else []
                likely_issue = args.get("likely_issue")
                if last_snippets and not likely_issue:
                    likely_issue = last_snippets[0].metadata.get("guide_title")
                summary = args.get("summary")
                if last_snippets and _looks_like_echo(summary, turn.query):
                    summary = _summary_from_context(
                        appliance_type=session_appliance_type,
                        likely_issue=likely_issue,
                        steps=last_steps,
                    )
                payload = _finalize_payload(
                    turn=turn,
                    sid=sid,
                    summary=summary,
                    clarifying_question=args.get("clarifying_question"),
                    likely_issue=likely_issue,
                    steps=last_steps or args.get("steps") or [],
                    parts=last_parts or args.get("parts_to_inspect") or [],
                    tools=last_tools or extract_required_tools(last_snippets) or args.get("tools_required") or [],
                    citations=_citations(last_snippets) if last_snippets else args.get("citations") or [],
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

    if last_search_attempted and not last_snippets:
        appliance_label = session_appliance_type or "the appliance"
        summary = (
            f"I could not find specific guidance for that {appliance_label} symptom "
            "in my indexed repair guides yet."
        )
        question = (
            f"Can you describe the {appliance_label} problem in more detail: any error codes, "
            "lights, sounds, smells, leaks, when it started, and anything you have already tried?"
        )
        payload = _finalize_payload(
            turn=turn,
            sid=sid,
            summary=summary,
            clarifying_question=question,
            likely_issue=None,
            steps=[],
            parts=[],
            tools=[],
            citations=[],
            snippets=[],
            trace=trace,
            safety_warning=safety_warning,
            guardrail_blocked=False,
            live_guides=[],
        )
        store.append_message(
            sid,
            role="assistant",
            content=payload.get("answer") or summary,
            structured=payload["structured"],
        )
        return payload

    if last_snippets:
        if not last_steps:
            forced_steps = StepByStepGuideFormatter().format(
                FormatGuideInput(retrieved_snippets=last_snippets, max_steps=8)
            )
            last_steps = [step.instruction for step in forced_steps.steps]
            trace.append(
                {
                    "tool": "step_by_step_formatter",
                    "forced": True,
                    "steps": len(last_steps),
                }
            )
        if not last_parts and not last_tools:
            forced_parts = PartIdentifier().identify(
                PartIdentifyInput(retrieved_snippets=last_snippets)
            )
            last_parts = [part.name for part in forced_parts.parts]
            last_tools = forced_parts.tools_required
            trace.append(
                {
                    "tool": "part_identifier",
                    "forced": True,
                    "parts_found": len(last_parts),
                }
            )
        likely_issue = last_snippets[0].metadata.get("guide_title") if last_snippets else None
        summary = _summary_from_context(
            appliance_type=session_appliance_type,
            likely_issue=likely_issue,
            steps=last_steps,
        )
        payload = _finalize_payload(
            turn=turn,
            sid=sid,
            summary=summary,
            clarifying_question=None,
            likely_issue=likely_issue,
            steps=last_steps,
            parts=last_parts,
            tools=last_tools or extract_required_tools(last_snippets),
            citations=_citations(last_snippets),
            snippets=_snippets_payload(last_snippets),
            trace=trace,
            safety_warning=safety_warning,
            guardrail_blocked=False,
            live_guides=[],
        )
        store.append_message(
            sid,
            role="assistant",
            content=payload.get("answer") or summary,
            structured=payload["structured"],
        )
        return payload

    fallback_query = turn.query
    if session_appliance_type and session_appliance_type.lower() not in fallback_query.lower():
        fallback_query = f"{session_appliance_type}: {turn.query}"
    fallback_turn = ChatTurnInput(
        query=fallback_query,
        appliance_category=turn.appliance_category,
        appliance_type=session_appliance_type,
        brand=turn.brand,
        model=turn.model,
        top_k=turn.top_k,
        session_id=sid,
        strategy_label=turn.strategy_label,
    )
    return run_agent_chat(
        fallback_turn,
        retrieve_fn=retrieve_fn,
        slm_generate=lambda query, chunks, live_guides: slm.generate_answer(
            query,
            chunks,
            live_guides,
            model_name=slm_model_name,
        ),
        live_ifixit_fn=live_ifixit_fn,
        provider_name="lmstudio",
        store=store,
        skip_user_append=True,
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
        answer = f"{summary or 'More detail is needed.'}\n\n{clarifying_question}"
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
