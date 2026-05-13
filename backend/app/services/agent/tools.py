"""Stage 4 agent tools — callable implementations with typed I/O."""

from __future__ import annotations

import re
from typing import Any, Callable

from app.services.agent.schemas import (
    ClarifyInput,
    ClarifyOutput,
    FormatGuideInput,
    FormatGuideOutput,
    GuideStep,
    ManualSearchInput,
    ManualSearchOutput,
    PartCandidate,
    PartIdentifyInput,
    PartIdentifyOutput,
    RetrievedSnippet,
    SafetyCheckInput,
    SafetyCheckOutput,
)
from app.services.guardrails import evaluate_query_safety

_PART_TERMS = (
    "filter",
    "pump",
    "belt",
    "thermal fuse",
    "fuse",
    "hose",
    "valve",
    "motor",
    "igniter",
    "element",
    "thermostat",
    "capacitor",
    "door latch",
    "seal",
    "lint",
    "vent",
    "coils",
    "compressor",
)


class ManualSearchTool:
    """Semantic search over indexed repair chunks."""

    def __init__(self, search_fn: Callable[..., Any]) -> None:
        self._search_fn = search_fn

    def run(self, inp: ManualSearchInput) -> ManualSearchOutput:
        chunks = self._search_fn(
            query=inp.query,
            appliance_category=inp.appliance_category,
            appliance_type=inp.appliance_type,
            brand=inp.brand,
            model=inp.model,
            top_k=inp.top_k,
        )
        snippets = [
            RetrievedSnippet(text=c.text, score=c.score, metadata=dict(c.metadata or {}))
            for c in chunks
        ]
        return ManualSearchOutput(snippets=snippets, retrieval_count=len(snippets))


class SafetyProtocolChecker:
    """Keyword safety gate before retrieval or repair guidance."""

    def check(self, inp: SafetyCheckInput) -> SafetyCheckOutput:
        decision = evaluate_query_safety(inp.user_query)
        return SafetyCheckOutput(
            allow_troubleshooting=decision.allow,
            reason=decision.reason,
            escalation_message=decision.escalation_message,
        )


class SymptomClarifier:
    """Detect underspecified symptoms that need a clarifying question."""

    _LEAK = re.compile(r"\bleak", re.I)
    _NOISE = re.compile(r"\bnoisy\b|\bnoise\b|\brattle\b", re.I)
    _VAGUE = re.compile(r"\bproblem\b|\bissue\b|\bweird\b|\bacting up\b", re.I)
    _LOCATION_TERMS = ("door", "bottom", "hose", "dispenser", "pump", "tub", "seal", "filter")
    _CONCRETE_TERMS = (
        "leak",
        "noise",
        "noisy",
        "rattle",
        "hum",
        "humming",
        "drain",
        "spin",
        "fill",
        "click",
        "grind",
        "heat",
        "cool",
        "warm",
        "cold",
        "smell",
        "smoke",
        "spark",
        "error",
        "latch",
        "start",
        "dry",
        "wet",
    )

    def analyze(self, inp: ClarifyInput) -> ClarifyOutput:
        q = inp.user_query.strip()
        lowered = q.lower()
        # Very short generic complaints
        if len(lowered.split()) <= 5 and any(
            x in lowered for x in ("not working", "broken", "won't start", "wont start")
        ):
            return ClarifyOutput(
                needs_clarification=True,
                clarifying_question=(
                    "What exactly happens when it tries to run "
                    "(any lights, sounds, error codes, or smells)?"
                ),
            )
        if self._VAGUE.search(q) and not any(t in lowered for t in self._CONCRETE_TERMS):
            return ClarifyOutput(
                needs_clarification=True,
                clarifying_question=(
                    "What exact symptom are you seeing (leak, no drain, no heat, noise, "
                    "error code, smell, or something else)?"
                ),
            )
        if self._LEAK.search(q) and not any(t in lowered for t in self._LOCATION_TERMS):
            return ClarifyOutput(
                needs_clarification=True,
                clarifying_question=(
                    "Where do you see water (front door, underneath, "
                    "supply hoses behind the unit, or dispenser area)?"
                ),
            )
        if self._NOISE.search(q) and not any(
            t in lowered for t in ("spin", "drain", "fill", "click", "grind", "hum")
        ):
            return ClarifyOutput(
                needs_clarification=True,
                clarifying_question=(
                    "When does the noise happen (during spin, drain, fill, or idle), "
                    "and is it a grind, hum, or bang?"
                ),
            )
        return ClarifyOutput(needs_clarification=False, clarifying_question=None)


_STEP_PREFIX_RE = re.compile(r"^\s*step\s+\d+\s*:\s*", re.I)


def extract_required_tools(snippets: list[RetrievedSnippet]) -> list[str]:
    tools: list[str] = []
    seen: set[str] = set()
    for snip in snippets:
        raw_tools = str((snip.metadata or {}).get("tools") or "")
        for piece in re.split(r"[,;]", raw_tools):
            name = piece.strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                tools.append(name)
    return tools[:12]


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class PartIdentifier:
    """Heuristic part candidates from chunk text only."""

    def identify(self, inp: PartIdentifyInput) -> PartIdentifyOutput:
        parts: list[PartCandidate] = []
        seen: set[str] = set()

        for snip in inp.retrieved_snippets:
            lowered = snip.text.lower()
            for term in _PART_TERMS:
                if term in lowered:
                    label = term.title()
                    lk = label.lower()
                    if lk not in seen:
                        seen.add(lk)
                        parts.append(
                            PartCandidate(
                                name=label,
                                confidence="low",
                                source_hint="chunk_text_keyword",
                            )
                        )

        return PartIdentifyOutput(
            parts=parts[:12],
            tools_required=extract_required_tools(inp.retrieved_snippets),
        )


class StepByStepGuideFormatter:
    """Turn retrieved chunks into numbered repair steps."""

    def format(self, inp: FormatGuideInput) -> FormatGuideOutput:
        steps: list[GuideStep] = []
        summary_lines: list[str] = []

        top_guide_id = None
        if inp.retrieved_snippets:
            top_guide_id = (inp.retrieved_snippets[0].metadata or {}).get("guide_id")

        selected: list[RetrievedSnippet] = []
        seen_keys: set[tuple[str, int, int]] = set()
        for snip in inp.retrieved_snippets:
            meta = snip.metadata or {}
            if top_guide_id and meta.get("guide_id") != top_guide_id:
                continue
            step_no = _safe_int(meta.get("step_number"))
            chunk_no = _safe_int(meta.get("chunk_number"))
            key = (str(meta.get("guide_id") or ""), step_no, chunk_no)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected.append(snip)

        selected.sort(
            key=lambda snip: (
                _safe_int((snip.metadata or {}).get("step_number")),
                _safe_int((snip.metadata or {}).get("chunk_number")),
            )
        )

        for i, snip in enumerate(selected[: inp.max_steps], start=1):
            meta = snip.metadata or {}
            title = str(meta.get("guide_title") or "Repair guide")
            step_no = meta.get("step_number")
            instruction = _STEP_PREFIX_RE.sub("", snip.text.strip())
            steps.append(
                GuideStep(
                    number=i,
                    instruction=instruction,
                    guide_title=title,
                    step_number=int(step_no) if step_no is not None else None,
                )
            )
            summary_lines.append(f"{i}. {instruction[:160]}{'…' if len(instruction) > 160 else ''}")

        return FormatGuideOutput(steps=steps, summary_lines=summary_lines)
