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
    _LOCATION_TERMS = ("door", "bottom", "hose", "dispenser", "pump", "tub", "seal", "filter")

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


class PartIdentifier:
    """Heuristic part candidates from tools metadata + chunk text."""

    def identify(self, inp: PartIdentifyInput) -> PartIdentifyOutput:
        parts: list[PartCandidate] = []
        seen: set[str] = set()

        for snip in inp.retrieved_snippets:
            meta = snip.metadata or {}
            raw_tools = str(meta.get("tools") or "")
            for piece in re.split(r"[,;]", raw_tools):
                name = piece.strip()
                if name and name.lower() not in seen:
                    seen.add(name.lower())
                    parts.append(
                        PartCandidate(
                            name=name,
                            confidence="medium",
                            source_hint="guide_tools_metadata",
                        )
                    )

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

        return PartIdentifyOutput(parts=parts[:12])


class StepByStepGuideFormatter:
    """Turn retrieved chunks into numbered repair steps."""

    def format(self, inp: FormatGuideInput) -> FormatGuideOutput:
        steps: list[GuideStep] = []
        summary_lines: list[str] = []
        for i, snip in enumerate(inp.retrieved_snippets[: inp.max_steps], start=1):
            meta = snip.metadata or {}
            title = str(meta.get("guide_title") or "Repair guide")
            step_no = meta.get("step_number")
            instruction = snip.text.strip()
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
