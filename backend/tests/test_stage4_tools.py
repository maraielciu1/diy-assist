"""Unit tests for Stage 4 agent tools."""

from app.services.agent.schemas import (
    ClarifyInput,
    FormatGuideInput,
    ManualSearchInput,
    PartIdentifyInput,
    RetrievedSnippet,
    SafetyCheckInput,
)
from app.services.agent.tools import (
    ManualSearchTool,
    PartIdentifier,
    SafetyProtocolChecker,
    StepByStepGuideFormatter,
    SymptomClarifier,
)


def test_safety_checker_blocks_gas() -> None:
    out = SafetyProtocolChecker().check(SafetyCheckInput(user_query="I smell gas near the dryer"))
    assert out.allow_troubleshooting is False
    assert out.reason


def test_symptom_clarifier_leaking() -> None:
    out = SymptomClarifier().analyze(ClarifyInput(user_query="my washer is leaking"))
    assert out.needs_clarification is True
    assert out.clarifying_question


def test_part_identifier_from_metadata_and_text() -> None:
    snippets = [
        RetrievedSnippet(
            text="Remove debris from the drain pump filter.",
            score=0.9,
            metadata={"tools": "Towel, Phillips screwdriver"},
        )
    ]
    out = PartIdentifier().identify(PartIdentifyInput(retrieved_snippets=snippets))
    names = {p.name.lower() for p in out.parts}
    assert "towel" in names or "phillips screwdriver" in names


def test_step_formatter_orders_chunks() -> None:
    snippets = [
        RetrievedSnippet(
            text="Step 1: Unplug.",
            score=0.9,
            metadata={"guide_title": "Demo", "step_number": 1},
        ),
        RetrievedSnippet(
            text="Step 2: Inspect hose.",
            score=0.85,
            metadata={"guide_title": "Demo", "step_number": 2},
        ),
    ]
    out = StepByStepGuideFormatter().format(FormatGuideInput(retrieved_snippets=snippets))
    assert len(out.steps) == 2
    assert "Unplug" in out.steps[0].instruction


def test_manual_search_tool_delegates() -> None:
    calls: list[dict] = []

    def fake_search(**kwargs):
        calls.append(kwargs)
        return []

    tool = ManualSearchTool(fake_search)
    tool.run(
        ManualSearchInput(
            query="test query here",
            appliance_category="Appliance",
            appliance_type="washer",
            top_k=2,
        )
    )
    assert calls and calls[0]["query"] == "test query here"
