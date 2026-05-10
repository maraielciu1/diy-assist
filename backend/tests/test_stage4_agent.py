"""Integration-style tests for Stage 4 agent chat orchestration."""

from dataclasses import dataclass

from app.services.agent.orchestrator import ChatTurnInput, run_agent_chat
from app.services.chat_store import ChatStore


@dataclass
class _Chunk:
    text: str
    score: float
    metadata: dict


def test_run_agent_chat_returns_structured_and_session(tmp_path) -> None:
    db = ChatStore(str(tmp_path / "chat.sqlite"))

    def retrieve_fn(**kwargs):
        return [
            _Chunk(
                text="Step 1: Clean lint screen.",
                score=0.95,
                metadata={
                    "guide_title": "Dryer Lint",
                    "source_url": "https://example.com",
                    "step_number": 1,
                    "tools": "Vacuum",
                },
            )
        ]

    out = run_agent_chat(
        ChatTurnInput(
            query="my dryer takes two cycles to dry towels",
            appliance_category="Appliance",
            appliance_type="dryer",
            brand=None,
            model=None,
            top_k=3,
            session_id=None,
            strategy_label="naive",
        ),
        retrieve_fn=retrieve_fn,
        slm_generate=lambda q, chunks, live=None: "Mock SLM answer.",
        live_ifixit_fn=lambda query, appliance_category=None: [],
        provider_name="ollama",
        store=db,
    )
    assert out["answer"] == "Mock SLM answer."
    assert out["agent_mode"] is True
    assert "session_id" in out
    assert out["structured"]["parts_list"]
    assert out["structured"]["steps"]
    assert any(t["tool"] == "Manual_Search_Tool" for t in out["structured"]["tool_trace"])


def test_run_agent_chat_escalates_hazard(tmp_path) -> None:
    db = ChatStore(str(tmp_path / "chat2.sqlite"))

    def boom(**kwargs):
        raise AssertionError("should not retrieve")

    out = run_agent_chat(
        ChatTurnInput(
            query="I smell gas near my dryer what should I do",
            appliance_category=None,
            appliance_type=None,
            brand=None,
            model=None,
            top_k=3,
            session_id=None,
            strategy_label="naive",
        ),
        retrieve_fn=boom,
        slm_generate=lambda *a, **k: "no",
        live_ifixit_fn=lambda **k: [],
        provider_name="ollama",
        store=db,
    )
    assert out["guardrail_blocked"] is True
    assert out["answer"] is None
