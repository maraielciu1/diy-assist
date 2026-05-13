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
        provider_name="lmstudio",
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
        provider_name="lmstudio",
        store=db,
    )
    assert out["guardrail_blocked"] is True
    assert out["answer"] is None


def test_run_agent_chat_general_question_does_not_retrieve(tmp_path) -> None:
    db = ChatStore(str(tmp_path / "chat3.sqlite"))

    def boom(**kwargs):
        raise AssertionError("general help should not retrieve")

    out = run_agent_chat(
        ChatTurnInput(
            query="what can you help me with?",
            appliance_category="Appliance",
            appliance_type=None,
            brand=None,
            model=None,
            top_k=5,
            session_id=None,
            strategy_label="naive",
        ),
        retrieve_fn=boom,
        slm_generate=lambda *a, **k: "no",
        live_ifixit_fn=lambda **k: [{"guide_title": "should not appear"}],
        provider_name="lmstudio",
        store=db,
    )

    assert out["intent"] == "general_help"
    assert out["retrieval_count"] == 0
    assert out["citations"] == []
    assert out["live_ifixit_guides"] == []
    assert "appliance" in out["answer"].lower()
    assert any(t["tool"] == "Intent_Router" for t in out["structured"]["tool_trace"])


def test_run_agent_chat_weak_retrieval_asks_for_details(tmp_path, monkeypatch) -> None:
    db = ChatStore(str(tmp_path / "chat4.sqlite"))

    from app.services.agent import orchestrator

    monkeypatch.setattr(orchestrator.settings, "retrieval_min_score", 0.55)

    def retrieve_fn(**kwargs):
        return [
            _Chunk(
                text="Unrelated fridge step.",
                score=0.21,
                metadata={
                    "guide_title": "Unrelated Guide",
                    "source_url": "https://example.com/unrelated",
                    "step_number": 1,
                },
            )
        ]

    out = run_agent_chat(
        ChatTurnInput(
            query="my washer has a weird intermittent problem",
            appliance_category="Appliance",
            appliance_type="washer",
            brand=None,
            model=None,
            top_k=5,
            session_id=None,
            strategy_label="naive",
        ),
        retrieve_fn=retrieve_fn,
        slm_generate=lambda *a, **k: "should not call model",
        live_ifixit_fn=lambda **k: [{"guide_title": "should not appear"}],
        provider_name="lmstudio",
        store=db,
    )

    assert out["uncertainty"] is True
    assert out["retrieval_count"] == 0
    assert out["citations"] == []
    assert out["live_ifixit_guides"] == []
    assert out["structured"]["steps"] == []
    assert out["structured"]["clarifying_question"]
