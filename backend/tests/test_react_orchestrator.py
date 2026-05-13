from dataclasses import dataclass

from app.services.agent.orchestrator import ChatTurnInput
from app.services.agent.react_orchestrator import run_react_agent_chat
from app.services.chat_store import ChatStore


@dataclass
class _Chunk:
    text: str
    score: float
    metadata: dict


class _ScriptedSLM:
    def __init__(self) -> None:
        self.calls = 0

    def chat_with_tools(self, messages, tools, tool_choice="auto"):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": "safe-1",
                        "name": "safety_protocol_checker",
                        "arguments": {"user_query": "my washer leaks from the front door"},
                    }
                ],
            }
        if self.calls == 2:
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": "search-1",
                        "name": "manual_search",
                        "arguments": {
                            "query": "washer leaks from front door",
                            "appliance_type": "washer",
                            "top_k": 3,
                        },
                    }
                ],
            }
        if self.calls == 3:
            return {
                "content": None,
                "tool_calls": [
                    {"id": "parts-1", "name": "part_identifier", "arguments": {}},
                    {"id": "steps-1", "name": "step_by_step_formatter", "arguments": {}},
                ],
            }
        return {
            "content": None,
            "tool_calls": [
                {
                    "id": "final-1",
                    "name": "finalize_answer",
                    "arguments": {
                        "summary": "Front-door leaks usually point to the door gasket or overloading.",
                        "likely_issue": "Washer Leaking From Front Door",
                        "steps": ["Unplug the washer and inspect the door gasket."],
                        "parts_to_inspect": ["Seal"],
                        "tools_required": ["Towel"],
                        "citations": [],
                    },
                }
            ],
        }

    def generate_answer(self, *args, **kwargs):
        return "fallback"


def test_react_orchestrator_dispatches_scripted_tool_calls(tmp_path) -> None:
    db = ChatStore(str(tmp_path / "chat.sqlite"))

    def retrieve_fn(**kwargs):
        assert kwargs["appliance_type"] == "washer"
        return [
            _Chunk(
                text="Inspect the rubber door seal for tears.",
                score=0.92,
                metadata={
                    "guide_id": "washer-door",
                    "guide_title": "Washer Leaking From Front Door",
                    "appliance_type": "washer",
                    "step_number": 1,
                    "chunk_number": 1,
                    "tools": "Towel",
                    "source_url": "https://www.ifixit.com/Guide/example",
                },
            )
        ]

    out = run_react_agent_chat(
        ChatTurnInput(
            query="my washer leaks from the front door",
            appliance_category="Appliance",
            appliance_type="washer",
            brand=None,
            model=None,
            top_k=3,
            session_id=None,
            strategy_label="naive",
        ),
        retrieve_fn=retrieve_fn,
        slm=_ScriptedSLM(),
        live_ifixit_fn=lambda **kwargs: [],
        store=db,
    )

    assert out["agent_mode"] == "react"
    assert out["structured"]["likely_issue"] == "Washer Leaking From Front Door"
    assert out["structured"]["parts_list"] == ["Seal"]
    assert out["structured"]["tools_required"] == ["Towel"]
    assert out["structured"]["tool_trace"][0]["tool"] == "safety_protocol_checker"
