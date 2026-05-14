"""OpenAI-compatible tool schemas exposed to the LM Studio ReAct loop."""

from __future__ import annotations

from typing import Any


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


AGENT_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "safety_protocol_checker",
            "description": "Check whether the user issue is safe for DIY troubleshooting.",
            "parameters": _object_schema(
                {
                    "user_query": {
                        "type": "string",
                        "description": "The user's current troubleshooting request.",
                    }
                },
                ["user_query"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "symptom_clarifier",
            "description": "Decide whether the symptom is underspecified and produce one clarifying question.",
            "parameters": _object_schema(
                {
                    "user_query": {
                        "type": "string",
                        "description": "The user's current troubleshooting request.",
                    }
                },
                ["user_query"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manual_search",
            "description": "Search indexed repair-guide chunks for the current appliance symptom.",
            "parameters": _object_schema(
                {
                    "query": {"type": "string"},
                    "appliance_type": {
                        "type": ["string", "null"],
                        "description": "washer, dryer, dishwasher, refrigerator, oven, microwave, or null.",
                    },
                    "brand": {"type": ["string", "null"]},
                    "model": {"type": ["string", "null"]},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                ["query"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "part_identifier",
            "description": "Identify likely appliance parts from the last manual_search result. Pass no arguments; the server uses the most recent retrieval.",
            "parameters": _object_schema({}),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "step_by_step_formatter",
            "description": "Format the last manual_search result into ordered repair steps. The server uses the most recent retrieval; only optionally cap the number of steps.",
            "parameters": _object_schema(
                {
                    "max_steps": {"type": "integer", "minimum": 1, "maximum": 20},
                },
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_answer",
            "description": (
                "Return the final structured answer for the UI. Call this exactly once at the end. "
                "Only author the natural-language fields; steps, parts, tools, and citations are taken "
                "server-side from prior tool calls and retrieved snippets — do not pass them here."
            ),
            "parameters": _object_schema(
                {
                    "summary": {"type": "string"},
                    "likely_issue": {"type": ["string", "null"]},
                    "clarifying_question": {"type": ["string", "null"]},
                    "safety_warning": {"type": ["string", "null"]},
                },
                ["summary"],
            ),
        },
    },
]
