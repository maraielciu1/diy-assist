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
            "description": "Identify likely appliance parts mentioned in retrieved snippets.",
            "parameters": _object_schema(
                {
                    "retrieved_snippets": {
                        "type": "array",
                        "description": "Retrieved snippets from manual_search. Leave empty to use the last search result.",
                        "items": {"type": "object"},
                    }
                },
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "step_by_step_formatter",
            "description": "Format retrieved guide snippets into ordered repair steps.",
            "parameters": _object_schema(
                {
                    "retrieved_snippets": {
                        "type": "array",
                        "description": "Retrieved snippets from manual_search. Leave empty to use the last search result.",
                        "items": {"type": "object"},
                    },
                    "max_steps": {"type": "integer", "minimum": 1, "maximum": 20},
                },
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_answer",
            "description": "Return the final structured answer for the UI. Call this only after safety and retrieval/clarification are complete.",
            "parameters": _object_schema(
                {
                    "summary": {"type": ["string", "null"]},
                    "likely_issue": {"type": ["string", "null"]},
                    "clarifying_question": {"type": ["string", "null"]},
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "parts_to_inspect": {"type": "array", "items": {"type": "string"}},
                    "tools_required": {"type": "array", "items": {"type": "string"}},
                    "safety_warning": {"type": ["string", "null"]},
                    "citations": {"type": "array", "items": {"type": "object"}},
                },
                ["summary", "steps", "parts_to_inspect", "tools_required", "citations"],
            ),
        },
    },
]
