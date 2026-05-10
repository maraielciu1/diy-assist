"""Typed tool input/output schemas for the Stage 4 agent."""

from typing import Any

from pydantic import BaseModel, Field


# --- Manual_Search_Tool ---
class ManualSearchInput(BaseModel):
    query: str = Field(min_length=3)
    appliance_category: str | None = None
    appliance_type: str | None = None
    brand: str | None = None
    model: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedSnippet(BaseModel):
    text: str
    score: float
    metadata: dict[str, Any]


class ManualSearchOutput(BaseModel):
    snippets: list[RetrievedSnippet]
    retrieval_count: int


# --- Safety_Protocol_Checker ---
class SafetyCheckInput(BaseModel):
    user_query: str = Field(min_length=1)


class SafetyCheckOutput(BaseModel):
    allow_troubleshooting: bool
    reason: str | None = None
    escalation_message: str | None = None


# --- Part_Identifier ---
class PartIdentifyInput(BaseModel):
    retrieved_snippets: list[RetrievedSnippet]


class PartCandidate(BaseModel):
    name: str
    confidence: str = Field(description="low|medium|high")
    source_hint: str | None = None


class PartIdentifyOutput(BaseModel):
    parts: list[PartCandidate]


# --- Symptom_Clarifier ---
class ClarifyInput(BaseModel):
    user_query: str


class ClarifyOutput(BaseModel):
    needs_clarification: bool
    clarifying_question: str | None = None


# --- Step_By_Step_Guide_Formatter ---
class FormatGuideInput(BaseModel):
    retrieved_snippets: list[RetrievedSnippet]
    max_steps: int = Field(default=8, ge=1, le=20)


class GuideStep(BaseModel):
    number: int
    instruction: str
    guide_title: str | None = None
    step_number: int | None = None


class FormatGuideOutput(BaseModel):
    steps: list[GuideStep]
    summary_lines: list[str]


# --- Agent structured response (API surface) ---
class StructuredChatPayload(BaseModel):
    answer_summary: str | None = None
    clarifying_question: str | None = None
    likely_issue: str | None = None
    steps: list[str] = Field(default_factory=list)
    parts_list: list[str] = Field(default_factory=list)
    retrieved_guide_snippets: list[dict[str, Any]] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
