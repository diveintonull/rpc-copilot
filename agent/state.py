"""Serializable contracts shared by Agent tools and future graph nodes."""

from __future__ import annotations

from typing import Literal, TypedDict


class Evidence(TypedDict):
    """One regulation clause returned by a deterministic tool."""

    parent_id: str
    source_id: str
    version: str
    section_number: str
    text: str
    score: float | None


class ClauseRef(TypedDict):
    """The three stable fields needed to locate one exact clause."""

    source_id: str
    version: str
    section_number: str


class ClauseComparison(TypedDict):
    """Left and right evidence kept separate for later explanation."""

    left: Evidence | None
    right: Evidence | None
    dimensions: list[str]


Intent = Literal[
    "regulation_qa",
    "clause_comparison",
    "gap_analysis",
    "unsupported",
]
FinalStatus = Literal["completed", "refused", "cancelled", "failed"]


class AgentState(TypedDict, total=False):
    """Serializable values passed between the future LangGraph nodes."""

    request_id: str
    query: str
    control_text: str
    intent: Intent
    active_skill: str
    skill_text: str
    tool_calls: list[dict]
    evidence: list[dict]
    answer: str
    citations_valid: bool
    citation_failures: list[dict]
    retry_action: str
    retry_count: int
    final_status: FinalStatus
    trace: list[dict]
