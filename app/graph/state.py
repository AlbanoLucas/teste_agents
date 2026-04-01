"""Shared workflow state."""

from __future__ import annotations

from typing import Any, TypedDict


class ProjectState(TypedDict, total=False):
    """Global state persisted across the LangGraph workflow."""

    user_request: str
    normalized_request: str
    article_goal: str
    audience: str
    tone: str
    knowledge_domain: str
    disciplinary_lens: str
    evidence_requirements: list[str]
    constraints: list[str]

    research_queries: list[str]
    research_notes: list[dict[str, Any]]
    research_summary: str
    source_references: list[dict[str, Any]]
    evidence_is_sufficient: bool
    evidence_confidence: float | str
    evidence_gaps: list[str]
    follow_up_queries: list[str]
    output_mode: str

    debate_prompt: str
    debate_shared_context: str
    debate_focus_axes: list[str]
    debate_domain_terms: list[str]
    debate_agent_specializations: dict[str, str]
    debate_min_rounds: int
    debate_max_rounds: int
    debate_completed_rounds: int
    debate_needs_more_rounds: bool
    debate_open_questions: list[str]
    debate_transcript: list[dict[str, Any]]
    debate_summary: str
    agent_positions: dict[str, str]

    final_outline_payload: dict[str, Any]
    final_outline: str
    section_units: list[dict[str, Any]]
    current_section_index: int
    section_retry_max: int
    quality_alerts: list[dict[str, Any]]
    final_article_md: str

    output_path: str
    metadata: dict[str, Any]
    errors: list[str]
