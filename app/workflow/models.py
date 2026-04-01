"""Typed runtime models used internally by the LangGraph workflow."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from app.llm.models import (
    DebateAgentPositions,
    DebatePromptPackOverride,
    DebateTurn,
    OutlinePayload,
    ResearchNote,
    ResolvedDebatePromptPack,
    SourceReference,
    StrictModel,
)


class PromptEnvelope(StrictModel):
    """Normalized prompt payload passed to the OpenAI client."""

    instructions: str
    prompt: str
    operation_name: str
    context_budget_hint: str = "standard"
    truncation_strategy: str = "none"


class WorkflowErrorRecord(StrictModel):
    """Structured workflow error persisted in state and logs."""

    node: str
    operation: str
    category: Literal["retryable_llm", "retryable_search", "schema_error", "terminal"]
    decision: str
    message: str
    attempt: int = Field(default=1, ge=1)


class QualityAlert(StrictModel):
    """Persistent quality warning attached to a section."""

    heading: str
    summary: str
    pending: list[str] = Field(default_factory=list)


class ResearchBundle(StrictModel):
    """Research artifacts accumulated by the workflow."""

    queries: list[str] = Field(default_factory=list)
    notes: list[ResearchNote] = Field(default_factory=list)
    summary: str = ""
    references: list[SourceReference] = Field(default_factory=list)
    evidence_is_sufficient: bool = False
    evidence_confidence: float | str = 0.0
    evidence_gaps: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)


class DebateBundle(StrictModel):
    """Debate artifacts and runtime control flags."""

    prompt: str = ""
    prompt_pack: ResolvedDebatePromptPack = Field(default_factory=ResolvedDebatePromptPack)
    min_rounds: int = Field(default=3, ge=1)
    max_rounds: int = Field(default=5, ge=1)
    completed_rounds: int = Field(default=0, ge=0)
    needs_more_rounds: bool = False
    open_questions: list[str] = Field(default_factory=list)
    transcript: list[DebateTurn] = Field(default_factory=list)
    summary: str = ""
    positions: DebateAgentPositions = Field(default_factory=DebateAgentPositions)


class OutlineBundle(StrictModel):
    """Structured outline plus rendered markdown preview."""

    payload: OutlinePayload | None = None
    markdown: str = ""


class SectionState(StrictModel):
    """Single section tracked across draft, review and recovery cycles."""

    id: str
    heading: str
    purpose: str
    bullets: list[str] = Field(default_factory=list)
    kind: Literal["short_form", "standard"] = "standard"
    status: Literal[
        "pending",
        "drafted",
        "approved",
        "needs_retry",
        "accepted_with_warnings",
    ] = "pending"
    target_words: int = Field(default=0, ge=0)
    draft_md: str = ""
    review_summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    revision_requirements: list[str] = Field(default_factory=list)
    prompt_improvements: list[str] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)
    recovery_problem_summary: str = ""
    section_research_queries: list[str] = Field(default_factory=list)
    section_research_notes: list[ResearchNote] = Field(default_factory=list)
    section_research_summary: str = ""
    section_debate_summary: str = ""
    section_agent_positions: DebateAgentPositions = Field(default_factory=DebateAgentPositions)
    section_debate_prompt: str = ""
    prompt_pack_override: DebatePromptPackOverride = Field(default_factory=DebatePromptPackOverride)
    resolved_prompt_pack: ResolvedDebatePromptPack = Field(default_factory=ResolvedDebatePromptPack)


class SectionDraftContext(StrictModel):
    """Compact context delivered to the section writer."""

    normalized_request: str
    article_goal: str
    audience: str
    tone: str
    knowledge_domain: str
    disciplinary_lens: str
    article_profile: str
    minimum_words: int
    headline: str
    editorial_angle: str
    section_index: int
    total_sections: int
    section: SectionState
    approved_section_summaries: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    research_summary: str = ""
    debate_summary: str = ""
    agent_positions: DebateAgentPositions = Field(default_factory=DebateAgentPositions)
    references: list[SourceReference] = Field(default_factory=list)


class SectionReviewContext(StrictModel):
    """Compact context delivered to the section reviewer."""

    normalized_request: str
    article_goal: str
    article_profile: str
    knowledge_domain: str
    disciplinary_lens: str
    minimum_words: int
    headline: str
    editorial_angle: str
    total_sections: int
    section: SectionState
    research_summary: str = ""
    debate_summary: str = ""
    references: list[SourceReference] = Field(default_factory=list)


class ArticleAssemblyContext(StrictModel):
    """Final assembly context for article or insufficiency report."""

    output_mode: Literal["article", "insufficiency_report"] = "article"
    normalized_request: str
    article_goal: str
    headline: str
    editorial_angle: str
    sections: list[SectionState] = Field(default_factory=list)
    references: list[SourceReference] = Field(default_factory=list)
    evidence_confidence: float | str = 0.0
    evidence_gaps: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)
    research_summary: str = ""
    quality_alerts: list[QualityAlert] = Field(default_factory=list)


class WorkflowStateEnvelope(StrictModel):
    """Internal source of truth for the workflow."""

    status: Literal["running", "failed", "completed"] = "running"
    failed_node: str | None = None
    terminal_error: WorkflowErrorRecord | None = None
    user_request: str = ""
    normalized_request: str = ""
    article_goal: str = ""
    audience: str = ""
    tone: str = ""
    knowledge_domain: str = ""
    disciplinary_lens: str = ""
    evidence_requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    output_mode: Literal["article", "insufficiency_report"] = "article"
    research: ResearchBundle = Field(default_factory=ResearchBundle)
    debate: DebateBundle = Field(default_factory=DebateBundle)
    outline: OutlineBundle = Field(default_factory=OutlineBundle)
    sections: list[SectionState] = Field(default_factory=list)
    current_section_index: int = Field(default=0, ge=0)
    section_retry_max: int = Field(default=3, ge=1)
    quality_alerts: list[QualityAlert] = Field(default_factory=list)
    final_article_md: str = ""
    output_path: str = "outputs/artigo_final.md"
    metadata: dict[str, Any] = Field(default_factory=dict)
    errors: list[WorkflowErrorRecord] = Field(default_factory=list)
