"""Structured data models shared across the workflow."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that forbids undeclared fields."""

    model_config = ConfigDict(extra="forbid")


class SourceReference(StrictModel):
    """Normalized citation collected during web research."""

    title: str
    source: str
    url: str
    snippet: str = ""


class ResearchNote(StrictModel):
    """Structured note consumed by debate and writing stages."""

    title: str
    source: str
    url: str
    summary: str
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)


class DebatePromptPack(StrictModel):
    """Topic-aware prompt enrichment created by the orchestrator for debate."""

    shared_context: str = ""
    focus_axes: list[str] = Field(default_factory=list)
    domain_terms: list[str] = Field(default_factory=list)
    Analitico: str = ""
    Critico: str = ""
    Estrategico: str = ""

    def agent_map(self) -> dict[str, str]:
        """Expose the explicit agent enrichments as the stable runtime mapping."""

        return {
            "Analitico": self.Analitico,
            "Critico": self.Critico,
            "Estrategico": self.Estrategico,
        }


class IntakePlan(StrictModel):
    """Editor-facing interpretation of the initial user request."""

    normalized_request: str
    article_goal: str
    audience: str
    tone: str
    knowledge_domain: str
    disciplinary_lens: str
    evidence_requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    research_queries: list[str] = Field(default_factory=list)
    debate_prompt: str
    debate_prompt_pack: DebatePromptPack = Field(default_factory=DebatePromptPack)


class ResearchSummaryPayload(StrictModel):
    """Research bundle passed to the debate stage."""

    research_summary: str
    notes: list[ResearchNote] = Field(default_factory=list)


class ResearchAnalysisPayload(StrictModel):
    """Research bundle plus evidence sufficiency for academic writing."""

    research_summary: str
    notes: list[ResearchNote] = Field(default_factory=list)
    evidence_is_sufficient: bool
    evidence_confidence: float = Field(ge=0.0, le=1.0)
    evidence_gaps: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)


class DebateTurn(StrictModel):
    """Single normalized debate turn produced by a specialist agent."""

    round: int = Field(ge=1)
    phase: str
    agent: str
    message: str
    stance: str = ""
    citations: list[str] = Field(default_factory=list)


class DebateAgentPositions(StrictModel):
    """Final position of each fixed debate specialist."""

    Analitico: str = ""
    Critico: str = ""
    Estrategico: str = ""

    def as_dict(self) -> dict[str, str]:
        """Expose positions as the plain mapping expected by the graph state."""

        return self.model_dump()

    @classmethod
    def from_mapping(cls, positions: dict[str, str] | None = None) -> "DebateAgentPositions":
        """Normalize a free-form mapping into the explicit fixed-agent schema."""

        return cls(**{key: value for key, value in (positions or {}).items() if key in cls.model_fields})


class DebateAssessmentPayload(StrictModel):
    """Structured synthesis of the debate transcript."""

    summary: str
    positions: DebateAgentPositions = Field(default_factory=DebateAgentPositions)
    needs_more_rounds: bool = False
    open_questions: list[str] = Field(default_factory=list)


class ArticleReviewPayload(StrictModel):
    """Structured editorial review used to trigger targeted rewrites."""

    review_summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    revision_requirements: list[str] = Field(default_factory=list)
    prompt_improvements: list[str] = Field(default_factory=list)
    needs_revision: bool = False
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SectionUnit(StrictModel):
    """Single outline section tracked through drafting, review and recovery."""

    id: str
    heading: str
    purpose: str
    bullets: list[str] = Field(default_factory=list)
    kind: str = "standard"
    status: str = "pending"
    target_words: int = Field(default=0, ge=0)
    draft_md: str = ""
    review_summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    revision_requirements: list[str] = Field(default_factory=list)
    prompt_improvements: list[str] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)
    section_research_queries: list[str] = Field(default_factory=list)
    section_research_notes: list[ResearchNote] = Field(default_factory=list)
    section_research_summary: str = ""
    section_debate_summary: str = ""
    section_agent_positions: DebateAgentPositions = Field(default_factory=DebateAgentPositions)
    section_debate_prompt: str = ""
    section_prompt_pack: DebatePromptPack = Field(default_factory=DebatePromptPack)


class SectionReviewPayload(StrictModel):
    """Review result for a single section draft."""

    review_summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    revision_requirements: list[str] = Field(default_factory=list)
    prompt_improvements: list[str] = Field(default_factory=list)
    needs_revision: bool = False
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SectionRecoveryPlan(StrictModel):
    """Focused recovery plan for a weak section."""

    problem_summary: str
    research_queries: list[str] = Field(default_factory=list, min_length=2, max_length=4)
    debate_prompt: str
    prompt_pack: DebatePromptPack = Field(default_factory=DebatePromptPack)


class OutlineSection(StrictModel):
    """Single section in the final editorial outline."""

    heading: str
    purpose: str
    bullets: list[str] = Field(default_factory=list)


class OutlinePayload(StrictModel):
    """Outline returned by the synthesis stage."""

    headline: str
    editorial_angle: str
    sections: list[OutlineSection] = Field(default_factory=list)
