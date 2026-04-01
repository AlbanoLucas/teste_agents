"""Structured data models shared across prompts, debate and OpenAI contracts."""

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


class ResolvedDebatePromptPack(StrictModel):
    """Fully resolved debate enrichment ready for execution."""

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


class DebatePromptPackOverride(StrictModel):
    """Partial prompt enrichment that must be merged with the global resolved pack."""

    shared_context: str | None = None
    focus_axes: list[str] | None = None
    domain_terms: list[str] | None = None
    Analitico: str | None = None
    Critico: str | None = None
    Estrategico: str | None = None


def merge_prompt_pack(
    global_pack: ResolvedDebatePromptPack,
    override: DebatePromptPackOverride | None = None,
) -> ResolvedDebatePromptPack:
    """Merge a partial override without allowing empty values to erase global context."""

    override = override or DebatePromptPackOverride()
    return ResolvedDebatePromptPack(
        shared_context=_pick_override(override.shared_context, global_pack.shared_context),
        focus_axes=_pick_list_override(override.focus_axes, global_pack.focus_axes),
        domain_terms=_pick_list_override(override.domain_terms, global_pack.domain_terms),
        Analitico=_pick_override(override.Analitico, global_pack.Analitico),
        Critico=_pick_override(override.Critico, global_pack.Critico),
        Estrategico=_pick_override(override.Estrategico, global_pack.Estrategico),
    )


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
    debate_prompt_pack: ResolvedDebatePromptPack = Field(default_factory=ResolvedDebatePromptPack)


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


class SectionReviewPayload(StrictModel):
    """Review result for a single section draft."""

    review_summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    revision_requirements: list[str] = Field(default_factory=list)
    prompt_improvements: list[str] = Field(default_factory=list)
    needs_revision: bool = False
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)


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


class SectionRecoveryPlan(StrictModel):
    """Focused recovery plan for a weak section."""

    problem_summary: str
    research_queries: list[str] = Field(default_factory=list, min_length=2, max_length=4)
    debate_prompt: str
    prompt_pack_override: DebatePromptPackOverride = Field(default_factory=DebatePromptPackOverride)


def _pick_override(candidate: str | None, fallback: str) -> str:
    if candidate is None:
        return fallback
    value = candidate.strip()
    return value if value else fallback


def _pick_list_override(candidate: list[str] | None, fallback: list[str]) -> list[str]:
    if candidate is None:
        return list(fallback)
    cleaned = [item.strip() for item in candidate if str(item).strip()]
    return cleaned if cleaned else list(fallback)
