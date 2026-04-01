"""Thin node composition for the LangGraph workflow."""

from __future__ import annotations

from app.graph.handlers.debate import DebateHandler
from app.graph.handlers.failure import FailureHandler
from app.graph.handlers.intake import IntakeHandler
from app.graph.handlers.research import ResearchHandler
from app.graph.handlers.save import SaveHandler
from app.graph.handlers.section_pipeline import (
    ArticleAssemblyHandler,
    SectionDebateHandler,
    SectionInitHandler,
    SectionRecoveryHandler,
    SectionReviewHandler,
    SectionWriteHandler,
)
from app.graph.handlers.synthesis import SynthesisHandler
from app.graph.services import WorkflowServices


def build_nodes(services: WorkflowServices) -> dict[str, callable]:
    """Instantiate the thin graph handlers."""

    return {
        "orchestrator_intake": IntakeHandler(services),
        "research_node": ResearchHandler(services),
        "debate_node": DebateHandler(services),
        "synthesis_node": SynthesisHandler(services),
        "failure_node": FailureHandler(services),
        "section_init_node": SectionInitHandler(services),
        "section_write_node": SectionWriteHandler(services),
        "section_review_node": SectionReviewHandler(services),
        "section_research_node": SectionRecoveryHandler(services),
        "section_debate_node": SectionDebateHandler(services),
        "article_assembly_node": ArticleAssemblyHandler(services),
        "save_file_node": SaveHandler(services),
    }
