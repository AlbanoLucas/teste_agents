"""Dependency container for the graph handlers."""

from __future__ import annotations

from dataclasses import dataclass

from app.debate.autogen_runner import DebateEngine
from app.llm.openai_client import OpenAIResponsesClient
from app.research.summarizer import ResearchSummarizer
from app.research.web_search import SearchBackend
from app.workflow.error_policy import WorkflowErrorPolicy
from app.workflow.section_recovery import SectionRecoveryService
from app.workflow.section_service import SectionStateService
from app.workflow.state_adapter import GraphStateAdapter
from app.writer.article_assembler import ArticleAssembler
from app.writer.markdown_saver import MarkdownSaver
from app.writer.outline import OutlineGenerator
from app.writer.section_reviewer import SectionReviewer
from app.writer.section_writer import SectionWriter


@dataclass(slots=True)
class WorkflowSettings:
    """Runtime settings for the graph."""

    default_output_path: str = "outputs/artigo_final.md"
    research_max_sources: int = 6
    debate_min_rounds: int = 3
    debate_max_rounds: int = 5
    section_retry_max: int = 3
    article_profile: str = "academic_rigid"
    article_min_words: int = 6000
    evidence_policy: str = "abort"


@dataclass(slots=True)
class WorkflowServices:
    """Concrete services used by graph handlers."""

    llm_client: OpenAIResponsesClient
    search_backend: SearchBackend
    research_summarizer: ResearchSummarizer
    debate_engine: DebateEngine
    outline_generator: OutlineGenerator
    section_writer: SectionWriter
    section_reviewer: SectionReviewer
    article_assembler: ArticleAssembler
    markdown_saver: MarkdownSaver
    state_adapter: GraphStateAdapter
    error_policy: WorkflowErrorPolicy
    section_service: SectionStateService
    section_recovery: SectionRecoveryService
    settings: WorkflowSettings
