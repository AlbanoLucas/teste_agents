"""CLI entry point for the multiagent article writer."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from app.debate.autogen_runner import DebateEngine
from app.graph.build_graph import build_graph
from app.graph.services import WorkflowServices, WorkflowSettings
from app.llm.openai_client import OpenAIResponsesClient
from app.logging_utils import configure_logging
from app.research.summarizer import ResearchSummarizer
from app.research.web_search import OpenAIWebSearchBackend
from app.workflow.error_policy import WorkflowErrorPolicy
from app.workflow.section_recovery import SectionRecoveryService
from app.workflow.section_service import SectionStateService
from app.workflow.state_adapter import GraphStateAdapter
from app.writer.article_assembler import ArticleAssembler
from app.writer.formatter import MarkdownFormatter
from app.writer.markdown_saver import MarkdownSaver
from app.writer.outline import OutlineGenerator
from app.writer.section_reviewer import SectionReviewer
from app.writer.section_writer import SectionWriter


@dataclass(slots=True)
class AppSettings:
    """Runtime configuration loaded from the environment."""

    api_key: str
    model: str
    output_path: str
    research_max_sources: int
    debate_min_rounds: int
    debate_max_rounds: int
    section_retry_max: int
    llm_temperature: float
    log_level: str
    log_style: str
    article_profile: str
    article_min_words: int
    evidence_policy: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera um artigo em Markdown a partir de um pedido.")
    parser.add_argument("--request", required=True, help="Solicitacao original do usuario.")
    parser.add_argument("--output-path", help="Sobrescreve o caminho do Markdown final.")
    parser.add_argument("--log-level", help="Nivel de log exibido no terminal.")
    parser.add_argument("--log-style", help="Estilo de log exibido no terminal.")
    return parser.parse_args()


def load_settings() -> AppSettings:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Defina OPENAI_API_KEY no ambiente antes de executar o fluxo.")
    return AppSettings(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        output_path=os.getenv("OUTPUT_PATH", "outputs/artigo_final.md"),
        research_max_sources=int(os.getenv("RESEARCH_MAX_SOURCES", "6")),
        debate_min_rounds=int(os.getenv("DEBATE_MIN_ROUNDS", "3")),
        debate_max_rounds=int(os.getenv("DEBATE_MAX_ROUNDS", "5")),
        section_retry_max=int(os.getenv("SECTION_RETRY_MAX", "3")),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_style=os.getenv("LOG_STYLE", "utf8_blocks"),
        article_profile=os.getenv("ARTICLE_PROFILE", "academic_rigid"),
        article_min_words=_normalize_article_min_words(os.getenv("ARTICLE_MIN_WORDS", "6000")),
        evidence_policy=os.getenv("EVIDENCE_POLICY", "abort"),
    )


def build_dependencies(settings: AppSettings) -> WorkflowServices:
    llm_client = OpenAIResponsesClient(
        model=settings.model,
        api_key=settings.api_key,
        temperature=settings.llm_temperature,
    )
    formatter = MarkdownFormatter()
    section_service = SectionStateService(formatter)
    error_policy = WorkflowErrorPolicy()
    search_backend = OpenAIWebSearchBackend(llm_client)
    research_summarizer = ResearchSummarizer(llm_client)
    debate_engine = DebateEngine(
        model=settings.model,
        api_key=settings.api_key,
        llm_client=llm_client,
    )
    return WorkflowServices(
        llm_client=llm_client,
        search_backend=search_backend,
        research_summarizer=research_summarizer,
        debate_engine=debate_engine,
        outline_generator=OutlineGenerator(llm_client),
        section_writer=SectionWriter(llm_client, formatter),
        section_reviewer=SectionReviewer(llm_client),
        article_assembler=ArticleAssembler(formatter),
        markdown_saver=MarkdownSaver(),
        state_adapter=GraphStateAdapter(),
        error_policy=error_policy,
        section_service=section_service,
        section_recovery=SectionRecoveryService(
            llm_client=llm_client,
            search_backend=search_backend,
            research_summarizer=research_summarizer,
            debate_engine=debate_engine,
            error_policy=error_policy,
            research_max_sources=settings.research_max_sources,
            article_profile=settings.article_profile,
        ),
        settings=WorkflowSettings(
            default_output_path=settings.output_path,
            research_max_sources=settings.research_max_sources,
            debate_min_rounds=settings.debate_min_rounds,
            debate_max_rounds=settings.debate_max_rounds,
            section_retry_max=settings.section_retry_max,
            article_profile=settings.article_profile,
            article_min_words=settings.article_min_words,
            evidence_policy=settings.evidence_policy,
        ),
    )


def _normalize_article_min_words(raw_value: str) -> int:
    """Clamp article length targets to a range that is ambitious but executable."""

    try:
        value = int(raw_value)
    except ValueError:
        value = 6000

    if value < 2500:
        return 2500
    if value > 12000:
        return 12000
    return value


def main() -> None:
    args = parse_args()
    settings = load_settings()
    configure_logging(args.log_level or settings.log_level, args.log_style or settings.log_style)
    deps = build_dependencies(settings)
    workflow = build_graph(deps)
    final_state = workflow.invoke(
        {
            "workflow": {
                "user_request": args.request,
                "output_path": args.output_path or settings.output_path,
                "errors": [],
                "metadata": {},
            }
        }
    )
    print(f"Artigo salvo em: {final_state['workflow']['output_path']}")


if __name__ == "__main__":
    main()
