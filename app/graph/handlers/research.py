"""Research handler for the workflow."""

from __future__ import annotations

import logging

from app.graph.handlers.base import BaseHandler
from app.logging_utils import log_block, preview_text
from app.research.parser import normalize_sources

logger = logging.getLogger(__name__)


class ResearchHandler(BaseHandler):
    """Run web research and populate the research bundle."""

    node_name = "research_node"

    def run(self, envelope):
        queries = envelope.research.queries
        log_block(
            logger,
            "Etapa: research_node",
            fields={"query_count": len(queries), "queries": queries},
        )
        results = [
            self.services.search_backend.search(
                query,
                max_results=self.services.settings.research_max_sources,
            )
            for query in queries
        ]
        references = normalize_sources(
            results,
            max_results=self.services.settings.research_max_sources,
        )
        for result in results:
            log_block(
                logger,
                "Pesquisa executada",
                fields={"query": result.query},
                body=preview_text(result.summary, limit=320),
            )

        payload = self.services.research_summarizer.summarize(
            topic=envelope.article_goal,
            knowledge_domain=envelope.knowledge_domain,
            disciplinary_lens=envelope.disciplinary_lens,
            evidence_requirements=envelope.evidence_requirements,
            results=results,
            references=references,
            minimum_words=self.services.settings.article_min_words,
            article_profile=self.services.settings.article_profile,
        )
        envelope.research.notes = payload.notes
        envelope.research.summary = payload.research_summary
        envelope.research.references = references
        envelope.research.evidence_is_sufficient = payload.evidence_is_sufficient
        envelope.research.evidence_confidence = payload.evidence_confidence
        envelope.research.evidence_gaps = payload.evidence_gaps
        envelope.research.follow_up_queries = payload.follow_up_queries
        envelope.output_mode = (
            "article"
            if payload.evidence_is_sufficient or self.services.settings.evidence_policy != "abort"
            else "insufficiency_report"
        )
        envelope.metadata["research_sources"] = len(references)
        log_block(
            logger,
            "Pesquisa consolidada",
            fields={
                "notas": len(payload.notes),
                "fontes": len(references),
                "evidencia_suficiente": payload.evidence_is_sufficient,
                "confianca": payload.evidence_confidence,
                "output_mode": envelope.output_mode,
            },
            body=[
                f"Resumo: {preview_text(payload.research_summary, limit=260)}",
                f"Lacunas: {', '.join(payload.evidence_gaps) or '(nenhuma)'}",
                f"Proximas queries: {', '.join(payload.follow_up_queries) or '(nenhuma)'}",
            ],
        )
        return envelope
