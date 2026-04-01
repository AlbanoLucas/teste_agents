"""Section pipeline handlers."""

from __future__ import annotations

import logging

from app.graph.handlers.base import BaseHandler
from app.logging_utils import log_block, preview_text
from app.workflow.models import ArticleAssemblyContext, SectionDraftContext, SectionReviewContext

logger = logging.getLogger(__name__)


class SectionInitHandler(BaseHandler):
    """Initialize typed sections from the outline."""

    node_name = "section_init_node"

    def run(self, envelope):
        log_block(
            logger,
            "Etapa: section_init_node",
            fields={"output_mode": envelope.output_mode},
        )
        if envelope.output_mode != "article":
            envelope.sections = []
            envelope.current_section_index = 0
            return envelope

        if envelope.outline.payload is None:
            raise ValueError("Outline ausente para inicializar as secoes.")

        envelope.sections = self.services.section_service.initialize_from_outline(
            envelope.outline.payload,
            minimum_words=self.services.settings.article_min_words,
        )
        envelope.current_section_index = self.services.section_service.next_pending_index(
            envelope.sections,
            0,
        )
        log_block(
            logger,
            "Plano de secoes preparado",
            fields={"secoes": len(envelope.sections), "ordem": [section.heading for section in envelope.sections]},
        )
        return envelope


class SectionWriteHandler(BaseHandler):
    """Write the current section from compact context."""

    node_name = "section_write_node"

    def run(self, envelope):
        log_block(
            logger,
            "Etapa: section_write_node",
            fields={"indice_atual": envelope.current_section_index},
        )
        if envelope.output_mode != "article":
            return envelope

        active = self.services.section_service.locate_active(
            envelope.sections,
            envelope.current_section_index,
        )
        if active is None:
            envelope.current_section_index = len(envelope.sections)
            return envelope

        index, section = active
        if envelope.outline.payload is None:
            raise ValueError("Outline ausente para redigir secoes.")

        context = SectionDraftContext(
            normalized_request=envelope.normalized_request,
            article_goal=envelope.article_goal,
            audience=envelope.audience,
            tone=envelope.tone,
            knowledge_domain=envelope.knowledge_domain,
            disciplinary_lens=envelope.disciplinary_lens,
            article_profile=self.services.settings.article_profile,
            minimum_words=self.services.settings.article_min_words,
            headline=envelope.outline.payload.headline,
            editorial_angle=envelope.outline.payload.editorial_angle,
            section_index=index,
            total_sections=len(envelope.sections),
            section=section,
            approved_section_summaries=self.services.section_service.approved_section_summaries(
                envelope.sections,
                before_index=index,
            ),
            evidence_requirements=envelope.evidence_requirements,
            constraints=envelope.constraints,
            research_summary=envelope.research.summary,
            debate_summary=envelope.debate.summary,
            agent_positions=envelope.debate.positions,
            references=_select_relevant_references(section, envelope.research.references),
        )
        section.draft_md = self.services.section_writer.write(context)
        section.status = "drafted"
        envelope.sections[index] = section
        envelope.current_section_index = index
        log_block(
            logger,
            f"Secao redigida: {section.heading}",
            fields={
                "indice": index,
                "retry_atual": section.retry_count,
                "meta_palavras": section.target_words,
            },
            body=preview_text(section.draft_md, limit=360),
        )
        return envelope


class SectionReviewHandler(BaseHandler):
    """Review the current section and decide whether to retry or advance."""

    node_name = "section_review_node"

    def run(self, envelope):
        log_block(
            logger,
            "Etapa: section_review_node",
            fields={"indice_atual": envelope.current_section_index},
        )
        if envelope.output_mode != "article":
            return envelope

        active = self.services.section_service.locate_active(
            envelope.sections,
            envelope.current_section_index,
        )
        if active is None:
            envelope.current_section_index = len(envelope.sections)
            return envelope

        index, section = active
        if envelope.outline.payload is None:
            raise ValueError("Outline ausente para revisar secoes.")

        context = SectionReviewContext(
            normalized_request=envelope.normalized_request,
            article_goal=envelope.article_goal,
            article_profile=self.services.settings.article_profile,
            knowledge_domain=envelope.knowledge_domain,
            disciplinary_lens=envelope.disciplinary_lens,
            minimum_words=self.services.settings.article_min_words,
            headline=envelope.outline.payload.headline,
            editorial_angle=envelope.outline.payload.editorial_angle,
            total_sections=len(envelope.sections),
            section=section,
            research_summary=envelope.research.summary,
            debate_summary=envelope.debate.summary,
            references=_select_relevant_references(section, envelope.research.references),
        )
        review = self.services.section_reviewer.review(context)
        sections, next_index, alerts = self.services.section_service.apply_review_result(
            sections=envelope.sections,
            index=index,
            review=review,
            retry_max=envelope.section_retry_max,
            quality_alerts=envelope.quality_alerts,
        )
        envelope.sections = sections
        envelope.current_section_index = next_index
        envelope.quality_alerts = alerts
        envelope.metadata["sections_reviewed"] = envelope.metadata.get("sections_reviewed", 0) + 1
        current_section = envelope.sections[index]
        log_block(
            logger,
            f"Revisao da secao: {current_section.heading}",
            fields={
                "status": current_section.status,
                "score": review.quality_score,
                "retry_atual": current_section.retry_count,
                "retry_max": envelope.section_retry_max,
            },
            body=[
                f"Resumo: {preview_text(review.review_summary, limit=240)}",
                f"Forcas: {', '.join(review.strengths) or '(nenhuma)'}",
                f"Fragilidades: {', '.join(review.weaknesses) or '(nenhuma)'}",
                f"Exigencias de revisao: {', '.join(review.revision_requirements) or '(nenhuma)'}",
            ],
        )
        return envelope


class SectionRecoveryHandler(BaseHandler):
    """Recover only the current weak section with focused research and debate."""

    node_name = "section_research_node"

    def run(self, envelope):
        log_block(
            logger,
            "Etapa: section_research_node",
            fields={"indice_atual": envelope.current_section_index},
        )
        active = self.services.section_service.locate_active(
            envelope.sections,
            envelope.current_section_index,
        )
        if active is None:
            return envelope

        index, section = active
        envelope, section = self.services.section_recovery.prepare_recovery(
            envelope=envelope,
            section=section,
            node_name=self.node_name,
        )
        envelope.sections[index] = section
        log_block(
            logger,
            f"Recuperacao focada da secao: {section.heading}",
            fields={
                "retry_atual": section.retry_count,
                "queries": section.section_research_queries,
            },
            body=[
                f"Problema: {preview_text(section.recovery_problem_summary, limit=220)}",
                f"Resumo da pesquisa focada: {preview_text(section.section_research_summary, limit=260)}",
            ],
        )
        return envelope


class SectionDebateHandler(BaseHandler):
    """Observability-only handler after focused recovery; debate already happened in the orchestrator."""

    node_name = "section_debate_node"

    def run(self, envelope):
        log_block(
            logger,
            "Etapa: section_debate_node",
            fields={"indice_atual": envelope.current_section_index},
        )
        active = self.services.section_service.locate_active(
            envelope.sections,
            envelope.current_section_index,
        )
        if active is None:
            return envelope

        index, section = active
        envelope, section = self.services.section_recovery.run_section_debate(
            envelope=envelope,
            section=section,
        )
        envelope.sections[index] = section
        log_block(
            logger,
            f"Mini-debate da secao: {section.heading}",
            fields={"rodadas": 3},
            body=preview_text(section.section_debate_summary, limit=300),
        )
        return envelope


class ArticleAssemblyHandler(BaseHandler):
    """Assemble the final markdown artifact."""

    node_name = "article_assembly_node"

    def run(self, envelope):
        log_block(
            logger,
            "Etapa: article_assembly_node",
            fields={"output_mode": envelope.output_mode},
        )
        context = ArticleAssemblyContext(
            output_mode=envelope.output_mode,
            normalized_request=envelope.normalized_request,
            article_goal=envelope.article_goal,
            headline=(
                envelope.outline.payload.headline
                if envelope.outline.payload
                else envelope.normalized_request
            ),
            editorial_angle=(
                envelope.outline.payload.editorial_angle
                if envelope.outline.payload
                else envelope.article_goal
            ),
            sections=envelope.sections,
            references=envelope.research.references,
            evidence_confidence=envelope.research.evidence_confidence,
            evidence_gaps=envelope.research.evidence_gaps,
            follow_up_queries=envelope.research.follow_up_queries,
            research_summary=envelope.research.summary,
            quality_alerts=envelope.quality_alerts,
        )
        envelope.final_article_md = self.services.article_assembler.assemble(context)
        log_block(
            logger,
            "Documento final montado",
            fields={
                "caracteres": len(envelope.final_article_md),
                "palavras_estimadas": len(envelope.final_article_md.split()),
                "alertas_qualidade": len(envelope.quality_alerts),
            },
        )
        return envelope


def _select_relevant_references(section, references):
    if not references:
        return []
    section_terms = {
        token.lower()
        for token in [section.heading, *section.bullets]
        for token in str(token).replace(":", " ").split()
        if len(token) > 3
    }
    ranked = [
        reference
        for reference in references
        if section_terms.intersection(
            {
                *reference.title.lower().split(),
                *reference.source.lower().split(),
                *reference.snippet.lower().split(),
            }
        )
    ]
    return ranked[:8] or references[:8]
