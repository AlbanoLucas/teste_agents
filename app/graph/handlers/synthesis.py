"""Synthesis handler for the workflow."""

from __future__ import annotations

import logging

from app.graph.handlers.base import BaseHandler
from app.logging_utils import log_block, preview_text

logger = logging.getLogger(__name__)


class SynthesisHandler(BaseHandler):
    """Combine research and debate into a structured outline."""

    node_name = "synthesis_node"

    def run(self, envelope):
        log_block(
            logger,
            "Etapa: synthesis_node",
            fields={"perfil_artigo": self.services.settings.article_profile},
        )
        payload, markdown = self.services.outline_generator.build(
            normalized_request=envelope.normalized_request,
            article_goal=envelope.article_goal,
            audience=envelope.audience,
            tone=envelope.tone,
            knowledge_domain=envelope.knowledge_domain,
            disciplinary_lens=envelope.disciplinary_lens,
            evidence_requirements=envelope.evidence_requirements,
            constraints=envelope.constraints,
            research_summary=envelope.research.summary,
            debate_summary=envelope.debate.summary,
            agent_positions=envelope.debate.positions.as_dict(),
            references=envelope.research.references,
            article_profile=self.services.settings.article_profile,
            minimum_words=self.services.settings.article_min_words,
        )
        envelope.outline.payload = payload
        envelope.outline.markdown = markdown
        log_block(
            logger,
            "Outline final gerado",
            fields={"secoes": len(payload.sections)},
            body=preview_text(markdown, limit=420),
        )
        return envelope
