"""Intake handler for the workflow."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from app.graph.handlers.base import BaseHandler
from app.llm.models import IntakePlan
from app.logging_utils import log_block, preview_text
from app.prompts.builders import build_intake_prompt

logger = logging.getLogger(__name__)


class IntakeHandler(BaseHandler):
    """Interpret the user request and initialize the workflow envelope."""

    node_name = "orchestrator_intake"

    def run(self, envelope):
        log_block(
            logger,
            "Etapa: orchestrator_intake",
            fields={"solicitacao": preview_text(envelope.user_request, limit=160)},
        )
        prompt_envelope = build_intake_prompt(envelope.user_request)
        plan = self.services.llm_client.generate_structured(
            envelope=prompt_envelope,
            schema_model=IntakePlan,
            temperature=0.2,
        )
        envelope.normalized_request = plan.normalized_request
        envelope.article_goal = plan.article_goal
        envelope.audience = plan.audience
        envelope.tone = plan.tone
        envelope.knowledge_domain = plan.knowledge_domain
        envelope.disciplinary_lens = plan.disciplinary_lens
        envelope.evidence_requirements = plan.evidence_requirements
        envelope.constraints = plan.constraints
        envelope.research.queries = plan.research_queries or [plan.normalized_request]
        envelope.debate.prompt = plan.debate_prompt
        envelope.debate.prompt_pack = plan.debate_prompt_pack
        envelope.debate.min_rounds = self.services.settings.debate_min_rounds
        envelope.debate.max_rounds = self.services.settings.debate_max_rounds
        envelope.section_retry_max = self.services.settings.section_retry_max
        envelope.metadata["started_at"] = datetime.now(UTC).isoformat()
        envelope.metadata["research_query_count"] = len(envelope.research.queries)
        log_block(
            logger,
            "Intake concluido",
            fields={
                "objetivo": plan.article_goal,
                "publico": plan.audience,
                "tom": plan.tone,
                "dominio": plan.knowledge_domain,
                "lente": plan.disciplinary_lens,
                "queries": envelope.research.queries,
            },
        )
        return envelope
