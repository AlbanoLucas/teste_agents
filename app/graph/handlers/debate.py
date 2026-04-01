"""Debate handler for the workflow."""

from __future__ import annotations

import logging

from app.graph.handlers.base import BaseHandler
from app.llm.models import DebateTurn
from app.logging_utils import log_block, preview_text

logger = logging.getLogger(__name__)


class DebateHandler(BaseHandler):
    """Run the debate engine for the global article debate."""

    node_name = "debate_node"

    def run(self, envelope):
        completed = envelope.debate.completed_rounds
        start_round = completed + 1 if completed else 1
        rounds_to_run = envelope.debate.min_rounds if completed == 0 else 1
        log_block(
            logger,
            "Etapa: debate_node",
            fields={
                "rodada_inicial": start_round,
                "rodadas_neste_ciclo": rounds_to_run,
                "pendencias": envelope.debate.open_questions,
                "eixos": envelope.debate.prompt_pack.focus_axes,
            },
            body=[
                f"Contexto do tema: {preview_text(envelope.debate.prompt_pack.shared_context, limit=220) or '(vazio)'}",
                f"Termos-chave: {', '.join(envelope.debate.prompt_pack.domain_terms) or '(nenhum)'}",
            ],
        )
        result = self.services.debate_engine.run(
            topic=envelope.debate.prompt or envelope.normalized_request,
            research_summary=envelope.research.summary,
            research_notes=[note.model_dump() for note in envelope.research.notes],
            rounds=rounds_to_run,
            prior_transcript=[turn.model_dump() for turn in envelope.debate.transcript],
            start_round=start_round,
            open_questions=envelope.debate.open_questions,
            shared_context=envelope.debate.prompt_pack.shared_context,
            focus_axes=envelope.debate.prompt_pack.focus_axes,
            domain_terms=envelope.debate.prompt_pack.domain_terms,
            agent_specializations=envelope.debate.prompt_pack.agent_map(),
        )
        capped = result["needs_more_rounds"] and result["completed_rounds"] >= envelope.debate.max_rounds
        envelope.debate.transcript = [
            DebateTurn.model_validate(item) for item in result["transcript"]
        ]
        envelope.debate.summary = str(result["summary"])
        if capped:
            envelope.debate.summary += " O debate atingiu o limite maximo de rodadas configurado."
        envelope.debate.positions = envelope.debate.positions.from_mapping(result["positions"])
        envelope.debate.completed_rounds = int(result["completed_rounds"])
        envelope.debate.needs_more_rounds = bool(result["needs_more_rounds"]) and not capped
        envelope.debate.open_questions = [] if capped else list(result["open_questions"])
        envelope.metadata["debate_rounds"] = envelope.debate.completed_rounds
        log_block(
            logger,
            "Debate concluido",
            fields={
                "rodadas": envelope.debate.completed_rounds,
                "continuar": envelope.debate.needs_more_rounds,
                "pendencias": envelope.debate.open_questions,
            },
            body=f"Resumo: {preview_text(envelope.debate.summary, limit=320)}",
        )
        return envelope
