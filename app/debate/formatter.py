"""Transcript normalization and debate synthesis helpers."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from app.debate.agents import DEBATE_AGENTS
from app.debate.prompts import phase_label
from app.llm.models import DebateAgentPositions, DebateAssessmentPayload, DebateTurn, ResearchNote
from app.llm.openai_client import OpenAIResponsesClient
from app.logging_utils import log_block, preview_text

logger = logging.getLogger(__name__)


class DebateFormatter:
    """Normalize AutoGen messages into workflow-friendly output."""

    def __init__(self, llm_client: OpenAIResponsesClient | None = None) -> None:
        self._llm_client = llm_client

    def normalize_messages(
        self,
        *,
        messages: list[Any],
        round_number: int,
    ) -> list[DebateTurn]:
        """Convert AutoGen chat messages into typed debate turns."""

        allowed_agents = {item.name for item in DEBATE_AGENTS}
        phase = phase_label(round_number)
        turns: list[DebateTurn] = []

        for message in messages:
            author = _message_author(message)
            content = _message_content(message)
            if author not in allowed_agents or not content:
                continue
            turns.append(
                DebateTurn(
                    round=round_number,
                    phase=phase,
                    agent=author,
                    message=content,
                    stance=content.split(".")[0].strip(),
                    citations=[],
                )
            )
        return turns

    def summarize(
        self,
        *,
        topic: str,
        transcript: list[DebateTurn],
        research_summary: str,
        research_notes: list[ResearchNote],
        completed_rounds: int,
    ) -> DebateAssessmentPayload:
        """Summarize the cumulative transcript and decide if more rounds are needed."""

        if not transcript:
            return DebateAssessmentPayload(
                summary="O debate nao produziu falas aproveitaveis.",
                positions=DebateAgentPositions.from_mapping(
                    {agent.name: "" for agent in DEBATE_AGENTS}
                ),
                needs_more_rounds=completed_rounds < 3,
                open_questions=["Nao houve falas suficientes no debate."],
            )

        positions = _extract_positions(transcript)
        if self._llm_client is None:
            return self._heuristic_assessment(
                transcript=transcript,
                positions=positions,
                completed_rounds=completed_rounds,
            )

        prompt = (
            f"Tema: {topic}\n"
            f"Rodadas concluidas: {completed_rounds}\n\n"
            f"Resumo da pesquisa:\n{research_summary}\n\n"
            f"Notas de pesquisa:\n"
            + "\n".join(
                f"- {note.title} | {note.source} | {note.summary}" for note in research_notes
            )
            + "\n\nTranscript:\n"
            + "\n".join(
                f"- Rodada {turn.round} [{turn.phase}] {turn.agent}: {turn.message}"
                for turn in transcript
            )
        )
        try:
            payload = self._llm_client.generate_structured(
                instructions=(
                    "Resuma debates editoriais multiagentes. "
                    "Retorne JSON com `summary`, `positions`, `needs_more_rounds` e `open_questions`."
                ),
                prompt=prompt,
                schema_model=DebateAssessmentPayload,
                temperature=0.2,
            )
        except Exception as exc:
            log_block(
                logger,
                "Fallback da avaliacao do debate",
                fields={
                    "motivo": exc.__class__.__name__,
                    "estrategia": "heuristica_local",
                },
                body=preview_text(str(exc), limit=320),
                level=logging.WARNING,
            )
            return self._heuristic_assessment(
                transcript=transcript,
                positions=positions,
                completed_rounds=completed_rounds,
            )
        if not any(payload.positions.as_dict().values()):
            payload.positions = DebateAgentPositions.from_mapping(positions)
        return payload

    def _heuristic_assessment(
        self,
        *,
        transcript: list[DebateTurn],
        positions: dict[str, str],
        completed_rounds: int,
    ) -> DebateAssessmentPayload:
        summary = " ".join(turn.message for turn in transcript[-3:])[:600]
        open_questions: list[str] = []
        if completed_rounds < 3:
            open_questions.append("O debate ainda nao completou as tres rodadas obrigatorias.")
        per_phase = defaultdict(int)
        for turn in transcript:
            per_phase[turn.phase] += 1
        if per_phase.get("critica_cruzada", 0) == 0 and completed_rounds >= 2:
            open_questions.append("As teses nao passaram por critica cruzada suficiente.")
        return DebateAssessmentPayload(
            summary=summary or "Debate concluido sem resumo estruturado.",
            positions=DebateAgentPositions.from_mapping(positions),
            needs_more_rounds=bool(open_questions),
            open_questions=open_questions,
        )


def _extract_positions(transcript: list[DebateTurn]) -> dict[str, str]:
    positions = {agent.name: "" for agent in DEBATE_AGENTS}
    for turn in transcript:
        positions[turn.agent] = turn.message
    return positions


def _message_author(message: Any) -> str:
    return str(
        getattr(message, "source", None)
        or getattr(message, "name", None)
        or (message.get("source") if isinstance(message, dict) else "")
        or (message.get("name") if isinstance(message, dict) else "")
    )


def _message_content(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(message, dict):
        raw = message.get("content")
        if isinstance(raw, str):
            return raw.strip()
    return ""
