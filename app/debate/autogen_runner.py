"""AutoGen-backed debate adapter used by the LangGraph debate node."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from app.debate.agents import DEBATE_AGENTS
from app.debate.formatter import DebateFormatter
from app.debate.prompts import build_agent_system_message, build_phase_task
from app.llm.models import DebateTurn, ResearchNote
from app.llm.openai_client import OpenAIResponsesClient
from app.logging_utils import log_block, preview_text

PhaseRunner = Callable[[int, str], list[Any]]
logger = logging.getLogger(__name__)


class DebateEngine:
    """Encapsulate AutoGen so the graph only sees a stable contract."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        llm_client: OpenAIResponsesClient | None = None,
        phase_runner: PhaseRunner | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._formatter = DebateFormatter(llm_client=llm_client)
        self._phase_runner = phase_runner

    def run(
        self,
        topic: str,
        research_summary: str,
        research_notes: list[dict] | list[ResearchNote],
        rounds: int = 3,
        prior_transcript: list | None = None,
        start_round: int = 1,
        open_questions: list[str] | None = None,
        shared_context: str = "",
        focus_axes: list[str] | None = None,
        domain_terms: list[str] | None = None,
        agent_specializations: dict[str, str] | None = None,
    ) -> dict:
        """Run one or more debate rounds and return a structured result."""

        typed_notes = [
            note if isinstance(note, ResearchNote) else ResearchNote.model_validate(note)
            for note in research_notes
        ]
        focus_axes = list(focus_axes or [])
        domain_terms = list(domain_terms or [])
        agent_specializations = dict(agent_specializations or {})
        cumulative_transcript: list[DebateTurn] = [
            item if isinstance(item, DebateTurn) else DebateTurn.model_validate(item)
            for item in (prior_transcript or [])
        ]
        log_block(
            logger,
            "DebateEngine acionado",
            fields={
                "topico": preview_text(topic, limit=140),
                "rodadas_neste_ciclo": rounds,
                "transcript_previo": len(cumulative_transcript),
                "eixos": focus_axes,
            },
            body=[
                f"Contexto compartilhado: {preview_text(shared_context, limit=220) or '(vazio)'}",
                f"Termos prioritarios: {', '.join(domain_terms) or '(nenhum)'}",
            ],
        )

        current_round = start_round
        target_round = start_round + max(rounds, 0)
        while current_round < target_round:
            log_block(
                logger,
                "Rodada de debate",
                fields={"rodada": current_round, "fase": _phase_name(current_round)},
            )
            task = build_phase_task(
                topic=topic,
                research_summary=research_summary,
                research_notes=typed_notes,
                round_number=current_round,
                prior_transcript=[turn.model_dump() for turn in cumulative_transcript],
                open_questions=open_questions,
                shared_context=shared_context,
                focus_axes=focus_axes,
                domain_terms=domain_terms,
            )
            log_block(
                logger,
                "Prompt da rodada",
                fields={"rodada": current_round},
                body=preview_text(task, limit=520),
                level=logging.DEBUG,
            )
            messages = self._run_phase(
                round_number=current_round,
                task=task,
                shared_context=shared_context,
                focus_axes=focus_axes,
                domain_terms=domain_terms,
                agent_specializations=agent_specializations,
            )
            normalized_turns = self._formatter.normalize_messages(
                messages=messages,
                round_number=current_round,
            )
            for turn in normalized_turns:
                log_block(
                    logger,
                    f"Agente: {turn.agent}",
                    fields={
                        "rodada": turn.round,
                        "fase": turn.phase,
                        "posicionamento": preview_text(turn.stance, limit=160),
                    },
                    body=turn.message,
                )
            cumulative_transcript.extend(normalized_turns)
            current_round += 1

        completed_rounds = max(start_round - 1, current_round - 1)
        assessment = self._formatter.summarize(
            topic=topic,
            transcript=cumulative_transcript,
            research_summary=research_summary,
            research_notes=typed_notes,
            completed_rounds=completed_rounds,
        )
        log_block(
            logger,
            "Avaliacao do debate",
            fields={
                "rodadas_concluidas": completed_rounds,
                "continuar": assessment.needs_more_rounds,
                "pendencias": assessment.open_questions,
            },
            body=preview_text(assessment.summary, limit=320),
        )
        return {
            "transcript": [turn.model_dump() for turn in cumulative_transcript],
            "summary": assessment.summary,
            "positions": assessment.positions.as_dict(),
            "completed_rounds": completed_rounds,
            "needs_more_rounds": assessment.needs_more_rounds,
            "open_questions": assessment.open_questions,
        }

    def _run_phase(
        self,
        *,
        round_number: int,
        task: str,
        shared_context: str,
        focus_axes: list[str],
        domain_terms: list[str],
        agent_specializations: dict[str, str],
    ) -> list[Any]:
        if self._phase_runner is not None:
            return self._phase_runner(round_number, task)
        return self._run_phase_with_autogen(
            round_number=round_number,
            task=task,
            shared_context=shared_context,
            focus_axes=focus_axes,
            domain_terms=domain_terms,
            agent_specializations=agent_specializations,
        )

    def _run_phase_with_autogen(
        self,
        *,
        round_number: int,
        task: str,
        shared_context: str,
        focus_axes: list[str],
        domain_terms: list[str],
        agent_specializations: dict[str, str],
    ) -> list[Any]:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import MaxMessageTermination
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        async def runner() -> list[Any]:
            model_client = OpenAIChatCompletionClient(
                model=self._model,
                api_key=self._api_key,
            )
            try:
                agents = [
                    AssistantAgent(
                        name=agent.name,
                        model_client=model_client,
                        system_message=build_agent_system_message(
                            agent,
                            shared_context=shared_context,
                            focus_axes=focus_axes,
                            domain_terms=domain_terms,
                            agent_specialization=agent_specializations.get(agent.name, ""),
                        ),
                    )
                    for agent in DEBATE_AGENTS
                ]
                team = RoundRobinGroupChat(
                    participants=agents,
                    termination_condition=MaxMessageTermination(max_messages=len(DEBATE_AGENTS)),
                )
                result = await team.run(task=task)
                return list(getattr(result, "messages", []))
            finally:
                close = getattr(model_client, "close", None)
                if close is not None:
                    maybe_coro = close()
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro

        return _run_sync(runner())


def _run_sync(coro: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    if not loop.is_running():
        return loop.run_until_complete(coro)

    result: Any = None
    error: BaseException | None = None

    def wrapped() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive bridge.
            error = exc

    import threading

    thread = threading.Thread(target=wrapped, daemon=True)
    thread.start()
    thread.join()
    if error is not None:
        raise error
    return result


def _phase_name(round_number: int) -> str:
    if round_number == 1:
        return "tese_inicial"
    if round_number == 2:
        return "critica_cruzada"
    if round_number == 3:
        return "refinamento_final"
    return f"follow_up_{round_number}"
