"""Workflow nodes for the LangGraph orchestrator."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from app.debate.autogen_runner import DebateEngine
from app.graph.state import ProjectState
from app.llm.models import (
    DebateAgentPositions,
    DebatePromptPack,
    IntakePlan,
    OutlinePayload,
    SectionRecoveryPlan,
    SectionUnit,
    SourceReference,
)
from app.llm.openai_client import OpenAIResponsesClient
from app.logging_utils import log_block, preview_text
from app.research.parser import normalize_sources
from app.research.summarizer import ResearchSummarizer
from app.research.web_search import SearchBackend
from app.writer.article_writer import MarkdownArticleWriter
from app.writer.markdown_saver import MarkdownSaver
from app.writer.outline import OutlineGenerator

logger = logging.getLogger(__name__)

_KEYWORDS_MARKERS = ("palavras-chave", "palavras chave", "keywords")
_ABSTRACT_MARKERS = ("resumo", "abstract")


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
class WorkflowDependencies:
    """Concrete services used by each workflow node."""

    llm_client: OpenAIResponsesClient
    search_backend: SearchBackend
    research_summarizer: ResearchSummarizer
    debate_engine: DebateEngine
    outline_generator: OutlineGenerator
    article_writer: MarkdownArticleWriter
    markdown_saver: MarkdownSaver
    settings: WorkflowSettings


def build_nodes(deps: WorkflowDependencies) -> dict[str, callable]:
    """Return the concrete node callables bound to the injected dependencies."""

    def orchestrator_intake(state: ProjectState) -> ProjectState:
        log_block(
            logger,
            "Etapa: orchestrator_intake",
            fields={"solicitacao": preview_text(state["user_request"], limit=160)},
        )
        plan = deps.llm_client.generate_structured(
            instructions=(
                "Interprete a solicitacao do usuario e prepare um plano editorial inicial "
                "para uma revisao academica ou tecnico-cientifica profissional, longa e bem embasada. "
                "Identifique o dominio do conhecimento e a lente disciplinar adequados ao tema. "
                "Detecte o assunto central e monte um pacote de enriquecimento para o debate entre agentes, "
                "incluindo contexto compartilhado do tema, eixos prioritarios de analise, termos-chave e "
                "instrucoes especificas para Analitico, Critico e Estrategico. "
                "Nao assuma que todo tema pertence a tecnologia, IA ou ciencias exatas. "
                "Evite framing escolar ou simplista. "
                "Retorne JSON com request normalizada, objetivo, publico, tom, "
                "dominio, lente disciplinar, exigencias de evidencia, restricoes, "
                "queries de pesquisa, prompt do debate e pacote de enriquecimento do debate."
            ),
            prompt=state["user_request"],
            schema_model=IntakePlan,
            temperature=0.2,
        )
        log_block(
            logger,
            "Intake concluido",
            fields={
                "objetivo": plan.article_goal,
                "publico": plan.audience,
                "tom": plan.tone,
                "dominio": plan.knowledge_domain,
                "lente": plan.disciplinary_lens,
                "queries": plan.research_queries or [plan.normalized_request],
                "perfil_artigo": deps.settings.article_profile,
                "meta_palavras": deps.settings.article_min_words,
            },
            body=[
                f"Exigencias de evidencia: {', '.join(plan.evidence_requirements) or '(nenhuma)'}",
                f"Eixos do debate: {', '.join(plan.debate_prompt_pack.focus_axes) or '(nenhum)'}",
                f"Termos-chave do debate: {', '.join(plan.debate_prompt_pack.domain_terms) or '(nenhum)'}",
                f"Prompt do debate: {preview_text(plan.debate_prompt, limit=280)}",
            ],
        )
        return {
            "normalized_request": plan.normalized_request,
            "article_goal": plan.article_goal,
            "audience": plan.audience,
            "tone": plan.tone,
            "knowledge_domain": plan.knowledge_domain,
            "disciplinary_lens": plan.disciplinary_lens,
            "evidence_requirements": plan.evidence_requirements,
            "constraints": plan.constraints,
            "research_queries": plan.research_queries or [plan.normalized_request],
            "debate_prompt": plan.debate_prompt,
            "debate_shared_context": plan.debate_prompt_pack.shared_context,
            "debate_focus_axes": plan.debate_prompt_pack.focus_axes,
            "debate_domain_terms": plan.debate_prompt_pack.domain_terms,
            "debate_agent_specializations": plan.debate_prompt_pack.agent_map(),
            "debate_min_rounds": state.get("debate_min_rounds", deps.settings.debate_min_rounds),
            "debate_max_rounds": state.get("debate_max_rounds", deps.settings.debate_max_rounds),
            "debate_completed_rounds": 0,
            "debate_needs_more_rounds": False,
            "debate_open_questions": [],
            "final_outline_payload": {},
            "final_outline": "",
            "section_units": [],
            "current_section_index": 0,
            "section_retry_max": state.get("section_retry_max", deps.settings.section_retry_max),
            "quality_alerts": [],
            "output_mode": "article",
            "metadata": {
                **state.get("metadata", {}),
                "started_at": datetime.now(UTC).isoformat(),
                "research_query_count": len(plan.research_queries or [plan.normalized_request]),
            },
            "errors": list(state.get("errors", [])),
        }

    def research_node(state: ProjectState) -> ProjectState:
        queries = state.get("research_queries", [])
        log_block(
            logger,
            "Etapa: research_node",
            fields={"query_count": len(queries), "queries": queries},
        )
        results = [
            deps.search_backend.search(query, max_results=deps.settings.research_max_sources)
            for query in queries
        ]
        references = normalize_sources(results, max_results=deps.settings.research_max_sources)
        for result in results:
            log_block(
                logger,
                "Pesquisa executada",
                fields={"query": result.query},
                body=preview_text(result.summary, limit=320),
            )
        payload = deps.research_summarizer.summarize(
            topic=state["article_goal"],
            knowledge_domain=state["knowledge_domain"],
            disciplinary_lens=state["disciplinary_lens"],
            evidence_requirements=state.get("evidence_requirements", []),
            results=results,
            references=references,
            minimum_words=deps.settings.article_min_words,
            article_profile=deps.settings.article_profile,
        )
        output_mode = (
            "article"
            if payload.evidence_is_sufficient or deps.settings.evidence_policy != "abort"
            else "insufficiency_report"
        )
        log_block(
            logger,
            "Pesquisa consolidada",
            fields={
                "notas": len(payload.notes),
                "fontes": len(references),
                "evidencia_suficiente": payload.evidence_is_sufficient,
                "confianca": payload.evidence_confidence,
                "output_mode": output_mode,
            },
            body=[
                f"Resumo: {preview_text(payload.research_summary, limit=260)}",
                f"Lacunas: {', '.join(payload.evidence_gaps) or '(nenhuma)'}",
                f"Proximas queries: {', '.join(payload.follow_up_queries) or '(nenhuma)'}",
            ],
        )
        return {
            "research_notes": [note.model_dump() for note in payload.notes],
            "research_summary": payload.research_summary,
            "source_references": [reference.model_dump() for reference in references],
            "evidence_is_sufficient": payload.evidence_is_sufficient,
            "evidence_confidence": payload.evidence_confidence,
            "evidence_gaps": payload.evidence_gaps,
            "follow_up_queries": payload.follow_up_queries,
            "output_mode": output_mode,
            "metadata": {
                **state.get("metadata", {}),
                "research_sources": len(references),
            },
        }

    def debate_node(state: ProjectState) -> ProjectState:
        completed = state.get("debate_completed_rounds", 0)
        start_round = completed + 1 if completed else 1
        rounds_to_run = (
            state.get("debate_min_rounds", deps.settings.debate_min_rounds)
            if completed == 0
            else 1
        )
        log_block(
            logger,
            "Etapa: debate_node",
            fields={
                "rodada_inicial": start_round,
                "rodadas_neste_ciclo": rounds_to_run,
                "pendencias": state.get("debate_open_questions", []),
                "eixos": state.get("debate_focus_axes", []),
            },
            body=[
                f"Contexto do tema: {preview_text(state.get('debate_shared_context', ''), limit=220) or '(vazio)'}",
                f"Termos-chave: {', '.join(state.get('debate_domain_terms', [])) or '(nenhum)'}",
            ],
        )
        result = deps.debate_engine.run(
            topic=state.get("debate_prompt", state["normalized_request"]),
            research_summary=state["research_summary"],
            research_notes=state.get("research_notes", []),
            rounds=rounds_to_run,
            prior_transcript=state.get("debate_transcript"),
            start_round=start_round,
            open_questions=state.get("debate_open_questions", []),
            shared_context=state.get("debate_shared_context", ""),
            focus_axes=state.get("debate_focus_axes", []),
            domain_terms=state.get("debate_domain_terms", []),
            agent_specializations=state.get("debate_agent_specializations", {}),
        )
        capped = (
            result["needs_more_rounds"]
            and result["completed_rounds"] >= state.get("debate_max_rounds", deps.settings.debate_max_rounds)
        )
        open_questions = [] if capped else list(result["open_questions"])
        summary = str(result["summary"])
        if capped:
            summary += " O debate atingiu o limite maximo de rodadas configurado."
        log_block(
            logger,
            "Debate concluido",
            fields={
                "rodadas": result["completed_rounds"],
                "continuar": bool(result["needs_more_rounds"]) and not capped,
                "pendencias": open_questions,
            },
            body=f"Resumo: {preview_text(summary, limit=320)}",
        )
        return {
            "debate_transcript": list(result["transcript"]),
            "debate_summary": summary,
            "agent_positions": dict(result["positions"]),
            "debate_completed_rounds": int(result["completed_rounds"]),
            "debate_needs_more_rounds": bool(result["needs_more_rounds"]) and not capped,
            "debate_open_questions": open_questions,
            "metadata": {
                **state.get("metadata", {}),
                "debate_rounds": int(result["completed_rounds"]),
            },
        }

    def synthesis_node(state: ProjectState) -> ProjectState:
        log_block(
            logger,
            "Etapa: synthesis_node",
            fields={"perfil_artigo": deps.settings.article_profile},
        )
        references = _references_from_state(state)
        outline_payload, outline_markdown = deps.outline_generator.build(
            normalized_request=state["normalized_request"],
            article_goal=state["article_goal"],
            audience=state["audience"],
            tone=state["tone"],
            knowledge_domain=state["knowledge_domain"],
            disciplinary_lens=state["disciplinary_lens"],
            evidence_requirements=state.get("evidence_requirements", []),
            constraints=state.get("constraints", []),
            research_summary=state["research_summary"],
            debate_summary=state["debate_summary"],
            agent_positions=state.get("agent_positions", {}),
            references=references,
            article_profile=deps.settings.article_profile,
            minimum_words=deps.settings.article_min_words,
        )
        log_block(
            logger,
            "Outline final gerado",
            fields={"secoes": len(outline_payload.sections)},
            body=preview_text(outline_markdown, limit=420),
        )
        return {
            "final_outline_payload": outline_payload.model_dump(),
            "final_outline": outline_markdown,
        }

    def section_init_node(state: ProjectState) -> ProjectState:
        log_block(
            logger,
            "Etapa: section_init_node",
            fields={"output_mode": state.get("output_mode", "article")},
        )
        if state.get("output_mode", "article") != "article":
            return {
                "section_units": [],
                "current_section_index": 0,
                "quality_alerts": list(state.get("quality_alerts", [])),
            }

        outline_payload = _outline_from_state(state)
        sections = _build_section_units(outline_payload, deps.settings.article_min_words)
        log_block(
            logger,
            "Plano de secoes preparado",
            fields={"secoes": len(sections), "ordem": [section.heading for section in sections]},
        )
        return {
            "section_units": _dump_sections(sections),
            "current_section_index": _next_pending_index(sections, 0),
            "section_retry_max": state.get("section_retry_max", deps.settings.section_retry_max),
            "quality_alerts": list(state.get("quality_alerts", [])),
        }

    def section_write_node(state: ProjectState) -> ProjectState:
        log_block(
            logger,
            "Etapa: section_write_node",
            fields={"indice_atual": state.get("current_section_index", 0)},
        )
        if state.get("output_mode", "article") != "article":
            return {}

        sections = _load_sections(state)
        active = _locate_active_section(sections, state.get("current_section_index", 0))
        if active is None:
            return {
                "section_units": _dump_sections(sections),
                "current_section_index": len(sections),
            }

        index, section = active
        outline_payload = _outline_from_state(state)
        references = _references_from_state(state)
        approved_before = [
            item
            for item in sections[:index]
            if item.status in {"approved", "accepted_with_warnings"}
        ]
        section_text = deps.article_writer.write_section(
            normalized_request=state["normalized_request"],
            article_goal=state["article_goal"],
            audience=state["audience"],
            tone=state["tone"],
            knowledge_domain=state["knowledge_domain"],
            disciplinary_lens=state["disciplinary_lens"],
            evidence_requirements=state.get("evidence_requirements", []),
            constraints=state.get("constraints", []),
            article_profile=deps.settings.article_profile,
            minimum_words=deps.settings.article_min_words,
            headline=outline_payload.headline,
            editorial_angle=outline_payload.editorial_angle,
            section=section,
            section_index=index,
            total_sections=len(sections),
            prior_sections=approved_before,
            research_summary=state["research_summary"],
            debate_summary=state.get("debate_summary", ""),
            agent_positions=state.get("agent_positions", {}),
            references=references,
        )
        section.draft_md = section_text
        section.status = "drafted"
        sections[index] = section
        log_block(
            logger,
            f"Secao redigida: {section.heading}",
            fields={
                "indice": index,
                "retry_atual": section.retry_count,
                "meta_palavras": section.target_words,
            },
            body=preview_text(section_text, limit=360),
        )
        return {
            "section_units": _dump_sections(sections),
            "current_section_index": index,
        }

    def section_review_node(state: ProjectState) -> ProjectState:
        log_block(
            logger,
            "Etapa: section_review_node",
            fields={"indice_atual": state.get("current_section_index", 0)},
        )
        if state.get("output_mode", "article") != "article":
            return {}

        sections = _load_sections(state)
        active = _locate_active_section(sections, state.get("current_section_index", 0))
        if active is None:
            return {
                "section_units": _dump_sections(sections),
                "current_section_index": len(sections),
            }

        index, section = active
        outline_payload = _outline_from_state(state)
        references = _references_from_state(state)
        review = deps.article_writer.review_section(
            normalized_request=state["normalized_request"],
            article_goal=state["article_goal"],
            article_profile=deps.settings.article_profile,
            knowledge_domain=state["knowledge_domain"],
            disciplinary_lens=state["disciplinary_lens"],
            minimum_words=deps.settings.article_min_words,
            headline=outline_payload.headline,
            editorial_angle=outline_payload.editorial_angle,
            section=section,
            total_sections=len(sections),
            research_summary=state["research_summary"],
            debate_summary=state.get("debate_summary", ""),
            references=references,
        )
        section.review_summary = review.review_summary
        section.strengths = review.strengths
        section.weaknesses = review.weaknesses
        section.revision_requirements = review.revision_requirements
        section.prompt_improvements = review.prompt_improvements

        quality_alerts = list(state.get("quality_alerts", []))
        next_index = index
        retry_budget = state.get("section_retry_max", deps.settings.section_retry_max)
        if not review.needs_revision:
            section.status = "approved"
            next_index = _next_pending_index(sections, index + 1)
        elif section.retry_count < retry_budget:
            section.retry_count += 1
            section.status = "needs_retry"
        else:
            section.status = "accepted_with_warnings"
            next_index = _next_pending_index(sections, index + 1)
            quality_alerts = _upsert_quality_alert(
                quality_alerts,
                heading=section.heading,
                summary=review.review_summary,
                pending=review.revision_requirements or review.weaknesses,
            )

        sections[index] = section
        log_block(
            logger,
            f"Revisao da secao: {section.heading}",
            fields={
                "status": section.status,
                "score": review.quality_score,
                "retry_atual": section.retry_count,
                "retry_max": retry_budget,
            },
            body=[
                f"Resumo: {preview_text(review.review_summary, limit=240)}",
                f"Forcas: {', '.join(review.strengths) or '(nenhuma)'}",
                f"Fragilidades: {', '.join(review.weaknesses) or '(nenhuma)'}",
                f"Exigencias de revisao: {', '.join(review.revision_requirements) or '(nenhuma)'}",
            ],
        )
        return {
            "section_units": _dump_sections(sections),
            "current_section_index": next_index,
            "quality_alerts": quality_alerts,
            "metadata": {
                **state.get("metadata", {}),
                "sections_reviewed": state.get("metadata", {}).get("sections_reviewed", 0) + 1,
            },
        }

    def section_research_node(state: ProjectState) -> ProjectState:
        log_block(
            logger,
            "Etapa: section_research_node",
            fields={"indice_atual": state.get("current_section_index", 0)},
        )
        sections = _load_sections(state)
        active = _locate_active_section(sections, state.get("current_section_index", 0))
        if active is None:
            return {"section_units": _dump_sections(sections)}

        index, section = active
        try:
            plan = deps.llm_client.generate_structured(
                instructions=(
                    "Voce cria um plano de recuperacao para uma secao fraca de um artigo academico. "
                    "Retorne JSON valido com `problem_summary`, `research_queries`, `debate_prompt` e `prompt_pack`. "
                    "As queries devem ser focadas, concretas e recuperar lacunas reais da secao."
                ),
                prompt=_build_section_recovery_prompt(state, section),
                schema_model=SectionRecoveryPlan,
                temperature=0.2,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback for live runs.
            log_block(
                logger,
                "Fallback do plano de recuperacao da secao",
                fields={"motivo": exc.__class__.__name__, "secao": section.heading},
                body=str(exc),
                level=logging.WARNING,
            )
            plan = _fallback_recovery_plan(state, section)

        max_results = max(2, min(4, deps.settings.research_max_sources))
        results = [
            deps.search_backend.search(query, max_results=max_results)
            for query in plan.research_queries
        ]
        references = normalize_sources(results, max_results=deps.settings.research_max_sources)
        summary_payload = deps.research_summarizer.summarize(
            topic=f"{state['normalized_request']} - secao {section.heading}",
            knowledge_domain=state["knowledge_domain"],
            disciplinary_lens=state["disciplinary_lens"],
            evidence_requirements=state.get("evidence_requirements", []),
            results=results,
            references=references,
            minimum_words=max(section.target_words, 1200),
            article_profile=deps.settings.article_profile,
        )
        section.section_research_queries = plan.research_queries
        section.section_research_notes = summary_payload.notes
        section.section_research_summary = summary_payload.research_summary
        section.section_debate_prompt = plan.debate_prompt
        section.section_prompt_pack = plan.prompt_pack
        sections[index] = section
        log_block(
            logger,
            f"Recuperacao focada da secao: {section.heading}",
            fields={
                "retry_atual": section.retry_count,
                "queries": plan.research_queries,
                "fontes_focadas": len(references),
            },
            body=[
                f"Problema: {preview_text(plan.problem_summary, limit=220)}",
                f"Resumo da pesquisa focada: {preview_text(summary_payload.research_summary, limit=260)}",
            ],
        )
        return {
            "section_units": _dump_sections(sections),
            "current_section_index": index,
        }

    def section_debate_node(state: ProjectState) -> ProjectState:
        log_block(
            logger,
            "Etapa: section_debate_node",
            fields={"indice_atual": state.get("current_section_index", 0)},
        )
        sections = _load_sections(state)
        active = _locate_active_section(sections, state.get("current_section_index", 0))
        if active is None:
            return {"section_units": _dump_sections(sections)}

        index, section = active
        prompt_pack = section.section_prompt_pack
        result = deps.debate_engine.run(
            topic=section.section_debate_prompt
            or f"Refine a secao '{section.heading}' do artigo sobre {state['normalized_request']}.",
            research_summary=section.section_research_summary or state["research_summary"],
            research_notes=[
                note.model_dump() if hasattr(note, "model_dump") else note
                for note in (section.section_research_notes or state.get("research_notes", []))
            ],
            rounds=3,
            prior_transcript=None,
            start_round=1,
            open_questions=section.revision_requirements or section.weaknesses,
            shared_context=prompt_pack.shared_context or state.get("debate_shared_context", ""),
            focus_axes=prompt_pack.focus_axes or state.get("debate_focus_axes", []),
            domain_terms=prompt_pack.domain_terms or state.get("debate_domain_terms", []),
            agent_specializations=prompt_pack.agent_map() or state.get("debate_agent_specializations", {}),
        )
        section.section_debate_summary = str(result["summary"])
        section.section_agent_positions = DebateAgentPositions.from_mapping(result["positions"])
        sections[index] = section
        log_block(
            logger,
            f"Mini-debate da secao: {section.heading}",
            fields={"rodadas": result["completed_rounds"]},
            body=preview_text(section.section_debate_summary, limit=300),
        )
        return {
            "section_units": _dump_sections(sections),
            "current_section_index": index,
        }

    def article_assembly_node(state: ProjectState) -> ProjectState:
        output_mode = state.get("output_mode", "article")
        log_block(
            logger,
            "Etapa: article_assembly_node",
            fields={"output_mode": output_mode},
        )
        references = _references_from_state(state)
        sections = _load_sections(state)
        outline_payload = _outline_from_state(state, allow_missing=True)
        article = deps.article_writer.assemble_article(
            output_mode=output_mode,
            normalized_request=state["normalized_request"],
            article_goal=state["article_goal"],
            headline=outline_payload.headline if outline_payload else state["normalized_request"],
            editorial_angle=outline_payload.editorial_angle if outline_payload else state["article_goal"],
            sections=sections,
            references=references,
            evidence_confidence=state.get("evidence_confidence", ""),
            evidence_gaps=state.get("evidence_gaps", []),
            follow_up_queries=state.get("follow_up_queries", []),
            research_summary=state.get("research_summary", ""),
            quality_alerts=list(state.get("quality_alerts", [])),
        )
        log_block(
            logger,
            "Documento final montado",
            fields={
                "caracteres": len(article),
                "palavras_estimadas": len(article.split()),
                "alertas_qualidade": len(state.get("quality_alerts", [])),
            },
        )
        return {"final_article_md": article}

    def save_file_node(state: ProjectState) -> ProjectState:
        output_path = state.get("output_path", deps.settings.default_output_path)
        log_block(
            logger,
            "Etapa: save_file_node",
            fields={"output_path": output_path},
        )
        path = deps.markdown_saver.save(
            markdown=state["final_article_md"],
            output_path=output_path,
        )
        log_block(
            logger,
            "Arquivo salvo",
            fields={"output_path": str(path)},
        )
        return {
            "output_path": str(path),
            "metadata": {
                **state.get("metadata", {}),
                "finished_at": datetime.now(UTC).isoformat(),
            },
        }

    return {
        "orchestrator_intake": orchestrator_intake,
        "research_node": research_node,
        "debate_node": debate_node,
        "synthesis_node": synthesis_node,
        "section_init_node": section_init_node,
        "section_write_node": section_write_node,
        "section_review_node": section_review_node,
        "section_research_node": section_research_node,
        "section_debate_node": section_debate_node,
        "article_assembly_node": article_assembly_node,
        "save_file_node": save_file_node,
    }


def _outline_from_state(
    state: ProjectState,
    *,
    allow_missing: bool = False,
) -> OutlinePayload | None:
    payload = state.get("final_outline_payload", {})
    if not payload:
        if allow_missing:
            return None
        raise ValueError("final_outline_payload ausente no estado.")
    return OutlinePayload.model_validate(payload)


def _references_from_state(state: ProjectState) -> list[SourceReference]:
    return [SourceReference.model_validate(item) for item in state.get("source_references", [])]


def _load_sections(state: ProjectState) -> list[SectionUnit]:
    return [SectionUnit.model_validate(item) for item in state.get("section_units", [])]


def _dump_sections(sections: list[SectionUnit]) -> list[dict]:
    return [section.model_dump() for section in sections]


def _locate_active_section(
    sections: list[SectionUnit],
    current_index: int,
) -> tuple[int, SectionUnit] | None:
    if current_index < len(sections) and sections[current_index].status not in {
        "approved",
        "accepted_with_warnings",
    }:
        return current_index, sections[current_index]

    next_index = _next_pending_index(sections, current_index)
    if next_index >= len(sections):
        return None
    return next_index, sections[next_index]


def _next_pending_index(sections: list[SectionUnit], start: int) -> int:
    for index in range(max(0, start), len(sections)):
        if sections[index].status not in {"approved", "accepted_with_warnings"}:
            return index
    return len(sections)


def _build_section_units(outline_payload: OutlinePayload, minimum_words: int) -> list[SectionUnit]:
    kinds = [_classify_section_kind(item.heading) for item in outline_payload.sections]
    short_targets = [
        _short_target_for_heading(item.heading) if kind == "short_form" else 0
        for item, kind in zip(outline_payload.sections, kinds, strict=False)
    ]
    standard_count = sum(1 for kind in kinds if kind == "standard")
    remaining_budget = max(minimum_words - sum(short_targets), 0)
    standard_target = max(450, remaining_budget // max(standard_count, 1)) if standard_count else 0

    units: list[SectionUnit] = []
    seen_ids: set[str] = set()
    for index, (section, kind) in enumerate(zip(outline_payload.sections, kinds, strict=False), start=1):
        section_id = _unique_section_id(_slugify(section.heading), seen_ids, index)
        target_words = _short_target_for_heading(section.heading) if kind == "short_form" else standard_target
        units.append(
            SectionUnit(
                id=section_id,
                heading=section.heading,
                purpose=section.purpose,
                bullets=section.bullets,
                kind=kind,
                status="pending",
                target_words=target_words,
            )
        )
    return units


def _classify_section_kind(heading: str) -> str:
    normalized = heading.strip().lower()
    if any(marker in normalized for marker in _KEYWORDS_MARKERS + _ABSTRACT_MARKERS):
        return "short_form"
    return "standard"


def _short_target_for_heading(heading: str) -> int:
    normalized = heading.strip().lower()
    if any(marker in normalized for marker in _KEYWORDS_MARKERS):
        return 40
    return 220


def _unique_section_id(base: str, seen_ids: set[str], index: int) -> str:
    candidate = base or f"secao-{index}"
    suffix = 1
    unique = candidate
    while unique in seen_ids:
        suffix += 1
        unique = f"{candidate}-{suffix}"
    seen_ids.add(unique)
    return unique


def _slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def _build_section_recovery_prompt(state: ProjectState, section: SectionUnit) -> str:
    return (
        f"Tema do artigo: {state['normalized_request']}\n"
        f"Objetivo do artigo: {state['article_goal']}\n"
        f"Dominio: {state['knowledge_domain']}\n"
        f"Lente disciplinar: {state['disciplinary_lens']}\n"
        f"Secao problematica: {section.heading}\n"
        f"Tipo da secao: {section.kind}\n"
        f"Finalidade: {section.purpose}\n"
        "Pontos-chave esperados:\n"
        + "\n".join(f"- {bullet}" for bullet in section.bullets)
        + "\n\n"
        f"Resumo global da pesquisa:\n{state.get('research_summary', '')}\n\n"
        f"Resumo global do debate:\n{state.get('debate_summary', '')}\n\n"
        f"Resumo da ultima revisao da secao:\n{section.review_summary}\n\n"
        "Fragilidades da secao:\n"
        + "\n".join(f"- {item}" for item in section.weaknesses)
        + "\n\n"
        "Requisitos de revisao:\n"
        + "\n".join(f"- {item}" for item in section.revision_requirements)
        + "\n\n"
        "Melhorias sugeridas para o prompt:\n"
        + "\n".join(f"- {item}" for item in section.prompt_improvements)
        + "\n\n"
        f"Rascunho atual da secao:\n{section.draft_md}"
    )


def _fallback_recovery_plan(state: ProjectState, section: SectionUnit) -> SectionRecoveryPlan:
    topic = state["normalized_request"]
    heading = section.heading
    problem_summary = section.review_summary or "A secao precisa de evidencia e enquadramento mais preciso."
    queries = [
        f"{topic} {heading} revisao sistematica",
        f"{topic} {heading} evidencias recentes",
        f"{topic} {heading} controversias e limitacoes",
    ]
    return SectionRecoveryPlan(
        problem_summary=problem_summary,
        research_queries=queries[:3],
        debate_prompt=(
            f"Refine a secao '{heading}' do artigo sobre '{topic}', cobrindo as lacunas "
            "identificadas na revisao e propondo a melhor formulacao editorial."
        ),
        prompt_pack=DebatePromptPack(
            shared_context=f"Recuperacao focada da secao {heading} no contexto do tema {topic}.",
            focus_axes=section.revision_requirements[:3] or section.weaknesses[:3],
            domain_terms=section.bullets[:3],
            Analitico="Reestruture a secao com logica mais rigorosa e encadeamento claro.",
            Critico="Aponte extrapolacoes, lacunas de evidencia e simplificacoes indevidas.",
            Estrategico="Conecte a secao ao argumento global e a implicacoes mais amplas do tema.",
        ),
    )


def _upsert_quality_alert(
    quality_alerts: list[dict[str, object]],
    *,
    heading: str,
    summary: str,
    pending: list[str],
) -> list[dict[str, object]]:
    updated: list[dict[str, object]] = []
    replaced = False
    for alert in quality_alerts:
        if alert.get("heading") == heading:
            updated.append({"heading": heading, "summary": summary, "pending": pending})
            replaced = True
        else:
            updated.append(alert)
    if not replaced:
        updated.append({"heading": heading, "summary": summary, "pending": pending})
    return updated
