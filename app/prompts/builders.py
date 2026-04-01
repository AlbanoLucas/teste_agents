"""Prompt builders that separate prompt engineering from execution."""

from __future__ import annotations

from app.llm.models import DebateTurn, ResearchNote, SourceReference
from app.prompts.fragments import (
    join_prompt_blocks,
    render_agent_positions_block,
    render_approved_sections_block,
    render_feedback_block,
    render_header_block,
    render_list_block,
    render_references_block,
)
from app.workflow.models import PromptEnvelope, SectionDraftContext, SectionReviewContext


def build_intake_prompt(user_request: str) -> PromptEnvelope:
    return PromptEnvelope(
        operation_name="intake_plan",
        context_budget_hint="compact",
        truncation_strategy="none",
        instructions=(
            "Interprete a solicitacao do usuario e prepare um plano editorial inicial "
            "para uma revisao academica ou tecnico-cientifica profissional, longa e bem embasada. "
            "Identifique dominio, lente disciplinar, exigencias de evidencia, restricoes, queries de pesquisa, "
            "prompt de debate e pacote completo de enriquecimento dos agentes."
        ),
        prompt=user_request,
    )


def build_outline_prompt(
    *,
    normalized_request: str,
    article_goal: str,
    audience: str,
    tone: str,
    knowledge_domain: str,
    disciplinary_lens: str,
    evidence_requirements: list[str],
    constraints: list[str],
    research_summary: str,
    debate_summary: str,
    agent_positions: dict[str, str],
    references: list[SourceReference],
    article_profile: str,
    minimum_words: int,
) -> PromptEnvelope:
    return PromptEnvelope(
        operation_name="outline_generation",
        context_budget_hint="medium",
        truncation_strategy="none",
        instructions=(
            "Voce define o outline final de um artigo academico-profissional em Markdown. "
            "Retorne JSON com `headline`, `editorial_angle` e `sections`, adaptando a estrutura ao dominio "
            "sem perder rigor, e incluindo secoes equivalentes a resumo, palavras-chave, introducao, "
            "desenvolvimento analitico, contrapontos, limitacoes e conclusao."
        ),
        prompt=join_prompt_blocks(
            render_header_block(
                "Contexto editorial",
                "\n".join(
                    [
                        f"Solicitacao normalizada: {normalized_request}",
                        f"Objetivo do artigo: {article_goal}",
                        f"Publico: {audience}",
                        f"Tom: {tone}",
                        f"Dominio do conhecimento: {knowledge_domain}",
                        f"Lente disciplinar: {disciplinary_lens}",
                        f"Perfil editorial: {article_profile}",
                        f"Meta minima de extensao: {minimum_words} palavras",
                        f"Restricoes editoriais: {', '.join(constraints[:10]) or 'Nenhuma'}",
                    ]
                ),
            ),
            render_list_block("Exigencias de evidencia e rigor", evidence_requirements, limit=10),
            render_header_block("Resumo da pesquisa", research_summary),
            render_header_block("Resumo do debate", debate_summary),
            render_agent_positions_block("Posicoes finais dos agentes", agent_positions),
            render_references_block("Fontes consideradas", references, limit=8),
        ),
    )


def build_section_draft_prompt(context: SectionDraftContext) -> PromptEnvelope:
    return PromptEnvelope(
        operation_name=f"section_draft:{context.section.id}",
        context_budget_hint="focused",
        truncation_strategy="approved_section_summaries_only",
        instructions=(
            "Voce esta escrevendo apenas uma secao de um artigo academico-profissional em Markdown. "
            "Nao reescreva secoes anteriores e nao inclua uma secao de referencias. "
            "A secao deve comecar exatamente com o heading correspondente, refletir o debate e a pesquisa "
            "e usar atribuicao inline de fontes sem inventar fatos."
        ),
        prompt=join_prompt_blocks(
            render_header_block(
                "Contexto editorial da secao",
                "\n".join(
                    [
                        f"Solicitacao normalizada: {context.normalized_request}",
                        f"Objetivo do artigo: {context.article_goal}",
                        f"Publico: {context.audience}",
                        f"Tom: {context.tone}",
                        f"Dominio: {context.knowledge_domain}",
                        f"Lente disciplinar: {context.disciplinary_lens}",
                        f"Perfil editorial: {context.article_profile}",
                        f"Meta minima do artigo: {context.minimum_words} palavras",
                        f"Titulo do artigo: {context.headline}",
                        f"Angulo editorial: {context.editorial_angle}",
                        f"Secao atual: {context.section.heading}",
                        f"Ordem da secao: {context.section_index + 1} de {context.total_sections}",
                        f"Tipo da secao: {context.section.kind}",
                        f"Meta da secao: {context.section.target_words} palavras",
                    ]
                ),
            ),
            render_list_block(
                "Escopo e intencao da secao",
                [f"Finalidade: {context.section.purpose}", *[f"Ponto-chave: {bullet}" for bullet in context.section.bullets]],
            ),
            render_list_block("Restricoes editoriais", context.constraints, limit=10),
            render_list_block("Exigencias de evidencia", context.evidence_requirements, limit=10),
            render_approved_sections_block(context.approved_section_summaries, limit=3),
            render_header_block("Resumo global da pesquisa", context.research_summary),
            render_header_block("Resumo global do debate", context.debate_summary),
            render_agent_positions_block("Posicoes globais dos agentes", context.agent_positions),
            render_header_block(
                "Pesquisa focada da secao",
                context.section.section_research_summary or "(nao houve recuperacao focada)",
            ),
            render_header_block(
                "Debate focado da secao",
                context.section.section_debate_summary or "(nao houve mini-debate focado)",
            ),
            render_agent_positions_block("Posicoes focadas dos agentes", context.section.section_agent_positions),
            render_references_block("Fontes relevantes", context.references, limit=8),
            render_feedback_block(context.section),
        ),
    )


def build_section_review_prompt(context: SectionReviewContext) -> PromptEnvelope:
    rubric = (
        "Rubrica short_form: verificar precisao terminologica, concisao e aderencia ao papel da secao."
        if context.section.kind == "short_form"
        else "Rubrica standard: verificar profundidade, rigor, aderencia ao escopo, uso de evidencias, "
        "conexao com o argumento central e maturidade academica."
    )
    return PromptEnvelope(
        operation_name=f"section_review:{context.section.id}",
        context_budget_hint="focused",
        truncation_strategy="none",
        instructions=(
            "Voce revisa apenas uma secao de um artigo academico em Markdown. "
            "Retorne JSON com `review_summary`, `strengths`, `weaknesses`, `revision_requirements`, "
            "`prompt_improvements`, `needs_revision` e `quality_score`."
        ),
        prompt=join_prompt_blocks(
            render_header_block(
                "Contexto da revisao",
                "\n".join(
                    [
                        f"Solicitacao normalizada: {context.normalized_request}",
                        f"Objetivo do artigo: {context.article_goal}",
                        f"Perfil editorial: {context.article_profile}",
                        f"Dominio: {context.knowledge_domain}",
                        f"Lente disciplinar: {context.disciplinary_lens}",
                        f"Meta minima do artigo: {context.minimum_words} palavras",
                        f"Titulo do artigo: {context.headline}",
                        f"Angulo editorial: {context.editorial_angle}",
                        f"Secao avaliada: {context.section.heading}",
                        f"Tipo da secao: {context.section.kind}",
                        f"Meta da secao: {context.section.target_words} palavras",
                        f"Quantidade total de secoes: {context.total_sections}",
                    ]
                ),
            ),
            render_list_block("Finalidade da secao", [context.section.purpose, *context.section.bullets]),
            render_header_block("Rubrica de avaliacao", rubric),
            render_header_block("Resumo global da pesquisa", context.research_summary),
            render_header_block("Resumo global do debate", context.debate_summary),
            render_references_block("Fontes relevantes", context.references, limit=8),
            render_header_block("Texto da secao", context.section.draft_md),
        ),
    )


def build_section_recovery_prompt(
    *,
    normalized_request: str,
    article_goal: str,
    knowledge_domain: str,
    disciplinary_lens: str,
    section_heading: str,
    section_kind: str,
    section_purpose: str,
    section_bullets: list[str],
    research_summary: str,
    debate_summary: str,
    review_summary: str,
    weaknesses: list[str],
    revision_requirements: list[str],
    prompt_improvements: list[str],
    draft_md: str,
) -> PromptEnvelope:
    return PromptEnvelope(
        operation_name=f"section_recovery_plan:{section_heading}",
        context_budget_hint="focused",
        truncation_strategy="none",
        instructions=(
            "Voce cria um plano de recuperacao para uma secao fraca de um artigo academico. "
            "Retorne JSON com `problem_summary`, `research_queries`, `debate_prompt` e `prompt_pack_override`. "
            "As queries devem ser focadas e as instrucoes dos agentes devem apenas complementar o contexto global."
        ),
        prompt=join_prompt_blocks(
            render_header_block(
                "Contexto do problema",
                "\n".join(
                    [
                        f"Tema do artigo: {normalized_request}",
                        f"Objetivo do artigo: {article_goal}",
                        f"Dominio: {knowledge_domain}",
                        f"Lente disciplinar: {disciplinary_lens}",
                        f"Secao problematica: {section_heading}",
                        f"Tipo da secao: {section_kind}",
                        f"Finalidade: {section_purpose}",
                    ]
                ),
            ),
            render_list_block("Pontos-chave esperados", section_bullets),
            render_header_block("Resumo global da pesquisa", research_summary),
            render_header_block("Resumo global do debate", debate_summary),
            render_header_block("Resumo da ultima revisao", review_summary),
            render_list_block("Fragilidades", weaknesses),
            render_list_block("Requisitos de revisao", revision_requirements),
            render_list_block("Melhorias de prompt sugeridas", prompt_improvements),
            render_header_block("Rascunho atual", draft_md),
        ),
    )


def build_debate_assessment_prompt(
    *,
    topic: str,
    transcript: list[DebateTurn],
    research_summary: str,
    research_notes: list[ResearchNote],
    completed_rounds: int,
) -> PromptEnvelope:
    return PromptEnvelope(
        operation_name="debate_assessment",
        context_budget_hint="medium",
        truncation_strategy="recent_turns_priority",
        instructions=(
            "Resuma debates editoriais multiagentes. "
            "Retorne JSON com `summary`, `positions`, `needs_more_rounds` e `open_questions`."
        ),
        prompt=join_prompt_blocks(
            render_header_block(
                "Contexto do debate",
                "\n".join(
                    [
                        f"Tema: {topic}",
                        f"Rodadas concluidas: {completed_rounds}",
                    ]
                ),
            ),
            render_header_block("Resumo da pesquisa", research_summary),
            render_list_block(
                "Notas de pesquisa",
                [f"{note.title} | {note.source} | {note.summary}" for note in research_notes],
                limit=8,
            ),
            render_list_block(
                "Transcript",
                [
                    f"Rodada {turn.round} [{turn.phase}] {turn.agent}: {turn.message}"
                    for turn in transcript[-12:]
                ],
            ),
        ),
    )
