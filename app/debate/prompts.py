"""Prompt builders for the debate adapter."""

from __future__ import annotations

from app.debate.agents import DebateAgentSpec
from app.llm.models import ResearchNote


def build_agent_system_message(
    spec: DebateAgentSpec,
    *,
    shared_context: str = "",
    focus_axes: list[str] | None = None,
    domain_terms: list[str] | None = None,
    agent_specialization: str = "",
) -> str:
    """Create the system prompt for a single specialist agent."""

    focus_block = "; ".join(focus_axes or []) or "Sem eixos adicionais."
    terms_block = ", ".join(domain_terms or []) or "Sem termos adicionais."

    return (
        f"Voce e o agente {spec.name}. "
        f"Responsabilidade: {spec.responsibility} "
        f"Objetivo: {spec.objective} "
        f"Estilo: {spec.style} "
        f"Contexto especializado do tema: {shared_context or 'Nao fornecido.'} "
        f"Eixos prioritarios de analise: {focus_block}. "
        f"Termos, atores ou conceitos que merecem atencao: {terms_block}. "
        f"Especializacao adicional deste agente para o tema atual: {agent_specialization or 'Manter a especializacao-base do papel.'} "
        f"Limite cada fala a aproximadamente {spec.max_words} palavras. "
        "Nao invente fatos especificos que nao estejam sustentados pelas notas de pesquisa. "
        "Adapte seu rigor ao dominio do tema: em ciencias empiricas, cobre mecanismos, resultados e limites metodologicos; "
        "em ciencias humanas e sociais, cobre enquadramento conceitual, escolas interpretativas, controversias e limites argumentativos. "
        "Evite repetir argumentos ja ditos e sempre contribua para melhorar o artigo final."
    )


def phase_label(round_number: int) -> str:
    """Return the purpose label for a debate round."""

    labels = {
        1: "tese_inicial",
        2: "critica_cruzada",
        3: "refinamento_final",
    }
    return labels.get(round_number, f"follow_up_{round_number}")


def build_phase_task(
    *,
    topic: str,
    research_summary: str,
    research_notes: list[ResearchNote],
    round_number: int,
    prior_transcript: list[dict] | None = None,
    open_questions: list[str] | None = None,
    shared_context: str = "",
    focus_axes: list[str] | None = None,
    domain_terms: list[str] | None = None,
) -> str:
    """Compose the user task sent to the AutoGen team for one round."""

    notes_block = "\n".join(
        f"- {note.title} | {note.source} | {note.url} | {note.summary}"
        for note in research_notes
    )
    prior_block = "\n".join(
        f"- Rodada {item['round']} [{item['phase']}] {item['agent']}: {item['message']}"
        for item in prior_transcript or []
    )
    phase = phase_label(round_number)

    if round_number == 1:
        round_goal = (
            "Apresente sua tese inicial para o artigo, destacando o angulo principal "
            "que merece entrar na estrutura final."
        )
    elif round_number == 2:
        round_goal = (
            "Critique as teses anteriores, aponte lacunas, riscos e simplificacoes, "
            "e force um texto mais rigoroso."
        )
    elif round_number == 3:
        round_goal = (
            "Revise sua posicao, integre o que aprendeu no confronto e proponha "
            "recomendacoes concretas para a versao final do artigo."
        )
    else:
        round_goal = (
            "Feche as questoes em aberto, reduza ambiguidades restantes e entregue "
            "orientacoes objetivas para o texto final."
        )

    open_questions_block = (
        "\n".join(f"- {item}" for item in open_questions or [])
        or "- Nenhuma questao adicional registrada."
    )
    prior_section = prior_block or "- Primeiro ciclo do debate."
    focus_block = "\n".join(f"- {item}" for item in (focus_axes or [])) or "- Nenhum eixo prioritario registrado."
    terms_block = ", ".join(domain_terms or []) or "Nenhum termo-chave adicional registrado."

    return (
        f"Tema do artigo: {topic}\n"
        f"Fase: {phase}\n"
        f"Objetivo desta rodada: {round_goal}\n\n"
        f"Contexto especializado do tema:\n{shared_context or '- Contexto adicional nao fornecido.'}\n\n"
        f"Eixos prioritarios de analise:\n{focus_block}\n\n"
        f"Termos, atores ou conceitos prioritarios:\n- {terms_block}\n\n"
        f"Resumo da pesquisa:\n{research_summary}\n\n"
        f"Notas estruturadas:\n{notes_block or '- Nenhuma nota disponivel.'}\n\n"
        f"Transcript anterior:\n{prior_section}\n\n"
        f"Questoes em aberto:\n{open_questions_block}\n\n"
        "Debatam em ordem, sem repetir o que ja esta resolvido. "
        "Cada fala deve trazer um ponto acionavel para melhorar a estrutura ou o conteudo do artigo. "
        "O objetivo e produzir uma revisao academica rigorosa adequada ao dominio do tema, e nao um texto generico."
    )
