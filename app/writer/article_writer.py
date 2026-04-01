"""Section-oriented Markdown writing, review and assembly helpers."""

from __future__ import annotations

import re

from app.llm.models import SectionReviewPayload, SectionUnit, SourceReference
from app.llm.openai_client import OpenAIResponsesClient


class MarkdownArticleWriter:
    """Generate, review and assemble article sections in Markdown."""

    def __init__(self, client: OpenAIResponsesClient) -> None:
        self._client = client

    def write_section(
        self,
        *,
        normalized_request: str,
        article_goal: str,
        audience: str,
        tone: str,
        knowledge_domain: str,
        disciplinary_lens: str,
        evidence_requirements: list[str],
        constraints: list[str],
        article_profile: str,
        minimum_words: int,
        headline: str,
        editorial_angle: str,
        section: SectionUnit,
        section_index: int,
        total_sections: int,
        prior_sections: list[SectionUnit],
        research_summary: str,
        debate_summary: str,
        agent_positions: dict[str, str],
        references: list[SourceReference],
    ) -> str:
        """Draft a single section while preserving the approved sections around it."""

        prompt = (
            f"Solicitacao normalizada: {normalized_request}\n"
            f"Objetivo do artigo: {article_goal}\n"
            f"Publico: {audience}\n"
            f"Tom: {tone}\n"
            f"Dominio do conhecimento: {knowledge_domain}\n"
            f"Lente disciplinar: {disciplinary_lens}\n"
            f"Perfil editorial: {article_profile}\n"
            f"Meta minima do artigo: {minimum_words} palavras\n"
            f"Titulo do artigo: {headline}\n"
            f"Angulo editorial: {editorial_angle}\n"
            f"Secao atual: {section.heading}\n"
            f"Ordem da secao: {section_index + 1} de {total_sections}\n"
            f"Tipo da secao: {section.kind}\n"
            f"Meta de extensao da secao: {section.target_words} palavras\n\n"
            "Escopo e intencao da secao:\n"
            f"- Finalidade: {section.purpose}\n"
            + "\n".join(f"- Ponto-chave: {bullet}" for bullet in section.bullets)
            + "\n\n"
            "Restricoes editoriais:\n"
            + "\n".join(f"- {item}" for item in constraints)
            + "\n\n"
            "Exigencias de evidencia:\n"
            + "\n".join(f"- {item}" for item in evidence_requirements)
            + "\n\n"
            f"Resumo global da pesquisa:\n{research_summary}\n\n"
            f"Resumo global do debate:\n{debate_summary}\n\n"
            "Posicoes globais dos agentes:\n"
            + "\n".join(f"- {agent}: {position}" for agent, position in agent_positions.items())
            + "\n\n"
            "Secoes ja aprovadas e congeladas:\n"
            + (
                "\n".join(
                    f"- {approved.heading}: {_preview_markdown(approved.draft_md, limit=220)}"
                    for approved in prior_sections
                )
                or "- Nenhuma"
            )
            + "\n\n"
            f"Pesquisa focada da secao:\n{section.section_research_summary or '(nao houve recuperacao focada)'}\n\n"
            f"Debate focado da secao:\n{section.section_debate_summary or '(nao houve mini-debate focado)'}\n\n"
            "Posicoes focadas dos agentes para a secao:\n"
            + "\n".join(
                f"- {agent}: {position}"
                for agent, position in section.section_agent_positions.as_dict().items()
                if position
            )
            + "\n\n"
            "Fontes disponiveis:\n"
            + "\n".join(f"- {reference.title} | {reference.source} | {reference.url}" for reference in references)
            + "\n\n"
            + self._render_section_feedback(section)
        )
        instructions = (
            "Voce esta escrevendo apenas uma secao de um artigo academico-profissional em Markdown. "
            "A secao precisa contribuir para um texto longo, analitico e coerente, sem parecer colagem. "
            "Nao reescreva secoes anteriores e nao adicione uma secao de referencias. "
            "Use atribuicao inline de fontes em afirmacoes factuais relevantes, com mencoes do tipo "
            "'Segundo OMS' ou 'De acordo com relatorio da ONU', sem inventar dados. "
            "A secao deve comecar exatamente com o heading em Markdown correspondente. "
            "Se a secao for 'Palavras-chave', entregue apenas uma linha concisa com termos separados por ponto e virgula. "
            "Se a secao for 'Resumo', escreva um resumo tecnico coeso e denso, nao um abstract escolar."
        )
        draft = self._client.generate_text(
            instructions=instructions,
            prompt=prompt,
            temperature=0.3,
        )
        return self._normalize_section_markdown(section=section, markdown=draft)

    def review_section(
        self,
        *,
        normalized_request: str,
        article_goal: str,
        article_profile: str,
        knowledge_domain: str,
        disciplinary_lens: str,
        minimum_words: int,
        headline: str,
        editorial_angle: str,
        section: SectionUnit,
        total_sections: int,
        research_summary: str,
        debate_summary: str,
        references: list[SourceReference],
    ) -> SectionReviewPayload:
        """Review a single section with a rubric suited to its section type."""

        rubric = (
            "Rubrica short_form: verificar se a secao e precisa, terminologicamente correta, "
            "coesa com o artigo e suficientemente informativa para seu papel curto."
            if section.kind == "short_form"
            else "Rubrica standard: verificar profundidade analitica, aderencia ao escopo, "
            "uso rigoroso de evidencias, conexao com o argumento central, ausencia de repeticao "
            "e maturidade academica."
        )
        prompt = (
            f"Solicitacao normalizada: {normalized_request}\n"
            f"Objetivo do artigo: {article_goal}\n"
            f"Perfil editorial: {article_profile}\n"
            f"Dominio do conhecimento: {knowledge_domain}\n"
            f"Lente disciplinar: {disciplinary_lens}\n"
            f"Meta minima do artigo completo: {minimum_words} palavras\n"
            f"Titulo do artigo: {headline}\n"
            f"Angulo editorial: {editorial_angle}\n"
            f"Secao avaliada: {section.heading}\n"
            f"Tipo da secao: {section.kind}\n"
            f"Meta da secao: {section.target_words} palavras\n"
            f"Posicao da secao: {total_sections}\n\n"
            "Finalidade da secao:\n"
            f"- {section.purpose}\n"
            + "\n".join(f"- {bullet}" for bullet in section.bullets)
            + "\n\n"
            f"{rubric}\n\n"
            f"Resumo global da pesquisa:\n{research_summary}\n\n"
            f"Resumo global do debate:\n{debate_summary}\n\n"
            "Fontes disponiveis:\n"
            + "\n".join(f"- {reference.title} | {reference.source} | {reference.url}" for reference in references)
            + "\n\n"
            f"Texto da secao:\n{section.draft_md}"
        )
        return self._client.generate_structured(
            instructions=(
                "Voce revisa apenas uma secao de um artigo academico em Markdown. "
                "Retorne JSON valido com `review_summary`, `strengths`, `weaknesses`, "
                "`revision_requirements`, `prompt_improvements`, `needs_revision` e `quality_score`."
            ),
            prompt=prompt,
            schema_model=SectionReviewPayload,
            temperature=0.2,
        )

    def assemble_article(
        self,
        *,
        output_mode: str,
        normalized_request: str,
        article_goal: str,
        headline: str,
        editorial_angle: str,
        sections: list[SectionUnit],
        references: list[SourceReference],
        evidence_confidence: float | str,
        evidence_gaps: list[str],
        follow_up_queries: list[str],
        research_summary: str,
        quality_alerts: list[dict[str, object]],
    ) -> str:
        """Assemble the final article or insufficiency report."""

        if output_mode != "article":
            return self.render_insufficiency_report(
                normalized_request=normalized_request,
                article_goal=article_goal,
                evidence_confidence=evidence_confidence,
                evidence_gaps=evidence_gaps,
                follow_up_queries=follow_up_queries,
                research_summary=research_summary,
                references=references,
            )

        lines = [f"# {headline}", "", f"Angulo editorial: {editorial_angle}", ""]
        for section in sections:
            cleaned = self._strip_references_block(section.draft_md.strip())
            if cleaned:
                lines.append(cleaned)
                lines.append("")

        if quality_alerts:
            lines.append("## Alertas de Qualidade")
            lines.append("")
            for alert in quality_alerts:
                heading = str(alert.get("heading", "Secao sem identificacao"))
                summary = str(alert.get("summary", "Persistem pendencias editoriais."))
                pending = alert.get("pending", [])
                pending_text = "; ".join(str(item) for item in pending) or "pendencias nao detalhadas"
                lines.append(f"- **{heading}**: {summary} Pendencias remanescentes: {pending_text}.")
            lines.append("")

        if references:
            lines.append(self.render_references(references))

        return "\n".join(line for line in lines if line is not None).strip()

    def render_references(self, references: list[SourceReference]) -> str:
        """Render the final references section once."""

        lines = ["## Referencias", ""]
        seen: set[str] = set()
        for reference in references:
            if reference.url in seen:
                continue
            seen.add(reference.url)
            lines.append(f"- {reference.source}. {reference.title}. Disponivel em: {reference.url}")
        return "\n".join(lines)

    def render_insufficiency_report(
        self,
        *,
        normalized_request: str,
        article_goal: str,
        evidence_confidence: float | str,
        evidence_gaps: list[str],
        follow_up_queries: list[str],
        research_summary: str,
        references: list[SourceReference],
    ) -> str:
        """Render a formal report when the evidence gate blocks article generation."""

        lines = [
            "# Relatorio de Insuficiencia de Evidencia",
            "",
            f"Solicitacao: {normalized_request}",
            "",
            f"Objetivo editorial pretendido: {article_goal}",
            "",
            "## Diagnostico",
            "",
            (
                "A geracao do artigo academico foi interrompida porque a base pesquisada nao "
                "sustenta com seguranca um texto profissional, longo e bem embasado."
            ),
            f"Confianca estimada da evidencia disponivel: {evidence_confidence}",
            "",
            "## Resumo da pesquisa disponivel",
            "",
            research_summary or "Nao ha resumo suficiente para sustentar o artigo.",
            "",
            "## Lacunas criticas",
            "",
        ]
        for gap in evidence_gaps or ["As fontes coletadas nao cobrem o tema com rigor suficiente."]:
            lines.append(f"- {gap}")
        lines.extend(["", "## Proximas pesquisas recomendadas", ""])
        for query in follow_up_queries or ["Aprofundar a busca por revisoes e fontes institucionais."]:
            lines.append(f"- {query}")
        if references:
            lines.extend(["", self.render_references(references)])
        return "\n".join(lines).strip()

    def _render_section_feedback(self, section: SectionUnit) -> str:
        lines = ["Feedback obrigatorio da revisao anterior:"]
        if not section.review_summary and not section.revision_requirements and not section.prompt_improvements:
            lines.append("- Nenhum feedback anterior. Produza a melhor versao inicial da secao.")
            return "\n".join(lines)

        lines.append(f"- Resumo da ultima revisao: {section.review_summary or '(sem resumo)'}")
        lines.extend(
            f"- Requisito de revisao: {item}" for item in (section.revision_requirements or [])
        )
        lines.extend(
            f"- Ajuste de prompt recomendado: {item}" for item in (section.prompt_improvements or [])
        )
        return "\n".join(lines)

    def _normalize_section_markdown(self, *, section: SectionUnit, markdown: str) -> str:
        cleaned = self._strip_references_block(markdown.strip())
        heading = f"## {section.heading}"
        if section.kind == "short_form" and "palavras-chave" in section.heading.lower():
            body = cleaned.replace(heading, "").strip()
            body = re.sub(r"(?i)^palavras-chave\s*:?\s*", "", body).strip()
            return f"{heading}\n{body or 'termos a definir'}".strip()
        if cleaned.startswith(heading):
            return cleaned
        body = cleaned.lstrip("#").strip()
        return f"{heading}\n\n{body}".strip()

    @staticmethod
    def _strip_references_block(markdown: str) -> str:
        return re.sub(
            r"\n## Referencias\b[\s\S]*$",
            "",
            markdown,
            flags=re.IGNORECASE,
        ).strip()


def _preview_markdown(markdown: str, *, limit: int = 220) -> str:
    compact = " ".join(markdown.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
