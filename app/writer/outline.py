"""Outline generation based on research and debate."""

from __future__ import annotations

from app.llm.models import OutlinePayload, OutlineSection, SourceReference
from app.llm.openai_client import OpenAIResponsesClient


class OutlineGenerator:
    """Create the final editorial outline before drafting the article."""

    def __init__(self, client: OpenAIResponsesClient) -> None:
        self._client = client

    def build(
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
        research_summary: str,
        debate_summary: str,
        agent_positions: dict[str, str],
        references: list[SourceReference],
        article_profile: str,
        minimum_words: int,
    ) -> tuple[OutlinePayload, str]:
        prompt = (
            f"Solicitacao normalizada: {normalized_request}\n"
            f"Objetivo do artigo: {article_goal}\n"
            f"Publico: {audience}\n"
            f"Tom: {tone}\n"
            f"Dominio do conhecimento: {knowledge_domain}\n"
            f"Lente disciplinar: {disciplinary_lens}\n"
            f"Perfil editorial: {article_profile}\n"
            f"Meta minima de extensao: {minimum_words} palavras\n"
            f"Restricoes editoriais: {', '.join(constraints) or 'Nenhuma'}\n\n"
            "Exigencias de evidencia e rigor:\n"
            + "\n".join(f"- {item}" for item in evidence_requirements)
            + "\n\n"
            f"Resumo da pesquisa:\n{research_summary}\n\n"
            f"Resumo do debate:\n{debate_summary}\n\n"
            "Posicoes finais dos agentes:\n"
            + "\n".join(f"- {name}: {position}" for name, position in agent_positions.items())
            + "\n\nFontes consideradas:\n"
            + "\n".join(f"- {ref.title} | {ref.source} | {ref.url}" for ref in references)
        )
        outline = self._client.generate_structured(
            instructions=(
                "Voce define o outline final de um artigo academico-profissional em Markdown. "
                "Adapte a forma do outline ao dominio informado sem perder rigor: "
                "use estrutura de revisao cientifica para temas empiricos e estrutura de revisao academica "
                "para temas conceituais, historicos, juridicos, sociais ou humanisticos. "
                "O artigo deve soar profissional, denso e bem embasado, nunca como trabalho escolar. "
                "Retorne JSON com `headline`, `editorial_angle` e `sections`. "
                "Inclua secoes equivalentes a resumo, palavras-chave, introducao, desenvolvimento "
                "analitico, contrapontos, limitacoes e conclusao."
            ),
            prompt=prompt,
            schema_model=OutlinePayload,
            temperature=0.2,
        )
        normalized_outline = self._ensure_academic_sections(outline, article_goal=article_goal)
        return normalized_outline, self.render(normalized_outline)

    @staticmethod
    def render(outline: OutlinePayload) -> str:
        """Render the outline as Markdown text for downstream stages."""

        lines = [f"# {outline.headline}", "", f"Angulo editorial: {outline.editorial_angle}", ""]
        for section in outline.sections:
            lines.append(f"## {section.heading}")
            lines.append(section.purpose)
            for bullet in section.bullets:
                lines.append(f"- {bullet}")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _ensure_academic_sections(
        outline: OutlinePayload,
        *,
        article_goal: str,
    ) -> OutlinePayload:
        """Guarantee the minimum academic-rigid sections even if the model omits some."""

        mandatory = {
            "Resumo": "Sintetizar objetivo, escopo, achados e conclusoes centrais.",
            "Palavras-chave": "Registrar termos-chave para enquadramento tecnico.",
            "Introducao": "Contextualizar o tema, justificar relevancia e delimitar o problema.",
            "Discussao": "Desenvolver a analise central com evidencias, comparacoes e nuance.",
            "Contrapontos e limitacoes": "Explicitar tensoes, controversias e limites da base.",
            "Conclusao": "Fechar o argumento central e indicar implicacoes praticas.",
        }
        existing = {section.heading.lower(): section for section in outline.sections}
        sections = list(outline.sections)
        for heading, purpose in mandatory.items():
            if heading.lower() in existing:
                continue
            sections.append(
                OutlineSection(
                    heading=heading,
                    purpose=purpose,
                    bullets=[article_goal],
                )
            )
        return OutlinePayload(
            headline=outline.headline,
            editorial_angle=outline.editorial_angle,
            sections=sections,
        )
