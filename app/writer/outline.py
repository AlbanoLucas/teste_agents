"""Outline generation based on research and debate."""

from __future__ import annotations

from app.llm.models import OutlinePayload, OutlineSection, SourceReference
from app.llm.openai_client import OpenAIResponsesClient
from app.prompts.builders import build_outline_prompt


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
        prompt_envelope = build_outline_prompt(
            normalized_request=normalized_request,
            article_goal=article_goal,
            audience=audience,
            tone=tone,
            knowledge_domain=knowledge_domain,
            disciplinary_lens=disciplinary_lens,
            evidence_requirements=evidence_requirements,
            constraints=constraints,
            research_summary=research_summary,
            debate_summary=debate_summary,
            agent_positions=agent_positions,
            references=references,
            article_profile=article_profile,
            minimum_words=minimum_words,
        )
        outline = self._client.generate_structured(
            envelope=prompt_envelope,
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
