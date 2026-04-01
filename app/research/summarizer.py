"""Research summarization module."""

from __future__ import annotations

from app.llm.models import ResearchAnalysisPayload, ResearchNote, SourceReference
from app.llm.openai_client import OpenAIResponsesClient
from app.research.parser import render_research_context
from app.research.web_search import SearchResult


class ResearchSummarizer:
    """Convert raw search results into structured research notes."""

    def __init__(self, client: OpenAIResponsesClient) -> None:
        self._client = client

    def summarize(
        self,
        *,
        topic: str,
        knowledge_domain: str,
        disciplinary_lens: str,
        evidence_requirements: list[str],
        results: list[SearchResult],
        references: list[SourceReference],
        minimum_words: int,
        article_profile: str,
    ) -> ResearchAnalysisPayload:
        if not references:
            return ResearchAnalysisPayload(
                research_summary=(
                    "Nenhuma fonte estruturada foi encontrada. O artigo deve sinalizar que "
                    "a pesquisa precisa ser aprofundada antes de afirmar fatos especificos."
                ),
                notes=[],
                evidence_is_sufficient=False,
                evidence_confidence=0.0,
                evidence_gaps=[
                    "Nao foram encontradas fontes estruturadas suficientes para sustentar o artigo.",
                ],
                follow_up_queries=[
                    f"estado da arte e revisoes cientificas sobre {topic}",
                    f"riscos, limitacoes e controversias sobre {topic}",
                    f"aplicacoes, impacto e evidencias recentes sobre {topic}",
                ],
            )

        prompt = (
            f"Tema do artigo: {topic}\n\n"
            f"Dominio do conhecimento: {knowledge_domain}\n"
            f"Lente disciplinar: {disciplinary_lens}\n"
            f"Perfil editorial: {article_profile}\n"
            f"Meta minima de extensao: {minimum_words} palavras.\n\n"
            "Exigencias de evidencia e rigor:\n"
            + "\n".join(f"- {item}" for item in evidence_requirements)
            + "\n\n"
            "Consolide a pesquisa em notas objetivas para um debate editorial entre agentes. "
            "Avalie se a base atual sustenta com seguranca um artigo academico rigido e longo, "
            "sem inflacao artificial e sem extrapolacao de fatos. "
            "Adapte o julgamento ao dominio: nem todo tema exige o mesmo tipo de evidencia, "
            "mas todo tema exige rigor proporcional ao seu campo.\n"
            "Cada nota precisa manter URL e nome da fonte. Produza um resumo que evidencie "
            "consensos, tensoes, riscos de simplificacao e lacunas de cobertura.\n\n"
            f"{render_research_context(results, references)}"
        )
        payload = self._client.generate_structured(
            instructions=(
                "Voce sintetiza pesquisas da web para um pipeline editorial. "
                "Retorne JSON valido com `research_summary`, `notes`, "
                "`evidence_is_sufficient`, `evidence_confidence`, `evidence_gaps` "
                "e `follow_up_queries`."
            ),
            prompt=prompt,
            schema_model=ResearchAnalysisPayload,
            temperature=0.2,
        )
        return self._repair_notes(payload=payload, references=references)

    @staticmethod
    def _repair_notes(
        *,
        payload: ResearchAnalysisPayload,
        references: list[SourceReference],
    ) -> ResearchAnalysisPayload:
        by_url = {item.url: item for item in references}
        fixed_notes: list[ResearchNote] = []
        for note in payload.notes:
            source = by_url.get(note.url)
            if source is None:
                fixed_notes.append(note)
                continue
            fixed_notes.append(
                ResearchNote(
                    title=note.title or source.title,
                    source=note.source or source.source,
                    url=note.url or source.url,
                    summary=note.summary,
                    relevance=note.relevance,
                )
            )
        return ResearchAnalysisPayload(
            research_summary=payload.research_summary,
            notes=fixed_notes,
            evidence_is_sufficient=payload.evidence_is_sufficient,
            evidence_confidence=payload.evidence_confidence,
            evidence_gaps=payload.evidence_gaps,
            follow_up_queries=payload.follow_up_queries,
        )
