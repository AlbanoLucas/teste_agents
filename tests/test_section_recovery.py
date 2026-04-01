from app.llm.models import (
    DebatePromptPackOverride,
    ResearchAnalysisPayload,
    ResearchNote,
    ResolvedDebatePromptPack,
    SectionRecoveryPlan,
    SourceReference,
)
from app.research.web_search import SearchResult
from app.workflow.error_policy import WorkflowErrorPolicy
from app.workflow.models import DebateBundle, ResearchBundle, SectionState, WorkflowStateEnvelope
from app.workflow.section_recovery import SectionRecoveryService


class FakeRecoveryLlmClient:
    def __init__(self, override: DebatePromptPackOverride) -> None:
        self.override = override

    def generate_structured(self, *, schema_model, envelope=None, instructions=None, prompt=None, temperature=None):
        assert schema_model is SectionRecoveryPlan
        return SectionRecoveryPlan(
            problem_summary="A secao precisa de mais evidencia comparativa.",
            research_queries=["query 1", "query 2"],
            debate_prompt="Mini-debate focado na secao.",
            prompt_pack_override=self.override,
        )


class FakeRecoverySearchBackend:
    def search(self, query: str, *, max_results: int) -> SearchResult:
        return SearchResult(
            query=query,
            summary=f"Resumo para {query}",
            sources=[
                {
                    "title": f"Fonte {query}",
                    "source": "Repositorio Y",
                    "url": f"https://example.com/{query.replace(' ', '-')}",
                    "snippet": "Snippet relevante.",
                }
            ][:max_results],
        )


class FakeRecoverySummarizer:
    def summarize(self, **kwargs) -> ResearchAnalysisPayload:
        references = kwargs["references"]
        return ResearchAnalysisPayload(
            research_summary="Resumo focado da pesquisa da secao.",
            notes=[
                ResearchNote(
                    title=references[0].title,
                    source=references[0].source,
                    url=references[0].url,
                    summary="Nota focada da secao.",
                    relevance=0.8,
                )
            ],
            evidence_is_sufficient=True,
            evidence_confidence=0.7,
            evidence_gaps=[],
            follow_up_queries=[],
        )


class RecordingDebateEngine:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {
            "transcript": [],
            "summary": "Mini-debate concluido.",
            "positions": {
                "Analitico": "Refinar a estrutura.",
                "Critico": "Explicitar lacunas.",
                "Estrategico": "Conectar ao argumento global.",
            },
            "completed_rounds": 3,
            "needs_more_rounds": False,
            "open_questions": [],
        }


def test_section_recovery_merges_blank_override_without_erasing_global_specialization() -> None:
    debate_engine = RecordingDebateEngine()
    orchestrator = SectionRecoveryService(
        llm_client=FakeRecoveryLlmClient(
            DebatePromptPackOverride(
                shared_context="   ",
                focus_axes=[],
                domain_terms=None,
                Analitico="",
                Critico=None,
                Estrategico="   ",
            )
        ),
        search_backend=FakeRecoverySearchBackend(),
        research_summarizer=FakeRecoverySummarizer(),
        debate_engine=debate_engine,
        error_policy=WorkflowErrorPolicy(),
        research_max_sources=3,
        article_profile="academic_rigid",
    )
    envelope = WorkflowStateEnvelope(
        normalized_request="Tema global",
        article_goal="Objetivo global",
        knowledge_domain="Politicas publicas",
        disciplinary_lens="Interdisciplinar",
        evidence_requirements=["Rigor"],
        research=ResearchBundle(
            summary="Pesquisa global resumida.",
            references=[SourceReference(title="Fonte base", source="Base", url="https://example.com/base")],
        ),
        debate=DebateBundle(
            prompt_pack=ResolvedDebatePromptPack(
                shared_context="Contexto global preservado.",
                focus_axes=["causas", "respostas"],
                domain_terms=["governanca", "impacto"],
                Analitico="Estruture com rigor.",
                Critico="Nao apague esta especializacao.",
                Estrategico="Conecte implicacoes amplas.",
            )
        ),
    )
    section = SectionState(
        id="discussao",
        heading="Discussao",
        purpose="Aprofundar a analise.",
        bullets=["comparar evidencias"],
        review_summary="Falta densidade.",
        weaknesses=["Pouca comparacao"],
        revision_requirements=["Adicionar evidencia comparativa"],
        prompt_improvements=["Explorar tensoes entre fontes"],
        draft_md="## Discussao\n\nTexto inicial.",
    )

    _, recovered_section = orchestrator.prepare_recovery(
        envelope=envelope,
        section=section,
        node_name="section_research_node",
    )

    assert debate_engine.calls == []
    assert recovered_section.resolved_prompt_pack.Critico == "Nao apague esta especializacao."

    _, debated_section = orchestrator.run_section_debate(
        envelope=envelope,
        section=recovered_section,
    )

    assert len(debate_engine.calls) == 1
    call = debate_engine.calls[0]
    assert call["shared_context"] == "Contexto global preservado."
    assert call["focus_axes"] == ["causas", "respostas"]
    assert call["domain_terms"] == ["governanca", "impacto"]
    assert call["agent_specializations"]["Critico"] == "Nao apague esta especializacao."
    assert debated_section.section_debate_summary == "Mini-debate concluido."
