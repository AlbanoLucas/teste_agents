from __future__ import annotations

from collections import Counter
from pathlib import Path
from uuid import uuid4

from app.graph.build_graph import build_graph
from app.graph.services import WorkflowServices, WorkflowSettings
from app.llm.models import (
    IntakePlan,
    OutlinePayload,
    OutlineSection,
    ResearchAnalysisPayload,
    ResearchNote,
    ResolvedDebatePromptPack,
    SectionReviewPayload,
    SourceReference,
)
from app.research.web_search import SearchResult
from app.workflow.error_policy import WorkflowErrorPolicy
from app.workflow.section_service import SectionStateService
from app.workflow.state_adapter import GraphStateAdapter
from app.writer.article_assembler import ArticleAssembler
from app.writer.formatter import MarkdownFormatter
from app.writer.markdown_saver import MarkdownSaver


class FakeLlmClient:
    def generate_structured(self, *, schema_model, envelope=None, instructions=None, prompt=None, temperature=None):
        if schema_model is IntakePlan:
            return IntakePlan(
                normalized_request="Tema de teste",
                article_goal="Produzir uma revisao tecnico-cientifica",
                audience="Publico profissional",
                tone="Analitico",
                knowledge_domain="Ciencias aplicadas",
                disciplinary_lens="Interdisciplinar",
                evidence_requirements=["Usar fontes verificaveis"],
                constraints=["Evitar generalizacoes"],
                research_queries=["tema de teste revisao"],
                debate_prompt="Debata o tema com rigor.",
                debate_prompt_pack=ResolvedDebatePromptPack(
                    shared_context="Contexto global do tema.",
                    focus_axes=["fundamentos", "limites"],
                    domain_terms=["evidencia", "impacto"],
                    Analitico="Estruture o argumento.",
                    Critico="Questione lacunas.",
                    Estrategico="Amplie a relevancia.",
                ),
            )
        raise AssertionError(f"schema_model inesperado: {schema_model}")


class FakeSearchBackend:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, *, max_results: int) -> SearchResult:
        self.queries.append(query)
        return SearchResult(
            query=query,
            summary=f"Resumo de busca para {query}",
            sources=[
                {
                    "title": "Fonte principal",
                    "source": "Instituicao X",
                    "url": "https://example.com/principal",
                    "snippet": "Evidencias sobre o tema.",
                }
            ][:max_results],
        )


class FakeResearchSummarizer:
    def __init__(self, *, evidence_is_sufficient: bool) -> None:
        self.evidence_is_sufficient = evidence_is_sufficient

    def summarize(self, **kwargs) -> ResearchAnalysisPayload:
        references = kwargs["references"]
        if not self.evidence_is_sufficient:
            return ResearchAnalysisPayload(
                research_summary="Base ainda insuficiente para um artigo academico longo.",
                notes=[],
                evidence_is_sufficient=False,
                evidence_confidence=0.22,
                evidence_gaps=["Faltam revisoes robustas e base comparativa."],
                follow_up_queries=["tema de teste revisao sistematica", "tema de teste estado da arte"],
            )
        return ResearchAnalysisPayload(
            research_summary="Pesquisa suficiente para seguir ao debate e a escrita.",
            notes=[
                ResearchNote(
                    title="Fonte principal",
                    source="Instituicao X",
                    url=references[0].url,
                    summary="Resumo objetivo da principal evidencia.",
                    relevance=0.9,
                )
            ],
            evidence_is_sufficient=True,
            evidence_confidence=0.91,
            evidence_gaps=[],
            follow_up_queries=[],
        )


class FakeDebateEngine:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {
            "transcript": [
                {"round": 1, "phase": "tese_inicial", "agent": "Analitico", "message": "Estrutura inicial.", "stance": "Estrutura inicial.", "citations": []},
                {"round": 1, "phase": "tese_inicial", "agent": "Critico", "message": "Cuidado com lacunas.", "stance": "Cuidado com lacunas.", "citations": []},
                {"round": 1, "phase": "tese_inicial", "agent": "Estrategico", "message": "Contexto amplo importa.", "stance": "Contexto amplo importa.", "citations": []},
            ],
            "summary": "Debate global consolidado.",
            "positions": {
                "Analitico": "Estruture por secoes claras.",
                "Critico": "Evite extrapolacoes.",
                "Estrategico": "Conecte implicacoes amplas.",
            },
            "completed_rounds": kwargs.get("rounds", 3),
            "needs_more_rounds": False,
            "open_questions": [],
        }


class FakeOutlineGenerator:
    def build(self, **kwargs) -> tuple[OutlinePayload, str]:
        payload = OutlinePayload(
            headline="Titulo do artigo",
            editorial_angle="Angulo editorial robusto",
            sections=[
                OutlineSection(
                    heading="Resumo",
                    purpose="Sintetizar o argumento central.",
                    bullets=["apresentar escopo", "indicar principais achados"],
                ),
                OutlineSection(
                    heading="Discussao",
                    purpose="Desenvolver a analise critica.",
                    bullets=["comparar evidencias", "explicitar limitacoes"],
                ),
            ],
        )
        return payload, "# Titulo do artigo\n\n## Resumo\n\n## Discussao"


class FakeSectionWriter:
    def __init__(self) -> None:
        self.calls = Counter()

    def write(self, context) -> str:
        self.calls[context.section.id] += 1
        if context.section.id == "resumo":
            return "## Resumo\n\nResumo aprovado na primeira rodada."
        if context.section.section_research_summary:
            return (
                "## Discussao\n\nDiscussao revisada com evidencia adicional e contrapontos "
                "mais claros."
            )
        return "## Discussao\n\nDiscussao inicial fraca e superficial."


class FakeSectionReviewer:
    def __init__(self, *, persistent_failure: bool) -> None:
        self.persistent_failure = persistent_failure

    def review(self, context) -> SectionReviewPayload:
        if context.section.id == "resumo":
            return SectionReviewPayload(
                review_summary="Secao adequada.",
                strengths=["Sintese clara"],
                weaknesses=[],
                revision_requirements=[],
                prompt_improvements=[],
                needs_revision=False,
                quality_score=0.9,
            )

        improved = "evidencia adicional" in context.section.draft_md
        needs_revision = self.persistent_failure or not improved
        return SectionReviewPayload(
            review_summary="A secao precisa de mais densidade analitica." if needs_revision else "Secao aprovada apos reforco.",
            strengths=["Cobertura basica do tema"] if needs_revision else ["Comparacao entre evidencias"],
            weaknesses=["Profundidade insuficiente"] if needs_revision else [],
            revision_requirements=["Adicionar evidencias comparativas"] if needs_revision else [],
            prompt_improvements=["Tornar os contrapontos mais explicitos"] if needs_revision else [],
            needs_revision=needs_revision,
            quality_score=0.42 if needs_revision else 0.86,
        )


class FakeSectionRecovery:
    def __init__(self) -> None:
        self.prepare_calls: list[str] = []
        self.debate_calls: list[str] = []

    def prepare_recovery(self, *, envelope, section, node_name):
        self.prepare_calls.append(section.id)
        section.recovery_problem_summary = "A secao carece de evidencia comparativa."
        section.section_research_queries = ["discussao evidencia comparativa", "discussao limitacoes"]
        section.section_research_summary = "Pesquisa focada com evidencias adicionais."
        return envelope, section

    def run_section_debate(self, *, envelope, section):
        self.debate_calls.append(section.id)
        section.section_debate_summary = "Mini-debate focado na densidade analitica da secao."
        section.section_agent_positions = envelope.debate.positions.from_mapping(
            {
                "Analitico": "Reestruturar a discussao.",
                "Critico": "Adicionar comparacao entre evidencias.",
                "Estrategico": "Conectar a secao ao argumento geral.",
            }
        )
        return envelope, section


class DummyComponent:
    def __getattr__(self, name: str):
        raise AssertionError(f"Componente nao deveria ser usado: {name}")


def _temp_dir() -> Path:
    path = Path("tests") / ".artifacts" / f"workflow-tests-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_services(
    tmp_path: Path,
    *,
    evidence_is_sufficient: bool,
    persistent_section_failure: bool = False,
) -> tuple[WorkflowServices, FakeDebateEngine, FakeSectionWriter, FakeSectionRecovery]:
    formatter = MarkdownFormatter()
    debate_engine = FakeDebateEngine()
    section_writer = FakeSectionWriter()
    section_recovery = FakeSectionRecovery()
    services = WorkflowServices(
        llm_client=FakeLlmClient(),
        search_backend=FakeSearchBackend(),
        research_summarizer=FakeResearchSummarizer(evidence_is_sufficient=evidence_is_sufficient),
        debate_engine=debate_engine,
        outline_generator=FakeOutlineGenerator(),
        section_writer=section_writer,
        section_reviewer=FakeSectionReviewer(persistent_failure=persistent_section_failure),
        article_assembler=ArticleAssembler(formatter),
        markdown_saver=MarkdownSaver(),
        state_adapter=GraphStateAdapter(),
        error_policy=WorkflowErrorPolicy(),
        section_service=SectionStateService(formatter),
        section_recovery=section_recovery,
        settings=WorkflowSettings(
            default_output_path=str(tmp_path / "artigo.md"),
            research_max_sources=3,
            debate_min_rounds=3,
            debate_max_rounds=5,
            section_retry_max=1,
            article_profile="academic_rigid",
            article_min_words=2500,
            evidence_policy="abort",
        ),
    )
    return services, debate_engine, section_writer, section_recovery


def test_graph_retries_only_the_failed_section_and_preserves_approved_ones() -> None:
    tmp_path = _temp_dir()
    services, debate_engine, section_writer, section_recovery = _make_services(
        tmp_path,
        evidence_is_sufficient=True,
        persistent_section_failure=False,
    )
    workflow = build_graph(services)
    output_path = tmp_path / "artigo_final.md"

    final_state = workflow.invoke(
        {
            "workflow": {
                "user_request": "Escreva sobre um tema de teste",
                "output_path": str(output_path),
                "errors": [],
                "metadata": {},
            }
        }
    )

    workflow_state = final_state["workflow"]
    assert len(debate_engine.calls) == 1
    assert section_recovery.prepare_calls == ["discussao"]
    assert section_recovery.debate_calls == ["discussao"]
    assert section_writer.calls["resumo"] == 1
    assert section_writer.calls["discussao"] == 2
    assert workflow_state["sections"][0]["status"] == "approved"
    assert workflow_state["sections"][1]["status"] == "approved"
    assert workflow_state["quality_alerts"] == []
    assert workflow_state["status"] == "completed"
    assert output_path.exists()


def test_graph_marks_section_with_quality_warning_after_retry_budget_is_exhausted() -> None:
    tmp_path = _temp_dir()
    services, _, section_writer, section_recovery = _make_services(
        tmp_path,
        evidence_is_sufficient=True,
        persistent_section_failure=True,
    )
    workflow = build_graph(services)
    output_path = tmp_path / "artigo_alerta.md"

    final_state = workflow.invoke(
        {
            "workflow": {
                "user_request": "Escreva sobre um tema de teste",
                "output_path": str(output_path),
                "errors": [],
                "metadata": {},
            }
        }
    )

    workflow_state = final_state["workflow"]
    assert section_recovery.prepare_calls == ["discussao"]
    assert section_recovery.debate_calls == ["discussao"]
    assert section_writer.calls["resumo"] == 1
    assert section_writer.calls["discussao"] == 2
    assert workflow_state["sections"][1]["status"] == "accepted_with_warnings"
    assert len(workflow_state["quality_alerts"]) == 1
    assert "## Alertas de Qualidade" in workflow_state["final_article_md"]


def test_graph_bypasses_debate_and_section_pipeline_when_evidence_is_insufficient() -> None:
    tmp_path = _temp_dir()
    services, debate_engine, section_writer, section_recovery = _make_services(
        tmp_path,
        evidence_is_sufficient=False,
    )
    workflow = build_graph(services)
    output_path = tmp_path / "relatorio.md"

    final_state = workflow.invoke(
        {
            "workflow": {
                "user_request": "Escreva sobre um tema de teste",
                "output_path": str(output_path),
                "errors": [],
                "metadata": {},
            }
        }
    )

    workflow_state = final_state["workflow"]
    assert debate_engine.calls == []
    assert section_recovery.prepare_calls == []
    assert section_recovery.debate_calls == []
    assert dict(section_writer.calls) == {}
    assert workflow_state["output_mode"] == "insufficiency_report"
    assert workflow_state["sections"] == []
    assert "Relatorio de Insuficiencia de Evidencia" in workflow_state["final_article_md"]
    assert output_path.exists()


class BoomSearchBackend:
    def search(self, query: str, *, max_results: int) -> SearchResult:
        raise RuntimeError(f"falha ao pesquisar {query}")


def test_graph_routes_terminal_failure_to_failure_node_and_saves_report() -> None:
    tmp_path = _temp_dir()
    formatter = MarkdownFormatter()
    workflow = build_graph(
        WorkflowServices(
            llm_client=FakeLlmClient(),
            search_backend=BoomSearchBackend(),
            research_summarizer=FakeResearchSummarizer(evidence_is_sufficient=True),
            debate_engine=FakeDebateEngine(),
            outline_generator=FakeOutlineGenerator(),
            section_writer=FakeSectionWriter(),
            section_reviewer=FakeSectionReviewer(persistent_failure=False),
            article_assembler=ArticleAssembler(formatter),
            markdown_saver=MarkdownSaver(),
            state_adapter=GraphStateAdapter(),
            error_policy=WorkflowErrorPolicy(),
            section_service=SectionStateService(formatter),
            section_recovery=FakeSectionRecovery(),
            settings=WorkflowSettings(
                default_output_path=str(tmp_path / "falha.md"),
                research_max_sources=3,
                debate_min_rounds=3,
                debate_max_rounds=5,
                section_retry_max=1,
                article_profile="academic_rigid",
                article_min_words=2500,
                evidence_policy="abort",
            ),
        )
    )
    output_path = tmp_path / "falha.md"

    final_state = workflow.invoke(
        {
            "workflow": {
                "user_request": "Escreva sobre um tema de teste",
                "output_path": str(output_path),
                "errors": [],
                "metadata": {},
            }
        }
    )

    workflow_state = final_state["workflow"]
    assert workflow_state["status"] == "failed"
    assert workflow_state["failed_node"] == "research_node"
    assert workflow_state["terminal_error"]["category"] == "terminal"
    assert "Relatorio de Falha do Workflow" in workflow_state["final_article_md"]
    assert output_path.exists()
