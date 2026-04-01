from __future__ import annotations

from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from app.graph.build_graph import build_graph
from app.graph.edges import route_after_section_init, route_after_section_review
from app.graph.nodes import WorkflowDependencies, WorkflowSettings, build_nodes
from app.llm.models import (
    DebatePromptPack,
    IntakePlan,
    OutlinePayload,
    OutlineSection,
    ResearchAnalysisPayload,
    ResearchNote,
    SectionRecoveryPlan,
    SectionReviewPayload,
    SectionUnit,
    SourceReference,
)
from app.research.web_search import SearchResult
from app.writer.article_writer import MarkdownArticleWriter
from app.writer.markdown_saver import MarkdownSaver
from app.writer.outline import OutlineGenerator


class CaptureClient:
    def __init__(self) -> None:
        self.last_prompt = ""

    def generate_text(self, *, prompt: str, **_: object) -> str:
        self.last_prompt = prompt
        return "Texto gerado."

    def generate_structured(self, *, schema_model, **_: object):
        return schema_model(
            review_summary="Secao aprovada.",
            strengths=["Coerente."],
            weaknesses=[],
            revision_requirements=[],
            prompt_improvements=[],
            needs_revision=False,
            quality_score=0.9,
        )


class FakeLLMClient:
    def generate_structured(self, *, schema_model, prompt: str, **_: object):
        if schema_model is IntakePlan:
            return IntakePlan(
                normalized_request="Artigo sobre energia limpa.",
                article_goal="Produzir uma revisao cientifica.",
                audience="Leitores especializados",
                tone="Formal e analitico",
                knowledge_domain="Politica energetica",
                disciplinary_lens="Revisao interdisciplinar",
                evidence_requirements=["Cruzar fontes tecnicas e institucionais."],
                constraints=["Evitar simplificacoes setoriais."],
                research_queries=["energia limpa revisao", "energia limpa regulacao"],
                debate_prompt="Debater fundamentos e implicacoes de energia limpa.",
                debate_prompt_pack=DebatePromptPack(
                    shared_context="Tema de transicao energetica.",
                    focus_axes=["regulacao", "escala tecnologica"],
                    domain_terms=["descarbonizacao", "matriz eletrica"],
                    Analitico="Estruture fundamentos e secoes.",
                    Critico="Aponte lacunas e trade-offs.",
                    Estrategico="Conecte impacto economico e institucional.",
                ),
            )
        if schema_model is SectionRecoveryPlan:
            heading = "Secao"
            for line in prompt.splitlines():
                if line.startswith("Secao problematica: "):
                    heading = line.split(": ", 1)[1]
                    break
            return SectionRecoveryPlan(
                problem_summary=f"Recuperar {heading}.",
                research_queries=[
                    f"{heading} evidencia 1",
                    f"{heading} evidencia 2",
                ],
                debate_prompt=f"Mini debate sobre {heading}.",
                prompt_pack=DebatePromptPack(
                    shared_context=f"Contexto da secao {heading}.",
                    focus_axes=[f"eixo {heading}"],
                    domain_terms=[heading.lower()],
                    Analitico=f"Estruture {heading}.",
                    Critico=f"Questione {heading}.",
                    Estrategico=f"Amplie a relevancia de {heading}.",
                ),
            )
        raise AssertionError(f"Schema inesperado: {schema_model}")


class FakeSearchBackend:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, *, max_results: int) -> SearchResult:
        self.queries.append(query)
        return SearchResult(
            query=query,
            summary=f"Resumo para {query}",
            sources=[
                {
                    "title": f"Fonte {query}",
                    "source": "Instituto Teste",
                    "url": f"https://example.com/{query.replace(' ', '-')}",
                    "snippet": "Trecho relevante.",
                }
            ],
        )


class FakeResearchSummarizer:
    def summarize(self, *, topic: str, **_: object) -> ResearchAnalysisPayload:
        return ResearchAnalysisPayload(
            research_summary=f"Resumo consolidado: {topic}",
            notes=[
                ResearchNote(
                    title=f"Nota {topic}",
                    source="Instituto Teste",
                    url=f"https://example.com/{topic.replace(' ', '-')}",
                    summary=f"Nota focada em {topic}.",
                    relevance=0.9,
                )
            ],
            evidence_is_sufficient=True,
            evidence_confidence=0.9,
            evidence_gaps=[],
            follow_up_queries=[],
        )


class FakeOutlineGenerator:
    def build(self, **_: object):
        payload = OutlinePayload(
            headline="Energia Limpa em Perspectiva Sistematica",
            editorial_angle="Artigo tecnico com foco em transicao, regulacao e limitacoes.",
            sections=[
                OutlineSection(
                    heading="Resumo",
                    purpose="Sintetizar objetivo e achados centrais.",
                    bullets=["Panorama do tema"],
                ),
                OutlineSection(
                    heading="Introducao",
                    purpose="Delimitar problema e relevancia.",
                    bullets=["Contexto regulatorio", "Motivacao do artigo"],
                ),
                OutlineSection(
                    heading="Discussao",
                    purpose="Aprofundar evidencias e contrapontos.",
                    bullets=["Trade-offs", "Escalabilidade"],
                ),
            ],
        )
        return payload, OutlineGenerator.render(payload)


class FakeDebateEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        topic = str(kwargs["topic"])
        return {
            "transcript": [],
            "summary": f"Debate concluido sobre {topic}.",
            "positions": {
                "Analitico": f"Estrutura para {topic}.",
                "Critico": f"Criticas para {topic}.",
                "Estrategico": f"Contexto para {topic}.",
            },
            "completed_rounds": int(kwargs.get("rounds", 3)),
            "needs_more_rounds": False,
            "open_questions": [],
        }


class FakeArticleWriter:
    def __init__(self) -> None:
        self.write_calls: list[str] = []
        self.review_calls: list[str] = []

    def write_section(self, *, section: SectionUnit, **_: object) -> str:
        self.write_calls.append(section.heading)
        return f"## {section.heading}\n\nVersao {section.retry_count} de {section.heading}."

    def review_section(self, *, section: SectionUnit, **_: object) -> SectionReviewPayload:
        self.review_calls.append(section.heading)
        if section.heading == "Introducao" and section.retry_count == 0:
            return SectionReviewPayload(
                review_summary="Introducao ainda superficial.",
                strengths=["Estrutura inicial visivel."],
                weaknesses=["Pouca profundidade analitica."],
                revision_requirements=["Adicionar evidencias e delimitar melhor o escopo."],
                prompt_improvements=["Reforcar causalidade e contexto regulatorio."],
                needs_revision=True,
                quality_score=0.45,
            )
        return SectionReviewPayload(
            review_summary=f"{section.heading} aprovada.",
            strengths=["Aderente ao outline."],
            weaknesses=[],
            revision_requirements=[],
            prompt_improvements=[],
            needs_revision=False,
            quality_score=0.9,
        )

    def assemble_article(
        self,
        *,
        output_mode: str,
        headline: str,
        editorial_angle: str,
        sections: list[SectionUnit],
        references: list[SourceReference],
        quality_alerts: list[dict[str, object]],
        **kwargs: object,
    ) -> str:
        if output_mode != "article":
            return "# Relatorio"
        lines = [f"# {headline}", "", f"Angulo editorial: {editorial_angle}", ""]
        for section in sections:
            lines.append(section.draft_md.strip())
            lines.append("")
        if quality_alerts:
            lines.append("## Alertas de Qualidade")
            lines.append("")
        lines.append("## Referencias")
        lines.append("")
        for reference in references:
            lines.append(f"- {reference.title} | {reference.url}")
        return "\n".join(lines).strip()


def _build_deps(tmp_dir: Path) -> tuple[WorkflowDependencies, FakeSearchBackend, FakeDebateEngine, FakeArticleWriter]:
    search_backend = FakeSearchBackend()
    debate_engine = FakeDebateEngine()
    article_writer = FakeArticleWriter()
    deps = WorkflowDependencies(
        llm_client=FakeLLMClient(),
        search_backend=search_backend,
        research_summarizer=FakeResearchSummarizer(),
        debate_engine=debate_engine,  # type: ignore[arg-type]
        outline_generator=FakeOutlineGenerator(),  # type: ignore[arg-type]
        article_writer=article_writer,  # type: ignore[arg-type]
        markdown_saver=MarkdownSaver(),
        settings=WorkflowSettings(
            default_output_path=str(tmp_dir / "artigo.md"),
            research_max_sources=3,
            debate_min_rounds=3,
            debate_max_rounds=5,
            section_retry_max=3,
            article_profile="academic_rigid",
            article_min_words=3000,
            evidence_policy="abort",
        ),
    )
    return deps, search_backend, debate_engine, article_writer


def test_route_after_section_init_and_review_cover_retry_paths():
    assert route_after_section_init({"output_mode": "article", "section_units": [{"status": "pending"}]}) == "section_write_node"
    assert route_after_section_init({"output_mode": "insufficiency_report", "section_units": []}) == "article_assembly_node"

    assert (
        route_after_section_review(
            {
                "output_mode": "article",
                "current_section_index": 0,
                "section_units": [{"status": "needs_retry"}],
            }
        )
        == "section_research_node"
    )
    assert (
        route_after_section_review(
            {
                "output_mode": "article",
                "current_section_index": 1,
                "section_units": [{"status": "approved"}, {"status": "pending"}],
            }
        )
        == "section_write_node"
    )
    assert (
        route_after_section_review(
            {
                "output_mode": "article",
                "current_section_index": 2,
                "section_units": [{"status": "approved"}, {"status": "accepted_with_warnings"}],
            }
        )
        == "article_assembly_node"
    )


def test_writer_section_prompt_includes_revision_feedback_and_assembly_adds_alerts():
    client = CaptureClient()
    writer = MarkdownArticleWriter(client)
    section = SectionUnit(
        id="introducao",
        heading="Introducao",
        purpose="Delimitar o problema.",
        bullets=["Contexto", "Escopo"],
        target_words=800,
        review_summary="A versao anterior ficou superficial.",
        revision_requirements=["Expandir a base empirica."],
        prompt_improvements=["Reforcar a delimitacao do escopo."],
    )
    markdown = writer.write_section(
        normalized_request="Artigo sobre seguranca alimentar.",
        article_goal="Produzir uma revisao rigorosa.",
        audience="Leitores especializados",
        tone="Formal",
        knowledge_domain="Saude publica",
        disciplinary_lens="Revisao interdisciplinar",
        evidence_requirements=["Cruzar revisoes e relatorios."],
        constraints=["Evitar simplificacoes."],
        article_profile="academic_rigid",
        minimum_words=6000,
        headline="Seguranca Alimentar",
        editorial_angle="Analise critica",
        section=section,
        section_index=1,
        total_sections=3,
        prior_sections=[],
        research_summary="Resumo da pesquisa.",
        debate_summary="Resumo do debate.",
        agent_positions={"Analitico": "Organizar o texto."},
        references=[SourceReference(title="Fonte", source="OMS", url="https://example.com")],
    )

    assembled = writer.assemble_article(
        output_mode="article",
        normalized_request="Artigo sobre seguranca alimentar.",
        article_goal="Produzir uma revisao rigorosa.",
        headline="Seguranca Alimentar",
        editorial_angle="Analise critica",
        sections=[section.model_copy(update={"draft_md": markdown, "status": "accepted_with_warnings"})],
        references=[SourceReference(title="Fonte", source="OMS", url="https://example.com")],
        evidence_confidence=0.7,
        evidence_gaps=[],
        follow_up_queries=[],
        research_summary="Resumo da pesquisa.",
        quality_alerts=[
            {
                "heading": "Introducao",
                "summary": "Ainda faltam nuances metodologicas.",
                "pending": ["Detalhar limitacoes das fontes."],
            }
        ],
    )

    assert "Feedback obrigatorio da revisao anterior" in client.last_prompt
    assert "Expandir a base empirica." in client.last_prompt
    assert markdown.startswith("## Introducao")
    assert "## Alertas de Qualidade" in assembled
    assert assembled.count("## Referencias") == 1


def test_section_graph_retries_only_failed_section_and_saves_article():
    tmp_path = Path("outputs") / f"test_graph_section_{uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        deps, search_backend, debate_engine, article_writer = _build_deps(tmp_path)
        workflow = build_graph(deps)

        final_state = workflow.invoke(
            {
                "user_request": "Crie um artigo sobre energia limpa.",
                "output_path": str(tmp_path / "artigo_final.md"),
                "errors": [],
                "metadata": {},
            }
        )

        output = Path(final_state["output_path"])
        assert output.exists()
        markdown = output.read_text(encoding="utf-8")
        assert "# Energia Limpa em Perspectiva Sistematica" in markdown
        assert "## Resumo" in markdown
        assert "## Introducao" in markdown
        assert "## Discussao" in markdown
        assert article_writer.write_calls == ["Resumo", "Introducao", "Introducao", "Discussao"]
        assert article_writer.review_calls == ["Resumo", "Introducao", "Introducao", "Discussao"]
        assert search_backend.queries[:2] == ["energia limpa revisao", "energia limpa regulacao"]
        assert any("Introducao evidencia 1" in query for query in search_backend.queries)
        assert any("Mini debate sobre Introducao." in str(call["topic"]) for call in debate_engine.calls)
        assert "## Referencias" in markdown
    finally:
        rmtree(tmp_path, ignore_errors=True)
