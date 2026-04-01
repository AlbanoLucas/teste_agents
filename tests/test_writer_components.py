from app.llm.models import ResearchNote, SectionReviewPayload, SourceReference
from app.workflow.models import ArticleAssemblyContext, QualityAlert, SectionState
from app.workflow.section_service import SectionStateService
from app.writer.article_assembler import ArticleAssembler
from app.writer.formatter import MarkdownFormatter


def test_article_assembler_renders_unique_references_and_quality_alerts() -> None:
    formatter = MarkdownFormatter()
    assembler = ArticleAssembler(formatter)
    references = [
        SourceReference(title="Fonte A", source="Org A", url="https://example.com/a"),
        SourceReference(title="Fonte A", source="Org A", url="https://example.com/a"),
        SourceReference(title="Fonte B", source="Org B", url="https://example.com/b"),
    ]
    context = ArticleAssemblyContext(
        output_mode="article",
        normalized_request="Tema de teste",
        article_goal="Explicar o tema",
        headline="Titulo Forte",
        editorial_angle="Angulo tecnico",
        sections=[
            SectionState(
                id="introducao",
                heading="Introducao",
                purpose="Apresentar o tema",
                draft_md="## Introducao\n\nTexto da introducao.\n\n## Referencias\n- nao deve ficar aqui",
            )
        ],
        references=references,
        quality_alerts=[
            QualityAlert(
                heading="Introducao",
                summary="Ainda falta contextualizacao historica.",
                pending=["Adicionar contexto historico", "Fortalecer framing inicial"],
            )
        ],
    )

    article = assembler.assemble(context)

    assert article.count("## Referencias") == 1
    assert article.count("https://example.com/a") == 1
    assert "## Alertas de Qualidade" in article
    assert "Pendencias remanescentes" in article
    assert "nao deve ficar aqui" not in article


def test_section_state_service_marks_warning_after_retry_budget_is_exhausted() -> None:
    service = SectionStateService(MarkdownFormatter())
    section = SectionState(
        id="discussao",
        heading="Discussao",
        purpose="Analisar limites e implicacoes.",
        draft_md="## Discussao\n\nTexto ainda fraco.",
        retry_count=1,
    )
    review = SectionReviewPayload(
        review_summary="A secao ainda esta rasa.",
        strengths=["Estrutura basica presente"],
        weaknesses=["Pouca profundidade"],
        revision_requirements=["Adicionar comparacao entre evidencias"],
        prompt_improvements=["Exigir contrapontos explicitos"],
        needs_revision=True,
        quality_score=0.45,
    )

    sections, next_index, alerts = service.apply_review_result(
        sections=[section],
        index=0,
        review=review,
        retry_max=1,
        quality_alerts=[],
    )

    assert sections[0].status == "accepted_with_warnings"
    assert next_index == 1
    assert len(alerts) == 1
    assert alerts[0].heading == "Discussao"
