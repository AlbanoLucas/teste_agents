"""Final Markdown assembly component."""

from __future__ import annotations

from app.workflow.models import ArticleAssemblyContext
from app.writer.formatter import MarkdownFormatter


class ArticleAssembler:
    """Assemble the final article or insufficiency report."""

    def __init__(self, formatter: MarkdownFormatter) -> None:
        self._formatter = formatter

    def assemble(self, context: ArticleAssemblyContext) -> str:
        if context.output_mode != "article":
            return self.render_insufficiency_report(context)

        lines = [f"# {context.headline}", "", f"Angulo editorial: {context.editorial_angle}", ""]
        for section in context.sections:
            cleaned = self._formatter.strip_references_block(section.draft_md.strip())
            if cleaned:
                lines.append(cleaned)
                lines.append("")

        if context.quality_alerts:
            lines.append("## Alertas de Qualidade")
            lines.append("")
            for alert in context.quality_alerts:
                pending_text = "; ".join(alert.pending) or "pendencias nao detalhadas"
                lines.append(
                    f"- **{alert.heading}**: {alert.summary} Pendencias remanescentes: {pending_text}."
                )
            lines.append("")

        if context.references:
            lines.append(self.render_references(context))

        return "\n".join(lines).strip()

    def render_references(self, context: ArticleAssemblyContext) -> str:
        lines = ["## Referencias", ""]
        seen: set[str] = set()
        for reference in context.references:
            if reference.url in seen:
                continue
            seen.add(reference.url)
            lines.append(f"- {reference.source}. {reference.title}. Disponivel em: {reference.url}")
        return "\n".join(lines)

    def render_insufficiency_report(self, context: ArticleAssemblyContext) -> str:
        lines = [
            "# Relatorio de Insuficiencia de Evidencia",
            "",
            f"Solicitacao: {context.normalized_request}",
            "",
            f"Objetivo editorial pretendido: {context.article_goal}",
            "",
            "## Diagnostico",
            "",
            (
                "A geracao do artigo academico foi interrompida porque a base pesquisada nao "
                "sustenta com seguranca um texto profissional, longo e bem embasado."
            ),
            f"Confianca estimada da evidencia disponivel: {context.evidence_confidence}",
            "",
            "## Resumo da pesquisa disponivel",
            "",
            context.research_summary or "Nao ha resumo suficiente para sustentar o artigo.",
            "",
            "## Lacunas criticas",
            "",
        ]
        for gap in context.evidence_gaps or ["As fontes coletadas nao cobrem o tema com rigor suficiente."]:
            lines.append(f"- {gap}")
        lines.extend(["", "## Proximas pesquisas recomendadas", ""])
        for query in context.follow_up_queries or ["Aprofundar a busca por revisoes e fontes institucionais."]:
            lines.append(f"- {query}")
        if context.references:
            lines.extend(["", self.render_references(context)])
        return "\n".join(lines).strip()
