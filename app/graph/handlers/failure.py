"""Failure handler that materializes workflow errors into Markdown."""

from __future__ import annotations

import logging

from app.graph.handlers.base import BaseHandler
from app.logging_utils import log_block, preview_text

logger = logging.getLogger(__name__)


class FailureHandler(BaseHandler):
    """Render a terminal workflow failure into the final Markdown artifact."""

    node_name = "failure_node"

    def run(self, envelope):
        error = envelope.terminal_error
        log_block(
            logger,
            "Etapa: failure_node",
            fields={
                "failed_node": envelope.failed_node or "(desconhecido)",
                "categoria": error.category if error else "(sem categoria)",
            },
            body=preview_text(error.message if error else "Falha sem detalhes.", limit=320),
            level=logging.ERROR,
        )

        lines = [
            "# Relatorio de Falha do Workflow",
            "",
            "O fluxo de geracao foi interrompido por uma falha terminal e nao concluiu o artefato final esperado.",
            "",
            "## Resumo da falha",
            "",
            f"- Status: {envelope.status}",
            f"- Node com falha: {envelope.failed_node or '(desconhecido)'}",
        ]
        if error is not None:
            lines.extend(
                [
                    f"- Operacao: {error.operation}",
                    f"- Categoria: {error.category}",
                    f"- Tentativa: {error.attempt}",
                    "",
                    "## Mensagem principal",
                    "",
                    error.message,
                ]
            )

        if envelope.errors:
            lines.extend(["", "## Historico de erros", ""])
            for item in envelope.errors:
                lines.append(
                    f"- `{item.node}` | `{item.operation}` | `{item.category}` | decisao `{item.decision}` | tentativa {item.attempt}: {item.message}"
                )

        envelope.final_article_md = "\n".join(lines).strip()
        return envelope
