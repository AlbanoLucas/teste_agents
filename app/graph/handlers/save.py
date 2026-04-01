"""Persistence handler for final markdown output."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from app.graph.handlers.base import BaseHandler
from app.logging_utils import log_block

logger = logging.getLogger(__name__)


class SaveHandler(BaseHandler):
    """Persist the final markdown artifact to disk."""

    node_name = "save_file_node"

    def run(self, envelope):
        log_block(
            logger,
            "Etapa: save_file_node",
            fields={"output_path": envelope.output_path},
        )
        path = self.services.markdown_saver.save(
            markdown=envelope.final_article_md,
            output_path=envelope.output_path or self.services.settings.default_output_path,
        )
        envelope.output_path = str(path)
        envelope.metadata["finished_at"] = datetime.now(UTC).isoformat()
        if envelope.status == "running":
            envelope.status = "completed"
        log_block(
            logger,
            "Arquivo salvo",
            fields={"output_path": envelope.output_path},
        )
        return envelope
