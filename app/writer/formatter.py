"""Markdown normalization helpers shared by writer components."""

from __future__ import annotations

import re


class MarkdownFormatter:
    """Normalize section markdown and final reference blocks."""

    def normalize_section(self, *, heading: str, kind: str, markdown: str) -> str:
        cleaned = self.strip_references_block(markdown.strip())
        heading_md = f"## {heading}"
        if kind == "short_form" and "palavras-chave" in heading.lower():
            body = cleaned.replace(heading_md, "").strip()
            body = re.sub(r"(?i)^palavras-chave\s*:?\s*", "", body).strip()
            return f"{heading_md}\n{body or 'termos a definir'}".strip()
        if cleaned.startswith(heading_md):
            return cleaned
        body = cleaned.lstrip("#").strip()
        return f"{heading_md}\n\n{body}".strip()

    def strip_references_block(self, markdown: str) -> str:
        return re.sub(
            r"\n## Referencias\b[\s\S]*$",
            "",
            markdown,
            flags=re.IGNORECASE,
        ).strip()

    def preview_markdown(self, markdown: str, *, limit: int = 220) -> str:
        compact = " ".join(markdown.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."
