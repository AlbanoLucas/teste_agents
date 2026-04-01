"""Persistence helpers for Markdown output."""

from __future__ import annotations

from pathlib import Path


class MarkdownSaver:
    """Save the final Markdown artifact to disk."""

    def save(self, *, markdown: str, output_path: str) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        return path
