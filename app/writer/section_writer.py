"""LLM-backed section drafting component."""

from __future__ import annotations

from app.llm.openai_client import OpenAIResponsesClient
from app.prompts.builders import build_section_draft_prompt
from app.workflow.models import SectionDraftContext
from app.writer.formatter import MarkdownFormatter


class SectionWriter:
    """Write a single article section from compact typed context."""

    def __init__(self, client: OpenAIResponsesClient, formatter: MarkdownFormatter) -> None:
        self._client = client
        self._formatter = formatter

    def write(self, context: SectionDraftContext) -> str:
        envelope = build_section_draft_prompt(context)
        draft = self._client.generate_text(
            envelope=envelope,
            temperature=0.3,
        )
        return self._formatter.normalize_section(
            heading=context.section.heading,
            kind=context.section.kind,
            markdown=draft,
        )
