"""LLM-backed section reviewer component."""

from __future__ import annotations

from app.llm.models import SectionReviewPayload
from app.llm.openai_client import OpenAIResponsesClient
from app.prompts.builders import build_section_review_prompt
from app.workflow.models import SectionReviewContext


class SectionReviewer:
    """Review a single section with a rubric adapted to its type."""

    def __init__(self, client: OpenAIResponsesClient) -> None:
        self._client = client

    def review(self, context: SectionReviewContext) -> SectionReviewPayload:
        envelope = build_section_review_prompt(context)
        return self._client.generate_structured(
            envelope=envelope,
            schema_model=SectionReviewPayload,
            temperature=0.2,
        )
