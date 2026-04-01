"""Section lifecycle helpers isolated from the graph nodes."""

from __future__ import annotations

import re

from app.llm.models import OutlinePayload, SectionReviewPayload
from app.workflow.models import QualityAlert, SectionState
from app.writer.formatter import MarkdownFormatter

_KEYWORDS_MARKERS = ("palavras-chave", "palavras chave", "keywords")
_ABSTRACT_MARKERS = ("resumo", "abstract")


class SectionStateService:
    """Manage section initialization, lookup and review decisions."""

    def __init__(self, formatter: MarkdownFormatter) -> None:
        self._formatter = formatter

    def initialize_from_outline(
        self,
        outline_payload: OutlinePayload,
        *,
        minimum_words: int,
    ) -> list[SectionState]:
        kinds = [self._classify_section_kind(item.heading) for item in outline_payload.sections]
        short_targets = [
            self._short_target_for_heading(item.heading) if kind == "short_form" else 0
            for item, kind in zip(outline_payload.sections, kinds, strict=False)
        ]
        standard_count = sum(1 for kind in kinds if kind == "standard")
        remaining_budget = max(minimum_words - sum(short_targets), 0)
        standard_target = max(450, remaining_budget // max(standard_count, 1)) if standard_count else 0

        sections: list[SectionState] = []
        seen_ids: set[str] = set()
        for index, (section, kind) in enumerate(zip(outline_payload.sections, kinds, strict=False), start=1):
            section_id = self._unique_section_id(self._slugify(section.heading), seen_ids, index)
            target_words = (
                self._short_target_for_heading(section.heading)
                if kind == "short_form"
                else standard_target
            )
            sections.append(
                SectionState(
                    id=section_id,
                    heading=section.heading,
                    purpose=section.purpose,
                    bullets=section.bullets,
                    kind=kind,
                    target_words=target_words,
                )
            )
        return sections

    def locate_active(
        self,
        sections: list[SectionState],
        current_index: int,
    ) -> tuple[int, SectionState] | None:
        if current_index < len(sections) and sections[current_index].status not in {
            "approved",
            "accepted_with_warnings",
        }:
            return current_index, sections[current_index]

        next_index = self.next_pending_index(sections, current_index)
        if next_index >= len(sections):
            return None
        return next_index, sections[next_index]

    def next_pending_index(self, sections: list[SectionState], start: int) -> int:
        for index in range(max(0, start), len(sections)):
            if sections[index].status not in {"approved", "accepted_with_warnings"}:
                return index
        return len(sections)

    def approved_section_summaries(
        self,
        sections: list[SectionState],
        *,
        before_index: int,
        limit: int = 220,
    ) -> list[str]:
        summaries: list[str] = []
        for item in sections[:before_index]:
            if item.status not in {"approved", "accepted_with_warnings"} or not item.draft_md:
                continue
            summaries.append(
                f"{item.heading}: {self._formatter.preview_markdown(item.draft_md, limit=limit)}"
            )
        return summaries

    def apply_review_result(
        self,
        *,
        sections: list[SectionState],
        index: int,
        review: SectionReviewPayload,
        retry_max: int,
        quality_alerts: list[QualityAlert],
    ) -> tuple[list[SectionState], int, list[QualityAlert]]:
        section = sections[index]
        section.review_summary = review.review_summary
        section.strengths = review.strengths
        section.weaknesses = review.weaknesses
        section.revision_requirements = review.revision_requirements
        section.prompt_improvements = review.prompt_improvements

        next_index = index
        if not review.needs_revision:
            section.status = "approved"
            next_index = self.next_pending_index(sections, index + 1)
        elif section.retry_count < retry_max:
            section.retry_count += 1
            section.status = "needs_retry"
        else:
            section.status = "accepted_with_warnings"
            next_index = self.next_pending_index(sections, index + 1)
            quality_alerts = self._upsert_quality_alert(
                quality_alerts,
                heading=section.heading,
                summary=review.review_summary,
                pending=review.revision_requirements or review.weaknesses,
            )

        sections[index] = section
        return sections, next_index, quality_alerts

    @staticmethod
    def _classify_section_kind(heading: str) -> str:
        normalized = heading.strip().lower()
        if any(marker in normalized for marker in _KEYWORDS_MARKERS + _ABSTRACT_MARKERS):
            return "short_form"
        return "standard"

    @staticmethod
    def _short_target_for_heading(heading: str) -> int:
        normalized = heading.strip().lower()
        if any(marker in normalized for marker in _KEYWORDS_MARKERS):
            return 40
        return 220

    @staticmethod
    def _unique_section_id(base: str, seen_ids: set[str], index: int) -> str:
        candidate = base or f"secao-{index}"
        suffix = 1
        unique = candidate
        while unique in seen_ids:
            suffix += 1
            unique = f"{candidate}-{suffix}"
        seen_ids.add(unique)
        return unique

    @staticmethod
    def _slugify(value: str) -> str:
        text = value.strip().lower()
        text = re.sub(r"[^a-z0-9]+", "-", text)
        return text.strip("-")

    @staticmethod
    def _upsert_quality_alert(
        quality_alerts: list[QualityAlert],
        *,
        heading: str,
        summary: str,
        pending: list[str],
    ) -> list[QualityAlert]:
        updated: list[QualityAlert] = []
        replaced = False
        for alert in quality_alerts:
            if alert.heading == heading:
                updated.append(QualityAlert(heading=heading, summary=summary, pending=pending))
                replaced = True
            else:
                updated.append(alert)
        if not replaced:
            updated.append(QualityAlert(heading=heading, summary=summary, pending=pending))
        return updated
