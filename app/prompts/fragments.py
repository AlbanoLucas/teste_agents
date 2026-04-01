"""Reusable prompt rendering helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def render_header_block(title: str, body: str | None) -> str:
    """Render a titled text block, omitting empty content."""

    content = (body or "").strip()
    if not content:
        return ""
    return f"{title}:\n{content}"


def render_list_block(title: str, items: Sequence[str], *, limit: int | None = None) -> str:
    """Render a titled bullet list, omitting empty items and empty blocks."""

    cleaned = _clean_text_items(items, limit=limit)
    if not cleaned:
        return ""
    return f"{title}:\n" + "\n".join(f"- {item}" for item in cleaned)


def render_references_block(title: str, references: Sequence[Any], *, limit: int = 8) -> str:
    """Render references in a compact stable format."""

    lines: list[str] = []
    for reference in list(references)[:limit]:
        title_value = str(getattr(reference, "title", "")).strip()
        source_value = str(getattr(reference, "source", "")).strip()
        url_value = str(getattr(reference, "url", "")).strip()
        if not (title_value or source_value or url_value):
            continue
        lines.append(" | ".join(part for part in [title_value, source_value, url_value] if part))
    return render_list_block(title, lines)


def render_agent_positions_block(
    title: str,
    positions: Mapping[str, str] | Any,
) -> str:
    """Render only non-empty agent positions."""

    mapping = positions.as_dict() if hasattr(positions, "as_dict") else dict(positions or {})
    lines = [f"{agent}: {position}" for agent, position in mapping.items() if str(position).strip()]
    return render_list_block(title, lines)


def render_feedback_block(section: Any) -> str:
    """Render prior review feedback for a section."""

    summary = str(getattr(section, "review_summary", "") or "").strip()
    revision_requirements = list(getattr(section, "revision_requirements", []) or [])
    prompt_improvements = list(getattr(section, "prompt_improvements", []) or [])

    if not summary and not revision_requirements and not prompt_improvements:
        return render_list_block(
            "Feedback obrigatorio da revisao anterior",
            ["Nenhum feedback anterior. Produza a melhor versao inicial da secao."],
        )

    items = [f"Resumo da ultima revisao: {summary or '(sem resumo)'}"]
    items.extend(f"Requisito de revisao: {item}" for item in revision_requirements)
    items.extend(f"Ajuste de prompt recomendado: {item}" for item in prompt_improvements)
    return render_list_block("Feedback obrigatorio da revisao anterior", items)


def render_approved_sections_block(summaries: Sequence[str], *, limit: int = 3) -> str:
    """Render compact summaries of already approved sections."""

    cleaned = _clean_text_items(summaries, limit=limit)
    if not cleaned:
        cleaned = ["Nenhuma secao aprovada anteriormente."]
    return render_list_block("Resumo das secoes aprovadas", cleaned)


def join_prompt_blocks(*blocks: str) -> str:
    """Join non-empty blocks with blank lines."""

    cleaned = [block.strip() for block in blocks if block and block.strip()]
    return "\n\n".join(cleaned)


def _clean_text_items(items: Sequence[str], *, limit: int | None = None) -> list[str]:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if limit is not None:
        return cleaned[:limit]
    return cleaned
