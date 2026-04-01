"""Normalization helpers for web research artifacts."""

from __future__ import annotations

from urllib.parse import urlparse

from app.llm.models import SourceReference
from app.research.web_search import SearchResult


def normalize_sources(results: list[SearchResult], *, max_results: int) -> list[SourceReference]:
    """Normalize and deduplicate sources from multiple search queries."""

    normalized: list[SourceReference] = []
    seen_urls: set[str] = set()

    for result in results:
        for source in result.sources:
            url = str(source.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            title = str(source.get("title", "")).strip() or url
            source_name = str(source.get("source", "")).strip() or _domain_from_url(url)
            snippet = str(source.get("snippet", "")).strip()
            normalized.append(
                SourceReference(
                    title=title,
                    source=source_name,
                    url=url,
                    snippet=snippet,
                )
            )
            if len(normalized) >= max_results:
                return normalized
    return normalized


def render_research_context(results: list[SearchResult], references: list[SourceReference]) -> str:
    """Create compact context text for research summarization."""

    blocks: list[str] = []
    for result in results:
        blocks.append(f"Query: {result.query}\nResumo da busca: {result.summary}")
    if references:
        blocks.append(
            "Fontes:\n"
            + "\n".join(
                f"- {item.title} | {item.source} | {item.url} | {item.snippet}"
                for item in references
            )
        )
    return "\n\n".join(blocks)


def _domain_from_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or "web"
