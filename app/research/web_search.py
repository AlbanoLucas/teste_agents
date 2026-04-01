"""Web search backend abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.llm.openai_client import OpenAIResponsesClient


@dataclass(slots=True)
class SearchResult:
    """Normalized result returned by a search backend."""

    query: str
    summary: str
    sources: list[dict[str, str]]


class SearchBackend(Protocol):
    """Contract for future search backends."""

    def search(self, query: str, *, max_results: int) -> SearchResult:
        """Run a search query and return a normalized result."""


class OpenAIWebSearchBackend:
    """OpenAI Responses API implementation for web search."""

    def __init__(self, client: OpenAIResponsesClient) -> None:
        self._client = client

    def search(self, query: str, *, max_results: int) -> SearchResult:
        payload = self._client.web_search(
            query=query,
            instructions=(
                "Pesquise na web fatos atuais e relevantes para apoiar a escrita de um artigo academico. "
                "Priorize fontes institucionais, revisoes, orgaos oficiais, publicacoes especializadas "
                "e materiais adequados ao dominio do tema. Resuma os pontos centrais e retorne resultados com links."
            ),
            max_results=max_results,
            operation_name=f"web_search:{query[:60]}",
        )
        return SearchResult(
            query=payload["query"],
            summary=payload["summary"],
            sources=payload["sources"],
        )
