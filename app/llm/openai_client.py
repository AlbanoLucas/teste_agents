"""Reusable OpenAI Responses API client with prompt envelopes and retry policy."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from app.workflow.error_policy import RetryableLLMError, RetryableSearchError, SchemaValidationError
from app.workflow.models import PromptEnvelope

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OperationPolicy:
    """Runtime policy for a specific OpenAI operation type."""

    timeout_seconds: float
    max_retries: int
    category: Literal["retryable_llm", "retryable_search"]


class OpenAIResponsesClient:
    """Thin adapter over the OpenAI Responses API."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
        client: Any | None = None,
        operation_policies: dict[str, OperationPolicy] | None = None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.temperature = temperature
        self._client = client or self._build_client(api_key=api_key)
        self._operation_policies = operation_policies or {
            "text": OperationPolicy(timeout_seconds=45.0, max_retries=2, category="retryable_llm"),
            "structured": OperationPolicy(timeout_seconds=45.0, max_retries=2, category="retryable_llm"),
            "web_search": OperationPolicy(timeout_seconds=30.0, max_retries=3, category="retryable_search"),
        }

    def generate_text(
        self,
        *,
        envelope: PromptEnvelope | None = None,
        instructions: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate free-form text for synthesis or article writing."""

        prompt_envelope = envelope or PromptEnvelope(
            instructions=instructions or "",
            prompt=prompt or "",
            operation_name="text_generation",
        )
        policy = self._operation_policies["text"]
        return self._run_with_retry(
            operation_name=prompt_envelope.operation_name,
            policy=policy,
            error_factory=RetryableLLMError,
            fn=lambda: self._extract_output_text(
                self._client.responses.create(
                    model=self.model,
                    temperature=self.temperature if temperature is None else temperature,
                    input=self._build_input(prompt_envelope.instructions, prompt_envelope.prompt),
                    timeout=policy.timeout_seconds,
                )
            ).strip(),
        )

    def generate_structured(
        self,
        *,
        schema_model: type[StructuredModelT],
        envelope: PromptEnvelope | None = None,
        instructions: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
    ) -> StructuredModelT:
        """Generate a structured payload validated against a Pydantic model."""

        prompt_envelope = envelope or PromptEnvelope(
            instructions=instructions or "",
            prompt=prompt or "",
            operation_name=f"structured:{schema_model.__name__}",
        )
        policy = self._operation_policies["structured"]
        response = self._run_with_retry(
            operation_name=prompt_envelope.operation_name,
            policy=policy,
            error_factory=RetryableLLMError,
            fn=lambda: self._create_structured_response(
                instructions=prompt_envelope.instructions,
                prompt=prompt_envelope.prompt,
                schema_model=schema_model,
                temperature=temperature,
                timeout_seconds=policy.timeout_seconds,
            ),
        )
        try:
            payload = self._extract_structured_payload(response)
            return schema_model.model_validate(payload)
        except (ValidationError, ValueError) as exc:
            raise SchemaValidationError(
                str(exc),
                operation_name=prompt_envelope.operation_name,
            ) from exc

    def web_search(
        self,
        *,
        query: str,
        instructions: str,
        max_results: int = 5,
        operation_name: str = "web_search",
    ) -> dict[str, Any]:
        """Run OpenAI web search and return a normalized response payload."""

        policy = self._operation_policies["web_search"]
        response = self._run_with_retry(
            operation_name=operation_name,
            policy=policy,
            error_factory=RetryableSearchError,
            fn=lambda: self._client.responses.create(
                model=self.model,
                temperature=0,
                input=self._build_input(instructions, query),
                tools=[{"type": "web_search"}],
                include=["web_search_call.action.sources"],
                timeout=policy.timeout_seconds,
            ),
        )
        return {
            "query": query,
            "summary": self._extract_output_text(response).strip(),
            "sources": self._extract_sources(response)[:max_results],
        }

    @staticmethod
    def _build_client(*, api_key: str | None) -> Any:
        from openai import OpenAI

        return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    @staticmethod
    def _build_input(instructions: str, prompt: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": instructions}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ]

    def _run_with_retry(
        self,
        *,
        operation_name: str,
        policy: OperationPolicy,
        error_factory: type[RetryableLLMError] | type[RetryableSearchError],
        fn: Any,
    ) -> Any:
        attempts = policy.max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                logger.debug(
                    "OpenAI operation started",
                    extra={
                        "operation_name": operation_name,
                        "attempt": attempt,
                        "timeout_seconds": policy.timeout_seconds,
                    },
                )
                return fn()
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "OpenAI operation failed",
                    extra={
                        "operation_name": operation_name,
                        "attempt": attempt,
                        "max_attempts": attempts,
                    },
                    exc_info=attempt == attempts,
                )
                if attempt >= attempts:
                    raise error_factory(
                        str(exc),
                        operation_name=operation_name,
                        attempt=attempt,
                    ) from exc
        raise error_factory(
            str(last_exc or "OpenAI operation failed"),
            operation_name=operation_name,
            attempt=attempts,
        )

    def _create_structured_response(
        self,
        *,
        instructions: str,
        prompt: str,
        schema_model: type[StructuredModelT],
        temperature: float | None = None,
        timeout_seconds: float,
    ) -> Any:
        responses_api = self._client.responses
        parse = getattr(responses_api, "parse", None)

        if callable(parse):
            try:
                return parse(
                    model=self.model,
                    temperature=self.temperature if temperature is None else temperature,
                    input=self._build_input(instructions, prompt),
                    text_format=schema_model,
                    timeout=timeout_seconds,
                )
            except TypeError:
                logger.debug(
                    "responses.parse nao suportado pelo client atual; voltando para responses.create.",
                    exc_info=True,
                )

        return responses_api.create(
            model=self.model,
            temperature=self.temperature if temperature is None else temperature,
            input=self._build_input(instructions, prompt),
            timeout=timeout_seconds,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_model.__name__,
                    "strict": True,
                    "schema": self._build_strict_schema(schema_model),
                }
            },
        )

    @classmethod
    def _build_strict_schema(cls, schema_model: type[StructuredModelT]) -> dict[str, Any]:
        """Convert a Pydantic schema into the stricter subset accepted by OpenAI."""

        schema = schema_model.model_json_schema()
        return cls._normalize_schema_node(schema)

    @classmethod
    def _normalize_schema_node(cls, node: Any) -> Any:
        """Recursively force OpenAI Structured Outputs compatibility."""

        if isinstance(node, dict):
            normalized = {
                key: cls._normalize_schema_node(value)
                for key, value in node.items()
                if key != "default"
            }

            if normalized.get("type") == "object":
                properties = normalized.get("properties", {})
                normalized["properties"] = properties
                normalized["required"] = list(properties.keys())
                normalized["additionalProperties"] = False

            if "$defs" in normalized and isinstance(normalized["$defs"], dict):
                normalized["$defs"] = {
                    key: cls._normalize_schema_node(value)
                    for key, value in normalized["$defs"].items()
                }

            if "items" in normalized:
                normalized["items"] = cls._normalize_schema_node(normalized["items"])

            if "anyOf" in normalized and isinstance(normalized["anyOf"], list):
                normalized["anyOf"] = [
                    cls._normalize_schema_node(item) for item in normalized["anyOf"]
                ]

            return normalized

        if isinstance(node, list):
            return [cls._normalize_schema_node(item) for item in node]

        return node

    def _extract_structured_payload(self, response: Any) -> Any:
        parsed = self._extract_parsed_output(response)
        if parsed is not None:
            return parsed

        text = self._extract_json_text(response)
        return self._load_json_payload(text)

    def _extract_parsed_output(self, response: Any) -> Any:
        for candidate in (
            getattr(response, "output_parsed", None),
            getattr(response, "parsed", None),
        ):
            normalized = self._normalize_structured_candidate(candidate)
            if normalized is not None:
                return normalized

        output = getattr(response, "output", None) or []
        for item in output:
            contents = []
            if hasattr(item, "content"):
                contents = list(getattr(item, "content", []) or [])
            elif isinstance(item, dict):
                contents = list(item.get("content", []) or [])

            for content in contents:
                parsed = (
                    getattr(content, "parsed", None)
                    if not isinstance(content, dict)
                    else content.get("parsed")
                )
                normalized = self._normalize_structured_candidate(parsed)
                if normalized is not None:
                    return normalized

        return None

    @staticmethod
    def _normalize_structured_candidate(candidate: Any) -> Any:
        if candidate is None:
            return None
        if hasattr(candidate, "model_dump"):
            return candidate.model_dump()
        if isinstance(candidate, (dict, list)):
            return candidate
        return None

    def _load_json_payload(self, text: str) -> Any:
        candidates: list[str] = []
        raw = text.strip()
        snippet = self._extract_json_snippet(raw)
        for candidate in (raw, snippet):
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        errors: list[Exception] = []
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                errors.append(exc)

            repaired = self._repair_json_text(candidate)
            if repaired != candidate:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError as exc:
                    errors.append(exc)

            python_like = self._pythonize_json_literals(repaired)
            try:
                return ast.literal_eval(python_like)
            except (SyntaxError, ValueError) as exc:
                errors.append(exc)

        preview = raw.replace("\n", " ")[:280]
        last_error = errors[-1] if errors else None
        raise ValueError(
            "Nao foi possivel interpretar a resposta estruturada da OpenAI. "
            f"Trecho recebido: {preview}"
        ) from last_error

    @staticmethod
    def _extract_json_snippet(text: str) -> str:
        if not text:
            return ""

        starts = [index for index in (text.find("{"), text.find("[")) if index != -1]
        if not starts:
            return text

        start = min(starts)
        end_object = text.rfind("}")
        end_array = text.rfind("]")
        end = max(end_object, end_array)
        if end <= start:
            return text[start:]
        return text[start : end + 1]

    @staticmethod
    def _repair_json_text(text: str) -> str:
        repaired = text.strip()
        repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
        repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        return repaired

    @staticmethod
    def _pythonize_json_literals(text: str) -> str:
        result: list[str] = []
        token: list[str] = []
        quote_char = ""
        escaped = False

        def flush_token() -> None:
            if not token:
                return
            literal = "".join(token)
            replacements = {"true": "True", "false": "False", "null": "None"}
            result.append(replacements.get(literal, literal))
            token.clear()

        for char in text:
            if quote_char:
                result.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote_char:
                    quote_char = ""
                continue

            if char in {'"', "'"}:
                flush_token()
                quote_char = char
                result.append(char)
                continue

            if char.isalpha():
                token.append(char)
                continue

            flush_token()
            result.append(char)

        flush_token()
        return "".join(result)

    def _extract_json_text(self, response: Any) -> str:
        text = self._extract_output_text(response).strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.startswith("```")]
            text = "\n".join(lines).strip()
        return text

    def _extract_output_text(self, response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return str(text)

        output = getattr(response, "output", None)
        if not output:
            return ""

        chunks: list[str] = []
        for item in output:
            if hasattr(item, "content"):
                for content in getattr(item, "content", []):
                    chunk = getattr(content, "text", None)
                    if chunk:
                        chunks.append(str(chunk))
            elif isinstance(item, dict):
                for content in item.get("content", []):
                    chunk = content.get("text")
                    if chunk:
                        chunks.append(str(chunk))
        return "\n".join(chunks)

    def _extract_sources(self, response: Any) -> list[dict[str, str]]:
        sources: list[dict[str, str]] = []
        seen: set[str] = set()

        def visit(value: Any) -> None:
            if value is None:
                return
            if hasattr(value, "model_dump"):
                visit(value.model_dump())
                return
            if isinstance(value, dict):
                nested_sources = value.get("sources")
                if isinstance(nested_sources, list):
                    for source in nested_sources:
                        if not isinstance(source, dict):
                            continue
                        url = str(source.get("url", "")).strip()
                        if not url or url in seen:
                            continue
                        seen.add(url)
                        sources.append(
                            {
                                "title": str(source.get("title", source.get("name", url))),
                                "source": str(source.get("source", source.get("domain", "web"))),
                                "url": url,
                                "snippet": str(source.get("snippet", source.get("text", ""))),
                            }
                        )
                for nested in value.values():
                    visit(nested)
                return
            if isinstance(value, list):
                for nested in value:
                    visit(nested)
                return
            if hasattr(value, "__dict__"):
                visit(vars(value))

        visit(response)
        return sources
