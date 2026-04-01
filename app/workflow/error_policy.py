"""Workflow error classification, recording and fallback decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from app.workflow.models import WorkflowErrorRecord, WorkflowStateEnvelope

ErrorCategory = Literal["retryable_llm", "retryable_search", "schema_error", "terminal"]


class WorkflowOperationError(RuntimeError):
    """Base exception for classified workflow failures."""

    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory,
        operation_name: str,
        attempt: int = 1,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.operation_name = operation_name
        self.attempt = attempt


class RetryableLLMError(WorkflowOperationError):
    """LLM call failed after retry budget was exhausted."""

    def __init__(self, message: str, *, operation_name: str, attempt: int = 1) -> None:
        super().__init__(
            message,
            category="retryable_llm",
            operation_name=operation_name,
            attempt=attempt,
        )


class RetryableSearchError(WorkflowOperationError):
    """Search call failed after retry budget was exhausted."""

    def __init__(self, message: str, *, operation_name: str, attempt: int = 1) -> None:
        super().__init__(
            message,
            category="retryable_search",
            operation_name=operation_name,
            attempt=attempt,
        )


class SchemaValidationError(WorkflowOperationError):
    """Structured payload could not be validated."""

    def __init__(self, message: str, *, operation_name: str, attempt: int = 1) -> None:
        super().__init__(
            message,
            category="schema_error",
            operation_name=operation_name,
            attempt=attempt,
        )


@dataclass(slots=True)
class WorkflowErrorPolicy:
    """Classify, record and optionally downgrade workflow failures."""

    def classify(self, exc: Exception) -> tuple[ErrorCategory, str, int]:
        if isinstance(exc, WorkflowOperationError):
            return exc.category, exc.operation_name, exc.attempt
        if isinstance(exc, ValidationError):
            return "schema_error", "validation", 1
        return "terminal", "runtime", 1

    def record(
        self,
        envelope: WorkflowStateEnvelope,
        *,
        node: str,
        operation: str,
        category: ErrorCategory,
        decision: str,
        message: str,
        attempt: int = 1,
    ) -> WorkflowStateEnvelope:
        envelope.errors.append(
            WorkflowErrorRecord(
                node=node,
                operation=operation,
                category=category,
                decision=decision,
                message=message,
                attempt=attempt,
            )
        )
        return envelope

    def handle_node_failure(
        self,
        envelope: WorkflowStateEnvelope,
        *,
        node: str,
        exc: Exception,
    ) -> WorkflowStateEnvelope:
        category, operation, attempt = self.classify(exc)
        envelope = self.record(
            envelope,
            node=node,
            operation=operation,
            category=category,
            decision="raise",
            message=str(exc),
            attempt=attempt,
        )
        envelope.status = "failed"
        envelope.failed_node = node
        envelope.terminal_error = envelope.errors[-1]
        return envelope

    def record_fallback(
        self,
        envelope: WorkflowStateEnvelope,
        *,
        node: str,
        exc: Exception,
        decision: str,
    ) -> WorkflowStateEnvelope:
        category, operation, attempt = self.classify(exc)
        return self.record(
            envelope,
            node=node,
            operation=operation,
            category=category,
            decision=decision,
            message=str(exc),
            attempt=attempt,
        )
