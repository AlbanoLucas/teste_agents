from types import SimpleNamespace

import pytest

from app.graph.handlers.base import BaseHandler
from app.graph.handlers.failure import FailureHandler
from app.llm.models import SectionReviewPayload
from app.llm.openai_client import OpenAIResponsesClient
from app.workflow.state_adapter import GraphStateAdapter
from app.workflow.error_policy import SchemaValidationError, WorkflowErrorPolicy
from app.workflow.models import PromptEnvelope, WorkflowStateEnvelope


class _FakeParsedResponse:
    output_parsed = {"review_summary": 123}


class _FakeResponsesApi:
    def parse(self, **_: object) -> _FakeParsedResponse:
        return _FakeParsedResponse()


class _FakeSdkClient:
    responses = _FakeResponsesApi()


def test_openai_client_raises_schema_validation_error_for_invalid_structured_payload() -> None:
    client = OpenAIResponsesClient(
        model="fake-model",
        api_key="fake-key",
        client=_FakeSdkClient(),
    )

    with pytest.raises(SchemaValidationError) as exc_info:
        client.generate_structured(
            schema_model=SectionReviewPayload,
            envelope=PromptEnvelope(
                operation_name="section_review:test",
                instructions="revise",
                prompt="texto",
            ),
        )

    assert exc_info.value.category == "schema_error"
    assert exc_info.value.operation_name == "section_review:test"


class _IdentityAdapter:
    def __init__(self, envelope: WorkflowStateEnvelope) -> None:
        self._envelope = envelope

    def from_graph_state(self, _: dict) -> WorkflowStateEnvelope:
        return self._envelope

    def to_graph_state(self, envelope: WorkflowStateEnvelope) -> dict:
        return {"workflow": envelope.model_dump(mode="json")}


class _BoomHandler(BaseHandler):
    node_name = "boom_node"

    def run(self, envelope: WorkflowStateEnvelope) -> WorkflowStateEnvelope:
        raise RuntimeError("falha terminal")


def test_terminal_error_marks_failed_state_and_records_error() -> None:
    envelope = WorkflowStateEnvelope()
    services = SimpleNamespace(
        state_adapter=_IdentityAdapter(envelope),
        error_policy=WorkflowErrorPolicy(),
    )
    handler = _BoomHandler(services)

    result = handler({})

    assert len(envelope.errors) == 1
    error = envelope.errors[0]
    assert error.node == "boom_node"
    assert error.category == "terminal"
    assert error.decision == "raise"
    assert envelope.status == "failed"
    assert envelope.failed_node == "boom_node"
    assert envelope.terminal_error == error
    assert result["workflow"]["status"] == "failed"


def test_graph_state_adapter_uses_only_nested_workflow_payload() -> None:
    adapter = GraphStateAdapter()
    envelope = WorkflowStateEnvelope(user_request="tema", output_path="outputs/teste.md")

    state = adapter.to_graph_state(envelope)

    assert set(state.keys()) == {"workflow"}
    assert state["workflow"]["user_request"] == "tema"
    assert "section_prompt_pack" not in str(state)


def test_failure_handler_materializes_terminal_error_as_markdown() -> None:
    envelope = WorkflowStateEnvelope(
        status="failed",
        failed_node="research_node",
        terminal_error=None,
    )
    error_policy = WorkflowErrorPolicy()
    envelope = error_policy.handle_node_failure(envelope, node="research_node", exc=RuntimeError("busca quebrou"))
    services = SimpleNamespace(
        state_adapter=_IdentityAdapter(envelope),
        error_policy=error_policy,
    )
    handler = FailureHandler(services)

    result = handler({})

    assert "Relatorio de Falha do Workflow" in result["workflow"]["final_article_md"]
    assert "research_node" in result["workflow"]["final_article_md"]
