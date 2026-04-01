"""Adapter between minimal graph state and typed workflow envelopes."""

from __future__ import annotations

from app.graph.state import ProjectState
from app.workflow.models import WorkflowStateEnvelope


class GraphStateAdapter:
    """Serialize the workflow envelope as the only graph payload."""

    def from_graph_state(self, state: ProjectState) -> WorkflowStateEnvelope:
        return WorkflowStateEnvelope.model_validate(state.get("workflow", {}))

    def to_graph_state(self, envelope: WorkflowStateEnvelope) -> ProjectState:
        return ProjectState(workflow=envelope.model_dump(mode="json"))
