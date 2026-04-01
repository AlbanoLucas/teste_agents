"""Base class for graph handlers with typed state adaptation."""

from __future__ import annotations

from app.graph.services import WorkflowServices
from app.graph.state import ProjectState


class BaseHandler:
    """Wrap a typed handler behind the flat LangGraph state contract."""

    node_name: str = "base"

    def __init__(self, services: WorkflowServices) -> None:
        self.services = services

    def __call__(self, state: ProjectState) -> ProjectState:
        envelope = self.services.state_adapter.from_graph_state(state)
        try:
            updated = self.run(envelope)
        except Exception as exc:
            updated = self.services.error_policy.handle_node_failure(
                envelope,
                node=self.node_name,
                exc=exc,
            )
        return self.services.state_adapter.to_graph_state(updated)

    def run(self, envelope):  # pragma: no cover - interface only.
        raise NotImplementedError
