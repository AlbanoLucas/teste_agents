"""Conditional edge logic for the LangGraph workflow."""

from __future__ import annotations

from app.graph.state import ProjectState


def route_after_fixed(next_node: str):
    """Return a router that forwards to a fixed node unless the workflow failed."""

    def router(state: ProjectState) -> str:
        workflow = _workflow(state)
        if workflow.get("status") == "failed":
            return "failure_node"
        return next_node

    return router


def route_after_research(state: ProjectState) -> str:
    """Decide whether the flow can continue to debate or must stop at a report."""

    workflow = _workflow(state)
    if workflow.get("status") == "failed":
        return "failure_node"
    if workflow.get("output_mode", "article") == "article":
        return "debate_node"
    return "article_assembly_node"


def route_after_debate(state: ProjectState) -> str:
    """Decide whether the graph should continue debating or move to synthesis."""

    workflow = _workflow(state)
    if workflow.get("status") == "failed":
        return "failure_node"

    debate = workflow.get("debate", {})
    completed = debate.get("completed_rounds", 0)
    maximum = debate.get("max_rounds", 5)
    needs_more = debate.get("needs_more_rounds", False)

    if needs_more and completed < maximum:
        return "debate_node"
    return "synthesis_node"


def route_after_section_init(state: ProjectState) -> str:
    """Start section drafting only when the outline produced section units."""

    workflow = _workflow(state)
    if workflow.get("status") == "failed":
        return "failure_node"
    if workflow.get("output_mode") != "article":
        return "article_assembly_node"
    if workflow.get("sections"):
        return "section_write_node"
    return "article_assembly_node"


def route_after_section_write(state: ProjectState) -> str:
    """Proceed to review unless the current write failed."""

    workflow = _workflow(state)
    if workflow.get("status") == "failed":
        return "failure_node"
    return "section_review_node"


def route_after_section_review(state: ProjectState) -> str:
    """Decide whether to retry the current section, move on, or assemble the article."""

    workflow = _workflow(state)
    if workflow.get("status") == "failed":
        return "failure_node"
    if workflow.get("output_mode") != "article":
        return "article_assembly_node"

    sections = workflow.get("sections", [])
    if not sections:
        return "article_assembly_node"

    current_index = workflow.get("current_section_index", 0)
    if current_index < len(sections) and sections[current_index].get("status") == "needs_retry":
        return "section_research_node"

    pending = any(
        section.get("status") not in {"approved", "accepted_with_warnings"}
        for section in sections
    )
    if pending:
        return "section_write_node"
    return "article_assembly_node"


def route_after_section_research(state: ProjectState) -> str:
    """Proceed from focused research to the real mini-debate unless the workflow failed."""

    workflow = _workflow(state)
    if workflow.get("status") == "failed":
        return "failure_node"
    return "section_debate_node"


def route_after_section_debate(state: ProjectState) -> str:
    """Proceed from the mini-debate to section rewriting unless the workflow failed."""

    workflow = _workflow(state)
    if workflow.get("status") == "failed":
        return "failure_node"
    return "section_write_node"


def route_after_article_assembly(state: ProjectState) -> str:
    """Proceed to persistence unless assembly failed."""

    workflow = _workflow(state)
    if workflow.get("status") == "failed":
        return "failure_node"
    return "save_file_node"


def _workflow(state: ProjectState) -> dict:
    return dict(state.get("workflow", {}))
