"""Conditional edge logic for the LangGraph workflow."""

from __future__ import annotations

from app.graph.state import ProjectState


def route_after_research(state: ProjectState) -> str:
    """Decide whether the flow can continue to debate or must stop at a report."""

    if state.get("output_mode", "article") == "article":
        return "debate_node"
    return "article_assembly_node"


def route_after_debate(state: ProjectState) -> str:
    """Decide whether the graph should continue debating or move to synthesis."""

    completed = state.get("debate_completed_rounds", 0)
    maximum = state.get("debate_max_rounds", 5)
    needs_more = state.get("debate_needs_more_rounds", False)

    if needs_more and completed < maximum:
        return "debate_node"
    return "synthesis_node"


def route_after_section_init(state: ProjectState) -> str:
    """Start section drafting only when the outline produced section units."""

    if state.get("output_mode") != "article":
        return "article_assembly_node"
    if state.get("section_units"):
        return "section_write_node"
    return "article_assembly_node"


def route_after_section_review(state: ProjectState) -> str:
    """Decide whether to retry the current section, move on, or assemble the article."""

    if state.get("output_mode") != "article":
        return "article_assembly_node"

    sections = state.get("section_units", [])
    if not sections:
        return "article_assembly_node"

    current_index = state.get("current_section_index", 0)
    if current_index < len(sections) and sections[current_index].get("status") == "needs_retry":
        return "section_research_node"

    pending = any(
        section.get("status") not in {"approved", "accepted_with_warnings"}
        for section in sections
    )
    if pending:
        return "section_write_node"
    return "article_assembly_node"
