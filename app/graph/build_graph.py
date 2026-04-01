"""Graph builder for the article generation workflow."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.graph.edges import (
    route_after_article_assembly,
    route_after_fixed,
    route_after_debate,
    route_after_research,
    route_after_section_debate,
    route_after_section_init,
    route_after_section_research,
    route_after_section_review,
    route_after_section_write,
)
from app.graph.nodes import build_nodes
from app.graph.services import WorkflowServices
from app.graph.state import ProjectState


def build_graph(services: WorkflowServices):
    """Compile and return the LangGraph workflow."""

    builder = StateGraph(ProjectState)
    nodes = build_nodes(services)

    builder.add_node("orchestrator_intake", nodes["orchestrator_intake"])
    builder.add_node("research_node", nodes["research_node"])
    builder.add_node("debate_node", nodes["debate_node"])
    builder.add_node("synthesis_node", nodes["synthesis_node"])
    builder.add_node("failure_node", nodes["failure_node"])
    builder.add_node("section_init_node", nodes["section_init_node"])
    builder.add_node("section_write_node", nodes["section_write_node"])
    builder.add_node("section_review_node", nodes["section_review_node"])
    builder.add_node("section_research_node", nodes["section_research_node"])
    builder.add_node("section_debate_node", nodes["section_debate_node"])
    builder.add_node("article_assembly_node", nodes["article_assembly_node"])
    builder.add_node("save_file_node", nodes["save_file_node"])

    builder.set_entry_point("orchestrator_intake")
    builder.add_conditional_edges(
        "orchestrator_intake",
        route_after_fixed("research_node"),
        {
            "research_node": "research_node",
            "failure_node": "failure_node",
        },
    )
    builder.add_conditional_edges(
        "research_node",
        route_after_research,
        {
            "debate_node": "debate_node",
            "article_assembly_node": "article_assembly_node",
            "failure_node": "failure_node",
        },
    )
    builder.add_conditional_edges(
        "debate_node",
        route_after_debate,
        {
            "debate_node": "debate_node",
            "synthesis_node": "synthesis_node",
            "failure_node": "failure_node",
        },
    )
    builder.add_conditional_edges(
        "synthesis_node",
        route_after_fixed("section_init_node"),
        {
            "section_init_node": "section_init_node",
            "failure_node": "failure_node",
        },
    )
    builder.add_conditional_edges(
        "section_init_node",
        route_after_section_init,
        {
            "section_write_node": "section_write_node",
            "article_assembly_node": "article_assembly_node",
            "failure_node": "failure_node",
        },
    )
    builder.add_conditional_edges(
        "section_write_node",
        route_after_section_write,
        {
            "section_review_node": "section_review_node",
            "failure_node": "failure_node",
        },
    )
    builder.add_conditional_edges(
        "section_review_node",
        route_after_section_review,
        {
            "section_research_node": "section_research_node",
            "section_write_node": "section_write_node",
            "article_assembly_node": "article_assembly_node",
            "failure_node": "failure_node",
        },
    )
    builder.add_conditional_edges(
        "section_research_node",
        route_after_section_research,
        {
            "section_debate_node": "section_debate_node",
            "failure_node": "failure_node",
        },
    )
    builder.add_conditional_edges(
        "section_debate_node",
        route_after_section_debate,
        {
            "section_write_node": "section_write_node",
            "failure_node": "failure_node",
        },
    )
    builder.add_conditional_edges(
        "article_assembly_node",
        route_after_article_assembly,
        {
            "save_file_node": "save_file_node",
            "failure_node": "failure_node",
        },
    )
    builder.add_edge("failure_node", "save_file_node")
    builder.add_edge("save_file_node", END)
    return builder.compile()
