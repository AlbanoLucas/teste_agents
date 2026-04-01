"""Graph builder for the article generation workflow."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.graph.edges import (
    route_after_debate,
    route_after_research,
    route_after_section_init,
    route_after_section_review,
)
from app.graph.nodes import WorkflowDependencies, build_nodes
from app.graph.state import ProjectState


def build_graph(deps: WorkflowDependencies):
    """Compile and return the LangGraph workflow."""

    builder = StateGraph(ProjectState)
    nodes = build_nodes(deps)

    builder.add_node("orchestrator_intake", nodes["orchestrator_intake"])
    builder.add_node("research_node", nodes["research_node"])
    builder.add_node("debate_node", nodes["debate_node"])
    builder.add_node("synthesis_node", nodes["synthesis_node"])
    builder.add_node("section_init_node", nodes["section_init_node"])
    builder.add_node("section_write_node", nodes["section_write_node"])
    builder.add_node("section_review_node", nodes["section_review_node"])
    builder.add_node("section_research_node", nodes["section_research_node"])
    builder.add_node("section_debate_node", nodes["section_debate_node"])
    builder.add_node("article_assembly_node", nodes["article_assembly_node"])
    builder.add_node("save_file_node", nodes["save_file_node"])

    builder.set_entry_point("orchestrator_intake")
    builder.add_edge("orchestrator_intake", "research_node")
    builder.add_conditional_edges(
        "research_node",
        route_after_research,
        {
            "debate_node": "debate_node",
            "article_assembly_node": "article_assembly_node",
        },
    )
    builder.add_conditional_edges(
        "debate_node",
        route_after_debate,
        {
            "debate_node": "debate_node",
            "synthesis_node": "synthesis_node",
        },
    )
    builder.add_edge("synthesis_node", "section_init_node")
    builder.add_conditional_edges(
        "section_init_node",
        route_after_section_init,
        {
            "section_write_node": "section_write_node",
            "article_assembly_node": "article_assembly_node",
        },
    )
    builder.add_edge("section_write_node", "section_review_node")
    builder.add_conditional_edges(
        "section_review_node",
        route_after_section_review,
        {
            "section_research_node": "section_research_node",
            "section_write_node": "section_write_node",
            "article_assembly_node": "article_assembly_node",
        },
    )
    builder.add_edge("section_research_node", "section_debate_node")
    builder.add_edge("section_debate_node", "section_write_node")
    builder.add_edge("article_assembly_node", "save_file_node")
    builder.add_edge("save_file_node", END)
    return builder.compile()
