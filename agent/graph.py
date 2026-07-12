"""LangGraph assembly and diagram export helpers for the GRC agent."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    execute_clause_comparison,
    execute_gap_analysis,
    execute_regulation_qa,
    execute_unsupported,
    finish,
    route_intent,
    select_workflow,
    verify,
)
from agent.state import AgentState


DEFAULT_MERMAID_PATH = Path("docs/agent_graph.mmd")


def create_graph_builder() -> StateGraph:
    """Return an empty builder bound to the stable AgentState schema."""
    return StateGraph(AgentState)


def build_graph(classify_intent, tools) -> Any:
    """Build the compiled routing graph from injected runtime dependencies."""
    builder = create_graph_builder()
    route_node = partial(route_intent, classify_intent=classify_intent)
    unsupported_node = partial(execute_unsupported, tools=tools)

    builder.add_node("route_intent", route_node)
    builder.add_node("execute_regulation_qa", execute_regulation_qa)
    builder.add_node("execute_clause_comparison", execute_clause_comparison)
    builder.add_node("execute_gap_analysis", execute_gap_analysis)
    builder.add_node("execute_unsupported", unsupported_node)
    builder.add_node("verify", verify)
    builder.add_node("finish", finish)

    builder.add_edge(START, "route_intent")
    builder.add_conditional_edges(
        "route_intent",
        select_workflow,
        {
            "execute_regulation_qa": "execute_regulation_qa",
            "execute_clause_comparison": "execute_clause_comparison",
            "execute_gap_analysis": "execute_gap_analysis",
            "execute_unsupported": "execute_unsupported",
        },
    )
    builder.add_edge("execute_regulation_qa", "verify")
    builder.add_edge("execute_clause_comparison", "verify")
    builder.add_edge("execute_gap_analysis", "verify")
    builder.add_edge("execute_unsupported", "verify")
    builder.add_edge("verify", "finish")
    builder.add_edge("finish", END)

    return builder.compile()


def export_graph_mermaid(
    graph: Any,
    path: Path = DEFAULT_MERMAID_PATH,
) -> Path:
    """Write Mermaid source for a compiled graph under ignored docs/."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    return path
