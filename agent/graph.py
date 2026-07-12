"""LangGraph assembly and diagram export helpers for the GRC agent."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    check_cancel,
    execute_clause_comparison,
    execute_gap_analysis,
    execute_regulation_qa,
    execute_unsupported,
    finish,
    prepare_retry,
    route_intent,
    select_after_post_workflow_cancel,
    select_after_pre_workflow_cancel,
    select_after_verify,
    verify,
)
from agent.state import AgentState


DEFAULT_MERMAID_PATH = Path("docs/agent_graph.mmd")
MAX_GRAPH_STEPS = 16


def create_graph_builder() -> StateGraph:
    """Return an empty builder bound to the stable AgentState schema."""
    return StateGraph(AgentState)


def build_graph(
    classify_intent,
    tools,
    llm,
    *,
    entailment_evaluator=None,
    is_cancelled=None,
) -> Any:
    """Build the compiled routing graph from injected runtime dependencies."""
    builder = create_graph_builder()
    route_node = partial(route_intent, classify_intent=classify_intent)
    unsupported_node = partial(execute_unsupported, tools=tools)
    regulation_qa_node = partial(
        execute_regulation_qa,
        tools=tools,
        llm=llm,
    )
    comparison_node = partial(
        execute_clause_comparison,
        tools=tools,
        llm=llm,
    )
    gap_analysis_node = partial(
        execute_gap_analysis,
        tools=tools,
        llm=llm,
    )
    verify_node = partial(
        verify,
        entailment_evaluator=entailment_evaluator,
    )
    cancellation_checker = is_cancelled or (lambda _request_id: False)
    pre_workflow_cancel_node = partial(
        check_cancel,
        is_cancelled=cancellation_checker,
        node_name="check_cancel_before_workflow",
    )
    post_workflow_cancel_node = partial(
        check_cancel,
        is_cancelled=cancellation_checker,
        node_name="check_cancel_after_workflow",
    )
    retry_node = partial(prepare_retry, llm=llm)

    builder.add_node("route_intent", route_node)
    builder.add_node(
        "check_cancel_before_workflow",
        pre_workflow_cancel_node,
    )
    builder.add_node("execute_regulation_qa", regulation_qa_node)
    builder.add_node("execute_clause_comparison", comparison_node)
    builder.add_node("execute_gap_analysis", gap_analysis_node)
    builder.add_node("execute_unsupported", unsupported_node)
    builder.add_node(
        "check_cancel_after_workflow",
        post_workflow_cancel_node,
    )
    builder.add_node("verify", verify_node)
    builder.add_node("prepare_retry", retry_node)
    builder.add_node("finish", finish)

    builder.add_edge(START, "route_intent")
    builder.add_edge("route_intent", "check_cancel_before_workflow")
    builder.add_conditional_edges(
        "check_cancel_before_workflow",
        select_after_pre_workflow_cancel,
        {
            "execute_regulation_qa": "execute_regulation_qa",
            "execute_clause_comparison": "execute_clause_comparison",
            "execute_gap_analysis": "execute_gap_analysis",
            "execute_unsupported": "execute_unsupported",
            "finish": "finish",
        },
    )
    builder.add_edge(
        "execute_regulation_qa",
        "check_cancel_after_workflow",
    )
    builder.add_edge(
        "execute_clause_comparison",
        "check_cancel_after_workflow",
    )
    builder.add_edge(
        "execute_gap_analysis",
        "check_cancel_after_workflow",
    )
    builder.add_edge(
        "execute_unsupported",
        "check_cancel_after_workflow",
    )
    builder.add_conditional_edges(
        "check_cancel_after_workflow",
        select_after_post_workflow_cancel,
        {"verify": "verify", "finish": "finish"},
    )
    builder.add_conditional_edges(
        "verify",
        select_after_verify,
        {"prepare_retry": "prepare_retry", "finish": "finish"},
    )
    builder.add_edge("prepare_retry", "check_cancel_before_workflow")
    builder.add_edge("finish", END)

    return builder.compile().with_config(
        {"recursion_limit": MAX_GRAPH_STEPS}
    )


def export_graph_mermaid(
    graph: Any,
    path: Path = DEFAULT_MERMAID_PATH,
) -> Path:
    """Write Mermaid source for a compiled graph under ignored docs/."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    return path
