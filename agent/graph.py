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
    load_skill_node,
    match_skill_node,
    prepare_retry,
    route_intent,
    select_after_post_workflow_cancel,
    select_after_pre_workflow_cancel,
    select_after_verify,
    verify,
)
from agent.skills import SkillCatalog, discover_skills
from agent.state import AgentState
from agent.tools import MCPCallTool, MCPStdioConfig, create_tool_backend


DEFAULT_MERMAID_PATH = Path("docs/agent_graph.mmd")
MAX_GRAPH_STEPS = 16


def create_graph_builder() -> StateGraph:
    """Return an empty builder bound to the stable AgentState schema."""
    return StateGraph(AgentState)


def _run_tool_node(
    state: AgentState,
    *,
    node,
    tools,
    llm,
    backend_name: str | None,
) -> dict:
    """Run one workflow node and mark newly recorded tool activity."""
    update = node(state, tools, llm)
    if backend_name not in {"local", "mcp"}:
        return update

    previous_call_count = len(state.get("tool_calls", []))
    tool_calls = update.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) <= previous_call_count:
        return update

    marked_calls = [
        dict(call) if isinstance(call, dict) else call
        for call in tool_calls
    ]
    for index in range(previous_call_count, len(marked_calls)):
        call = marked_calls[index]
        if isinstance(call, dict):
            call["tool_backend"] = backend_name

    marked_trace = [
        dict(event) if isinstance(event, dict) else event
        for event in update.get("trace", [])
    ]
    previous_trace_count = len(state.get("trace", []))
    for index in range(previous_trace_count, len(marked_trace)):
        event = marked_trace[index]
        if isinstance(event, dict):
            event["tool_backend"] = backend_name

    return {
        **update,
        "tool_calls": marked_calls,
        "trace": marked_trace,
    }


def build_graph(
    classify_intent,
    tools,
    llm,
    *,
    entailment_evaluator=None,
    is_cancelled=None,
    skill_catalog: SkillCatalog | None = None,
    tool_backend: str = "local",
    mcp_call_tool: MCPCallTool | None = None,
    mcp_stdio_config: MCPStdioConfig | None = None,
) -> Any:
    """Build the compiled routing graph from injected runtime dependencies."""
    builder = create_graph_builder()

    if tools is not None and (
        tool_backend != "local"
        or mcp_call_tool is not None
        or mcp_stdio_config is not None
    ):
        raise ValueError(
            "pass either injected tools or tool backend configuration, not both"
        )

    selected_tools = (
        tools
        if tools is not None
        else create_tool_backend(
            tool_backend,
            mcp_call_tool=mcp_call_tool,
            mcp_stdio_config=mcp_stdio_config,
        )
    )
    backend_name = getattr(selected_tools, "backend_name", None)

    selected_skill_catalog = (
        skill_catalog
        if skill_catalog is not None
        else discover_skills(Path("skills"))
    )

    route_node = partial(route_intent, classify_intent=classify_intent)

    match_node = partial(match_skill_node, catalog=selected_skill_catalog)
    load_node = partial(load_skill_node, catalog=selected_skill_catalog)

    unsupported_node = partial(execute_unsupported, tools=selected_tools)
    regulation_qa_node = partial(
        _run_tool_node,
        node=execute_regulation_qa,
        tools=selected_tools,
        llm=llm,
        backend_name=backend_name,
    )
    comparison_node = partial(
        _run_tool_node,
        node=execute_clause_comparison,
        tools=selected_tools,
        llm=llm,
        backend_name=backend_name,
    )
    gap_analysis_node = partial(
        _run_tool_node,
        node=execute_gap_analysis,
        tools=selected_tools,
        llm=llm,
        backend_name=backend_name,
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
    builder.add_node("match_skill", match_node)
    builder.add_node("load_skill", load_node)
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
    builder.add_edge("route_intent", "match_skill")
    builder.add_edge("match_skill", "load_skill")
    builder.add_edge("load_skill", "check_cancel_before_workflow")
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
