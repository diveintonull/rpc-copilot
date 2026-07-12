"""Contract tests for the GRC agent routing graph."""

from __future__ import annotations

import pytest

from agent.graph import build_graph
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


class FakeTools:
    """Fake Task11 tools whose call log can prove a path stayed tool-free."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search_regulation(
        self,
        query: str,
        source_ids: list[str] | None = None,
    ) -> list[dict]:
        self.calls.append(
            {
                "tool": "search_regulation",
                "query": query,
                "source_ids": source_ids,
            }
        )
        return []

    def get_clause(
        self,
        source_id: str,
        version: str,
        section_number: str,
    ) -> dict | None:
        self.calls.append(
            {
                "tool": "get_clause",
                "source_id": source_id,
                "version": version,
                "section_number": section_number,
            }
        )
        return None

    def compare_clauses(
        self,
        left: dict,
        right: dict,
        dimensions: list[str],
    ) -> dict:
        self.calls.append(
            {
                "tool": "compare_clauses",
                "left": left,
                "right": right,
                "dimensions": dimensions,
            }
        )
        return {"left": None, "right": None, "dimensions": dimensions}

@pytest.fixture
def fake_tools() -> FakeTools:
    return FakeTools()


def test_route_intent_returns_regulation_qa(
    fake_tools: FakeTools,
) -> None:
    classifier_calls: list[tuple[str, str]] = []

    def fake_classifier(query: str, control_text: str) -> str:
        classifier_calls.append((query, control_text))
        return "regulation_qa"

    state: AgentState = {
        "request_id": "req-route-qa",
        "query": "等保 2.0 对身份鉴别有什么要求？",
        "control_text": "",
        "trace": [{"node": "received"}],
    }

    update = route_intent(state, fake_classifier)

    assert update["intent"] == "regulation_qa"
    assert classifier_calls == [(state["query"], state["control_text"])]
    assert fake_tools.calls == []


def test_route_intent_appends_predictable_trace_event() -> None:
    def fake_classifier(_query: str, _control_text: str) -> str:
        return "regulation_qa"

    state: AgentState = {
        "request_id": "req-route-trace",
        "query": "等保 2.0 对身份鉴别有什么要求？",
        "control_text": "",
        "trace": [{"node": "received"}],
    }

    update = route_intent(state, fake_classifier)

    assert update["trace"] == [
        {"node": "received"},
        {"node": "route_intent", "intent": "regulation_qa"},
    ]


def test_select_workflow_routes_regulation_qa_to_qa_node() -> None:
    state: AgentState = {"intent": "regulation_qa"}

    next_node = select_workflow(state)

    assert next_node == "execute_regulation_qa"


def test_select_workflow_routes_comparison_to_comparison_node() -> None:
    state: AgentState = {"intent": "clause_comparison"}

    next_node = select_workflow(state)

    assert next_node == "execute_clause_comparison"


def test_select_workflow_routes_gap_analysis_to_gap_node() -> None:
    state: AgentState = {"intent": "gap_analysis"}

    next_node = select_workflow(state)

    assert next_node == "execute_gap_analysis"


def test_select_workflow_routes_unsupported_to_unsupported_node() -> None:
    state: AgentState = {"intent": "unsupported"}

    next_node = select_workflow(state)

    assert next_node == "execute_unsupported"


@pytest.mark.parametrize(
    ("intent", "workflow_node", "expected_answer", "expected_node"),
    [
        (
            "regulation_qa",
            execute_regulation_qa,
            "fake regulation_qa result",
            "execute_regulation_qa",
        ),
        (
            "clause_comparison",
            execute_clause_comparison,
            "fake clause_comparison result",
            "execute_clause_comparison",
        ),
        (
            "gap_analysis",
            execute_gap_analysis,
            "fake gap_analysis result",
            "execute_gap_analysis",
        ),
    ],
)
def test_supported_fake_workflow_returns_answer_and_trace(
    intent: str,
    workflow_node,
    expected_answer: str,
    expected_node: str,
) -> None:
    state: AgentState = {
        "intent": intent,
        "trace": [{"node": "route_intent", "intent": intent}],
    }

    update = workflow_node(state)

    assert update["answer"] == expected_answer
    assert update["trace"] == [
        {"node": "route_intent", "intent": intent},
        {"node": expected_node},
    ]


def test_unsupported_workflow_refuses_without_calling_tools(
    fake_tools: FakeTools,
) -> None:
    state: AgentState = {
        "intent": "unsupported",
        "trace": [
            {"node": "route_intent", "intent": "unsupported"}
        ],
    }

    update = execute_unsupported(state, fake_tools)

    assert update["answer"] == "unsupported request"
    assert update["trace"] == [
        {"node": "route_intent", "intent": "unsupported"},
        {"node": "execute_unsupported"},
    ]
    assert fake_tools.calls == []


@pytest.mark.parametrize(
    ("intent", "expected_valid"),
    [
        ("regulation_qa", True),
        ("unsupported", False),
    ],
)
def test_verify_records_deterministic_result(
    intent: str,
    expected_valid: bool,
) -> None:
    state: AgentState = {
        "intent": intent,
        "trace": [{"node": "execute_workflow"}],
    }

    update = verify(state)

    assert update["citations_valid"] is expected_valid
    assert update["trace"] == [
        {"node": "execute_workflow"},
        {"node": "verify", "citations_valid": expected_valid},
    ]


@pytest.mark.parametrize(
    ("intent", "expected_status"),
    [
        ("regulation_qa", "completed"),
        ("unsupported", "refused"),
    ],
)
def test_finish_sets_explicit_status_and_trace(
    intent: str,
    expected_status: str,
) -> None:
    state: AgentState = {
        "intent": intent,
        "trace": [{"node": "verify"}],
    }

    update = finish(state)

    assert update["final_status"] == expected_status
    assert update["trace"] == [
        {"node": "verify"},
        {"node": "finish", "final_status": expected_status},
    ]


@pytest.mark.parametrize(
    (
        "intent",
        "workflow_node",
        "expected_status",
        "expected_valid",
    ),
    [
        (
            "regulation_qa",
            "execute_regulation_qa",
            "completed",
            True,
        ),
        (
            "clause_comparison",
            "execute_clause_comparison",
            "completed",
            True,
        ),
        (
            "gap_analysis",
            "execute_gap_analysis",
            "completed",
            True,
        ),
        (
            "unsupported",
            "execute_unsupported",
            "refused",
            False,
        ),
    ],
)
def test_graph_routes_each_intent_through_verify_and_finish(
    intent: str,
    workflow_node: str,
    expected_status: str,
    expected_valid: bool,
    fake_tools: FakeTools,
) -> None:
    def fake_classifier(_query: str, _control_text: str) -> str:
        return intent

    graph = build_graph(fake_classifier, fake_tools)

    result = graph.invoke(
        {
            "request_id": f"req-{intent}",
            "query": "route this request",
            "control_text": "",
            "trace": [{"node": "received"}],
        }
    )

    assert [event["node"] for event in result["trace"]] == [
        "received",
        "route_intent",
        workflow_node,
        "verify",
        "finish",
    ]
    assert result["intent"] == intent
    assert result["citations_valid"] is expected_valid
    assert result["final_status"] == expected_status

    if intent == "unsupported":
        assert fake_tools.calls == []


def test_graph_diagram_exposes_all_conditional_paths(
    fake_tools: FakeTools,
) -> None:
    graph = build_graph(
        lambda _query, _control_text: "regulation_qa",
        fake_tools,
    )
    edges = {
        (edge.source, edge.target)
        for edge in graph.get_graph().edges
    }

    assert {
        ("route_intent", "execute_regulation_qa"),
        ("route_intent", "execute_clause_comparison"),
        ("route_intent", "execute_gap_analysis"),
        ("route_intent", "execute_unsupported"),
    } <= edges
