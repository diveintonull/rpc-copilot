"""Routing and workflow nodes for the GRC agent graph."""

from __future__ import annotations

from agent.state import AgentState


def route_intent(state: AgentState, classify_intent) -> dict:
    """Classify one request and return its intent state update."""
    intent = classify_intent(state["query"], state["control_text"])

    old_trace = state["trace"]
    new_event = {"node": "route_intent", "intent": intent}
    new_trace = old_trace + [new_event]

    return {
        "intent": intent,
        "trace": new_trace,
    }


def select_workflow(state: AgentState) -> str:
    """Return the next workflow node name for the classified intent."""
    intent = state["intent"]

    if intent == "regulation_qa":
        return "execute_regulation_qa"
    elif intent == "clause_comparison":
        return "execute_clause_comparison"
    elif intent == "gap_analysis":
        return "execute_gap_analysis"
    elif intent == "unsupported":
        return "execute_unsupported"
    else:
        raise ValueError(f"unknown intent: {intent}")


def execute_regulation_qa(state: AgentState) -> dict:
    """Return a deterministic placeholder for the regulation-QA route."""
    return {
        "answer": "fake regulation_qa result",
        "trace": state["trace"] + [{"node": "execute_regulation_qa"}],
    }


def execute_clause_comparison(state: AgentState) -> dict:
    """Return a deterministic placeholder for the comparison route."""
    return {
        "answer": "fake clause_comparison result",
        "trace": state["trace"] + [{"node": "execute_clause_comparison"}],
    }


def execute_gap_analysis(state: AgentState) -> dict:
    """Return a deterministic placeholder for the gap-analysis route."""
    return {
        "answer": "fake gap_analysis result",
        "trace": state["trace"] + [{"node": "execute_gap_analysis"}],
    }


def execute_unsupported(state: AgentState, tools) -> dict:
    """Refuse an unsupported request without calling a retrieval tool."""
    return {
        "answer": "unsupported request",
        "trace": state["trace"] + [{"node": "execute_unsupported"}],
    }


def verify(state: AgentState) -> dict:
    """Return the deterministic verification update for a routed request."""
    citations_valid = state["intent"] != "unsupported"
    return {
        "citations_valid": citations_valid,
        "trace": state["trace"] + [
            {
                "node": "verify",
                "citations_valid": citations_valid,
            }
        ],
    }


def finish(state: AgentState) -> dict:
    """Return the final status update and terminal trace event."""
    final_status = "refused" if state["intent"] == "unsupported" else "completed"
    return {
        "final_status": final_status,
        "trace": state["trace"] + [
            {
                "node": "finish",
                "final_status": final_status,
            }
        ],
    }
