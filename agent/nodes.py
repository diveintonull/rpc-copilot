"""Routing and workflow nodes for the GRC agent graph."""

from __future__ import annotations

from html import escape

from agent.skills import (
    SkillCatalog,
    load_skill as load_skill_from_catalog,
    match_skill as match_skill_from_catalog,
)
from agent.state import AgentState
from rag.citations import EntailmentEvaluator, validate_citations


GAP_MATRIX_FIELDS = (
    "requirement",
    "current_state",
    "gap",
    "risk",
    "recommendation",
)
GAP_ANALYSIS_DISCLAIMER = "以下为初步差距分析，最终结果需要人工确认。"
GAP_ANALYSIS_OVERCLAIM_REFUSAL = (
    "差距分析包含越权结论，已拒绝输出；最终结果需要人工确认。"
)
GAP_ANALYSIS_UNSAFE_CLAIMS = (
    "企业已经合规",
    "企业已合规",
    "企业已经违法",
    "企业已违法",
    "the enterprise is compliant",
    "the enterprise is illegal",
)
EVIDENCE_START = "<evidence>"
EVIDENCE_END = "</evidence>"
REPAIRABLE_CITATION_FAILURES = {
    "uncited_claim",
    "unsupported_claim",
    "unknown_citation",
}


def _evidence_key(item: dict) -> tuple[object, object, object]:
    return (
        item.get("source_id"),
        item.get("version"),
        item.get("section_number"),
    )


def _bound_evidence_item(item: dict) -> dict:
    bounded = dict(item)
    text = bounded.get("text")
    if isinstance(text, str):
        safe_text = escape(text, quote=False)
        bounded["text"] = (
            f"{EVIDENCE_START}\n{safe_text}\n{EVIDENCE_END}"
        )
    return bounded


def _bound_evidence(evidence: list[dict]) -> list[dict]:
    return [_bound_evidence_item(item) for item in evidence]


def _restore_gap_evidence(
    gap_matrix: list[dict],
    regulation_evidence: list[dict],
) -> list[dict]:
    raw_by_key = {
        _evidence_key(item): item for item in regulation_evidence
    }
    restored = []
    for row in gap_matrix:
        restored_row = dict(row)
        restored_row["evidence"] = [
            raw_by_key[key]
            for item in row.get("evidence", [])
            if (key := _evidence_key(item)) in raw_by_key
        ]
        restored.append(restored_row)
    return restored


def format_gap_matrix(gap_matrix: list[dict]) -> str:
    """Format grounded gap rows with an explicit human-review boundary."""
    lines = [GAP_ANALYSIS_DISCLAIMER]

    for index, row in enumerate(gap_matrix, start=1):
        lines.append(f"gap_item: {index}")
        for field in GAP_MATRIX_FIELDS:
            lines.append(f"{field}: {row.get(field, '')}")

        references = []
        for item in row.get("evidence", []):
            source_id = item.get("source_id", "")
            version = item.get("version", "")
            section_number = item.get("section_number", "")
            references.append(f"{source_id}@{version}#{section_number}")
        lines.append(f"evidence: {', '.join(references)}")

    return "\n".join(lines)


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


def match_skill_node(
    state: AgentState,
    catalog: SkillCatalog,
) -> dict:
    """Match one routed intent to at most one available Skill."""
    matched_skill = match_skill_from_catalog(state["intent"], catalog)

    return {
        "active_skill": matched_skill or "",
        "trace": state["trace"] + [
            {
                "node": "match_skill",
                "intent": state["intent"],
                "matched_skill": matched_skill,
                "catalog_tokens": catalog.catalog_tokens,
            }
        ],
    }


def load_skill_node(
    state: AgentState,
    catalog: SkillCatalog,
) -> dict:
    """Load the matched Skill body after a successful selection."""
    matched_skill = state.get("active_skill") or None

    if matched_skill is None:
        return {
            "active_skill": "",
            "skill_text": "",
            "trace": state["trace"] + [
                {
                    "node": "load_skill",
                    "matched_skill": None,
                    "loaded": False,
                    "body_tokens": 0,
                    "resource_tokens": 0,
                }
            ],
        }

    loaded = load_skill_from_catalog(matched_skill, catalog)

    return {
        "active_skill": loaded.name,
        "skill_text": loaded.text,
        "trace": state["trace"] + [
            {
                "node": "load_skill",
                "matched_skill": loaded.name,
                "loaded": True,
                "body_tokens": loaded.token_usage["body"],
                "resource_tokens": loaded.token_usage["resources"],
            }
        ],
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


def execute_regulation_qa(state: AgentState, tools, llm) -> dict:
    """Search regulation evidence and produce an evidence-aware answer."""
    query = state["query"]
    if state.get("retry_action") == "repair_answer" and state.get("evidence"):
        evidence = state["evidence"]
        failures = state.get("citation_failures", [])
        answer = llm.repair_regulation_answer(
            query,
            state.get("answer", ""),
            _bound_evidence(evidence),
            failures,
            skill_text=state.get("skill_text", ""),
        )
        return {
            "answer": answer,
            "evidence": evidence,
            "retry_action": "",
            "citation_failures": [],
            "tool_calls": state.get("tool_calls", []),
            "trace": state["trace"] + [
                {
                    "node": "execute_regulation_qa",
                    "action": "repair_answer",
                    "result_count": len(evidence),
                }
            ],
        }

    evidence = tools.search_regulation(query, None)

    tool_call = {
        "tool": "search_regulation",
        "query": query,
        "source_ids": None,
        "result_count": len(evidence),
    }
    new_trace = {
        "node": "execute_regulation_qa",
        "tool": "search_regulation",
        "result_count": len(evidence),
    }

    if evidence:
        answer = llm.answer_regulation(
            query,
            _bound_evidence(evidence),
            skill_text=state.get("skill_text", ""),
        )
    else:
        answer = "insufficient regulation evidence"

    return {
        "answer": answer,
        "retry_action": "",
        "citation_failures": [],
        "tool_calls": state.get("tool_calls", []) + [tool_call],
        "evidence": evidence,
        "trace": state["trace"] + [new_trace],
    }


def execute_clause_comparison(state: AgentState, tools, llm) -> dict:
    """Compare two precisely identified clauses when both sides exist."""
    query = state["query"]
    plan = llm.plan_comparison(
        query,
        skill_text=state.get("skill_text", ""),
    )

    left_plan = plan["left"]
    right_plan = plan["right"]

    def is_exact(reference: dict) -> bool:
        return all(
            isinstance(reference.get(field), str)
            and bool(reference[field].strip())
            for field in ("source_id", "version", "section_number")
        )

    if is_exact(left_plan) and is_exact(right_plan):
        comparison = tools.compare_clauses(
            left_plan,
            right_plan,
            plan["dimensions"],
        )
        tool_calls = [
            {
                "tool": "compare_clauses",
                "left": left_plan,
                "right": right_plan,
                "dimensions": plan["dimensions"],
            }
        ]
        trace_tool = "compare_clauses"
    else:
        side_results = []
        tool_calls = []
        for side_plan in (left_plan, right_plan):
            source_id = side_plan.get("source_id", "").strip()
            source_ids = [source_id] if source_id else None
            search_query = side_plan.get("search_query", "").strip() or query
            candidates = tools.search_regulation(search_query, source_ids)
            requested_version = side_plan.get("version", "").strip()
            if requested_version:
                candidates = [
                    item
                    for item in candidates
                    if item.get("version") == requested_version
                ]
            side_results.append(candidates[0] if candidates else None)
            tool_calls.append(
                {
                    "tool": "search_regulation",
                    "query": search_query,
                    "source_ids": source_ids,
                    "result_count": len(candidates),
                }
            )
        comparison = {
            "left": side_results[0],
            "right": side_results[1],
            "dimensions": list(plan["dimensions"]),
        }
        if (
            comparison["left"] is not None
            and comparison["right"] is not None
            and comparison["left"].get("parent_id")
            == comparison["right"].get("parent_id")
        ):
            comparison["right"] = None
        trace_tool = "search_regulation"

    left_evidence = comparison["left"]
    right_evidence = comparison["right"]
    left_found = left_evidence is not None
    right_found = right_evidence is not None

    evidence = [
        item
        for item in [left_evidence, right_evidence]
        if item is not None
    ]
    for tool_call in tool_calls:
        tool_call["left_found"] = left_found
        tool_call["right_found"] = right_found
    new_trace = {
        "node": "execute_clause_comparison",
        "tool": trace_tool,
        "left_found": left_found,
        "right_found": right_found,
    }

    if left_found and right_found:
        bounded_comparison = {
            "left": _bound_evidence_item(left_evidence),
            "right": _bound_evidence_item(right_evidence),
            "dimensions": comparison["dimensions"],
        }
        answer = llm.answer_comparison(
            query,
            bounded_comparison,
            skill_text=state.get("skill_text", ""),
        )
    else:
        answer = "incomplete comparison evidence"

    return {
        "answer": answer,
        "tool_calls": state.get("tool_calls", []) + tool_calls,
        "evidence": evidence,
        "trace": state["trace"] + [new_trace],
    }


def execute_gap_analysis(state: AgentState, tools, llm) -> dict:
    """Build an evidence-linked control gap matrix for human review."""
    query = state["query"]
    control_text = state.get("control_text", "").strip()
    previous_tool_calls = state.get("tool_calls", [])

    if not control_text:
        return {
            "answer": "control description required",
            "tool_calls": previous_tool_calls,
            "evidence": [],
            "trace": state["trace"] + [
                {
                    "node": "execute_gap_analysis",
                    "reason": "missing_control_text",
                    "gap_count": 0,
                }
            ],
        }

    controls = llm.extract_controls(
        control_text,
        skill_text=state.get("skill_text", ""),
    )
    regulation_evidence = tools.search_regulation(query, None)
    tool_call = {
        "tool": "search_regulation",
        "query": query,
        "source_ids": None,
        "result_count": len(regulation_evidence),
    }

    if regulation_evidence:
        generated_gap_matrix = llm.map_gaps(
            query,
            controls,
            _bound_evidence(regulation_evidence),
            skill_text=state.get("skill_text", ""),
        )
        gap_matrix = _restore_gap_evidence(
            generated_gap_matrix,
            regulation_evidence,
        )
    else:
        gap_matrix = []

    if gap_matrix:
        answer = format_gap_matrix(gap_matrix)
    elif regulation_evidence:
        answer = "insufficient gap analysis evidence"
    else:
        answer = "insufficient regulation evidence"

    return {
        "answer": answer,
        "tool_calls": previous_tool_calls + [tool_call],
        "evidence": gap_matrix,
        "trace": state["trace"] + [
            {
                "node": "execute_gap_analysis",
                "result_count": len(regulation_evidence),
                "gap_count": len(gap_matrix),
                "human_confirmation_required": True,
            }
        ],
    }


def execute_unsupported(state: AgentState, tools) -> dict:
    """Refuse an unsupported request without calling a retrieval tool."""
    return {
        "answer": "unsupported request",
        "trace": state["trace"] + [{"node": "execute_unsupported"}],
    }


def check_cancel(state: AgentState, is_cancelled, node_name: str) -> dict:
    """Record an external cancellation decision at one graph boundary."""
    request_id = state.get("request_id", "")
    cancelled = bool(is_cancelled(request_id))
    update = {
        "trace": state["trace"] + [
            {"node": node_name, "cancelled": cancelled}
        ]
    }
    if cancelled:
        update["final_status"] = "cancelled"
    return update


def select_after_pre_workflow_cancel(state: AgentState) -> str:
    """Dispatch a workflow unless cancellation already won the race."""
    if state.get("final_status") == "cancelled":
        return "finish"
    return select_workflow(state)


def select_after_post_workflow_cancel(state: AgentState) -> str:
    """Skip verification when cancellation arrives after a workflow."""
    if state.get("final_status") == "cancelled":
        return "finish"
    return "verify"


def prepare_retry(state: AgentState, llm) -> dict:
    """Repair a grounded answer or re-retrieve when evidence itself failed."""
    failures = []
    if state.get("trace"):
        failures = state["trace"][-1].get("citation_failures", [])
    query = state["query"]
    failure_codes = {
        failure.get("code")
        for failure in failures
        if isinstance(failure, dict)
    }
    can_repair_answer = (
        state.get("intent") == "regulation_qa"
        and bool(state.get("evidence"))
        and bool(failure_codes)
        and failure_codes <= REPAIRABLE_CITATION_FAILURES
    )
    retry_count = state.get("retry_count", 0) + 1

    if can_repair_answer:
        return {
            "citation_failures": failures,
            "citations_valid": False,
            "retry_action": "repair_answer",
            "retry_count": retry_count,
            "trace": state["trace"] + [
                {
                    "node": "prepare_retry",
                    "action": "repair_answer",
                    "retry_count": retry_count,
                }
            ],
        }

    rewritten_query = llm.rewrite_query(query, failures).strip()
    if not rewritten_query:
        rewritten_query = query
    return {
        "query": rewritten_query,
        "answer": "",
        "evidence": [],
        "citation_failures": failures,
        "citations_valid": False,
        "retry_action": "retrieve",
        "retry_count": retry_count,
        "trace": state["trace"] + [
            {
                "node": "prepare_retry",
                "action": "retrieve",
                "retry_count": retry_count,
                "query": rewritten_query,
            }
        ],
    }


def select_after_verify(state: AgentState) -> str:
    """Pass, retry once, or refuse after deterministic verification."""
    if state.get("citations_valid"):
        return "finish"
    if (
        state.get("intent") != "unsupported"
        and state.get("retry_count", 0) < 1
    ):
        return "prepare_retry"
    return "finish"


def verify(
    state: AgentState,
    entailment_evaluator: EntailmentEvaluator | None = None,
) -> dict:
    """Return the deterministic verification update for a routed request."""
    intent = state["intent"]
    evidence = state.get("evidence", [])
    has_unsafe_claim = False
    citation_failures = []

    if intent == "regulation_qa":
        citations_valid = bool(evidence)
    elif intent == "clause_comparison":
        citations_valid = len(evidence) == 2
    elif intent == "gap_analysis":
        answer = state.get("answer", "")
        normalized_answer = answer.casefold()
        has_unsafe_claim = any(
            claim.casefold() in normalized_answer
            for claim in GAP_ANALYSIS_UNSAFE_CLAIMS
        )
        every_gap_has_evidence = bool(evidence) and all(
            isinstance(row, dict) and bool(row.get("evidence"))
            for row in evidence
        )
        has_human_boundary = "人工确认" in answer
        citations_valid = (
            every_gap_has_evidence
            and has_human_boundary
            and not has_unsafe_claim
        )
    else:
        citations_valid = False

    if (
        entailment_evaluator is not None
        and intent in {"regulation_qa", "clause_comparison"}
        and citations_valid
    ):
        validation = validate_citations(
            state.get("answer", ""),
            evidence,
            entailment_evaluator=entailment_evaluator,
            joint_citations=True,
        )
        citations_valid = validation["valid"]
        citation_failures = validation["failures"]

    trace_event = {
        "node": "verify",
        "citations_valid": citations_valid,
    }
    if (
        entailment_evaluator is not None
        and intent in {"regulation_qa", "clause_comparison"}
    ):
        trace_event.update(
            {
                "citation_failures": citation_failures,
                "validation_action": (
                    "pass" if citations_valid else "refuse"
                ),
            }
        )

    update = {
        "citations_valid": citations_valid,
        "trace": state["trace"] + [trace_event],
    }
    if intent == "gap_analysis" and has_unsafe_claim:
        update["answer"] = GAP_ANALYSIS_OVERCLAIM_REFUSAL
    return update


def finish(state: AgentState) -> dict:
    """Return the final status update and terminal trace event."""
    if state.get("final_status") == "cancelled":
        final_status = "cancelled"
    else:
        citations_valid = state["citations_valid"]
        final_status = "completed" if citations_valid else "refused"

    return {
        "final_status": final_status,
        "trace": state["trace"] + [
            {
                "node": "finish",
                "final_status": final_status,
            }
        ],
    }
