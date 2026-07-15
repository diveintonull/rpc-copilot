"""Contract tests for deterministic Agent tools."""

from __future__ import annotations

import json
import inspect
from typing import get_type_hints

import pytest

from agent import tools as tools_module
from agent.state import AgentState
from agent.tools import (
    LocalToolBackend,
    MCPBackendUnavailableError,
    MCPToolBackend,
    MCPToolError,
    create_tool_backend,
    search_regulation,
)
from rag.types import SearchHit


def hit(
    parent_id: str,
    *,
    source_id: str,
    version: str,
    section_number: str,
    text: str,
    score: float,
) -> SearchHit:
    """Build one small, hand-checkable retrieval result."""
    return SearchHit(
        chunk_id=f"{parent_id}:0",
        parent_id=parent_id,
        score=score,
        text=text,
        source_id=source_id,
        version=version,
        section_number=section_number,
    )


def test_search_regulation_filters_source_and_serializes_evidence() -> None:
    calls: list[str] = []

    requested_filters: list[list[str] | None] = []

    def fake_search(
        query: str, source_ids: list[str] | None
    ) -> list[SearchHit]:
        calls.append(query)
        requested_filters.append(source_ids)
        return [
            hit(
                "GBT-22239@2019#7.1.4.1",
                source_id="GBT-22239",
                version="2019",
                section_number="7.1.4.1",
                text="应对登录用户进行身份鉴别。",
                score=0.91,
            ),
            hit(
                "GDPR@2016-679#32",
                source_id="GDPR",
                version="2016-679",
                section_number="32",
                text="Security of processing.",
                score=0.88,
            ),
        ]

    result = search_regulation(
        "身份鉴别",
        source_ids=["GBT-22239"],
        search_backend=fake_search,
    )

    assert calls == ["身份鉴别"]
    assert requested_filters == [["GBT-22239"]]
    assert result == [
        {
            "parent_id": "GBT-22239@2019#7.1.4.1",
            "source_id": "GBT-22239",
            "version": "2019",
            "section_number": "7.1.4.1",
            "text": "应对登录用户进行身份鉴别。",
            "score": 0.91,
        }
    ]


def test_search_regulation_rejects_blank_query_before_searching() -> None:
    def must_not_search(
        _query: str, _source_ids: list[str] | None
    ) -> list[SearchHit]:
        raise AssertionError("blank query must not reach the search backend")

    with pytest.raises(ValueError, match="query"):
        search_regulation("   ", search_backend=must_not_search)


def test_search_regulation_without_source_filter_preserves_all_hits() -> None:
    hits = [
        hit(
            "left",
            source_id="GBT-22239",
            version="2019",
            section_number="7.1",
            text="left",
            score=0.9,
        ),
        hit(
            "right",
            source_id="GDPR",
            version="2016-679",
            section_number="32",
            text="right",
            score=0.8,
        ),
    ]

    result = search_regulation(
        "security", search_backend=lambda _query, _source_ids: hits
    )

    assert [item["parent_id"] for item in result] == ["left", "right"]


def test_search_regulation_returns_empty_when_source_does_not_match() -> None:
    result = search_regulation(
        "security",
        source_ids=["not-present"],
        search_backend=lambda _query, _source_ids: [
            hit(
                "gdpr",
                source_id="GDPR",
                version="2016-679",
                section_number="32",
                text="security",
                score=0.8,
            )
        ],
    )

    assert result == []


def test_search_regulation_does_not_hide_backend_failure() -> None:
    def failing_search(
        _query: str, _source_ids: list[str] | None
    ) -> list[SearchHit]:
        raise RuntimeError("qdrant unavailable")

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        search_regulation("security", search_backend=failing_search)


def test_search_regulation_uses_default_backend_when_none_is_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_default(
        query: str, source_ids: list[str] | None
    ) -> list[SearchHit]:
        calls.append(query)
        assert source_ids == ["GBT-22239"]
        return []

    monkeypatch.setattr(tools_module, "_default_search", fake_default, raising=False)

    assert search_regulation("身份鉴别", ["GBT-22239"]) == []
    assert calls == ["身份鉴别"]


def test_get_clause_matches_source_version_and_section() -> None:
    parent_store = {
        "GBT-22239@2019#7.1.4.1": {
            "text": "2019 版身份鉴别要求",
            "number": "7.1.4.1",
            "metadata": {"source_id": "GBT-22239", "version": "2019"},
        },
        "GBT-22239@2019#7.1.4.2": {
            "text": "2019 版访问控制要求",
            "number": "7.1.4.2",
            "metadata": {"source_id": "GBT-22239", "version": "2019"},
        },
        "GBT-22239@2024#7.1.4.1": {
            "text": "2024 版身份鉴别要求",
            "number": "7.1.4.1",
            "metadata": {"source_id": "GBT-22239", "version": "2024"},
        },
    }

    result = tools_module.get_clause(
        "GBT-22239",
        "2019",
        "7.1.4.1",
        parent_store=parent_store,
    )

    assert result == {
        "parent_id": "GBT-22239@2019#7.1.4.1",
        "source_id": "GBT-22239",
        "version": "2019",
        "section_number": "7.1.4.1",
        "text": "2019 版身份鉴别要求",
        "score": None,
    }


def test_get_clause_returns_none_when_clause_is_missing() -> None:
    parent_store = {
        "GDPR@2016-679#32": {
            "text": "Security of processing.",
            "number": "32",
            "metadata": {"source_id": "GDPR", "version": "2016-679"},
        }
    }

    assert (
        tools_module.get_clause(
            "GBT-22239",
            "2019",
            "7.1.4.1",
            parent_store=parent_store,
        )
        is None
    )


def test_get_clause_rejects_ambiguous_duplicate_matches() -> None:
    parent_store = {
        "duplicate-a": {
            "text": "first copy",
            "number": "7.1.4.1",
            "metadata": {"source_id": "GBT-22239", "version": "2019"},
        },
        "duplicate-b": {
            "text": "second copy",
            "number": "7.1.4.1",
            "metadata": {"source_id": "GBT-22239", "version": "2019"},
        },
    }

    with pytest.raises(tools_module.AmbiguousClauseError):
        tools_module.get_clause(
            "GBT-22239",
            "2019",
            "7.1.4.1",
            parent_store=parent_store,
        )


def test_get_clause_reports_the_id_of_a_malformed_parent_record() -> None:
    parent_store = {
        "broken-parent": {
            "text": "missing metadata",
            "number": "7.1.4.1",
        }
    }

    with pytest.raises(ValueError, match="broken-parent"):
        tools_module.get_clause(
            "GBT-22239",
            "2019",
            "7.1.4.1",
            parent_store=parent_store,
        )


def test_get_clause_loads_default_parent_store_when_none_is_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_store = {
        "GBT-22239@2019#7.1.4.1": {
            "text": "身份鉴别要求",
            "number": "7.1.4.1",
            "metadata": {"source_id": "GBT-22239", "version": "2019"},
        }
    }
    loads = 0

    def fake_load():
        nonlocal loads
        loads += 1
        return parent_store

    monkeypatch.setattr(tools_module, "_load_parent_store", fake_load, raising=False)

    result = tools_module.get_clause("GBT-22239", "2019", "7.1.4.1")

    assert result["parent_id"] == "GBT-22239@2019#7.1.4.1"
    assert loads == 1


@pytest.mark.parametrize(
    ("source_id", "version", "section_number", "message"),
    [
        (" ", "2019", "7.1.4.1", "source_id"),
        ("GBT-22239", " ", "7.1.4.1", "version"),
        ("GBT-22239", "2019", " ", "section_number"),
    ],
)
def test_get_clause_rejects_blank_locator_fields(
    source_id: str,
    version: str,
    section_number: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        tools_module.get_clause(
            source_id,
            version,
            section_number,
            parent_store={},
        )


def test_compare_clauses_preserves_left_and_right_evidence() -> None:
    left_ref = {
        "source_id": "GBT-22239",
        "version": "2019",
        "section_number": "7.1.4.1",
    }
    right_ref = {
        "source_id": "GDPR",
        "version": "2016-679",
        "section_number": "32",
    }
    left_evidence = {
        "parent_id": "GBT-22239@2019#7.1.4.1",
        "source_id": "GBT-22239",
        "version": "2019",
        "section_number": "7.1.4.1",
        "text": "身份鉴别要求",
        "score": None,
    }
    right_evidence = {
        "parent_id": "GDPR@2016-679#32",
        "source_id": "GDPR",
        "version": "2016-679",
        "section_number": "32",
        "text": "Security of processing.",
        "score": None,
    }
    calls: list[dict[str, str]] = []

    def fake_lookup(
        source_id: str,
        version: str,
        section_number: str,
    ):
        ref = {
            "source_id": source_id,
            "version": version,
            "section_number": section_number,
        }
        calls.append(ref)
        return left_evidence if source_id == "GBT-22239" else right_evidence

    result = tools_module.compare_clauses(
        left_ref,
        right_ref,
        ["scope", "obligation"],
        clause_lookup=fake_lookup,
    )

    assert result == {
        "left": left_evidence,
        "right": right_evidence,
        "dimensions": ["scope", "obligation"],
    }
    assert calls == [left_ref, right_ref]


def test_compare_clauses_uses_get_clause_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refs = [
        {
            "source_id": "GBT-22239",
            "version": "2019",
            "section_number": "7.1.4.1",
        },
        {
            "source_id": "GDPR",
            "version": "2016-679",
            "section_number": "32",
        },
    ]
    calls: list[dict[str, str]] = []

    def fake_get_clause(source_id: str, version: str, section_number: str):
        ref = {
            "source_id": source_id,
            "version": version,
            "section_number": section_number,
        }
        calls.append(ref)
        return {
            "parent_id": f"{source_id}@{version}#{section_number}",
            **ref,
            "text": section_number,
            "score": None,
        }

    monkeypatch.setattr(tools_module, "get_clause", fake_get_clause)

    result = tools_module.compare_clauses(refs[0], refs[1], ["obligation"])

    assert calls == refs
    assert result["left"]["source_id"] == "GBT-22239"
    assert result["right"]["source_id"] == "GDPR"


def test_agent_state_records_routing_evidence_and_final_outcome() -> None:
    hints = get_type_hints(AgentState)

    assert set(hints) == {
        "request_id",
        "query",
        "control_text",
        "intent",
        "active_skill",
        "skill_text",
        "tool_calls",
        "evidence",
        "answer",
        "citations_valid",
        "citation_failures",
        "retry_action",
        "retry_count",
        "final_status",
        "trace",
    }

    state: AgentState = {
        "request_id": "req-001",
        "query": "对比身份鉴别要求",
        "control_text": "",
        "intent": "clause_comparison",
        "active_skill": "compare-regulations",
        "skill_text": "preserve both sides",
        "tool_calls": [],
        "evidence": [],
        "answer": "",
        "citations_valid": False,
        "retry_count": 0,
        "final_status": "completed",
        "trace": [{"node": "route_intent"}],
    }

    assert json.loads(json.dumps(state, ensure_ascii=False)) == state


def test_agent_package_exports_stable_public_contract() -> None:
    import agent

    assert {
        "AgentState",
        "ClauseComparison",
        "ClauseRef",
        "Evidence",
        "compare_clauses",
        "get_clause",
        "search_regulation",
    } <= set(dir(agent))


def test_tool_core_has_no_llm_or_langgraph_dependency() -> None:
    source = inspect.getsource(tools_module).casefold()
    parameters = inspect.signature(tools_module.compare_clauses).parameters

    assert "openai" not in source
    assert "langgraph" not in source
    assert "llm" not in parameters


def test_local_and_mcp_backends_return_the_same_domain_structures() -> None:
    clause = {
        "parent_id": "GBT-22239@2019#7.1.4.1",
        "source_id": "GBT-22239",
        "version": "2019",
        "section_number": "7.1.4.1",
        "text": "应对登录用户进行身份鉴别。",
        "score": 0.91,
    }
    comparison = {
        "left": clause,
        "right": None,
        "dimensions": ["scope"],
    }
    left_ref = {
        "source_id": "GBT-22239",
        "version": "2019",
        "section_number": "7.1.4.1",
    }
    right_ref = {
        "source_id": "GDPR",
        "version": "2016-679",
        "section_number": "32",
    }
    local = LocalToolBackend(
        search_tool=lambda _query, _source_ids: [clause],
        clause_tool=lambda _source_id, _version, _section: clause,
        comparison_tool=lambda _left, _right, _dimensions: comparison,
    )
    mcp_calls: list[tuple[str, dict]] = []

    def call_mcp_tool(name: str, arguments: dict) -> dict:
        mcp_calls.append((name, arguments))
        data = {
            "search_regulation": [clause],
            "get_clause": clause,
            "compare_clauses": comparison,
        }[name]
        return {"ok": True, "data": data, "error": None}

    mcp = MCPToolBackend(call_mcp_tool=call_mcp_tool)

    local_results = {
        "search": local.search_regulation("身份鉴别", ["GBT-22239"]),
        "get": local.get_clause("GBT-22239", "2019", "7.1.4.1"),
        "compare": local.compare_clauses(
            left_ref,
            right_ref,
            ["scope"],
        ),
    }
    mcp_results = {
        "search": mcp.search_regulation("身份鉴别", ["GBT-22239"]),
        "get": mcp.get_clause("GBT-22239", "2019", "7.1.4.1"),
        "compare": mcp.compare_clauses(
            left_ref,
            right_ref,
            ["scope"],
        ),
    }

    assert mcp_results == local_results
    assert mcp_calls == [
        (
            "search_regulation",
            {"query": "身份鉴别", "source_ids": ["GBT-22239"]},
        ),
        (
            "get_clause",
            {
                "source_id": "GBT-22239",
                "version": "2019",
                "section_number": "7.1.4.1",
            },
        ),
        (
            "compare_clauses",
            {
                "left": left_ref,
                "right": right_ref,
                "dimensions": ["scope"],
            },
        ),
    ]


def test_mcp_backend_unavailable_is_explicit_and_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def local_must_not_run(*_args, **_kwargs):
        raise AssertionError("MCP mode must not fall back to local tools")

    def unavailable_mcp(_name: str, _arguments: dict) -> dict:
        raise ConnectionError("server process exited")

    monkeypatch.setattr(tools_module, "search_regulation", local_must_not_run)
    backend = MCPToolBackend(call_mcp_tool=unavailable_mcp)

    with pytest.raises(
        MCPBackendUnavailableError,
        match=r"mcp backend unavailable.*search_regulation.*server process exited",
    ):
        backend.search_regulation("身份鉴别")


def test_mcp_backend_preserves_structured_tool_error() -> None:
    backend = MCPToolBackend(
        call_mcp_tool=lambda _name, _arguments: {
            "ok": False,
            "data": None,
            "error": {
                "code": "invalid_argument",
                "message": "query must not be blank",
                "details": {"exception": "ValueError"},
            },
        }
    )

    with pytest.raises(MCPToolError) as error_info:
        backend.search_regulation("   ")

    assert error_info.value.code == "invalid_argument"
    assert error_info.value.details == {"exception": "ValueError"}
    assert "query must not be blank" in str(error_info.value)


def test_tool_backend_factory_defaults_to_local_and_rejects_unknown_mode() -> None:
    assert isinstance(create_tool_backend(), LocalToolBackend)
    assert isinstance(
        create_tool_backend(
            "mcp",
            mcp_call_tool=lambda _name, _arguments: {
                "ok": True,
                "data": None,
                "error": None,
            },
        ),
        MCPToolBackend,
    )

    with pytest.raises(ValueError, match="unknown tool backend"):
        create_tool_backend("remote")

    with pytest.raises(ValueError, match="MCP options require"):
        create_tool_backend(
            mcp_call_tool=lambda _name, _arguments: {
                "ok": True,
                "data": None,
                "error": None,
            }
        )
