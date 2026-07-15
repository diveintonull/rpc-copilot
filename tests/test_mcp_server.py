"""Protocol contracts for the GRC MCP server."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import anyio
import pytest

from mcp_server import server as server_module
from mcp_server.server import ToolBackend, create_server


Evidence = dict[str, object]


def evidence(
    parent_id: str,
    *,
    source_id: str,
    version: str,
    section_number: str,
    text: str,
    score: float | None,
) -> Evidence:
    return {
        "parent_id": parent_id,
        "source_id": source_id,
        "version": version,
        "section_number": section_number,
        "text": text,
        "score": score,
    }


def make_backend(
    *,
    search: Callable | None = None,
    get: Callable | None = None,
    compare: Callable | None = None,
) -> ToolBackend:
    return ToolBackend(
        search_regulation=search or (lambda _query, _source_ids=None: []),
        get_clause=get or (
            lambda _source_id, _version, _section_number: None
        ),
        compare_clauses=compare or (
            lambda _left, _right, dimensions: {
                "left": None,
                "right": None,
                "dimensions": list(dimensions),
            }
        ),
    )


def call_tool(server, name: str, arguments: dict) -> dict:
    result = anyio.run(lambda: server.call_tool(name, arguments))
    assert isinstance(result, tuple)
    _content, structured = result
    assert isinstance(structured, dict)
    return structured


def test_server_exposes_stable_tool_names_and_input_schemas() -> None:
    server = create_server(backend=make_backend())

    tools = {
        tool.name: tool for tool in anyio.run(server.list_tools)
    }

    assert set(tools) == {
        "search_regulation",
        "get_clause",
        "compare_clauses",
    }
    assert set(tools["search_regulation"].inputSchema["properties"]) == {
        "query",
        "source_ids",
    }
    assert tools["search_regulation"].inputSchema["required"] == ["query"]
    assert set(tools["get_clause"].inputSchema["properties"]) == {
        "source_id",
        "version",
        "section_number",
    }
    assert tools["get_clause"].inputSchema["required"] == [
        "source_id",
        "version",
        "section_number",
    ]
    assert set(tools["compare_clauses"].inputSchema["properties"]) == {
        "left",
        "right",
        "dimensions",
    }
    assert tools["compare_clauses"].inputSchema["required"] == [
        "left",
        "right",
        "dimensions",
    ]
    for tool in tools.values():
        assert set(tool.outputSchema["properties"]) == {
            "ok",
            "data",
            "error",
        }
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.idempotentHint is True
        assert tool.annotations.openWorldHint is False


@pytest.mark.parametrize(
    ("tool_name", "arguments", "message"),
    [
        (
            "search_regulation",
            {"query": "   ", "source_ids": None},
            "query",
        ),
        (
            "get_clause",
            {
                "source_id": " ",
                "version": "2019",
                "section_number": "7.1.4.1",
            },
            "source_id",
        ),
        (
            "compare_clauses",
            {
                "left": {
                    "source_id": " ",
                    "version": "2019",
                    "section_number": "7.1.4.1",
                },
                "right": {
                    "source_id": "GDPR",
                    "version": "2016-679",
                    "section_number": "32",
                },
                "dimensions": ["scope"],
            },
            "source_id",
        ),
    ],
)
def test_invalid_domain_arguments_return_structured_errors(
    tool_name: str,
    arguments: dict,
    message: str,
) -> None:
    result = call_tool(create_server(), tool_name, arguments)

    assert result["ok"] is False
    assert result["data"] in (None, [])
    assert result["error"]["code"] == "invalid_argument"
    assert message in result["error"]["message"]
    assert result["error"]["details"]["exception"] in {
        "KeyError",
        "TypeError",
        "ValueError",
    }


def test_backend_failure_is_a_structured_tool_error() -> None:
    def fail_search(_query: str, _source_ids: list[str] | None = None):
        raise RuntimeError("qdrant unavailable")

    server = create_server(backend=make_backend(search=fail_search))

    result = call_tool(
        server,
        "search_regulation",
        {"query": "identity", "source_ids": None},
    )

    assert result == {
        "ok": False,
        "data": [],
        "error": {
            "code": "tool_failure",
            "message": "qdrant unavailable",
            "details": {"exception": "RuntimeError"},
        },
    }


def test_search_result_preserves_versioned_evidence() -> None:
    item = evidence(
        "GBT-22239@2019#7.1.4.1",
        source_id="GBT-22239",
        version="2019",
        section_number="7.1.4.1",
        text="身份鉴别要求",
        score=0.91,
    )
    calls = []

    def search(query: str, source_ids: list[str] | None = None):
        calls.append((query, source_ids))
        return [item]

    server = create_server(backend=make_backend(search=search))

    result = call_tool(
        server,
        "search_regulation",
        {"query": "身份鉴别", "source_ids": ["GBT-22239"]},
    )

    assert calls == [("身份鉴别", ["GBT-22239"])]
    assert result == {"ok": True, "data": [item], "error": None}


def test_get_clause_result_preserves_locator_and_text() -> None:
    item = evidence(
        "GDPR@2016-679#32",
        source_id="GDPR",
        version="2016-679",
        section_number="32",
        text="Security of processing.",
        score=None,
    )
    calls = []

    def get(source_id: str, version: str, section_number: str):
        calls.append((source_id, version, section_number))
        return item

    server = create_server(backend=make_backend(get=get))

    result = call_tool(
        server,
        "get_clause",
        {
            "source_id": "GDPR",
            "version": "2016-679",
            "section_number": "32",
        },
    )

    assert calls == [("GDPR", "2016-679", "32")]
    assert result == {"ok": True, "data": item, "error": None}


def test_compare_result_preserves_both_sides_and_dimensions() -> None:
    left = evidence(
        "GBT-22239@2019#7.1.4.1",
        source_id="GBT-22239",
        version="2019",
        section_number="7.1.4.1",
        text="身份鉴别要求",
        score=None,
    )
    right = evidence(
        "GDPR@2016-679#32",
        source_id="GDPR",
        version="2016-679",
        section_number="32",
        text="Security of processing.",
        score=None,
    )
    calls = []

    def compare(left_ref: dict, right_ref: dict, dimensions: list[str]):
        calls.append((left_ref, right_ref, dimensions))
        return {"left": left, "right": right, "dimensions": dimensions}

    server = create_server(backend=make_backend(compare=compare))
    arguments = {
        "left": {
            "source_id": "GBT-22239",
            "version": "2019",
            "section_number": "7.1.4.1",
        },
        "right": {
            "source_id": "GDPR",
            "version": "2016-679",
            "section_number": "32",
        },
        "dimensions": ["scope", "obligation"],
    }

    result = call_tool(server, "compare_clauses", arguments)

    assert calls == [
        (arguments["left"], arguments["right"], arguments["dimensions"])
    ]
    assert result == {
        "ok": True,
        "data": {
            "left": left,
            "right": right,
            "dimensions": ["scope", "obligation"],
        },
        "error": None,
    }


def test_mcp_layer_does_not_reimplement_rag_or_parent_store_logic() -> None:
    source = inspect.getsource(server_module)

    assert "from rag" not in source
    assert "PARENTS_STORE" not in source
    assert "qdrant" not in source.casefold()
    assert "agent_tools.search_regulation" in source
    assert "agent_tools.get_clause" in source
    assert "agent_tools.compare_clauses" in source
