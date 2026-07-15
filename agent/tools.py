"""Thin deterministic tools over the existing RAG and parent store."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

from agent.state import Evidence
from rag.types import SearchHit

SearchBackend = Callable[[str, list[str] | None], list[SearchHit]]
ParentStore = Mapping[str, Mapping[str, Any]]
ClauseLookup = Callable[[str, str, str], Evidence | None]
ClauseMatch = tuple[str, Mapping[str, Any]]
ToolBackendName = Literal["local", "mcp"]
SearchTool = Callable[[str, list[str] | None], list[Evidence]]
ClauseTool = Callable[[str, str, str], Evidence | None]
ComparisonTool = Callable[[dict, dict, list[str]], dict]
MCPCallTool = Callable[[str, dict[str, Any]], Mapping[str, Any]]


class AmbiguousClauseError(LookupError):
    """More than one parent clause matches the same locator."""


def _default_search(
    query: str, source_ids: list[str] | None
) -> list[SearchHit]:
    """Use Task10's selected parent-child Dense + Rerank pipeline."""
    from rag.retrieve import retrieve
    from rag.types import RetrievalConfig

    return retrieve(
        query,
        RetrievalConfig(
            use_sparse=False,
            use_rerank=True,
            expand_parent=True,
        ),
        source_ids=source_ids,
    )


def _searchhit_to_evidence(hit: SearchHit) -> Evidence:
    return {
        "parent_id": hit.parent_id,
        "source_id": hit.source_id,
        "version": hit.version,
        "section_number": hit.section_number,
        "text": hit.text,
        "score": hit.score,
    }


def search_regulation(
    query: str,
    source_ids: list[str] | None = None,
    *,
    search_backend: SearchBackend | None = None,
) -> list[Evidence]:
    """Search regulation evidence, optionally restricted to one source."""
    if not query.strip():
        raise ValueError("query must not be blank")

    backend = search_backend if search_backend is not None else _default_search
    hits = backend(query, source_ids)
    selected_hits = []

    for hit in hits:
        if source_ids is None:
            selected_hits.append(hit)
        elif hit.source_id in source_ids:
            selected_hits.append(hit)

    return [_searchhit_to_evidence(hit) for hit in selected_hits]


def _record_to_evidence(match: ClauseMatch) -> Evidence:
    parent_id, record = match
    return {
        "parent_id": parent_id,
        "source_id": record["metadata"]["source_id"],
        "version": record["metadata"]["version"],
        "section_number": record["number"],
        "text": record["text"],
        "score": None,
    }


def _load_parent_store() -> ParentStore:
    """Read the real parent store only when a caller does not inject one."""
    from ingest.index import PARENTS_STORE

    payload = json.loads(PARENTS_STORE.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("parent store must be a JSON object")
    return payload


def get_clause(
    source_id: str,
    version: str,
    section_number: str,
    *,
    parent_store: ParentStore | None = None,
) -> Evidence | None:
    """Return the one clause matching source, version, and section exactly."""
    for field_name, value in (
        ("source_id", source_id),
        ("version", version),
        ("section_number", section_number),
    ):
        if not value.strip():
            raise ValueError(f"{field_name} must not be blank")

    store = parent_store if parent_store is not None else _load_parent_store()
    matches: list[ClauseMatch] = []

    for parent_id, record in store.items():
        try:
            metadata = record["metadata"]
            record_source_id = metadata["source_id"]
            record_version = metadata["version"]
            record_number = record["number"]
            record_text = record["text"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"invalid parent record {parent_id}: {exc}"
            ) from exc
        if not all(
            isinstance(value, str)
            for value in (
                record_source_id,
                record_version,
                record_number,
                record_text,
            )
        ):
            raise ValueError(
                f"invalid parent record {parent_id}: fields must be strings"
            )

        if (
            record_source_id == source_id
            and record_version == version
            and record_number == section_number
        ):
            matches.append((parent_id, record))

    if not matches:
        return None
    if len(matches) > 1:
        raise AmbiguousClauseError(
            "ambiguous clause: "
            f"{source_id}@{version}#{section_number} matched {len(matches)} records"
        )

    return _record_to_evidence(matches[0])


def compare_clauses(
    left: dict,
    right: dict,
    dimensions: list[str],
    *,
    clause_lookup: ClauseLookup | None = None,
) -> dict:
    """Return left and right clause evidence without using an LLM."""
    lookup = clause_lookup if clause_lookup is not None else get_clause
    left_evidence = lookup(
        left["source_id"],
        left["version"],
        left["section_number"],
    )
    right_evidence = lookup(
        right["source_id"],
        right["version"],
        right["section_number"],
    )
    return {
        "left": left_evidence,
        "right": right_evidence,
        "dimensions": list(dimensions),
    }


class AgentToolBackend(Protocol):
    """Common domain interface consumed by the Agent graph."""

    backend_name: ToolBackendName

    def search_regulation(
        self,
        query: str,
        source_ids: list[str] | None = None,
    ) -> list[Evidence]: ...

    def get_clause(
        self,
        source_id: str,
        version: str,
        section_number: str,
    ) -> Evidence | None: ...

    def compare_clauses(
        self,
        left: dict,
        right: dict,
        dimensions: list[str],
    ) -> dict: ...


class ToolBackendError(RuntimeError):
    """Base error for a configured Agent tool backend."""


class MCPBackendUnavailableError(ToolBackendError):
    """The configured MCP transport or server could not complete a call."""

    def __init__(self, tool_name: str, cause: Exception) -> None:
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(
            f"mcp backend unavailable while calling {tool_name}: {cause}"
        )


class MCPToolError(ToolBackendError):
    """An MCP tool returned a structured domain or protocol error."""

    def __init__(
        self,
        tool_name: str,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.code = code
        self.details = dict(details or {})
        super().__init__(f"mcp tool {tool_name} failed [{code}]: {message}")


class LocalToolBackend:
    """Call the in-process deterministic tools without protocol overhead."""

    backend_name: Literal["local"] = "local"

    def __init__(
        self,
        *,
        search_tool: SearchTool = search_regulation,
        clause_tool: ClauseTool = get_clause,
        comparison_tool: ComparisonTool = compare_clauses,
    ) -> None:
        self._search_tool = search_tool
        self._clause_tool = clause_tool
        self._comparison_tool = comparison_tool

    def search_regulation(
        self,
        query: str,
        source_ids: list[str] | None = None,
    ) -> list[Evidence]:
        return self._search_tool(query, source_ids)

    def get_clause(
        self,
        source_id: str,
        version: str,
        section_number: str,
    ) -> Evidence | None:
        return self._clause_tool(source_id, version, section_number)

    def compare_clauses(
        self,
        left: dict,
        right: dict,
        dimensions: list[str],
    ) -> dict:
        return self._comparison_tool(left, right, dimensions)


@dataclass(frozen=True)
class MCPStdioConfig:
    """Process configuration for one official MCP stdio client call."""

    command: str = sys.executable
    args: tuple[str, ...] = ("-m", "mcp_server.server")
    cwd: str | Path | None = None
    read_timeout_seconds: float = 300.0


class MCPStdioToolCaller:
    """Call an MCP tool through a short-lived official stdio session."""

    def __init__(self, config: MCPStdioConfig | None = None) -> None:
        self.config = config or MCPStdioConfig()

    def __call__(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Mapping[str, Any]:
        import anyio

        return anyio.run(self._call_async, tool_name, arguments)

    async def _call_async(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Mapping[str, Any]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        parameters = StdioServerParameters(
            command=self.config.command,
            args=list(self.config.args),
            cwd=self.config.cwd,
            encoding="utf-8",
            encoding_error_handler="strict",
        )
        timeout = timedelta(seconds=self.config.read_timeout_seconds)

        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timeout,
            ) as session:
                await session.initialize()
                result = await session.call_tool(
                    tool_name,
                    arguments=arguments,
                    read_timeout_seconds=timeout,
                )

        if result.isError:
            messages = [
                item.text
                for item in result.content
                if hasattr(item, "text")
            ]
            raise MCPToolError(
                tool_name,
                "protocol_error",
                "\n".join(messages) or "MCP tool call failed",
            )

        payload = result.structuredContent
        if not isinstance(payload, Mapping):
            raise MCPToolError(
                tool_name,
                "invalid_response",
                "MCP response has no structured content",
            )
        return dict(payload)


class MCPToolBackend:
    """Expose MCP calls through the same domain interface as local tools."""

    backend_name: Literal["mcp"] = "mcp"

    def __init__(
        self,
        *,
        call_mcp_tool: MCPCallTool | None = None,
        stdio_config: MCPStdioConfig | None = None,
    ) -> None:
        self._call_mcp_tool = (
            call_mcp_tool
            if call_mcp_tool is not None
            else MCPStdioToolCaller(stdio_config)
        )

    def _call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        try:
            payload = self._call_mcp_tool(tool_name, arguments)
        except ToolBackendError:
            raise
        except Exception as exc:
            raise MCPBackendUnavailableError(tool_name, exc) from exc

        if not isinstance(payload, Mapping):
            raise MCPToolError(
                tool_name,
                "invalid_response",
                "MCP tool response must be an object",
            )

        if payload.get("ok") is not True:
            error = payload.get("error")
            if not isinstance(error, Mapping):
                raise MCPToolError(
                    tool_name,
                    "invalid_response",
                    "failed MCP response has no structured error",
                )
            code = error.get("code")
            message = error.get("message")
            details = error.get("details")
            raise MCPToolError(
                tool_name,
                code if isinstance(code, str) else "tool_failure",
                message if isinstance(message, str) else "MCP tool failed",
                details if isinstance(details, Mapping) else None,
            )

        return payload.get("data")

    def search_regulation(
        self,
        query: str,
        source_ids: list[str] | None = None,
    ) -> list[Evidence]:
        data = self._call(
            "search_regulation",
            {"query": query, "source_ids": source_ids},
        )
        if not isinstance(data, list) or not all(
            isinstance(item, Mapping) for item in data
        ):
            raise MCPToolError(
                "search_regulation",
                "invalid_response",
                "search data must be a list of evidence objects",
            )
        return [dict(item) for item in data]

    def get_clause(
        self,
        source_id: str,
        version: str,
        section_number: str,
    ) -> Evidence | None:
        data = self._call(
            "get_clause",
            {
                "source_id": source_id,
                "version": version,
                "section_number": section_number,
            },
        )
        if data is None:
            return None
        if not isinstance(data, Mapping):
            raise MCPToolError(
                "get_clause",
                "invalid_response",
                "clause data must be an evidence object or null",
            )
        return dict(data)

    def compare_clauses(
        self,
        left: dict,
        right: dict,
        dimensions: list[str],
    ) -> dict:
        data = self._call(
            "compare_clauses",
            {
                "left": dict(left),
                "right": dict(right),
                "dimensions": list(dimensions),
            },
        )
        if not isinstance(data, Mapping):
            raise MCPToolError(
                "compare_clauses",
                "invalid_response",
                "comparison data must be an object",
            )
        return dict(data)


def create_tool_backend(
    backend: str = "local",
    *,
    mcp_call_tool: MCPCallTool | None = None,
    mcp_stdio_config: MCPStdioConfig | None = None,
) -> AgentToolBackend:
    """Build the configured tool backend without implicit fallback."""
    if backend == "local":
        if mcp_call_tool is not None or mcp_stdio_config is not None:
            raise ValueError("MCP options require the mcp tool backend")
        return LocalToolBackend()
    if backend == "mcp":
        return MCPToolBackend(
            call_mcp_tool=mcp_call_tool,
            stdio_config=mcp_stdio_config,
        )
    raise ValueError(f"unknown tool backend: {backend}")
