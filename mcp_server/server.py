"""Expose the deterministic GRC tools through the MCP protocol."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict

from agent import tools as agent_tools


class ClauseRefInput(BaseModel):
    """Exact versioned locator for one regulation clause."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    version: str
    section_number: str


class EvidencePayload(BaseModel):
    """Protocol-safe copy of the stable Agent evidence structure."""

    model_config = ConfigDict(extra="forbid")

    parent_id: str
    source_id: str
    version: str
    section_number: str
    text: str
    score: float | None


ErrorCode = Literal[
    "invalid_argument",
    "ambiguous_clause",
    "tool_failure",
]


class ErrorPayload(BaseModel):
    """Machine-readable failure returned inside every tool response."""

    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    message: str
    details: dict[str, str]


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    data: list[EvidencePayload]
    error: ErrorPayload | None


class GetClauseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    data: EvidencePayload | None
    error: ErrorPayload | None


class ComparisonPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: EvidencePayload | None
    right: EvidencePayload | None
    dimensions: list[str]


class CompareResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    data: ComparisonPayload | None
    error: ErrorPayload | None


SearchFn = Callable[[str, list[str] | None], list[dict]]
GetClauseFn = Callable[[str, str, str], dict | None]
CompareFn = Callable[[dict, dict, list[str]], dict]


@dataclass(frozen=True, slots=True)
class ToolBackend:
    """Injectable references to the existing deterministic domain tools."""

    search_regulation: SearchFn
    get_clause: GetClauseFn
    compare_clauses: CompareFn


DEFAULT_BACKEND = ToolBackend(
    search_regulation=agent_tools.search_regulation,
    get_clause=agent_tools.get_clause,
    compare_clauses=agent_tools.compare_clauses,
)

READ_ONLY_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


def _error_payload(exc: Exception) -> ErrorPayload:
    if isinstance(exc, agent_tools.AmbiguousClauseError):
        code: ErrorCode = "ambiguous_clause"
    elif isinstance(exc, (KeyError, TypeError, ValueError)):
        code = "invalid_argument"
    else:
        code = "tool_failure"
    return ErrorPayload(
        code=code,
        message=str(exc),
        details={"exception": type(exc).__name__},
    )


def _evidence_payload(item: dict) -> EvidencePayload:
    return EvidencePayload.model_validate(item)


def create_server(backend: ToolBackend | None = None) -> FastMCP:
    """Build a stdio-ready server over one injectable domain backend."""
    selected = backend if backend is not None else DEFAULT_BACKEND
    mcp = FastMCP(
        name="grc-copilot",
        instructions=(
            "Read-only access to versioned regulation search, exact clause "
            "lookup, and deterministic clause comparison."
        ),
        json_response=True,
    )

    @mcp.tool(
        name="search_regulation",
        annotations=READ_ONLY_ANNOTATIONS,
        structured_output=True,
    )
    def search_regulation(
        query: str,
        source_ids: list[str] | None = None,
    ) -> SearchResult:
        """Search versioned regulation evidence with an optional source filter."""
        try:
            raw_items = selected.search_regulation(query, source_ids)
            items = [_evidence_payload(item) for item in raw_items]
            return SearchResult(ok=True, data=items, error=None)
        except Exception as exc:
            return SearchResult(ok=False, data=[], error=_error_payload(exc))

    @mcp.tool(
        name="get_clause",
        annotations=READ_ONLY_ANNOTATIONS,
        structured_output=True,
    )
    def get_clause(
        source_id: str,
        version: str,
        section_number: str,
    ) -> GetClauseResult:
        """Get one clause by exact source, version, and section number."""
        try:
            raw_item = selected.get_clause(source_id, version, section_number)
            item = _evidence_payload(raw_item) if raw_item is not None else None
            return GetClauseResult(ok=True, data=item, error=None)
        except Exception as exc:
            return GetClauseResult(
                ok=False,
                data=None,
                error=_error_payload(exc),
            )

    @mcp.tool(
        name="compare_clauses",
        annotations=READ_ONLY_ANNOTATIONS,
        structured_output=True,
    )
    def compare_clauses(
        left: ClauseRefInput,
        right: ClauseRefInput,
        dimensions: list[str],
    ) -> CompareResult:
        """Resolve two exact clauses and preserve both sides for comparison."""
        try:
            raw = selected.compare_clauses(
                left.model_dump(),
                right.model_dump(),
                dimensions,
            )
            data = ComparisonPayload(
                left=(
                    _evidence_payload(raw["left"])
                    if raw["left"] is not None
                    else None
                ),
                right=(
                    _evidence_payload(raw["right"])
                    if raw["right"] is not None
                    else None
                ),
                dimensions=list(raw["dimensions"]),
            )
            return CompareResult(ok=True, data=data, error=None)
        except Exception as exc:
            return CompareResult(
                ok=False,
                data=None,
                error=_error_payload(exc),
            )

    return mcp


server = create_server()


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
