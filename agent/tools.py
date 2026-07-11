"""Thin deterministic tools over the existing RAG and parent store."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from agent.state import Evidence
from rag.types import SearchHit

SearchBackend = Callable[[str, list[str] | None], list[SearchHit]]
ParentStore = Mapping[str, Mapping[str, Any]]
ClauseLookup = Callable[[str, str, str], Evidence | None]
ClauseMatch = tuple[str, Mapping[str, Any]]


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
