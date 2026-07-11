"""Unified retrieval entry point with small-to-big parent expansion."""

from __future__ import annotations

import json
from dataclasses import replace

from ingest.index import PARENTS_STORE
from rag.dense import search_dense
from rag.fusion import reciprocal_rank_fusion
from rag.rerank import rerank_hits
from rag.sparse import search_sparse
from rag.types import RetrievalConfig, SearchHit


def expand_parent_hits(
    child_hits: list[SearchHit], parent_store: dict
) -> list[SearchHit]:
    """Replace child text with full parent text and keep the first hit per parent."""
    seen: set[str] = set()
    expanded: list[SearchHit] = []
    for hit in child_hits:
        if hit.parent_id in seen or hit.parent_id not in parent_store:
            continue
        seen.add(hit.parent_id)
        expanded.append(
            SearchHit(
                chunk_id=hit.chunk_id,
                parent_id=hit.parent_id,
                score=hit.score,
                text=parent_store[hit.parent_id]["text"],
                source_id=hit.source_id,
                version=hit.version,
                section_number=hit.section_number,
                dense_rank=hit.dense_rank,
                sparse_rank=hit.sparse_rank,
                rerank_score=hit.rerank_score,
            )
        )
    return expanded


def retrieve(
    query: str,
    config: RetrievalConfig,
    *,
    source_ids: list[str] | None = None,
) -> list[SearchHit]:
    """Run switchable Dense, Sparse, fusion, parent expansion, and reranking."""
    dense_kwargs = {"k": config.dense_k}
    if source_ids is not None:
        dense_kwargs["source_ids"] = source_ids
    dense_hits = search_dense(query, **dense_kwargs)
    if config.use_sparse:
        sparse_kwargs = {"k": config.sparse_k}
        if source_ids is not None:
            sparse_kwargs["source_ids"] = source_ids
        sparse_hits = search_sparse(query, **sparse_kwargs)
        candidates = reciprocal_rank_fusion(
            dense_hits, sparse_hits, limit=config.fused_k
        )
    else:
        candidates = [
            replace(hit, dense_rank=rank)
            for rank, hit in enumerate(dense_hits[: config.fused_k], start=1)
        ]

    if config.expand_parent:
        parent_store = json.loads(PARENTS_STORE.read_text(encoding="utf-8"))
        candidates = expand_parent_hits(candidates, parent_store)

    if config.use_rerank:
        return rerank_hits(query, candidates, limit=config.rerank_k)
    return candidates[: config.fused_k]
