"""Chinese BM25 retrieval over the same child chunks used by Dense search."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import math
import re

import jieba
from rank_bm25 import BM25Okapi

from ingest.index import build_corpus
from rag.types import SearchHit


def tokenize_zh(text: str) -> list[str]:
    """Tokenize Chinese without relying on whitespace-separated words."""
    return [
        token.casefold()
        for raw in jieba.lcut(text, cut_all=False)
        if (token := raw.strip()) and re.search(r"\w", token, re.UNICODE)
    ]


class SparseIndex:
    """Small in-memory BM25 index whose documents are stable SearchHit values."""

    def __init__(self, hits: list[SearchHit]) -> None:
        self._hits = list(hits)
        corpus = [tokenize_zh(hit.text) for hit in self._hits]
        self._bm25 = BM25Okapi(corpus) if corpus and any(corpus) else None

    def search(
        self,
        query: str,
        *,
        k: int,
        source_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        if k <= 0 or self._bm25 is None:
            return []
        if source_ids == []:
            return []
        tokens = tokenize_zh(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        allowed_sources = set(source_ids) if source_ids is not None else None
        ranked = sorted(
            (
                (index, float(score))
                for index, score in enumerate(scores)
                if allowed_sources is None
                or self._hits[index].source_id in allowed_sources
                if math.isfinite(float(score)) and float(score) > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )[:k]
        return [
            replace(self._hits[index], score=score, sparse_rank=rank)
            for rank, (index, score) in enumerate(ranked, start=1)
        ]


@lru_cache(maxsize=1)
def _get_sparse_index() -> SparseIndex:
    _parents, children = build_corpus()
    hits = [
        SearchHit(
            chunk_id=chunk.id,
            parent_id=chunk.metadata["parent_id"],
            score=0.0,
            text=chunk.text,
            source_id=chunk.metadata.get(
                "source_id", chunk.metadata.get("source", "")
            ),
            version=chunk.metadata.get("version", ""),
            section_number=chunk.metadata.get("section_number", ""),
        )
        for chunk in children
    ]
    return SparseIndex(hits)


def search_sparse(
    query: str,
    *,
    k: int,
    source_ids: list[str] | None = None,
) -> list[SearchHit]:
    """Search the cached real-corpus BM25 index."""
    return _get_sparse_index().search(query, k=k, source_ids=source_ids)
