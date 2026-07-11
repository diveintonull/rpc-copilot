"""Injectable cross-encoder reranking with deterministic fallback."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import lru_cache
import math
import os

from rag.types import SearchHit
from runtime.device import model_kwargs_for, select_model_device

RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RerankScorer = Callable[[str, list[str]], Sequence[float]]


@lru_cache(maxsize=1)
def get_rerank_scorer() -> RerankScorer:
    """Load the real cross-encoder only when reranking is enabled."""
    from sentence_transformers import CrossEncoder

    model_path = RERANK_MODEL_NAME
    if os.environ.get("RERANK_MODEL_SOURCE", "modelscope") == "modelscope":
        from modelscope import snapshot_download

        model_path = snapshot_download(RERANK_MODEL_NAME)
    device = select_model_device("bge-reranker-v2-m3")
    model = CrossEncoder(
        model_path,
        device=device,
        max_length=512,
        model_kwargs=model_kwargs_for(device),
    )

    def score(query: str, documents: list[str]) -> Sequence[float]:
        return model.predict(
            [(query, document) for document in documents],
            batch_size=4,
            show_progress_bar=False,
        )

    return score


def rerank_hits(
    query: str,
    hits: list[SearchHit],
    *,
    limit: int,
    scorer: RerankScorer | None = None,
    strict: bool = False,
) -> list[SearchHit]:
    """Rerank candidates; optionally expose failures to evaluation callers."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if not hits or limit == 0:
        return []
    try:
        active_scorer = scorer or get_rerank_scorer()
        raw_scores = list(active_scorer(query, [hit.text for hit in hits]))
        if len(raw_scores) != len(hits):
            raise ValueError("reranker returned a different number of scores")
        scores = [float(score) for score in raw_scores]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("reranker returned a non-finite score")
    except Exception:
        if strict:
            raise
        return hits[:limit]

    scored = [
        (index, replace(hit, rerank_score=scores[index]))
        for index, hit in enumerate(hits)
    ]
    scored.sort(key=lambda item: (-scores[item[0]], item[0]))
    return [hit for _index, hit in scored[:limit]]
