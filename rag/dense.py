"""Dense child-chunk retrieval over the existing Qdrant collection."""

from __future__ import annotations

from functools import lru_cache

from ingest.index import COLLECTION, _client, embed, get_model
from rag.types import SearchHit


@lru_cache(maxsize=1)
def _get_cached_model():
    return get_model()


def search_dense(
    query: str,
    *,
    k: int,
    source_ids: list[str] | None = None,
) -> list[SearchHit]:
    """Embed one query and convert Qdrant child points to stable SearchHit values."""
    if source_ids == []:
        return []

    query_vector = embed(_get_cached_model(), [query])[0]
    query_filter = None
    if source_ids is not None:
        from qdrant_client import models

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source_id",
                    match=models.MatchAny(any=source_ids),
                )
            ]
        )
    points = _client().query_points(
        COLLECTION,
        query=query_vector.tolist(),
        query_filter=query_filter,
        limit=k,
        with_payload=True,
    ).points
    return [
        SearchHit(
            chunk_id=point.payload["chunk_id"],
            parent_id=point.payload["parent_id"],
            score=float(point.score),
            text=point.payload["text"],
            source_id=point.payload.get("source_id", point.payload.get("source", "")),
            version=point.payload.get("version", ""),
            section_number=point.payload.get("section_number", ""),
        )
        for point in points
    ]
