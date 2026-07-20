"""Fuse clause retrieval and page-image retrieval into one evidence list."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

from ingest.visual import VISUAL_PAGES_ROOT
from rag.visual import VisualHit, search_visual


TextSearch = Callable[[str, list[str] | None], list[dict]]
VisualSearch = Callable[[str, list[str] | None], list[VisualHit]]
DEFAULT_FUSION_LIMIT = 5
DEFAULT_RRF_K = 60


def visual_hit_to_evidence(
    hit: VisualHit,
    *,
    pages_root: Path = VISUAL_PAGES_ROOT,
) -> dict:
    """Map a visual page hit onto the Agent's versioned evidence contract."""
    public_path = quote(hit.image_relpath.replace("\\", "/"), safe="/")
    return {
        "parent_id": hit.visual_id,
        "source_id": hit.source_id,
        "version": hit.version,
        "section_number": f"page {hit.page_number}",
        "text": hit.text or f"Rendered page {hit.page_number} of {hit.title}",
        "score": hit.score,
        "modality": "image",
        "page_number": hit.page_number,
        "title": hit.title,
        "image_path": str(pages_root / Path(hit.image_relpath)),
        "image_url": f"/visual-assets/{public_path}",
    }


def reciprocal_rank_fuse_evidence(
    text_evidence: list[dict],
    visual_evidence: list[dict],
    *,
    limit: int = DEFAULT_FUSION_LIMIT,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """Fuse modalities by rank and normalize scores for display."""
    if limit < 1 or rrf_k < 1:
        raise ValueError("fusion limit and rrf_k must be positive")
    combined: dict[str, dict] = {}
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    sequence = 0
    for modality, evidence in (("text", text_evidence), ("image", visual_evidence)):
        for rank, item in enumerate(evidence, start=1):
            parent_id = str(item.get("parent_id", ""))
            if not parent_id:
                continue
            if parent_id not in combined:
                combined[parent_id] = {**item, "modality": item.get("modality", modality)}
                first_seen[parent_id] = sequence
                sequence += 1
            scores[parent_id] = scores.get(parent_id, 0.0) + 1.0 / (rrf_k + rank)

    ranked_ids = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))[:limit]
    if not ranked_ids:
        return []
    top_score = scores[ranked_ids[0]]
    result = []
    for parent_id in ranked_ids:
        item = dict(combined[parent_id])
        item["retrieval_score"] = item.get("score")
        item["score"] = scores[parent_id] / top_score
        result.append(item)
    return result


def search_multimodal_evidence(
    query: str,
    source_ids: list[str] | None = None,
    *,
    text_search: TextSearch,
    visual_search: VisualSearch = search_visual,
    pages_root: Path = VISUAL_PAGES_ROOT,
    limit: int = DEFAULT_FUSION_LIMIT,
) -> list[dict]:
    """Retrieve both modalities and return one citation-order evidence list."""
    text_evidence = text_search(query, source_ids)
    visual_hits = visual_search(query, source_ids)
    visual_evidence = [
        visual_hit_to_evidence(hit, pages_root=pages_root) for hit in visual_hits
    ]
    return reciprocal_rank_fuse_evidence(
        text_evidence,
        visual_evidence,
        limit=limit,
    )
