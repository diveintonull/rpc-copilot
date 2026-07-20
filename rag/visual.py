"""ColQwen2 page retrieval backed by Qdrant multivectors.

The dense vector is a cheap mean-pooled prefetch representation.  The `late`
multivector keeps every ColQwen page token and is reranked by Qdrant MaxSim.

CLI:
  python -m rag.visual index
  python -m rag.visual search "show the table about authentication"
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from ingest.index import DEFAULT_QDRANT_URL
from ingest.visual import VISUAL_PAGES_ROOT, VisualPage, load_visual_manifest


VISUAL_COLLECTION = "grc_visual_pages"
DEFAULT_VISUAL_MODEL = "vidore/colqwen2-v1.0-hf"
VISUAL_VECTOR_DIM = 128
DEFAULT_VISUAL_TOP_K = 5
DEFAULT_VISUAL_PREFETCH_K = 20


class VisualEncoder(Protocol):
    """Small injectable boundary around page/query multivector encoding."""

    def embed_images(self, image_paths: list[Path]) -> list[list[list[float]]]: ...

    def embed_queries(self, queries: list[str]) -> list[list[list[float]]]: ...


@dataclass(frozen=True, slots=True)
class VisualHit:
    visual_id: str
    score: float
    document_id: str
    source_id: str
    version: str
    title: str
    page_number: int
    image_relpath: str
    text: str


def _tensor_to_matrices(output: Any) -> list[list[list[float]]]:
    """Convert HF tensor/model output and remove zero padding token rows."""
    tensor = output
    if not hasattr(tensor, "detach"):
        tensor = getattr(output, "embeddings", None)
    if tensor is None or not hasattr(tensor, "detach"):
        raise TypeError("visual model output does not contain embeddings")
    batches = tensor.detach().float().cpu().tolist()
    result: list[list[list[float]]] = []
    for matrix in batches:
        rows = [
            [float(value) for value in row]
            for row in matrix
            if any(abs(float(value)) > 1e-12 for value in row)
        ]
        if not rows:
            raise ValueError("visual encoder returned an empty multivector")
        result.append(rows)
    return result


class ColQwen2VisualEncoder:
    """Lazy Hugging Face adapter for the native ColQwen2 retrieval model."""

    def __init__(
        self,
        model_name: str = DEFAULT_VISUAL_MODEL,
        *,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import ColQwen2ForRetrieval, ColQwen2Processor

        selected_device = device or os.environ.get("MODEL_DEVICE", "").strip()
        if not selected_device:
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        if selected_device.startswith("cuda") and not torch.cuda.is_available():
            selected_device = "cpu"
        dtype = torch.float16 if selected_device.startswith("cuda") else torch.float32
        self.device = selected_device
        self._inference_lock = Lock()
        self.processor = ColQwen2Processor.from_pretrained(model_name)
        self.model = ColQwen2ForRetrieval.from_pretrained(
            model_name,
            dtype=dtype,
        ).to(selected_device)
        self.model.eval()

    def embed_images(self, image_paths: list[Path]) -> list[list[list[float]]]:
        from PIL import Image
        import torch

        images = []
        try:
            for path in image_paths:
                with Image.open(path) as source:
                    source.load()
                    images.append(source.convert("RGB"))
            with self._inference_lock:
                batch = self.processor.process_images(images).to(self.device)
                with torch.no_grad():
                    output = self.model(**batch)
            return _tensor_to_matrices(output)
        finally:
            for image in images:
                image.close()

    def embed_queries(self, queries: list[str]) -> list[list[list[float]]]:
        import torch

        with self._inference_lock:
            batch = self.processor.process_queries(queries).to(self.device)
            with torch.no_grad():
                output = self.model(**batch)
        return _tensor_to_matrices(output)


@lru_cache(maxsize=2)
def get_visual_encoder(
    model_name: str = DEFAULT_VISUAL_MODEL,
    device: str | None = None,
) -> ColQwen2VisualEncoder:
    """Reuse the expensive retrieval model across requests and index batches."""
    return ColQwen2VisualEncoder(model_name, device=device)


def mean_pool_normalize(matrix: list[list[float]]) -> list[float]:
    """Create a unit dense vector used only for candidate prefetch."""
    if not matrix:
        raise ValueError("cannot pool an empty multivector")
    dimension = len(matrix[0])
    if dimension < 1 or any(len(row) != dimension for row in matrix):
        raise ValueError("multivector rows must have one non-zero dimension")
    pooled = [sum(row[index] for row in matrix) / len(matrix) for index in range(dimension)]
    norm = math.sqrt(sum(value * value for value in pooled))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero pooled vector")
    return [value / norm for value in pooled]


def _client():
    from qdrant_client import QdrantClient

    return QdrantClient(url=os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL))


def _point_id(visual_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"grc-copilot:{visual_id}"))


def build_visual_index(
    pages: list[VisualPage] | None = None,
    *,
    encoder: VisualEncoder | None = None,
    client: Any | None = None,
    pages_root: Path = VISUAL_PAGES_ROOT,
    batch_size: int = 2,
) -> int:
    """Replace the dedicated visual collection with manifest page vectors."""
    if batch_size < 1:
        raise ValueError("visual batch size must be positive")
    from qdrant_client import models

    selected_pages = pages if pages is not None else load_visual_manifest()
    selected_encoder = encoder or get_visual_encoder(
        os.environ.get("VISUAL_MODEL", DEFAULT_VISUAL_MODEL),
        os.environ.get("MODEL_DEVICE", "").strip() or None,
    )
    selected_client = client or _client()
    if selected_client.collection_exists(VISUAL_COLLECTION):
        selected_client.delete_collection(VISUAL_COLLECTION)
    selected_client.create_collection(
        VISUAL_COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=VISUAL_VECTOR_DIM,
                distance=models.Distance.COSINE,
            ),
            "late": models.VectorParams(
                size=VISUAL_VECTOR_DIM,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(m=0),
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        },
    )

    for start in range(0, len(selected_pages), batch_size):
        batch_pages = selected_pages[start : start + batch_size]
        matrices = selected_encoder.embed_images(
            [page.image_path(pages_root) for page in batch_pages]
        )
        if len(matrices) != len(batch_pages):
            raise ValueError("visual encoder result count does not match page batch")
        points = []
        for page, matrix in zip(batch_pages, matrices, strict=True):
            if not matrix or len(matrix[0]) != VISUAL_VECTOR_DIM:
                raise ValueError(
                    f"visual vector dimension must be {VISUAL_VECTOR_DIM}"
                )
            points.append(
                models.PointStruct(
                    id=_point_id(page.visual_id),
                    vector={
                        "dense": mean_pool_normalize(matrix),
                        "late": matrix,
                    },
                    payload=page.to_payload(),
                )
            )
        selected_client.upsert(VISUAL_COLLECTION, points=points)
    return len(selected_pages)


def search_visual(
    query: str,
    source_ids: list[str] | None = None,
    *,
    limit: int = DEFAULT_VISUAL_TOP_K,
    prefetch_k: int = DEFAULT_VISUAL_PREFETCH_K,
    encoder: VisualEncoder | None = None,
    client: Any | None = None,
) -> list[VisualHit]:
    """Dense-prefetch and MaxSim-rerank visual pages for one query."""
    if not query.strip():
        raise ValueError("query must not be blank")
    if source_ids == []:
        return []
    if limit < 1 or prefetch_k < limit:
        raise ValueError("visual prefetch_k must be at least limit")
    from qdrant_client import models

    selected_encoder = encoder or get_visual_encoder(
        os.environ.get("VISUAL_MODEL", DEFAULT_VISUAL_MODEL),
        os.environ.get("MODEL_DEVICE", "").strip() or None,
    )
    selected_client = client or _client()
    query_matrix = selected_encoder.embed_queries([query])[0]
    query_filter = None
    if source_ids:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source_id",
                    match=models.MatchAny(any=source_ids),
                )
            ]
        )
    result = selected_client.query_points(
        VISUAL_COLLECTION,
        prefetch=models.Prefetch(
            query=mean_pool_normalize(query_matrix),
            using="dense",
            limit=prefetch_k,
            filter=query_filter,
        ),
        query=query_matrix,
        using="late",
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    hits: list[VisualHit] = []
    for point in result.points:
        payload = point.payload or {}
        hits.append(
            VisualHit(
                visual_id=str(payload["visual_id"]),
                score=float(point.score),
                document_id=str(payload["document_id"]),
                source_id=str(payload["source_id"]),
                version=str(payload["version"]),
                title=str(payload["title"]),
                page_number=int(payload["page_number"]),
                image_relpath=str(payload["image_relpath"]),
                text=str(payload.get("text", "")),
            )
        )
    return hits


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "index":
        print(f"indexed {build_visual_index()} pages into {VISUAL_COLLECTION}")
        return
    if len(sys.argv) >= 3 and sys.argv[1] == "search":
        query = " ".join(sys.argv[2:])
        for hit in search_visual(query):
            print(f"[{hit.score:.3f}] {hit.visual_id} {hit.title}")
        return
    raise SystemExit("usage: python -m rag.visual index|search <query>")


if __name__ == "__main__":
    main()
