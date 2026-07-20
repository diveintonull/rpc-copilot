"""Tests for ColQwen-style Qdrant multivector retrieval."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ingest.visual import VisualPage
from rag.visual import VISUAL_COLLECTION, build_visual_index, mean_pool_normalize, search_visual


class FakeEncoder:
    def embed_images(self, paths: list[Path]) -> list[list[list[float]]]:
        return [[[1.0] + [0.0] * 127, [0.0, 1.0] + [0.0] * 126] for _ in paths]

    def embed_queries(self, queries: list[str]) -> list[list[list[float]]]:
        assert queries == ["身份鉴别表格"]
        return [[[1.0] + [0.0] * 127, [0.0, 1.0] + [0.0] * 126]]


class FakeClient:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.created: list[tuple[str, dict]] = []
        self.upserted = []
        self.query_kwargs = None

    def collection_exists(self, name: str) -> bool:
        assert name == VISUAL_COLLECTION
        return True

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)

    def create_collection(self, name: str, *, vectors_config) -> None:
        self.created.append((name, vectors_config))

    def upsert(self, name: str, *, points) -> None:
        self.upserted.extend(points)

    def query_points(self, name: str, **kwargs):
        assert name == VISUAL_COLLECTION
        self.query_kwargs = kwargs
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=8.5,
                    payload={
                        "visual_id": "GBT-22239@2019#page=12",
                        "document_id": "GBT-22239@2019",
                        "source_id": "GBT-22239",
                        "version": "2019",
                        "title": "网络安全等级保护基本要求",
                        "page_number": 12,
                        "image_relpath": "GBT-22239-2019/page-0012.png",
                        "text": "身份鉴别要求表",
                    },
                )
            ]
        )


def page() -> VisualPage:
    return VisualPage(
        visual_id="GBT-22239@2019#page=12",
        document_id="GBT-22239@2019",
        source_id="GBT-22239",
        version="2019",
        title="网络安全等级保护基本要求",
        page_number=12,
        image_relpath="GBT-22239-2019/page-0012.png",
        text="身份鉴别要求表",
        content_hash="a" * 64,
    )


def test_mean_pool_normalize_returns_unit_vector() -> None:
    pooled = mean_pool_normalize([[1.0, 0.0], [0.0, 1.0]])
    assert pooled == pytest.approx([2 ** -0.5, 2 ** -0.5])


def test_build_visual_index_creates_dense_and_maxsim_named_vectors(tmp_path: Path) -> None:
    client = FakeClient()

    count = build_visual_index(
        [page()], encoder=FakeEncoder(), client=client, pages_root=tmp_path
    )

    assert count == 1
    assert client.deleted == [VISUAL_COLLECTION]
    vectors = client.created[0][1]
    assert set(vectors) == {"dense", "late"}
    assert vectors["late"].multivector_config.comparator == "max_sim"
    assert vectors["late"].hnsw_config.m == 0
    assert client.upserted[0].payload["page_number"] == 12
    assert set(client.upserted[0].vector) == {"dense", "late"}


def test_search_visual_prefetches_dense_then_reranks_late() -> None:
    client = FakeClient()

    hits = search_visual(
        "身份鉴别表格",
        ["GBT-22239"],
        limit=1,
        prefetch_k=4,
        encoder=FakeEncoder(),
        client=client,
    )

    assert hits[0].visual_id == "GBT-22239@2019#page=12"
    assert hits[0].page_number == 12
    assert client.query_kwargs["using"] == "late"
    assert client.query_kwargs["prefetch"].using == "dense"
    assert client.query_kwargs["prefetch"].limit == 4
    assert client.query_kwargs["query_filter"] is not None
