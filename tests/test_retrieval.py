"""Contract tests for Dense retrieval and parent expansion."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rag import dense
from rag import retrieve as retrieve_module
from rag.retrieve import expand_parent_hits
from rag.types import RetrievalConfig, SearchHit


def child_hit(
    chunk_id: str,
    parent_id: str,
    *,
    score: float,
    text: str = "child text",
    source_id: str = "GBT-22239",
    version: str = "2019",
    section_number: str = "7.1.4.1",
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        parent_id=parent_id,
        score=score,
        text=text,
        source_id=source_id,
        version=version,
        section_number=section_number,
    )


def load_task9_module(name: str):
    module_name = f"rag.{name}"
    assert importlib.util.find_spec(module_name) is not None, (
        f"{module_name} should exist"
    )
    return importlib.import_module(module_name)


def test_retrieval_config_has_stable_dense_baseline_defaults() -> None:
    config = RetrievalConfig()

    assert config.dense_k == 20
    assert config.fused_k == 20
    assert config.expand_parent is True


def test_expand_parent_hits_deduplicates_in_first_hit_order() -> None:
    hits = [
        child_hit("p1:0", "p1", score=0.9),
        child_hit("p1:1", "p1", score=0.8),
        child_hit("p2:0", "p2", score=0.7),
    ]
    store = {
        "p1": {"text": "full parent one"},
        "p2": {"text": "full parent two"},
    }

    expanded = expand_parent_hits(hits, store)

    assert [hit.parent_id for hit in expanded] == ["p1", "p2"]
    assert [hit.chunk_id for hit in expanded] == ["p1:0", "p2:0"]
    assert [hit.text for hit in expanded] == ["full parent one", "full parent two"]


def test_dense_search_returns_search_hits_from_qdrant_payload(monkeypatch) -> None:
    point = SimpleNamespace(
        score=0.87,
        payload={
            "chunk_id": "GBT-22239@2019#7.1.4.1:0",
            "parent_id": "GBT-22239@2019#7.1.4.1",
            "text": "身份鉴别子块",
            "source_id": "GBT-22239",
            "version": "2019",
            "section_number": "7.1.4.1",
        },
    )
    calls = {}

    class Vector:
        def tolist(self):
            return [0.1, 0.2]

    class Client:
        def query_points(self, collection, **kwargs):
            calls.update(collection=collection, **kwargs)
            return SimpleNamespace(points=[point])

    monkeypatch.setattr(dense, "get_model", lambda: object())
    monkeypatch.setattr(dense, "embed", lambda _model, _texts: [Vector()])
    monkeypatch.setattr(dense, "_client", lambda: Client())
    if hasattr(dense, "_get_cached_model"):
        dense._get_cached_model.cache_clear()

    hits = dense.search_dense("身份鉴别", k=7)

    assert hits == [
        SearchHit(
            chunk_id="GBT-22239@2019#7.1.4.1:0",
            parent_id="GBT-22239@2019#7.1.4.1",
            score=0.87,
            text="身份鉴别子块",
            source_id="GBT-22239",
            version="2019",
            section_number="7.1.4.1",
        )
    ]
    assert calls["limit"] == 7
    assert calls["with_payload"] is True


def test_dense_search_reuses_the_embedding_model(monkeypatch) -> None:
    loads = 0

    class Vector:
        def tolist(self):
            return [0.1]

    class Client:
        def query_points(self, _collection, **_kwargs):
            return SimpleNamespace(points=[])

    def load_model():
        nonlocal loads
        loads += 1
        return object()

    monkeypatch.setattr(dense, "get_model", load_model)
    monkeypatch.setattr(dense, "embed", lambda _model, _texts: [Vector()])
    monkeypatch.setattr(dense, "_client", lambda: Client())
    dense._get_cached_model.cache_clear()

    dense.search_dense("first", k=1)
    dense.search_dense("second", k=1)

    assert loads == 1


def test_retrieve_uses_config_and_expands_parents(monkeypatch, tmp_path) -> None:
    parent_store = tmp_path / "parents.json"
    parent_store.write_text(
        json.dumps({"p1": {"text": "full parent"}}), encoding="utf-8"
    )
    calls = {}

    def fake_search(query: str, *, k: int) -> list[SearchHit]:
        calls.update(query=query, k=k)
        return [child_hit("p1:0", "p1", score=0.9)]

    monkeypatch.setattr(retrieve_module, "PARENTS_STORE", parent_store)
    monkeypatch.setattr(retrieve_module, "search_dense", fake_search)

    hits = retrieve_module.retrieve(
        "身份鉴别",
        RetrievalConfig(
            dense_k=7,
            fused_k=5,
            use_sparse=False,
            use_rerank=False,
        ),
    )

    assert calls == {"query": "身份鉴别", "k": 7}
    assert [hit.text for hit in hits] == ["full parent"]


def test_chinese_tokenizer_does_not_require_spaces() -> None:
    sparse = load_task9_module("sparse")

    tokens = sparse.tokenize_zh("网络安全日志")

    assert len(tokens) >= 2
    assert "日志" in tokens
    assert all(token.strip() == token and token for token in tokens)


def test_sparse_search_recalls_a_no_space_chinese_clause() -> None:
    sparse = load_task9_module("sparse")
    index = sparse.SparseIndex(
        [
            child_hit(
                "logging:0",
                "logging",
                score=0.0,
                text="网络运行日志至少保存六个月",
            ),
            child_hit(
                "identity:0",
                "identity",
                score=0.0,
                text="用户登录时应当进行身份鉴别",
            ),
            child_hit(
                "backup:0",
                "backup",
                score=0.0,
                text="重要数据应当定期进行异地备份",
            ),
        ]
    )

    hits = index.search("日志保存期限", k=2)

    assert [hit.parent_id for hit in hits] == ["logging"]
    assert hits[0].sparse_rank == 1
    assert hits[0].score > 0


def test_sparse_index_treats_punctuation_only_corpus_as_empty() -> None:
    sparse = load_task9_module("sparse")

    index = sparse.SparseIndex(
        [child_hit("punctuation:0", "punctuation", score=0.0, text="？！……")]
    )

    assert index.search("日志", k=5) == []


def test_rrf_has_hand_calculable_order_and_preserves_source_ranks() -> None:
    fusion = load_task9_module("fusion")
    a = child_hit("a", "a", score=0.9, text="A")
    b = child_hit("b", "b", score=0.8, text="B")
    c = child_hit("c", "c", score=0.7, text="C")

    fused = fusion.reciprocal_rank_fusion([a, b], [b, c], rrf_k=0)

    assert [hit.chunk_id for hit in fused] == ["b", "a", "c"]
    assert fused[0].score == 1.5
    assert fused[0].dense_rank == 2
    assert fused[0].sparse_rank == 1


def test_rrf_preserves_dense_candidates_when_sparse_is_empty() -> None:
    fusion = load_task9_module("fusion")
    dense_hits = [
        child_hit("a", "a", score=0.9),
        child_hit("b", "b", score=0.8),
    ]

    fused = fusion.reciprocal_rank_fusion(dense_hits, [], limit=20)

    assert [hit.chunk_id for hit in fused] == ["a", "b"]
    assert [hit.dense_rank for hit in fused] == [1, 2]
    assert all(hit.sparse_rank is None for hit in fused)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rrf_k": -1}, "rrf_k"),
        ({"limit": -1}, "limit"),
    ],
)
def test_rrf_rejects_negative_parameters(kwargs, message) -> None:
    fusion = load_task9_module("fusion")

    with pytest.raises(ValueError, match=message):
        fusion.reciprocal_rank_fusion([], [], **kwargs)


def test_reranker_uses_an_injected_scorer() -> None:
    rerank = load_task9_module("rerank")
    hits = [
        child_hit("a", "a", score=0.9, text="A"),
        child_hit("b", "b", score=0.8, text="B"),
    ]

    reranked = rerank.rerank_hits(
        "query",
        hits,
        limit=2,
        scorer=lambda _query, _documents: [0.1, 0.9],
    )

    assert [hit.chunk_id for hit in reranked] == ["b", "a"]
    assert [hit.rerank_score for hit in reranked] == [0.9, 0.1]


def test_reranker_failure_falls_back_to_the_input_order() -> None:
    rerank = load_task9_module("rerank")
    hits = [
        child_hit("a", "a", score=0.9),
        child_hit("b", "b", score=0.8),
        child_hit("c", "c", score=0.7),
    ]

    def failing_scorer(_query, _documents):
        raise RuntimeError("reranker unavailable")

    reranked = rerank.rerank_hits(
        "query", hits, limit=2, scorer=failing_scorer
    )

    assert [hit.chunk_id for hit in reranked] == ["a", "b"]
    assert all(hit.rerank_score is None for hit in reranked)


def test_reranker_strict_mode_propagates_failure_for_evaluation() -> None:
    rerank = load_task9_module("rerank")
    hits = [child_hit("a", "a", score=0.9)]

    def failing_scorer(_query, _documents):
        raise RuntimeError("reranker unavailable")

    with pytest.raises(RuntimeError, match="reranker unavailable"):
        rerank.rerank_hits(
            "query",
            hits,
            limit=1,
            scorer=failing_scorer,
            strict=True,
        )


@pytest.mark.parametrize(
    "scores",
    [
        [0.1],
        ["not-a-number", 0.2],
        [float("nan"), 0.2],
        [float("inf"), 0.2],
    ],
)
def test_reranker_invalid_outputs_fall_back(scores) -> None:
    rerank = load_task9_module("rerank")
    hits = [
        child_hit("a", "a", score=0.9),
        child_hit("b", "b", score=0.8),
    ]

    result = rerank.rerank_hits(
        "query", hits, limit=2, scorer=lambda _query, _documents: scores
    )

    assert result == hits


def test_reranker_preserves_input_order_when_scores_tie() -> None:
    rerank = load_task9_module("rerank")
    hits = [
        child_hit("a", "a", score=0.9),
        child_hit("b", "b", score=0.8),
    ]

    result = rerank.rerank_hits(
        "query", hits, limit=2, scorer=lambda _query, _documents: [0.5, 0.5]
    )

    assert [hit.chunk_id for hit in result] == ["a", "b"]


def test_real_reranker_caps_cross_encoder_input_at_512_tokens(monkeypatch) -> None:
    rerank = load_task9_module("rerank")
    init_calls = {}
    predict_calls = {}

    class FakeCrossEncoder:
        def __init__(self, model_path, **kwargs):
            init_calls.update(model_path=model_path, **kwargs)

        def predict(self, pairs, **kwargs):
            predict_calls.update(pairs=pairs, **kwargs)
            return [0.5]

    monkeypatch.setenv("RERANK_MODEL_SOURCE", "hf")
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    monkeypatch.setattr(
        rerank, "select_model_device", lambda _name: "cuda", raising=False
    )
    monkeypatch.setattr(
        rerank,
        "model_kwargs_for",
        lambda _device: {"dtype": "float16"},
        raising=False,
    )
    rerank.get_rerank_scorer.cache_clear()

    scorer = rerank.get_rerank_scorer()
    assert list(scorer("query", ["document"])) == [0.5]

    assert init_calls == {
        "model_path": "BAAI/bge-reranker-v2-m3",
        "device": "cuda",
        "max_length": 512,
        "model_kwargs": {"dtype": "float16"},
    }
    assert predict_calls == {
        "pairs": [("query", "document")],
        "batch_size": 4,
        "show_progress_bar": False,
    }
    rerank.get_rerank_scorer.cache_clear()


def test_real_reranker_passes_cpu_without_model_kwargs(monkeypatch) -> None:
    rerank = load_task9_module("rerank")
    calls = {}

    class FakeCrossEncoder:
        def __init__(self, model_path, **kwargs):
            calls.update(model_path=model_path, **kwargs)

        def predict(self, _pairs, **_kwargs):
            return []

    monkeypatch.setenv("RERANK_MODEL_SOURCE", "hf")
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    monkeypatch.setattr(rerank, "select_model_device", lambda _name: "cpu")
    monkeypatch.setattr(rerank, "model_kwargs_for", lambda _device: None)
    rerank.get_rerank_scorer.cache_clear()

    rerank.get_rerank_scorer()

    assert calls == {
        "model_path": "BAAI/bge-reranker-v2-m3",
        "device": "cpu",
        "max_length": 512,
        "model_kwargs": None,
    }
    rerank.get_rerank_scorer.cache_clear()


def test_retrieve_can_disable_sparse_and_rerank(monkeypatch, tmp_path) -> None:
    parent_store = tmp_path / "parents.json"
    parent_store.write_text(
        json.dumps({"p1": {"text": "full parent"}}), encoding="utf-8"
    )
    calls: list[str] = []
    dense_hit = child_hit("p1:0", "p1", score=0.9)

    monkeypatch.setattr(retrieve_module, "PARENTS_STORE", parent_store)
    monkeypatch.setattr(
        retrieve_module,
        "search_dense",
        lambda _query, *, k: calls.append(f"dense:{k}") or [dense_hit],
    )
    monkeypatch.setattr(
        retrieve_module,
        "search_sparse",
        lambda _query, *, k: calls.append(f"sparse:{k}") or [],
        raising=False,
    )
    monkeypatch.setattr(
        retrieve_module,
        "rerank_hits",
        lambda *_args, **_kwargs: calls.append("rerank") or [],
        raising=False,
    )

    hits = retrieve_module.retrieve(
        "身份鉴别",
        RetrievalConfig(use_sparse=False, use_rerank=False),
    )

    assert calls == ["dense:20"]
    assert [hit.parent_id for hit in hits] == ["p1"]
    assert hits[0].dense_rank == 1


def test_expand_parent_hits_preserves_rank_provenance() -> None:
    hit = replace(
        child_hit("p1:0", "p1", score=0.7),
        dense_rank=4,
        sparse_rank=2,
        rerank_score=0.6,
    )

    expanded = expand_parent_hits([hit], {"p1": {"text": "full parent"}})

    assert expanded == [replace(hit, text="full parent")]


def test_retrieve_runs_sparse_fusion_without_rerank(monkeypatch) -> None:
    calls: list[str] = []
    dense_hit = child_hit("dense", "dense", score=0.9)
    sparse_hit = child_hit("sparse", "sparse", score=2.0)
    fused_hit = child_hit("fused", "fused", score=0.5)

    monkeypatch.setattr(
        retrieve_module, "search_dense", lambda _query, *, k: [dense_hit]
    )
    def fake_sparse(_query, *, k):
        calls.append(f"sparse:{k}")
        return [sparse_hit]

    monkeypatch.setattr(retrieve_module, "search_sparse", fake_sparse, raising=False)

    def fake_fusion(dense_hits, sparse_hits, *, limit):
        calls.append(f"fusion:{limit}:{dense_hits[0].chunk_id}:{sparse_hits[0].chunk_id}")
        return [fused_hit]

    monkeypatch.setattr(
        retrieve_module, "reciprocal_rank_fusion", fake_fusion, raising=False
    )

    hits = retrieve_module.retrieve(
        "日志",
        RetrievalConfig(
            sparse_k=7,
            fused_k=3,
            use_sparse=True,
            use_rerank=False,
            expand_parent=False,
        ),
    )

    assert calls == ["sparse:7", "fusion:3:dense:sparse"]
    assert hits == [fused_hit]


def test_retrieve_can_rerank_dense_children_without_parent_store(monkeypatch) -> None:
    calls: list[str] = []
    dense_hit = child_hit("dense", "dense", score=0.9)
    reranked_hit = child_hit("reranked", "reranked", score=1.0)

    monkeypatch.setattr(
        retrieve_module, "search_dense", lambda _query, *, k: [dense_hit]
    )

    def fake_rerank(query, hits, *, limit):
        calls.append(f"rerank:{query}:{limit}:{hits[0].chunk_id}")
        return [reranked_hit]

    monkeypatch.setattr(retrieve_module, "rerank_hits", fake_rerank, raising=False)
    monkeypatch.setattr(
        retrieve_module,
        "PARENTS_STORE",
        SimpleNamespace(read_text=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("parent store read"))),
    )

    hits = retrieve_module.retrieve(
        "日志",
        RetrievalConfig(
            fused_k=8,
            rerank_k=2,
            use_sparse=False,
            use_rerank=True,
            expand_parent=False,
        ),
    )

    assert calls == ["rerank:日志:2:dense"]
    assert hits == [reranked_hit]
