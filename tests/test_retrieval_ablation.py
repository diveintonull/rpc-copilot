"""Contracts for the reproducible Task10 retrieval ablation matrix."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest
import numpy as np

from evals.run_retrieval_ablation import (
    FIXED_COLLECTION,
    PARENT_CHILD_COLLECTION,
    AblationConfig,
    AblationRuntime,
    CaseResult,
    ConfigResult,
    CorpusItem,
    SectionSpan,
    aggregate_config,
    build_parser,
    build_ablation_index,
    config_hash,
    evaluate_config,
    fixed_items_from_text,
    generate_config_matrix,
    locate_section_spans,
    load_cases_from_bytes,
    main,
    parent_child_items_from_parts,
    reciprocal_rank_for_hits,
    recall_at_k_for_hits,
    render_report,
    retrieve_for_ablation,
    run_matrix,
    select_default,
)


def test_matrix_contains_each_binary_combination_once_with_stable_hashes() -> None:
    matrix = generate_config_matrix()

    assert len(matrix) == 8
    combinations = {
        (config.chunking, config.use_sparse, config.use_rerank)
        for config in matrix
    }
    assert combinations == {
        (chunking, sparse, rerank)
        for chunking in ("fixed_window", "parent_child")
        for sparse in (False, True)
        for rerank in (False, True)
    }
    hashes = [config_hash(config) for config in matrix]
    assert len(set(hashes)) == 8
    assert hashes == [config_hash(config) for config in generate_config_matrix()]
    assert all(len(value) == 64 for value in hashes)


def test_config_hash_changes_when_a_retrieval_parameter_changes() -> None:
    config = AblationConfig("fixed_window", use_sparse=False, use_rerank=False)

    assert config_hash(config) != config_hash(replace(config, dense_k=10))


def test_fixed_window_records_every_parent_section_it_overlaps() -> None:
    text = "AAAAABBBBB"
    sections = (
        SectionSpan("law@v1#one", 0, 5),
        SectionSpan("law@v1#two", 5, 10),
    )

    items = fixed_items_from_text(
        text,
        doc_id="law@v1",
        sections=sections,
        size=6,
        overlap=2,
        source_id="law",
        version="v1",
    )

    assert [(item.text, item.parent_ids) for item in items] == [
        ("AAAAAB", ("law@v1#one", "law@v1#two")),
        ("ABBBBB", ("law@v1#one", "law@v1#two")),
    ]


def test_section_locator_preserves_document_order_for_repeated_text() -> None:
    text = "heading same middle heading same"
    sections = [
        SimpleNamespace(id="first", text="heading same"),
        SimpleNamespace(id="second", text="heading same"),
    ]

    assert locate_section_spans(text, sections) == (
        SectionSpan("first", 0, 12),
        SectionSpan("second", 20, 32),
    )


def test_parent_child_items_keep_child_text_and_single_parent_identity() -> None:
    parents = [
        SimpleNamespace(
            id="law@v1#one",
            text="full section",
            number="one",
            metadata={"source_id": "law", "version": "v1"},
        )
    ]
    children = [
        SimpleNamespace(
            id="law@v1#one:0",
            text="child text",
            metadata={
                "parent_id": "law@v1#one",
                "source_id": "law",
                "version": "v1",
                "section_number": "one",
            },
        )
    ]

    items, parent_items = parent_child_items_from_parts(parents, children)

    assert items == [
        CorpusItem(
            "law@v1#one:0",
            "child text",
            ("law@v1#one",),
            "law",
            "v1",
            "one",
        )
    ]
    assert parent_items["law@v1#one"].text == "full section"


def test_retrieval_metrics_are_hand_calculable_for_multi_parent_hits() -> None:
    hits = [
        CorpusItem("first", "x", ("wrong",), "law", "v1", ""),
        CorpusItem("second", "y", ("gold-a", "gold-b"), "law", "v1", ""),
    ]

    assert recall_at_k_for_hits(hits, ("gold-a", "gold-b"), k=1) == 0.0
    assert recall_at_k_for_hits(hits, ("gold-a", "gold-b"), k=2) == 1.0
    assert reciprocal_rank_for_hits(hits, ("gold-a", "gold-b")) == 0.5


def test_aggregate_excludes_empty_gold_from_quality_but_keeps_its_latency() -> None:
    config = AblationConfig("fixed_window", use_sparse=False, use_rerank=False)
    cases = (
        CaseResult("answer", ("gold",), (("gold",),), 10.0),
        CaseResult("refusal", (), (), 100.0),
    )

    result = aggregate_config(config, index_size=2, cases=cases)

    assert result.recall_at_5 == 1.0
    assert result.mrr == 1.0
    assert result.p95_latency_ms == 100.0
    assert result.answer_cases == 1
    assert result.total_cases == 2


class FakeQdrantClient:
    def __init__(self, count: int) -> None:
        self.expected_count = count
        self.deleted: list[str] = []
        self.created: list[str] = []
        self.upserted: list[tuple[str, int]] = []

    def collection_exists(self, name: str) -> bool:
        return True

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)

    def create_collection(self, name: str, **_kwargs) -> None:
        self.created.append(name)

    def upsert(self, name: str, *, points) -> None:
        self.upserted.append((name, len(points)))

    def count(self, name: str):
        return type("Count", (), {"count": self.expected_count})()


def test_ablation_index_uses_only_dedicated_collection_and_checks_count() -> None:
    items = [CorpusItem("one", "text", ("p1",), "law", "v1", "1")]
    client = FakeQdrantClient(count=1)

    count = build_ablation_index(
        items,
        collection_name=FIXED_COLLECTION,
        client=client,
        vectors=[[0.1, 0.2]],
    )

    assert count == 1
    assert client.deleted == [FIXED_COLLECTION]
    assert client.created == [FIXED_COLLECTION]
    assert client.upserted == [(FIXED_COLLECTION, 1)]
    assert FIXED_COLLECTION != "grc_kb"
    assert PARENT_CHILD_COLLECTION != "grc_kb"


def test_ablation_index_rejects_qdrant_count_mismatch() -> None:
    client = FakeQdrantClient(count=0)
    items = [CorpusItem("one", "text", ("p1",), "law", "v1", "1")]

    with pytest.raises(RuntimeError, match="count mismatch"):
        build_ablation_index(
            items,
            collection_name=PARENT_CHILD_COLLECTION,
            client=client,
            vectors=[[0.1, 0.2]],
        )


def test_ablation_index_accepts_real_numpy_embedding_matrix() -> None:
    client = FakeQdrantClient(count=1)
    items = [CorpusItem("one", "text", ("p1",), "law", "v1", "1")]

    assert build_ablation_index(
        items,
        collection_name=FIXED_COLLECTION,
        client=client,
        vectors=np.asarray([[0.1, 0.2]], dtype=np.float32),
    ) == 1


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.dense = [SimpleNamespace(chunk_id="dense")]
        self.sparse = [SimpleNamespace(chunk_id="sparse")]

    def dense_search(self, query, config):
        self.calls.append(f"dense:{query}:{config.dense_k}")
        return self.dense

    def sparse_search(self, query, config):
        self.calls.append(f"sparse:{query}:{config.sparse_k}")
        return self.sparse

    def fuse(self, dense_hits, sparse_hits, config):
        self.calls.append(f"fuse:{len(dense_hits)}:{len(sparse_hits)}:{config.fused_k}")
        return [*dense_hits, *sparse_hits]

    def materialize(self, hits, chunking):
        self.calls.append(f"materialize:{chunking}:{len(hits)}")
        return [CorpusItem(hit.chunk_id, hit.chunk_id, (hit.chunk_id,), "", "", "") for hit in hits]

    def rerank(self, query, items, limit):
        self.calls.append(f"rerank:{query}:{limit}")
        return list(reversed(items))[:limit]


def test_retrieve_pipeline_skips_disabled_sparse_and_rerank() -> None:
    runtime = FakeRuntime()
    config = AblationConfig("fixed_window", False, False, fused_k=7)

    hits = retrieve_for_ablation("query", config, runtime)

    assert [hit.item_id for hit in hits] == ["dense"]
    assert runtime.calls == [
        "dense:query:20",
        "materialize:fixed_window:1",
    ]


def test_retrieve_pipeline_runs_hybrid_parent_expansion_and_rerank() -> None:
    runtime = FakeRuntime()
    config = AblationConfig("parent_child", True, True, rerank_k=2)

    hits = retrieve_for_ablation("query", config, runtime)

    assert [hit.item_id for hit in hits] == ["sparse", "dense"]
    assert runtime.calls == [
        "dense:query:20",
        "sparse:query:20",
        "fuse:1:1:20",
        "materialize:parent_child:2",
        "rerank:query:2",
    ]


def test_real_ablation_runtime_requests_strict_rerank(monkeypatch) -> None:
    from rag import rerank as rerank_module

    captured = {}

    def failing_rerank(query, hits, *, limit, strict):
        captured.update(query=query, hits=hits, limit=limit, strict=strict)
        raise RuntimeError("reranker unavailable")

    monkeypatch.setattr(rerank_module, "rerank_hits", failing_rerank)
    item = CorpusItem("one", "text", ("p1",), "law", "v1", "1")

    with pytest.raises(RuntimeError, match="reranker unavailable"):
        AblationRuntime.rerank("query", [item], 1)

    assert captured["strict"] is True


def test_evaluate_config_keeps_case_errors_as_failed_retrievals() -> None:
    config = AblationConfig("fixed_window", False, False)
    cases = [
        SimpleNamespace(id="ok", question="q1", gold_citations=("gold",)),
        SimpleNamespace(id="bad", question="q2", gold_citations=("gold",)),
    ]

    def retrieve(question, _config, _runtime):
        if question == "q2":
            raise RuntimeError("boom")
        return [CorpusItem("one", "text", ("gold",), "", "", "")]

    result = evaluate_config(
        cases,
        config=config,
        index_size=1,
        runtime=SimpleNamespace(),
        retrieve_fn=retrieve,
    )

    assert result.total_cases == 2
    assert result.error_count == 1
    assert result.recall_at_5 == 0.5
    assert result.cases[1].hit_parent_ids == ()
    assert result.cases[1].error == "RuntimeError: boom"


def _result(config: AblationConfig, *, recall: float = 0.5) -> ConfigResult:
    return ConfigResult(
        config=config,
        config_sha256=config_hash(config),
        index_size=10,
        recall_at_5=recall,
        mrr=0.5,
        p95_latency_ms=20.0,
        answer_cases=1,
        total_cases=1,
        error_count=0,
        cases=(CaseResult("one", ("gold",), (("gold",),), 20.0),),
    )


def test_run_matrix_executes_each_unique_config_once() -> None:
    configs = generate_config_matrix()
    seen: list[str] = []

    results = run_matrix(
        configs,
        evaluate=lambda config: seen.append(config_hash(config)) or _result(config),
    )

    assert len(results) == 8
    assert seen == [config_hash(config) for config in configs]


def test_run_matrix_rejects_duplicate_config_hashes() -> None:
    config = AblationConfig("fixed_window", False, False)

    with pytest.raises(ValueError, match="duplicate configuration"):
        run_matrix([config, config], evaluate=lambda value: _result(value))


def test_default_selection_prioritizes_recall_then_mrr_then_latency() -> None:
    configs = generate_config_matrix()[:4]
    results = [
        replace(_result(configs[0]), recall_at_5=0.7, mrr=0.6, p95_latency_ms=10),
        replace(_result(configs[1]), recall_at_5=0.8, mrr=0.5, p95_latency_ms=5),
        replace(_result(configs[2]), recall_at_5=0.8, mrr=0.7, p95_latency_ms=20),
        replace(_result(configs[3]), recall_at_5=0.8, mrr=0.7, p95_latency_ms=15),
    ]

    assert select_default(results) == results[3]


def test_report_keeps_errors_and_writes_one_conclusion_per_config() -> None:
    configs = generate_config_matrix()
    results = [_result(config, recall=0.5 + index / 100) for index, config in enumerate(configs)]
    failed_case = CaseResult("failed", ("gold",), (), 30.0, "RuntimeError: boom")
    results[-1] = replace(results[-1], error_count=1, cases=(failed_case,))

    report = render_report(
        results,
        dataset_path="evals/dataset.jsonl",
        dataset_sha256="abc123",
        started_at="2026-07-11T00:00:00+00:00",
        finished_at="2026-07-11T00:01:00+00:00",
        device="cuda: RTX test",
        embedding_model="embed-test",
        reranker_model="rerank-test",
        collection_sizes={FIXED_COLLECTION: 9, PARENT_CHILD_COLLECTION: 10},
    )

    assert report.count("逐行结论：") == 8
    assert "默认配置" in report
    assert "failed" in report
    assert "RuntimeError: boom" in report
    assert "abc123" in report
    assert "embed-test" in report
    assert "rerank-test" in report
    assert FIXED_COLLECTION in report
    assert '"dense_k": 20' in report
    assert "## 完整逐题记录" in report
    assert "| `one` |" in report
    assert "| `failed` |" in report
    lines = report.splitlines()
    table_separator = lines.index("|---|---|---|---|---:|---:|---:|---:|---:|")
    assert all(
        line.startswith("| `") for line in lines[table_separator + 1 : table_separator + 9]
    )


def test_report_calls_out_sparse_regression_within_same_chunking_strategy() -> None:
    fixed_dense = _result(AblationConfig("fixed_window", False, False), recall=0.5)
    parent_dense = replace(
        _result(AblationConfig("parent_child", False, False), recall=0.7),
        mrr=0.5,
    )
    parent_hybrid = replace(
        _result(AblationConfig("parent_child", True, False), recall=0.6),
        mrr=0.6,
    )

    report = render_report(
        [fixed_dense, parent_dense, parent_hybrid],
        dataset_path="evals/dataset.jsonl",
        dataset_sha256="hash",
        started_at="start",
        finished_at="finish",
        device="cuda",
    )

    assert "相对同切片 Dense 退化" in report
    assert "MRR +0.1000" in report


def test_cli_defaults_to_sixty_case_dataset_and_task10_report() -> None:
    args = build_parser().parse_args([])

    assert args.dataset.as_posix() == "evals/dataset.jsonl"
    assert args.output.as_posix() == "results/retrieval_ablation.md"
    assert args.window_size == 500
    assert args.window_overlap == 100


def test_dataset_cases_and_hash_are_derived_from_the_same_frozen_bytes() -> None:
    payload = {
        "id": "one",
        "question": "问题？",
        "task_type": "regulation_qa",
        "gold_points": ["要点"],
        "gold_citations": ["law@v1#one"],
        "should_refuse": False,
        "source_versions": ["law@v1"],
        "tags": ["single_regulation"],
    }
    frozen = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

    cases, digest = load_cases_from_bytes(frozen)

    assert [case.id for case in cases] == ["one"]
    assert digest == hashlib.sha256(frozen).hexdigest()


def test_cli_rejects_window_parameters_that_parent_child_index_cannot_honor() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--window-size", "1000"])


def test_main_writes_injected_report_and_returns_error_status(tmp_path) -> None:
    output = tmp_path / "report.md"

    exit_code = main(
        ["--output", str(output)],
        execute=lambda _args: ("generated report", 2),
    )

    assert exit_code == 1
    assert output.read_text(encoding="utf-8") == "generated report"
