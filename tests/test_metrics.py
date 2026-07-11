"""Hand-calculable contracts for the reproducible RAG evaluator."""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone

import pytest


def metrics_module():
    return importlib.import_module("evals.metrics")


def runner_module():
    return importlib.import_module("evals.run_eval")


def test_recall_at_5_has_a_hand_calculable_half_recall_sample() -> None:
    metrics = metrics_module()
    ranked = ["a", "b", "c", "d", "e", "f"]
    gold = ["c", "f"]

    assert metrics.recall_at_k(ranked, gold, k=5) == pytest.approx(1 / 2)


def test_recall_at_20_has_a_hand_calculable_half_recall_sample() -> None:
    metrics = metrics_module()
    ranked = [f"clause-{number}" for number in range(1, 22)]
    gold = ["clause-20", "clause-21"]

    assert metrics.recall_at_k(ranked, gold, k=20) == pytest.approx(1 / 2)


def test_mrr_has_a_hand_calculable_third_rank_sample() -> None:
    metrics = metrics_module()

    assert metrics.mean_reciprocal_rank(["x", "y", "gold"], ["gold"]) == pytest.approx(
        1 / 3
    )


def test_citation_precision_has_a_hand_calculable_half_correct_sample() -> None:
    metrics = metrics_module()
    predicted = ["gold-a", "wrong"]
    gold = ["gold-a", "gold-b"]

    assert metrics.citation_precision(predicted, gold) == pytest.approx(1 / 2)


def test_citation_coverage_has_a_hand_calculable_half_covered_sample() -> None:
    metrics = metrics_module()
    predicted = ["gold-a"]
    gold = ["gold-a", "gold-b"]

    assert metrics.citation_coverage(predicted, gold) == pytest.approx(1 / 2)


def test_refusal_accuracy_has_a_hand_calculable_three_of_four_sample() -> None:
    metrics = metrics_module()
    expected = [True, True, False, False]
    predicted = [True, False, False, False]

    assert metrics.refusal_accuracy(expected, predicted) == pytest.approx(3 / 4)


def test_p50_latency_uses_the_hand_calculable_nearest_rank() -> None:
    metrics = metrics_module()

    assert metrics.percentile_latency([10.0, 20.0, 30.0, 40.0], 50) == 20.0


def test_p95_latency_uses_the_hand_calculable_nearest_rank() -> None:
    metrics = metrics_module()

    assert metrics.percentile_latency([10.0, 20.0, 30.0, 40.0], 95) == 40.0


def test_average_tokens_has_a_hand_calculable_mean_sample() -> None:
    metrics = metrics_module()

    assert metrics.average_tokens([100, 200, 300]) == pytest.approx(200.0)


def test_numbered_answer_citations_map_to_sources_and_keep_invalid_numbers() -> None:
    runner = runner_module()
    sources = [
        {"n": 1, "parent_id": "law@v1#one"},
        {"n": 2, "parent_id": "law@v1#two"},
    ]

    assert runner.cited_parent_ids("主张一[2]，重复引用[2]，坏引用[9]。", sources) == (
        "law@v1#two",
        "invalid:[9]",
    )


def test_run_metadata_records_model_parameters_dataset_hash_and_runtime(tmp_path) -> None:
    runner = runner_module()
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_bytes(b'{"id":"one"}\n')
    started = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 7, 10, 8, 0, 2, 500000, tzinfo=timezone.utc)

    metadata = runner.build_run_metadata(
        dataset,
        model="test-model",
        parameters={"temperature": 0, "dense_k": 20},
        started_at=started,
        finished_at=finished,
    )

    assert metadata.model == "test-model"
    assert metadata.parameters == {"temperature": 0, "dense_k": 20}
    assert metadata.dataset_sha256 == (
        "ce0cf703fcedc0186b777a8b5e4bc49a9fac282be6c47f953573a44f45ac71fa"
    )
    assert metadata.duration_seconds == pytest.approx(2.5)
    assert metadata.started_at == "2026-07-10T08:00:00+00:00"
    assert metadata.finished_at == "2026-07-10T08:00:02.500000+00:00"


def test_report_keeps_failed_cases_in_the_full_results_and_analysis(tmp_path) -> None:
    metrics = metrics_module()
    runner = runner_module()
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    metadata = runner.build_run_metadata(
        dataset,
        model="test-model",
        parameters={"temperature": 0},
        started_at=now,
        finished_at=now,
    )
    records = [
        metrics.EvaluationRecord(
            case_id="passing-case",
            question="pass?",
            should_refuse=False,
            predicted_refused=False,
            gold_citations=("law#one",),
            retrieved_citations=("law#one",),
            predicted_citations=("law#one",),
            answer="supported [1]",
            latency_ms=10.0,
            total_tokens=100,
        ),
        metrics.EvaluationRecord(
            case_id="failed-case",
            question="fail?",
            should_refuse=False,
            predicted_refused=False,
            gold_citations=("law#missing",),
            retrieved_citations=("law#wrong",),
            predicted_citations=("law#wrong",),
            answer="wrong [1]",
            latency_ms=20.0,
            total_tokens=200,
        ),
    ]

    report = runner.render_report(metadata, records)

    assert "passing-case" in report
    assert "failed-case" in report
    assert "## 失败案例分析" in report
    assert "检索未在 Top-20 找到全部 gold 引用" in report


def test_unified_runner_rejects_duplicate_case_ids(tmp_path) -> None:
    runner = runner_module()
    payload = {
        "id": "duplicate",
        "question": "数据如何保护？",
        "task_type": "regulation_qa",
        "gold_points": ["应采取保护措施"],
        "gold_citations": ["law@v1#one"],
        "should_refuse": False,
        "source_versions": ["law@v1"],
        "tags": ["single_regulation"],
    }
    dataset = tmp_path / "dataset.jsonl"
    row = json.dumps(payload, ensure_ascii=False)
    dataset.write_text(f"{row}\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate id 'duplicate'"):
        runner.load_cases(dataset)


def test_parameters_record_generation_and_embedding_models() -> None:
    runner = runner_module()
    types = importlib.import_module("rag.types")
    config = types.RetrievalConfig(
        dense_k=20,
        fused_k=20,
        use_sparse=False,
        use_rerank=False,
        expand_parent=True,
    )

    parameters = runner.build_parameters(
        config,
        generation_model="generator-v1",
        embedding_model="embedder-v1",
        temperature=0.0,
        max_tokens=4096,
    )

    assert parameters["generation_model"] == "generator-v1"
    assert parameters["embedding_model"] == "embedder-v1"
    assert parameters["dense_k"] == 20
    assert parameters["max_context_hits"] == 6
