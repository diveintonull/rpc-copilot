"""Hand-calculable contracts for the reproducible RAG evaluator."""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

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


def test_literal_evidence_ids_are_counted_for_gap_matrix_answers() -> None:
    runner = runner_module()
    sources = [
        {"n": 1, "parent_id": "law@v1#one"},
        {"n": 2, "parent_id": "law@v1#two"},
    ]

    assert runner.cited_parent_ids(
        "evidence: law@v1#two, law@v1#one", sources
    ) == ("law@v1#two", "law@v1#one")


def test_stable_hash_ignores_dictionary_insertion_order() -> None:
    runner = runner_module()

    left = {"model": "m", "parameters": {"b": 2, "a": 1}}
    right = {"parameters": {"a": 1, "b": 2}, "model": "m"}

    assert runner.stable_hash(left) == runner.stable_hash(right)


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
    assert len(metadata.prompt_sha256) == 64
    assert metadata.skill_sha256
    assert len(metadata.config_sha256) == 64
    assert metadata.git_commit


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
    assert parameters["rerank_k"] == 5
    assert parameters["use_rerank"] is False
    assert parameters["retrieval_metric_depth"] == 20
    assert parameters["generation_context_k"] == 5
    assert parameters["router_max_tokens"] == 2048


def test_final_summary_measures_skill_trigger_from_predicted_route() -> None:
    metrics = metrics_module()
    runner = runner_module()
    records = [
        runner.FinalEvaluationRecord(
            case_id="qa",
            question="question",
            should_refuse=False,
            predicted_refused=False,
            gold_citations=("law#one",),
            retrieved_citations=("law#one",),
            predicted_citations=("law#one",),
            answer="answer [1]",
            latency_ms=10,
            total_tokens=10,
            task_type="regulation_qa",
            predicted_intent="regulation_qa",
            active_skill="regulation-qa",
        ),
        runner.FinalEvaluationRecord(
            case_id="gap-routed-wrongly",
            question="question",
            should_refuse=False,
            predicted_refused=False,
            gold_citations=("law#two",),
            retrieved_citations=("law#two",),
            predicted_citations=("law#two",),
            answer="answer",
            latency_ms=10,
            total_tokens=10,
            task_type="gap_analysis",
            predicted_intent="regulation_qa",
            active_skill="regulation-qa",
        ),
    ]

    summary = runner.summarize_final(records)

    assert summary.metrics == metrics.summarize(records)
    assert summary.skill_trigger_accuracy == pytest.approx(0.5)
    assert summary.intent_accuracy == pytest.approx(0.5)


def test_router_output_must_be_one_exact_supported_label() -> None:
    runner = runner_module()

    assert runner.parse_intent_label("regulation_qa") == "regulation_qa"
    assert runner.parse_intent_label('"unsupported"') == "unsupported"
    with pytest.raises(ValueError, match="invalid intent label"):
        runner.parse_intent_label("I think this is regulation_qa")


def test_final_cli_defaults_to_selected_single_variable_configuration() -> None:
    runner = runner_module()

    args = runner.build_parser().parse_args([])

    assert args.rerank_k == 20
    assert args.generation_context_k == 5


def test_evaluation_retains_twenty_hits_but_generation_sees_only_three() -> None:
    runner = runner_module()
    hits = [
        runner.SearchHit(
            chunk_id=f"chunk-{number}",
            parent_id=f"law@v1#{number}",
            score=1 / number,
            text=f"evidence {number}",
            source_id="law",
            version="v1",
            section_number=str(number),
        )
        for number in range(1, 21)
    ]
    case = runner.EvaluationCase(
        id="case",
        question="question",
        task_type="regulation_qa",
        gold_points=("point",),
        gold_citations=("law@v1#1",),
        should_refuse=False,
        source_versions=("law@v1",),
        tags=("single_regulation",),
    )
    captured_messages = []

    def create(**kwargs):
        captured_messages.append(kwargs["messages"])
        return SimpleNamespace(
            usage=SimpleNamespace(total_tokens=20),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="answer [1]"),
                    finish_reason="stop",
                )
            ],
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    config = runner.RetrievalConfig(
        dense_k=20,
        fused_k=20,
        rerank_k=20,
        use_sparse=False,
        use_rerank=True,
        expand_parent=True,
    )

    record = runner.evaluate_case(
        case,
        config=config,
        client=client,
        model="model",
        temperature=0,
        max_tokens=100,
        generation_context_k=3,
        retrieve_fn=lambda _query, _config: hits,
        classify_fn=lambda _question: ("regulation_qa", 10),
    )

    assert len(record.retrieved_citations) == 20
    user_message = captured_messages[0][-1]["content"]
    assert "parent_id=law@v1#3" in user_message
    assert "parent_id=law@v1#4" not in user_message
    assert record.predicted_citations == ("law@v1#1",)


def test_retrieval_depth_must_support_recall_at_twenty() -> None:
    runner = runner_module()

    runner.validate_depths(rerank_k=20, generation_context_k=3)
    with pytest.raises(ValueError, match="at least 20"):
        runner.validate_depths(rerank_k=5, generation_context_k=3)
    with pytest.raises(ValueError, match="cannot exceed"):
        runner.validate_depths(rerank_k=20, generation_context_k=21)


def test_failure_attribution_uses_the_first_observable_divergence() -> None:
    runner = runner_module()

    def record(*, retrieved, predicted, predicted_intent="regulation_qa"):
        return runner.FinalEvaluationRecord(
            case_id="case",
            question="question",
            should_refuse=False,
            predicted_refused=False,
            gold_citations=("gold",),
            retrieved_citations=tuple(retrieved),
            predicted_citations=tuple(predicted),
            answer="answer",
            latency_ms=1,
            total_tokens=1,
            task_type="regulation_qa",
            predicted_intent=predicted_intent,
            active_skill="regulation-qa",
        )

    assert runner.failure_attribution(
        record(
            retrieved=("wrong-1", "wrong-2", "wrong-3", "wrong-4", "wrong-5", "gold"),
            predicted=(),
        )
    ) == "检索"
    assert runner.failure_attribution(
        record(retrieved=("gold",), predicted=("wrong",))
    ) == "生成"
    assert runner.failure_attribution(
        record(
            retrieved=(),
            predicted=(),
            predicted_intent="unsupported",
        )
    ) == "Skill"
