"""Dependency-free metrics for retrieval and grounded-answer evaluation."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass


def _unique(values: Sequence[str]) -> set[str]:
    return set(values)


def recall_at_k(
    ranked_citations: Sequence[str], gold_citations: Collection[str], *, k: int
) -> float:
    """Return the fraction of unique gold citations found in the first *k* ranks."""
    if k <= 0:
        raise ValueError("k must be positive")
    gold = set(gold_citations)
    if not gold:
        return 0.0
    return len(_unique(ranked_citations[:k]) & gold) / len(gold)


def mean_reciprocal_rank(
    ranked_citations: Sequence[str], gold_citations: Collection[str]
) -> float:
    """Return reciprocal rank of the first relevant citation (MRR for one case)."""
    gold = set(gold_citations)
    for rank, citation in enumerate(ranked_citations, start=1):
        if citation in gold:
            return 1.0 / rank
    return 0.0


def citation_precision(
    predicted_citations: Sequence[str], gold_citations: Collection[str]
) -> float:
    """Return the share of unique predicted citations that are gold citations."""
    predicted = _unique(predicted_citations)
    if not predicted:
        return 0.0
    return len(predicted & set(gold_citations)) / len(predicted)


def citation_coverage(
    predicted_citations: Sequence[str], gold_citations: Collection[str]
) -> float:
    """Return the share of unique gold citations used by the answer."""
    gold = set(gold_citations)
    if not gold:
        return 1.0
    return len(_unique(predicted_citations) & gold) / len(gold)


def refusal_accuracy(
    expected: Sequence[bool], predicted: Sequence[bool | None]
) -> float:
    """Return exact binary refusal accuracy; an unavailable prediction is incorrect."""
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have equal lengths")
    if not expected:
        return 0.0
    return sum(
        actual is not None and wanted == actual
        for wanted, actual in zip(expected, predicted, strict=True)
    ) / len(expected)


def percentile_latency(latencies_ms: Sequence[float], percentile: float) -> float:
    """Return a nearest-rank percentile from latency values in milliseconds."""
    if not latencies_ms:
        return 0.0
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    ordered = sorted(float(value) for value in latencies_ms)
    rank = math.ceil((percentile / 100) * len(ordered))
    return ordered[rank - 1]


def average_tokens(total_tokens: Sequence[int]) -> float:
    """Return mean total tokens per evaluation case, including zero-token refusals."""
    if not total_tokens:
        return 0.0
    return sum(total_tokens) / len(total_tokens)


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """Observable inputs and outputs retained for one evaluation case."""

    case_id: str
    question: str
    should_refuse: bool
    predicted_refused: bool | None
    gold_citations: tuple[str, ...]
    retrieved_citations: tuple[str, ...]
    predicted_citations: tuple[str, ...]
    answer: str
    latency_ms: float
    total_tokens: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MetricSummary:
    total_cases: int
    answer_cases: int
    recall_at_5: float
    recall_at_20: float
    mrr: float
    citation_precision: float
    citation_coverage: float
    refusal_accuracy: float
    p50_latency_ms: float
    p95_latency_ms: float
    average_tokens: float


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(records: Sequence[EvaluationRecord]) -> MetricSummary:
    """Aggregate records using only citation-bearing cases for RAG quality metrics."""
    answer_records = [
        record
        for record in records
        if not record.should_refuse and record.gold_citations
    ]
    return MetricSummary(
        total_cases=len(records),
        answer_cases=len(answer_records),
        recall_at_5=_mean(
            [
                recall_at_k(
                    record.retrieved_citations, record.gold_citations, k=5
                )
                for record in answer_records
            ]
        ),
        recall_at_20=_mean(
            [
                recall_at_k(
                    record.retrieved_citations, record.gold_citations, k=20
                )
                for record in answer_records
            ]
        ),
        mrr=_mean(
            [
                mean_reciprocal_rank(
                    record.retrieved_citations, record.gold_citations
                )
                for record in answer_records
            ]
        ),
        citation_precision=_mean(
            [
                citation_precision(
                    record.predicted_citations, record.gold_citations
                )
                for record in answer_records
            ]
        ),
        citation_coverage=_mean(
            [
                citation_coverage(
                    record.predicted_citations, record.gold_citations
                )
                for record in answer_records
            ]
        ),
        refusal_accuracy=refusal_accuracy(
            [record.should_refuse for record in records],
            [record.predicted_refused for record in records],
        ),
        p50_latency_ms=percentile_latency(
            [record.latency_ms for record in records], 50
        ),
        p95_latency_ms=percentile_latency(
            [record.latency_ms for record in records], 95
        ),
        average_tokens=average_tokens(
            [record.total_tokens for record in records]
        ),
    )
