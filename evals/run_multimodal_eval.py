"""Compare text-only, visual-only, and fused retrieval on page-aware cases.

Dataset JSONL fields:
  id, question, source_ids, gold_text_ids, gold_page_ids

The two gold lists are deliberately separate: clause retrieval and page
retrieval use different stable identifiers. Hybrid relevance is their union.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from evals.metrics import mean_reciprocal_rank, percentile_latency, recall_at_k


@dataclass(frozen=True, slots=True)
class MultimodalCase:
    id: str
    question: str
    source_ids: tuple[str, ...]
    gold_text_ids: tuple[str, ...]
    gold_page_ids: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict) -> "MultimodalCase":
        expected = {
            "id",
            "question",
            "source_ids",
            "gold_text_ids",
            "gold_page_ids",
        }
        if set(payload) != expected:
            raise ValueError(
                "multimodal case fields must be exactly "
                + ", ".join(sorted(expected))
            )
        for name in ("id", "question"):
            if not isinstance(payload[name], str) or not payload[name].strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("source_ids", "gold_text_ids", "gold_page_ids"):
            value = payload[name]
            if not isinstance(value, list) or any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                raise ValueError(f"{name} must be a list of non-empty strings")
        if not payload["gold_text_ids"] and not payload["gold_page_ids"]:
            raise ValueError("at least one gold evidence ID is required")
        return cls(
            id=payload["id"],
            question=payload["question"],
            source_ids=tuple(payload["source_ids"]),
            gold_text_ids=tuple(payload["gold_text_ids"]),
            gold_page_ids=tuple(payload["gold_page_ids"]),
        )


@dataclass(frozen=True, slots=True)
class RetrievalRun:
    case_id: str
    retriever: str
    retrieved_ids: tuple[str, ...]
    gold_ids: tuple[str, ...]
    recall_at_5: float
    reciprocal_rank: float
    latency_ms: float


Retriever = Callable[[MultimodalCase], list[str]]


def load_cases(path: Path) -> list[MultimodalCase]:
    cases = []
    seen = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("row must be an object")
            case = MultimodalCase.from_dict(payload)
            if case.id in seen:
                raise ValueError(f"duplicate id {case.id!r}")
            seen.add(case.id)
            cases.append(case)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid multimodal dataset line {line_number}: {exc}"
            ) from exc
    if not cases:
        raise ValueError("multimodal dataset is empty")
    return cases


def evaluate_retriever(
    cases: list[MultimodalCase],
    name: str,
    retriever: Retriever,
    gold_selector: Callable[[MultimodalCase], tuple[str, ...]],
) -> list[RetrievalRun]:
    runs = []
    for case in cases:
        started = perf_counter()
        retrieved = tuple(retriever(case))
        latency_ms = (perf_counter() - started) * 1000
        gold = gold_selector(case)
        runs.append(
            RetrievalRun(
                case_id=case.id,
                retriever=name,
                retrieved_ids=retrieved,
                gold_ids=gold,
                recall_at_5=recall_at_k(retrieved, gold, k=5),
                reciprocal_rank=mean_reciprocal_rank(retrieved, gold),
                latency_ms=latency_ms,
            )
        )
    return runs


def summarize(runs: list[RetrievalRun]) -> dict:
    if not runs:
        return {
            "cases": 0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }
    return {
        "cases": len(runs),
        "recall_at_5": sum(run.recall_at_5 for run in runs) / len(runs),
        "mrr": sum(run.reciprocal_rank for run in runs) / len(runs),
        "p50_latency_ms": percentile_latency(
            [run.latency_ms for run in runs], 50
        ),
        "p95_latency_ms": percentile_latency(
            [run.latency_ms for run in runs], 95
        ),
    }


def _real_retrievers() -> dict[str, tuple[Retriever, Callable]]:
    from agent.tools import search_regulation
    from rag.multimodal import search_multimodal_evidence
    from rag.visual import search_visual

    def text(case: MultimodalCase) -> list[str]:
        return [
            item["parent_id"]
            for item in search_regulation(case.question, list(case.source_ids) or None)
        ]

    def visual(case: MultimodalCase) -> list[str]:
        return [
            item.visual_id
            for item in search_visual(case.question, list(case.source_ids) or None)
        ]

    def hybrid(case: MultimodalCase) -> list[str]:
        return [
            item["parent_id"]
            for item in search_multimodal_evidence(
                case.question,
                list(case.source_ids) or None,
                text_search=search_regulation,
            )
        ]

    return {
        "text": (text, lambda case: case.gold_text_ids),
        "visual": (visual, lambda case: case.gold_page_ids),
        "hybrid": (
            hybrid,
            lambda case: case.gold_text_ids + case.gold_page_ids,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    all_runs = []
    summary = {}
    for name, (retriever, gold_selector) in _real_retrievers().items():
        runs = evaluate_retriever(cases, name, retriever, gold_selector)
        all_runs.extend(runs)
        summary[name] = summarize(runs)
    report = {
        "summary": summary,
        "records": [asdict(run) for run in all_runs],
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
