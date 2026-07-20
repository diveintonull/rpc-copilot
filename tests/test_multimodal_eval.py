"""Tests for page-aware multimodal retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.run_multimodal_eval import evaluate_retriever, load_cases, summarize


def test_load_and_evaluate_multimodal_case(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "mm-001",
                "question": "图中的身份鉴别表格有什么要求？",
                "source_ids": ["GBT-22239"],
                "gold_text_ids": ["GBT-22239@2019#8.1.4.1"],
                "gold_page_ids": ["GBT-22239@2019#page=12"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cases = load_cases(dataset)

    runs = evaluate_retriever(
        cases,
        "visual",
        lambda _case: ["wrong", "GBT-22239@2019#page=12"],
        lambda case: case.gold_page_ids,
    )

    assert runs[0].recall_at_5 == 1.0
    assert runs[0].reciprocal_rank == 0.5
    assert summarize(runs)["mrr"] == 0.5


def test_multimodal_dataset_requires_gold_evidence(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "mm-001",
                "question": "question",
                "source_ids": [],
                "gold_text_ids": [],
                "gold_page_ids": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least one gold evidence ID"):
        load_cases(dataset)
