"""Contract tests for the GRC evaluation dataset."""

from __future__ import annotations

import json

import pytest

from evals.schema import EvaluationCase
from evals.validate_dataset import main, validate_dataset


def valid_payload() -> dict:
    return {
        "id": "eval-001",
        "question": "等保三级对身份鉴别有什么要求？",
        "task_type": "regulation_qa",
        "gold_points": ["应对登录用户进行身份标识和鉴别"],
        "gold_citations": ["GBT-22239@2019#7.1.4.1"],
        "should_refuse": False,
        "source_versions": ["GBT-22239@2019"],
        "tags": ["single_regulation"],
    }


def test_valid_evaluation_case_passes() -> None:
    case = EvaluationCase.from_dict(valid_payload())

    assert case.id == "eval-001"
    assert case.gold_citations == ("GBT-22239@2019#7.1.4.1",)


def test_answerable_case_without_citations_fails() -> None:
    payload = valid_payload()
    payload["gold_citations"] = []

    with pytest.raises(ValueError, match="gold_citations"):
        EvaluationCase.from_dict(payload)


def test_answerable_case_without_gold_points_fails() -> None:
    payload = valid_payload()
    payload["gold_points"] = []

    with pytest.raises(ValueError, match="gold_points"):
        EvaluationCase.from_dict(payload)


def test_version_trap_without_source_versions_fails() -> None:
    payload = valid_payload()
    payload["source_versions"] = []
    payload["tags"] = ["version_trap"]

    with pytest.raises(ValueError, match="source_versions"):
        EvaluationCase.from_dict(payload)


def test_unknown_task_type_fails() -> None:
    payload = valid_payload()
    payload["task_type"] = "make_it_up"

    with pytest.raises(ValueError, match="task_type"):
        EvaluationCase.from_dict(payload)


def test_unknown_schema_field_fails() -> None:
    payload = valid_payload()
    payload["agent_answer"] = "不能把当前 Agent 的答案当作 gold"

    with pytest.raises(ValueError, match="unknown fields"):
        EvaluationCase.from_dict(payload)


def test_missing_schema_field_fails() -> None:
    payload = valid_payload()
    payload.pop("question")

    with pytest.raises(ValueError, match="missing fields"):
        EvaluationCase.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gold_points", "not-a-list"),
        ("gold_citations", "not-a-list"),
        ("source_versions", "not-a-list"),
        ("tags", "not-a-list"),
        ("should_refuse", "false"),
    ],
)
def test_schema_rejects_wrong_json_field_types(field, value) -> None:
    payload = valid_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        EvaluationCase.from_dict(payload)


def test_citation_must_belong_to_declared_source_version() -> None:
    payload = valid_payload()
    payload["gold_citations"] = ["GBT-22239@2025#7.1.4.1"]

    with pytest.raises(ValueError, match="declared source_versions"):
        EvaluationCase.from_dict(payload)


def test_validate_dataset_counts_valid_jsonl_records(tmp_path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    second = valid_payload()
    second["id"] = "eval-002"
    dataset.write_text(
        "\n".join(
            json.dumps(payload, ensure_ascii=False)
            for payload in (valid_payload(), second)
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_dataset(dataset)

    assert report.valid == 2
    assert report.invalid == 0
    assert report.errors == ()


def test_validate_dataset_rejects_duplicate_ids(tmp_path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    row = json.dumps(valid_payload(), ensure_ascii=False)
    dataset.write_text(f"{row}\n{row}\n", encoding="utf-8")

    report = validate_dataset(dataset)

    assert report.valid == 1
    assert report.invalid == 1
    assert "duplicate id" in report.errors[0]


def test_validator_cli_prints_required_summary(tmp_path, monkeypatch, capsys) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(valid_payload(), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr("sys.argv", ["validate_dataset", str(dataset)])

    assert main() == 0

    assert capsys.readouterr().out.strip() == "valid=1 invalid=0"
