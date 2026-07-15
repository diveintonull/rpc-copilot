"""Contract tests for deterministic answer-citation validation."""

from __future__ import annotations

from rag.citations import (
    extract_claims,
    parse_numbered_citations,
    validate_citations,
)


class FakeEntailmentEvaluator:
    """Return a configured decision and record every evaluated claim."""

    def __init__(self, supported: bool) -> None:
        self.supported = supported
        self.calls: list[dict] = []

    def __call__(self, claim: str, evidence: dict) -> bool:
        self.calls.append({"claim": claim, "evidence": evidence})
        return self.supported


def regulation_evidence(
    *,
    parent_id: str = "GBT-22239@2019#8.1.4.1",
    version: str = "2019",
) -> dict:
    return {
        "parent_id": parent_id,
        "source_id": "GBT-22239",
        "version": version,
        "section_number": "8.1.4.1",
        "text": "应采用两种或两种以上组合的鉴别技术。",
        "score": 0.91,
    }


def failure_codes(result: dict) -> list[str]:
    return [failure["code"] for failure in result["failures"]]


def test_numbered_citation_parser_preserves_first_seen_order() -> None:
    assert parse_numbered_citations("要求一[2]，要求二[1]，重复[2]。") == [2, 1]


def test_claim_parser_splits_english_periods_without_splitting_sections() -> None:
    claims = extract_claims(
        "**Left**\nSection 8.1.4.1 requires authentication [1]. "
        "A second claim is unsupported."
    )

    assert claims == [
        ("**Left**", []),
        ("Section 8.1.4.1 requires authentication", [1]),
        ("A second claim is unsupported", []),
    ]


def test_claim_parser_does_not_split_common_vs_abbreviation() -> None:
    claims = extract_claims(
        "适用对象（数据安全事件 vs. 网络安全事件）不同[1][2]。"
    )

    assert claims == [
        ("适用对象（数据安全事件 vs. 网络安全事件）不同", [1, 2])
    ]


def test_markdown_comparison_labels_are_not_factual_claims() -> None:
    evaluator = FakeEntailmentEvaluator(supported=True)

    result = validate_citations(
        "**Left**\nAuthentication is required [1].",
        [regulation_evidence()],
        entailment_evaluator=evaluator,
    )

    assert result == {"valid": True, "failures": []}


def test_short_chinese_comparison_labels_are_not_factual_claims() -> None:
    evaluator = FakeEntailmentEvaluator(supported=True)

    result = validate_citations(
        "**左:**\n需要身份鉴别[1]。\n### 限制",
        [regulation_evidence()],
        entailment_evaluator=evaluator,
    )

    assert result == {"valid": True, "failures": []}


def test_regulation_answer_section_labels_are_not_factual_claims() -> None:
    evaluator = FakeEntailmentEvaluator(supported=True)

    result = validate_citations(
        "**直接回答**\n管理员应采用组合身份鉴别技术[1]。\n\n"
        "### 版本说明\n该要求来自指定版本[1]。\n\n"
        "限制说明\n现有证据仅支持上述要求[1]。",
        [regulation_evidence()],
        entailment_evaluator=evaluator,
    )

    assert result == {"valid": True, "failures": []}
    assert [call["claim"] for call in evaluator.calls] == [
        "管理员应采用组合身份鉴别技术",
        "该要求来自指定版本",
        "现有证据仅支持上述要求",
    ]


def test_version_meta_conclusion_still_requires_direct_support() -> None:
    evaluator = FakeEntailmentEvaluator(supported=False)

    result = validate_citations(
        "版本说明\n所有证据均来自2019版，未发现版本冲突[1]。",
        [regulation_evidence()],
        entailment_evaluator=evaluator,
    )

    assert result["valid"] is False
    assert failure_codes(result) == ["unsupported_claim"]


def test_unknown_citation_number_fails_without_entailment_call() -> None:
    evaluator = FakeEntailmentEvaluator(supported=True)

    result = validate_citations(
        "管理员应启用多因素认证[2]。",
        [regulation_evidence()],
        entailment_evaluator=evaluator,
    )

    assert result["valid"] is False
    assert failure_codes(result) == ["unknown_citation"]
    assert result["failures"][0]["citation"] == 2
    assert evaluator.calls == []


def test_cited_evidence_version_mismatch_fails_deterministically() -> None:
    evaluator = FakeEntailmentEvaluator(supported=True)
    inconsistent = regulation_evidence(
        parent_id="GBT-22239@2020#8.1.4.1",
        version="2019",
    )

    result = validate_citations(
        "管理员应启用多因素认证[1]。",
        [inconsistent],
        entailment_evaluator=evaluator,
    )

    assert result["valid"] is False
    assert failure_codes(result) == ["version_mismatch"]
    assert result["failures"][0]["citation"] == 1
    assert evaluator.calls == []


def test_factual_claim_without_citation_fails() -> None:
    evaluator = FakeEntailmentEvaluator(supported=True)

    result = validate_citations(
        "管理员应启用多因素认证。",
        [regulation_evidence()],
        entailment_evaluator=evaluator,
    )

    assert result["valid"] is False
    assert failure_codes(result) == ["uncited_claim"]
    assert result["failures"][0]["citation"] is None
    assert evaluator.calls == []


def test_cited_but_unsupported_claim_fails_entailment() -> None:
    evaluator = FakeEntailmentEvaluator(supported=False)
    evidence = regulation_evidence()

    result = validate_citations(
        "管理员必须每天更换密码[1]。",
        [evidence],
        entailment_evaluator=evaluator,
    )

    assert result["valid"] is False
    assert failure_codes(result) == ["unsupported_claim"]
    assert result["failures"][0]["citation"] == 1
    assert evaluator.calls == [
        {"claim": "管理员必须每天更换密码", "evidence": evidence}
    ]


def test_supported_cited_claim_passes() -> None:
    evaluator = FakeEntailmentEvaluator(supported=True)
    evidence = regulation_evidence()

    result = validate_citations(
        "管理员应采用组合身份鉴别技术[1]。",
        [evidence],
        entailment_evaluator=evaluator,
    )

    assert result == {"valid": True, "failures": []}
    assert evaluator.calls == [
        {"claim": "管理员应采用组合身份鉴别技术", "evidence": evidence}
    ]


def test_pure_human_review_disclaimer_needs_no_citation() -> None:
    evaluator = FakeEntailmentEvaluator(supported=False)

    result = validate_citations(
        "以下为初步分析，最终结果需要人工确认。",
        [],
        entailment_evaluator=evaluator,
    )

    assert result == {"valid": True, "failures": []}
    assert evaluator.calls == []


def test_comparison_can_validate_a_claim_against_joint_citations() -> None:
    evaluator = FakeEntailmentEvaluator(supported=True)
    left = regulation_evidence()
    right = {
        "parent_id": "cybersecurity-law@2025-amended#第二十一条",
        "source_id": "cybersecurity-law",
        "version": "2025-amended",
        "section_number": "第二十一条",
        "text": "网络运营者应当履行网络安全保护义务。",
        "score": 0.88,
    }

    result = validate_citations(
        "两条款的规范对象和措施不同[1][2]。",
        [left, right],
        entailment_evaluator=evaluator,
        joint_citations=True,
    )

    assert result == {"valid": True, "failures": []}
    assert len(evaluator.calls) == 1
    combined = evaluator.calls[0]["evidence"]
    assert left["text"] in combined["text"]
    assert right["text"] in combined["text"]
    assert combined["parent_id"] == (
        f"{left['parent_id']} + {right['parent_id']}"
    )
    assert combined["joint"] is True
