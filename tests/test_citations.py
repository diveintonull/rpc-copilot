"""Contract tests for deterministic answer-citation validation."""

from __future__ import annotations

from rag.citations import parse_numbered_citations, validate_citations


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
