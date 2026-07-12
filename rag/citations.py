"""Deterministic parsing contracts for answer-citation validation."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal, TypedDict


FailureCode = Literal[
    "unknown_citation",
    "version_mismatch",
    "uncited_claim",
    "unsupported_claim",
]


class CitationFailure(TypedDict):
    code: FailureCode
    claim: str
    citation: int | None


class CitationValidation(TypedDict):
    valid: bool
    failures: list[CitationFailure]


EntailmentEvaluator = Callable[[str, dict], bool]
_NUMBERED_CITATION = re.compile(r"\[(\d+)]")
_SENTENCE_BOUNDARY = re.compile(r"[。！？!?]+")


def parse_numbered_citations(text: str) -> list[int]:
    """Return unique numbered citations in their first-seen order."""
    citations = []
    seen = set()
    for match in _NUMBERED_CITATION.finditer(text):
        number = int(match.group(1))
        if number not in seen:
            seen.add(number)
            citations.append(number)
    return citations


def extract_claims(answer: str) -> list[tuple[str, list[int]]]:
    """Split an answer into non-empty claims and their citation numbers."""
    claims = []
    for sentence in _SENTENCE_BOUNDARY.split(answer):
        sentence = sentence.strip()
        if not sentence:
            continue
        citations = parse_numbered_citations(sentence)
        claim = _NUMBERED_CITATION.sub("", sentence).strip(" ，,；;:\n\t")
        if claim:
            claims.append((claim, citations))
    return claims


def _is_disclaimer(claim: str) -> bool:
    return "人工确认" in claim and (
        "初步分析" in claim or "免责声明" in claim
    )


def _parent_version(parent_id: object) -> str | None:
    if not isinstance(parent_id, str):
        return None
    prefix, separator, _section = parent_id.partition("#")
    if not separator:
        return None
    _source_id, separator, version = prefix.rpartition("@")
    if not separator or not version:
        return None
    return version


def validate_citations(
    answer: str,
    evidence: list[dict],
    *,
    entailment_evaluator: EntailmentEvaluator,
) -> CitationValidation:
    """Return the validation result for answer claims and ordered evidence."""
    failures: list[CitationFailure] = []

    for claim, citations in extract_claims(answer):
        if _is_disclaimer(claim):
            continue
        if not citations:
            failures.append(
                {
                    "code": "uncited_claim",
                    "claim": claim,
                    "citation": None,
                }
            )
            continue

        for citation in citations:
            if citation < 1 or citation > len(evidence):
                failures.append(
                    {
                        "code": "unknown_citation",
                        "claim": claim,
                        "citation": citation,
                    }
                )
                continue

            cited_evidence = evidence[citation - 1]
            parent_version = _parent_version(
                cited_evidence.get("parent_id")
            )
            if parent_version != cited_evidence.get("version"):
                failures.append(
                    {
                        "code": "version_mismatch",
                        "claim": claim,
                        "citation": citation,
                    }
                )
                continue

            if not entailment_evaluator(claim, cited_evidence):
                failures.append(
                    {
                        "code": "unsupported_claim",
                        "claim": claim,
                        "citation": citation,
                    }
                )

    return {"valid": not failures, "failures": failures}
