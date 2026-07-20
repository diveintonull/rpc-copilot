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
_SENTENCE_BOUNDARY = re.compile(
    r"[。！？!?]+|(?<!\d)(?<!vs)(?<!Vs)(?<!VS)\.(?!\d)|[\r\n]+"
)
_STRUCTURE_LABELS = {
    "left",
    "right",
    "comparison",
    "limitation",
    "左",
    "右",
    "左侧",
    "右侧",
    "比较",
    "限制",
    "局限性",
    "direct answer",
    "version note",
    "version notes",
    "直接回答",
    "直接答案",
    "版本说明",
    "限制说明",
}


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


def _is_structure_label(claim: str) -> bool:
    normalized = claim.strip().lstrip("#").strip()
    normalized = normalized.replace("*", "").strip()
    normalized = normalized.rstrip(":：").strip().casefold()
    return normalized in _STRUCTURE_LABELS


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
    joint_citations: bool = False,
) -> CitationValidation:
    """Return the validation result for answer claims and ordered evidence."""
    failures: list[CitationFailure] = []

    for claim, citations in extract_claims(answer):
        if _is_disclaimer(claim) or _is_structure_label(claim):
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

        cited_items = []
        deterministic_failure = False
        for citation in citations:
            if citation < 1 or citation > len(evidence):
                failures.append(
                    {
                        "code": "unknown_citation",
                        "claim": claim,
                        "citation": citation,
                    }
                )
                deterministic_failure = True
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
                deterministic_failure = True
                continue
            cited_items.append((citation, cited_evidence))

        if deterministic_failure:
            continue

        if joint_citations and len(cited_items) > 1:
            combined_evidence = {
                "parent_id": " + ".join(
                    str(item.get("parent_id", ""))
                    for _citation, item in cited_items
                ),
                "source_id": " + ".join(
                    str(item.get("source_id", ""))
                    for _citation, item in cited_items
                ),
                "version": " + ".join(
                    str(item.get("version", ""))
                    for _citation, item in cited_items
                ),
                "section_number": " + ".join(
                    str(item.get("section_number", ""))
                    for _citation, item in cited_items
                ),
                "text": "\n\n".join(
                    f"[{citation}] {item.get('text', '')}"
                    for citation, item in cited_items
                ),
                "score": None,
                "joint": True,
                "visual_evidence": [
                    {
                        "citation": citation,
                        **{
                            key: item[key]
                            for key in (
                                "parent_id",
                                "source_id",
                                "version",
                                "section_number",
                                "page_number",
                                "image_path",
                                "image_url",
                            )
                            if key in item
                        },
                    }
                    for citation, item in cited_items
                    if item.get("modality") == "image"
                    and isinstance(item.get("image_path"), str)
                ],
            }
            if not entailment_evaluator(claim, combined_evidence):
                failures.append(
                    {
                        "code": "unsupported_claim",
                        "claim": claim,
                        "citation": None,
                    }
                )
            continue

        for citation, cited_evidence in cited_items:
            if not entailment_evaluator(claim, cited_evidence):
                failures.append(
                    {
                        "code": "unsupported_claim",
                        "claim": claim,
                        "citation": citation,
                    }
                )

    return {"valid": not failures, "failures": failures}
