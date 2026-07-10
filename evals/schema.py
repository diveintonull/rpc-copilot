"""Strict, JSON-friendly contracts for GRC evaluation cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TASK_TYPES = frozenset(
    {"regulation_qa", "clause_comparison", "gap_analysis", "unsupported"}
)
CASE_FIELDS = frozenset(
    {
        "id",
        "question",
        "task_type",
        "gold_points",
        "gold_citations",
        "should_refuse",
        "source_versions",
        "tags",
    }
)


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    question: str
    task_type: str
    gold_points: tuple[str, ...]
    gold_citations: tuple[str, ...]
    should_refuse: bool
    source_versions: tuple[str, ...]
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.task_type not in TASK_TYPES:
            raise ValueError(
                f"unknown task_type {self.task_type!r}; expected one of {sorted(TASK_TYPES)}"
            )
        if not self.should_refuse and not self.gold_points:
            raise ValueError("gold_points are required when should_refuse is false")
        if not self.should_refuse and not self.gold_citations:
            raise ValueError("gold_citations are required when should_refuse is false")
        if "version_trap" in self.tags and not self.source_versions:
            raise ValueError("source_versions are required for version_trap cases")
        undeclared = [
            citation
            for citation in self.gold_citations
            if not any(
                citation.startswith(f"{source_version}#")
                for source_version in self.source_versions
            )
        ]
        if undeclared:
            raise ValueError(
                "gold_citations must belong to declared source_versions: "
                + ", ".join(undeclared)
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationCase":
        """Create an immutable case from one decoded JSON object."""
        unknown = sorted(set(payload) - CASE_FIELDS)
        if unknown:
            raise ValueError(f"unknown fields: {', '.join(unknown)}")
        missing = sorted(CASE_FIELDS - set(payload))
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")
        for name in ("id", "question", "task_type"):
            if not isinstance(payload[name], str) or not payload[name].strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("gold_points", "gold_citations", "source_versions", "tags"):
            value = payload[name]
            if not isinstance(value, list) or any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                raise ValueError(f"{name} must be a list of non-empty strings")
        if type(payload["should_refuse"]) is not bool:
            raise ValueError("should_refuse must be a boolean")
        return cls(
            id=payload["id"],
            question=payload["question"],
            task_type=payload["task_type"],
            gold_points=tuple(payload["gold_points"]),
            gold_citations=tuple(payload["gold_citations"]),
            should_refuse=payload["should_refuse"],
            source_versions=tuple(payload["source_versions"]),
            tags=tuple(payload["tags"]),
        )
