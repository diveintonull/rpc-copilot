"""Validate the line-oriented GRC evaluation dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from evals.schema import EvaluationCase


@dataclass(frozen=True, slots=True)
class ValidationReport:
    valid: int
    invalid: int
    errors: tuple[str, ...]


def validate_dataset(path: Path) -> ValidationReport:
    """Validate every non-empty JSONL row and retain line-scoped errors."""
    valid = 0
    errors: list[str] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("row must be a JSON object")
            case = EvaluationCase.from_dict(payload)
            if case.id in seen_ids:
                raise ValueError(f"duplicate id {case.id!r}")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"line {line_number}: {exc}")
        else:
            seen_ids.add(case.id)
            valid += 1
    return ValidationReport(valid=valid, invalid=len(errors), errors=tuple(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("evals/dataset.jsonl"))
    args = parser.parse_args()

    report = validate_dataset(args.dataset)
    for error in report.errors:
        print(error)
    print(f"valid={report.valid} invalid={report.invalid}")
    return 1 if report.invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
