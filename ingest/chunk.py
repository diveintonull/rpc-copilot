"""Naive fixed-size chunker (P1-04).

Splits text into fixed-length, overlapping windows and wraps each in a uniform
Chunk (id / text / metadata). This is the baseline; the parent-child chunker
(P1-05) builds on it. Chinese is split by character count, not tokens.

CLI: uv run python -m ingest.chunk   (prints per-document chunk stats)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PARSED = Path("data/parsed")

DEFAULT_SIZE = 500
DEFAULT_OVERLAP = 100


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def iter_spans(length: int, size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[tuple[int, int]]:
    """(start, end) windows over [0, length) with the given size and overlap.

    Stops as soon as a window reaches the end, so there is no redundant tail.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < size, got overlap={overlap}, size={size}")
    if length <= 0:
        return []

    stride = size - overlap
    spans: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + size, length)
        spans.append((start, end))
        if end >= length:
            break
        start += stride
    return spans


def chunk_document(
    text: str,
    *,
    doc_id: str,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split `text` into Chunks with stable ids and span metadata."""
    base = metadata or {}
    chunks: list[Chunk] = []
    for i, (start, end) in enumerate(iter_spans(len(text), size=size, overlap=overlap)):
        meta = {**base, "chunk_index": i, "char_start": start, "char_end": end}
        chunks.append(Chunk(id=f"{doc_id}:{i}", text=text[start:end], metadata=meta))
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict:
    """Count and length summary for a list of chunks."""
    lengths = [len(c.text) for c in chunks]
    n = len(chunks)
    return {
        "count": n,
        "avg_length": (sum(lengths) / n) if n else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }


def main() -> None:
    files = sorted(PARSED.glob("*.md"))
    if not files:
        print(f"No parsed markdown in {PARSED}/ — run the Day-1 parsers first.")
        return

    header = f"{'file':32s} {'chunks':>7s} {'avg_len':>8s} {'min':>5s} {'max':>5s}"
    print(header)
    print("-" * len(header))

    all_chunks: list[Chunk] = []
    for f in files:
        chunks = chunk_document(f.read_text(encoding="utf-8"), doc_id=f.stem)
        all_chunks.extend(chunks)
        s = chunk_stats(chunks)
        print(f"{f.name:32s} {s['count']:7d} {s['avg_length']:8.1f} {s['min_length']:5d} {s['max_length']:5d}")

    t = chunk_stats(all_chunks)
    print("-" * len(header))
    print(f"{'TOTAL':32s} {t['count']:7d} {t['avg_length']:8.1f} {t['min_length']:5d} {t['max_length']:5d}")


if __name__ == "__main__":
    main()
