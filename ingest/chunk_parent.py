"""Parent-child segmenter (P1-05).

Day-1 finding: our parsed Markdown has flat `##` headings — the real hierarchy
lives in the *numbering* (`6.1.9.1`) for GB/T standards and in `第X条` markers
for PRC statutes, not in the `#` depth. So we reconstruct sections from those,
not from heading levels.

Model:
* A **parent** (`Section`) is a coherent unit — a numbered subsection (等保) or an
  article (法律) — stored whole and NOT embedded.
* A **child** is a naive fixed-size chunk of that section's text, carrying
  `parent_id` in its metadata. Only children go into the vector store; at query
  time a child hit is expanded to its parent for full context (small-to-big).

CLI: uv run python -m ingest.chunk_parent   (writes 等保 parent-child JSON)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ingest.chunk import Chunk, chunk_document

PARSED = Path("data/parsed")
DENGBAO = PARSED / "GBT+22239-2019.md"

_HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*$")
_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)")
# Article markers like 第一条 / 第十条 / 第八十一条 at the start of a line.
_ARTICLE_RE = re.compile(r"^(第[一二三四五六七八九十百千零〇两]+条)")


@dataclass
class Section:
    id: str
    title: str
    number: str
    level: int
    text: str


def _heading_title(line: str) -> str | None:
    m = _HEADING_RE.match(line)
    return m.group(1) if m else None


def segment_by_headings(text: str, *, doc_id: str) -> list[Section]:
    """Split Markdown into sections at `#` headings; depth from the dotted number.

    Content before the first heading (title page / front matter) is dropped.
    """
    lines = text.splitlines(keepends=True)
    # Indices of heading lines.
    starts = [i for i, ln in enumerate(lines) if _heading_title(ln) is not None]

    sections: list[Section] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        title = _heading_title(lines[start]) or ""
        body = "".join(lines[start:end])
        m = _NUMBER_RE.match(title)
        number = m.group(1) if m else ""
        level = len(number.split(".")) if number else 0
        sid = f"{doc_id}#{number}" if number else f"{doc_id}#h{idx}"
        sections.append(Section(id=sid, title=title, number=number, level=level, text=body))
    return sections


def segment_by_articles(text: str, *, doc_id: str) -> list[Section]:
    """Split a PRC statute into articles at `第X条` markers (parents = articles).

    Content before the first article (chapter titles / preamble) is dropped.
    """
    lines = text.splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines) if _ARTICLE_RE.match(ln.strip())]

    sections: list[Section] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        number = _ARTICLE_RE.match(lines[start].strip()).group(1)
        body = "".join(lines[start:end])
        sections.append(Section(id=f"{doc_id}#{number}", title=number, number=number, level=1, text=body))
    return sections


_STRATEGIES = {
    "headings": segment_by_headings,
    "articles": segment_by_articles,
}


def build_parent_child(
    text: str,
    *,
    doc_id: str,
    strategy: str = "headings",
    child_size: int = 500,
    child_overlap: int = 100,
) -> tuple[list[Section], list[Chunk]]:
    """Segment into parents, then split each parent into children tagged with parent_id."""
    sections = _STRATEGIES[strategy](text, doc_id=doc_id)
    children: list[Chunk] = []
    for sec in sections:
        children.extend(
            chunk_document(
                sec.text,
                doc_id=sec.id,
                size=child_size,
                overlap=child_overlap,
                metadata={
                    "parent_id": sec.id,
                    "section_number": sec.number,
                    "section_title": sec.title,
                },
            )
        )
    return sections, children


def to_json(parents: list[Section], children: list[Chunk]) -> dict:
    kids_by_parent: dict[str, list[str]] = {}
    for c in children:
        kids_by_parent.setdefault(c.metadata["parent_id"], []).append(c.id)
    return {
        "parents": [
            {
                "id": p.id,
                "number": p.number,
                "level": p.level,
                "title": p.title,
                "text": p.text,
                "child_ids": kids_by_parent.get(p.id, []),
            }
            for p in parents
        ],
        "children": [
            {
                "id": c.id,
                "parent_id": c.metadata["parent_id"],
                "char_start": c.metadata["char_start"],
                "char_end": c.metadata["char_end"],
            }
            for c in children
        ],
    }


def main() -> None:
    if not DENGBAO.exists():
        print(f"{DENGBAO} not found — run the Day-1 parsers first.")
        return

    parents, children = build_parent_child(DENGBAO.read_text(encoding="utf-8"), doc_id=DENGBAO.stem)
    out = PARSED / f"{DENGBAO.stem}.parents.json"
    out.write_text(json.dumps(to_json(parents, children), ensure_ascii=False, indent=2), encoding="utf-8")

    numbered = [p for p in parents if p.number]
    print(f"parents={len(parents)} (numbered={len(numbered)})  children={len(children)}")
    print(f"wrote {out}")
    print("--- 5 sample child -> parent traces (ids only) ---")
    for c in children[:5]:
        print(f"  {c.id}  ->  {c.metadata['parent_id']}")


if __name__ == "__main__":
    main()
