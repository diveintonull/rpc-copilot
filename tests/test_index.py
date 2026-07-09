"""Tests for the pure helpers in the indexing module (P1-06)."""

from ingest.chunk import Chunk
from ingest.chunk_parent import Section
from ingest.index import filter_stubs, strip_heading


def test_strip_heading_removes_leading_heading_line():
    assert strip_heading("## 6.1.1 安全物理环境\n本项要求包括：\na）应…") == "本项要求包括：\na）应…"


def test_strip_heading_is_empty_when_only_a_heading():
    assert strip_heading("## 6 第一级安全要求\n") == ""


def test_filter_stubs_drops_heading_only_sections():
    parents = [
        Section(id="d#6", title="6 x", number="6", level=1, text="## 6 x\n"),
        Section(id="d#6.1.1", title="6.1.1 y", number="6.1.1", level=3, text="## 6.1.1 y\nreal requirement body"),
    ]
    children = [
        Chunk(id="d#6:0", text="## 6 x\n", metadata={"parent_id": "d#6"}),
        Chunk(id="d#6.1.1:0", text="## 6.1.1 y\nreal requirement body", metadata={"parent_id": "d#6.1.1"}),
    ]
    kept_parents, kept_children = filter_stubs(parents, children, min_body=5)
    assert [p.id for p in kept_parents] == ["d#6.1.1"]
    assert [c.id for c in kept_children] == ["d#6.1.1:0"]
