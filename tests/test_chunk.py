"""Tests for the naive fixed-size chunker (P1-04)."""

import pytest

from ingest.chunk import Chunk, chunk_document, chunk_stats, iter_spans


def test_iter_spans_splits_into_overlapping_windows():
    # length 1000, window 400, overlap 100 -> stride 300, stop once a window hits the end
    assert iter_spans(1000, size=400, overlap=100) == [(0, 400), (300, 700), (600, 1000)]


def test_consecutive_chunks_share_overlap_chars():
    text = "".join(str(i % 10) for i in range(1000))  # 1000 distinct-ish chars
    chunks = chunk_document(text, doc_id="doc", size=400, overlap=100)
    first, second = chunks[0].text, chunks[1].text
    assert first[-100:] == second[:100]  # the 100-char overlap region matches


def test_text_shorter_than_size_is_one_chunk():
    text = "abc" * 10  # 30 chars, well under the window
    chunks = chunk_document(text, doc_id="d", size=500, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_empty_text_yields_no_chunks():
    assert chunk_document("", doc_id="d") == []


def test_chunk_carries_id_and_span_metadata():
    text = "x" * 1000
    chunks = chunk_document(
        text, doc_id="etc", size=400, overlap=100, metadata={"source_file": "a.md"}
    )
    c0 = chunks[0]
    assert isinstance(c0, Chunk)
    assert c0.id == "etc:0"
    assert c0.metadata["chunk_index"] == 0
    assert c0.metadata["char_start"] == 0
    assert c0.metadata["char_end"] == 400
    assert c0.metadata["source_file"] == "a.md"  # caller metadata is preserved
    assert len({c.id for c in chunks}) == len(chunks)  # ids are unique


def test_overlap_must_be_less_than_size():
    with pytest.raises(ValueError):
        chunk_document("abc", doc_id="d", size=100, overlap=100)


def test_chunk_stats_reports_count_and_average_length():
    chunks = chunk_document("y" * 1000, doc_id="d", size=400, overlap=100)
    stats = chunk_stats(chunks)
    assert stats["count"] == 3
    assert stats["avg_length"] == 400
