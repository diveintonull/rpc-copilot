"""Tests for multimodal fusion and Agent routing."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.nodes import execute_regulation_qa
from agent.tools import MultimodalLocalToolBackend
from rag.multimodal import reciprocal_rank_fuse_evidence, search_multimodal_evidence
from rag.visual import VisualHit


def text_evidence(parent_id: str, score: float) -> dict:
    return {
        "parent_id": parent_id,
        "source_id": "GBT-22239",
        "version": "2019",
        "section_number": "8.1.4.1",
        "text": "文本条款",
        "score": score,
    }


def visual_hit(page: int, score: float = 7.2) -> VisualHit:
    return VisualHit(
        visual_id=f"GBT-22239@2019#page={page}",
        score=score,
        document_id="GBT-22239@2019",
        source_id="GBT-22239",
        version="2019",
        title="网络安全等级保护基本要求",
        page_number=page,
        image_relpath=f"GBT-22239-2019/page-{page:04d}.png",
        text="页面中的身份鉴别表格",
    )


def test_rrf_interleaves_text_and_visual_evidence() -> None:
    fused = reciprocal_rank_fuse_evidence(
        [text_evidence("GBT-22239@2019#8.1.4.1", 0.91), text_evidence("GBT-22239@2019#8.1.4.2", 0.8)],
        [
            {"parent_id": "GBT-22239@2019#page=12", "score": 8.2, "modality": "image"},
            {"parent_id": "GBT-22239@2019#page=13", "score": 7.1, "modality": "image"},
        ],
        limit=4,
    )

    assert [item["parent_id"] for item in fused] == [
        "GBT-22239@2019#8.1.4.1",
        "GBT-22239@2019#page=12",
        "GBT-22239@2019#8.1.4.2",
        "GBT-22239@2019#page=13",
    ]
    assert fused[0]["score"] == pytest.approx(1.0)
    assert fused[1]["modality"] == "image"


def test_search_multimodal_builds_private_and_public_image_paths(tmp_path: Path) -> None:
    result = search_multimodal_evidence(
        "身份鉴别表格",
        text_search=lambda _query, _sources: [text_evidence("GBT-22239@2019#8.1.4.1", 0.9)],
        visual_search=lambda _query, _sources: [visual_hit(12)],
        pages_root=tmp_path,
        limit=2,
    )

    image = next(item for item in result if item["modality"] == "image")
    assert image["parent_id"] == "GBT-22239@2019#page=12"
    assert image["page_number"] == 12
    assert image["image_url"] == "/visual-assets/GBT-22239-2019/page-0012.png"
    assert image["image_path"] == str(tmp_path / "GBT-22239-2019/page-0012.png")


def test_multimodal_backend_keeps_plain_search_for_other_workflows() -> None:
    calls = []
    backend = MultimodalLocalToolBackend(
        search_tool=lambda _query, _sources: [text_evidence("text", 0.8)],
        multimodal_search_tool=lambda _query, _sources: calls.append("visual") or [text_evidence("visual", 0.9)],
    )

    assert backend.search_regulation("q")[0]["parent_id"] == "text"
    assert backend.search_regulation_multimodal("q")[0]["parent_id"] == "visual"
    assert calls == ["visual"]


def test_regulation_node_prefers_optional_multimodal_tool() -> None:
    evidence = text_evidence("GBT-22239@2019#page=12", 0.9)

    class Tools:
        def search_regulation_multimodal(self, _query, _source_ids):
            return [evidence]

        def search_regulation(self, *_args):
            raise AssertionError("plain search should not run")

    class LLM:
        def answer_regulation(self, _query, supplied, skill_text=""):
            assert supplied[0]["parent_id"] == evidence["parent_id"]
            return "页面支持该要求[1]"

    update = execute_regulation_qa(
        {"query": "身份鉴别表格", "trace": [], "tool_calls": []},
        Tools(),
        LLM(),
    )

    assert update["trace"][-1]["tool"] == "search_regulation_multimodal"
    assert update["tool_calls"][-1]["tool"] == "search_regulation_multimodal"
