"""Tests for governed PDF page preparation."""

from __future__ import annotations

import json
from pathlib import Path

from ingest.visual import load_mineru_page_text, load_visual_manifest, prepare_visual_pages


def test_load_mineru_page_text_keeps_text_lists_tables_and_captions(tmp_path: Path) -> None:
    root = tmp_path / "mineru"
    content = root / "GBT+22239-2019" / "ocr" / "GBT+22239-2019_content_list.json"
    content.parent.mkdir(parents=True)
    content.write_text(
        json.dumps(
            [
                {"type": "text", "text": "身份鉴别", "page_idx": 0},
                {"type": "list", "list_items": ["要求一", "要求二"], "page_idx": 0},
                {"type": "table", "table_caption": ["表 1"], "table_body": "控制 | 要求", "page_idx": 1},
                {"type": "image", "image_caption": ["图 1 架构"], "page_idx": 1},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pages = load_mineru_page_text("GBT+22239-2019", root)

    assert pages[1] == "身份鉴别\n要求一\n要求二"
    assert pages[2] == "表 1\n控制 | 要求\n图 1 架构"


def test_prepare_visual_pages_writes_versioned_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pdf = raw / "GBT+22239-2019.pdf"
    pdf.write_bytes(b"fake governed pdf")

    mineru = tmp_path / "mineru"
    content = mineru / pdf.stem / "ocr" / f"{pdf.stem}_content_list.json"
    content.parent.mkdir(parents=True)
    content.write_text(
        json.dumps([{"type": "text", "text": "第一页要求", "page_idx": 0}], ensure_ascii=False),
        encoding="utf-8",
    )

    def renderer(_pdf: Path, target: Path, _dpi: int, _force: bool) -> list[Path]:
        target.mkdir(parents=True)
        pages = [target / "page-0001.png", target / "page-0002.png"]
        for page in pages:
            page.write_bytes(b"png")
        return pages

    visual_root = tmp_path / "visual"
    pages = prepare_visual_pages(
        raw_root=raw,
        mineru_root=mineru,
        visual_root=visual_root,
        renderer=renderer,
    )

    assert [page.visual_id for page in pages] == [
        "GBT-22239@2019#page=1",
        "GBT-22239@2019#page=2",
    ]
    assert pages[0].text == "第一页要求"
    assert pages[0].image_relpath == "GBT-22239-2019/page-0001.png"
    assert len(pages[0].content_hash) == 64
    assert load_visual_manifest(visual_root / "manifest.json") == pages
