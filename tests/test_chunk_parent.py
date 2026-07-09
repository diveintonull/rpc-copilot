"""Tests for the parent-child segmenter (P1-05)."""

from ingest.chunk_parent import (
    Section,
    build_parent_child,
    segment_by_articles,
    segment_by_headings,
)


def test_segment_by_headings_reconstructs_number_and_level():
    md = (
        "## 6 第一级安全要求\n"
        "intro\n"
        "## 6.1 安全通用要求\n"
        "body\n"
        "## 6.1.1 安全物理环境\n"
        "detail\n"
    )
    secs = segment_by_headings(md, doc_id="etc")
    assert [s.number for s in secs] == ["6", "6.1", "6.1.1"]
    assert [s.level for s in secs] == [1, 2, 3]  # depth from the dotted number, not the # count
    assert secs[0].title == "6 第一级安全要求"
    assert secs[2].id == "etc#6.1.1"


def test_segment_by_headings_keeps_body_within_its_own_section():
    md = "## 6.1 通用\nbody-A\n## 6.1.1 物理\nbody-B\n"
    secs = segment_by_headings(md, doc_id="etc")
    assert "body-A" in secs[0].text and "body-B" not in secs[0].text
    assert "body-B" in secs[1].text


def test_build_parent_child_links_every_child_to_an_existing_parent():
    md = (
        "## 6.1.1 安全物理环境\n"
        + "物" * 1200  # long body -> several children
        + "\n## 6.1.2 安全通信网络\nshort\n"
    )
    parents, children = build_parent_child(md, doc_id="etc", child_size=500, child_overlap=100)

    parent_ids = {p.id for p in parents}
    assert all(c.metadata["parent_id"] in parent_ids for c in children)

    p0 = parents[0].id  # "etc#6.1.1"
    kids0 = [c for c in children if c.metadata["parent_id"] == p0]
    assert len(kids0) >= 2  # the long section split into multiple children
    assert all(c.id.startswith(p0 + ":") for c in kids0)


def test_segment_by_articles_splits_chinese_statutes_on_article_markers():
    md = (
        "第一章　总则\n"
        "第一条　为了规范数据处理活动。\n"
        "第二条　在中华人民共和国境内。\n"
        "第三条　本法所称数据。\n"
    )
    secs = segment_by_articles(md, doc_id="dsl")
    assert isinstance(secs[0], Section)
    assert [s.number for s in secs] == ["第一条", "第二条", "第三条"]  # chapter line is not an article
    assert secs[0].id == "dsl#第一条"
    assert "为了规范" in secs[0].text
