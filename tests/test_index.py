"""Tests for the indexing helpers and repeatable collection rebuild."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from ingest.chunk import Chunk
from ingest.chunk_parent import Section
from ingest import index as index_module
from ingest.index import document_meta_for, filter_stubs, strip_heading
from ingest.schema import content_sha256


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


def test_document_meta_for_builds_versioned_identity_from_corpus_stem():
    meta = document_meta_for(Path("GBT+22239-2019.md"), "standard body")

    assert meta.document_id == "GBT-22239@2019"
    assert meta.jurisdiction == "CN"
    assert meta.content_hash == content_sha256("standard body")


def test_document_meta_for_does_not_repeat_gdpr_version_in_source_id():
    meta = document_meta_for(Path("CELEX_32016R0679_EN_TXT.md"), "regulation body")

    assert meta.document_id == "GDPR@2016-679"


def test_bge_model_loader_passes_cuda_and_fp16(monkeypatch):
    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, model_path, **kwargs):
            calls.update(model_path=model_path, **kwargs)

    monkeypatch.setenv("EMBED_MODEL_SOURCE", "hf")
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setattr(
        index_module, "select_model_device", lambda _name: "cuda", raising=False
    )
    monkeypatch.setattr(
        index_module,
        "model_kwargs_for",
        lambda _device: {"dtype": "float16"},
        raising=False,
    )

    index_module.get_model()

    assert calls == {
        "model_path": index_module.MODEL_NAME,
        "device": "cuda",
        "model_kwargs": {"dtype": "float16"},
    }


def test_bge_model_loader_passes_cpu_without_model_kwargs(monkeypatch):
    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, model_path, **kwargs):
            calls.update(model_path=model_path, **kwargs)

    monkeypatch.setenv("EMBED_MODEL_SOURCE", "hf")
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setattr(index_module, "select_model_device", lambda _name: "cpu")
    monkeypatch.setattr(index_module, "model_kwargs_for", lambda _device: None)

    index_module.get_model()

    assert calls == {
        "model_path": index_module.MODEL_NAME,
        "device": "cpu",
        "model_kwargs": None,
    }


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakeQdrantClient:
    def __init__(self):
        self.exists = True
        self.delete_calls = 0
        self.create_calls = 0
        self.points = []

    def collection_exists(self, _name):
        return self.exists

    def delete_collection(self, _name):
        self.delete_calls += 1
        self.exists = False
        self.points = []

    def create_collection(self, _name, *, vectors_config):
        self.create_calls += 1
        self.exists = True
        self.vectors_config = vectors_config

    def upsert(self, _name, *, points):
        self.points.extend(points)


def test_index_rebuild_is_repeatable_and_embeds_only_children(monkeypatch, tmp_path):
    parent = Section(
        id="GBT-22239@2019#6.1.1",
        title="6.1.1 安全物理环境",
        number="6.1.1",
        level=3,
        text="完整父块，不应参与嵌入",
    )
    child = Chunk(
        id="GBT-22239@2019#6.1.1:0",
        text="用于检索的子块",
        metadata={
            "parent_id": parent.id,
            "source": "GBT-22239",
            "version": "2019",
            "section_number": "6.1.1",
            "section_title": "安全物理环境",
        },
    )
    client = FakeQdrantClient()
    embedded_batches = []

    def fake_embed(_model, texts):
        embedded_batches.append(texts)
        return [FakeVector([0.1] * index_module.BGE_M3_DIM) for _ in texts]

    monkeypatch.setattr(index_module, "PARENTS_STORE", tmp_path / "parents.json")
    monkeypatch.setattr(index_module, "build_corpus", lambda: ([parent], [child]))
    monkeypatch.setattr(index_module, "get_model", lambda: object())
    monkeypatch.setattr(index_module, "embed", fake_embed)
    monkeypatch.setattr(index_module, "_client", lambda: client)

    assert index_module.index() == 1
    assert index_module.index() == 1

    assert embedded_batches == [[child.text], [child.text]]
    assert client.delete_calls == 2
    assert client.create_calls == 2
    assert len(client.points) == 1
    assert client.points[0].payload["version"] == "2019"
    stored = json.loads(index_module.PARENTS_STORE.read_text(encoding="utf-8"))
    assert stored[parent.id]["text"] == parent.text
