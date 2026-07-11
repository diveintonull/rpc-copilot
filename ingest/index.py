"""Embed child chunks with bge-m3 and index them in Qdrant.

Only children are embedded and searched; parents are written to a separate JSON
store (not vectorised) for small-to-big expansion at query time.

Heavy deps (torch / sentence-transformers / qdrant-client) are imported lazily so
the pure helpers stay unit-testable without loading a 2 GB model.

CLI:
  uv run python -m ingest.index          # build + embed + index, print count
  uv run python -m ingest.index "query"  # search demo, print top-5
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from ingest.chunk import Chunk
from ingest.chunk_parent import PARSED, Section, build_parent_child
from ingest.schema import DocumentMeta, content_sha256
from runtime.device import model_kwargs_for, select_model_device

# Per-document segmentation strategy (GB/T standards + GDPR are heading-numbered;
# PRC statutes are article-numbered).
STRATEGY = {
    "GBT+22239-2019": "headings",
    "GBT+35273-2020": "headings",
    "CELEX_32016R0679_EN_TXT": "headings",
    "cybersecurity-law": "articles",
    "data-security-law": "articles",
}

DOCUMENT_SPECS = {
    "GBT+22239-2019": {
        "source_id": "GBT-22239",
        "title": "网络安全等级保护基本要求",
        "jurisdiction": "CN",
        "version": "2019",
        "effective_date": None,
        "source_url": "https://openstd.samr.gov.cn/",
    },
    "GBT+35273-2020": {
        "source_id": "GBT-35273",
        "title": "个人信息安全规范",
        "jurisdiction": "CN",
        "version": "2020",
        "effective_date": None,
        "source_url": "https://openstd.samr.gov.cn/",
    },
    "CELEX_32016R0679_EN_TXT": {
        "source_id": "GDPR",
        "title": "General Data Protection Regulation",
        "jurisdiction": "EU",
        "version": "2016-679",
        "effective_date": date(2018, 5, 25),
        "source_url": "https://eur-lex.europa.eu/eli/reg/2016/679/oj",
    },
    "cybersecurity-law": {
        "source_id": "cybersecurity-law",
        "title": "中华人民共和国网络安全法",
        "jurisdiction": "CN",
        "version": "2025-amended",
        "effective_date": None,
        "source_url": "https://www.cac.gov.cn/",
    },
    "data-security-law": {
        "source_id": "data-security-law",
        "title": "中华人民共和国数据安全法",
        "jurisdiction": "CN",
        "version": "2021",
        "effective_date": date(2021, 9, 1),
        "source_url": "https://www.cac.gov.cn/",
    },
}

COLLECTION = "grc_kb"
MODEL_NAME = "BAAI/bge-m3"
BGE_M3_DIM = 1024
QDRANT_URL = "http://localhost:6333"
PARENTS_STORE = PARSED / "_parents_store.json"


def document_meta_for(path: Path, text: str) -> DocumentMeta:
    """Build provenance for a parsed corpus file, with a safe local fallback."""
    spec = DOCUMENT_SPECS.get(path.stem)
    if spec is None:
        spec = {
            "source_id": path.stem,
            "title": path.stem,
            "jurisdiction": "UNKNOWN",
            "version": "unversioned",
            "effective_date": None,
            "source_url": f"file://{path.as_posix()}",
        }
    return DocumentMeta(**spec, content_hash=content_sha256(text))


def strip_heading(text: str) -> str:
    """Section body with its leading Markdown heading line removed."""
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def filter_stubs(
    parents: list[Section], children: list[Chunk], min_body: int = 10
) -> tuple[list[Section], list[Chunk]]:
    """Drop structural stub sections (heading with (almost) no body) and their children."""
    stub_ids = {p.id for p in parents if len(strip_heading(p.text)) < min_body}
    kept_parents = [p for p in parents if p.id not in stub_ids]
    kept_children = [c for c in children if c.metadata["parent_id"] not in stub_ids]
    return kept_parents, kept_children


def build_corpus() -> tuple[list[Section], list[Chunk]]:
    """All (parents, children) across data/parsed, per-doc strategy, stubs removed."""
    parents_all: list[Section] = []
    children_all: list[Chunk] = []
    for md in sorted(PARSED.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        document = document_meta_for(md, text)
        strategy = STRATEGY.get(md.stem, "headings")
        parents, children = build_parent_child(
            text,
            doc_id=document.document_id,
            strategy=strategy,
            metadata=document.to_payload(),
        )
        for c in children:
            c.metadata["source"] = document.source_id
            c.metadata["document_id"] = document.document_id
        parents_all.extend(parents)
        children_all.extend(children)
    return filter_stubs(parents_all, children_all)


def get_model():
    """Load bge-m3. Defaults to ModelScope (HuggingFace is unreachable here);
    set EMBED_MODEL_SOURCE=hf to load straight from HuggingFace instead."""
    import os

    from sentence_transformers import SentenceTransformer

    device = select_model_device("bge-m3")
    model_kwargs = model_kwargs_for(device)

    if os.environ.get("EMBED_MODEL_SOURCE", "modelscope") == "modelscope":
        from modelscope import snapshot_download

        model_path = snapshot_download(MODEL_NAME)
    else:
        model_path = MODEL_NAME
    return SentenceTransformer(
        model_path,
        device=device,
        model_kwargs=model_kwargs,
    )


def embed(model, texts: list[str]):
    return model.encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True
    )


def _client():
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL)


def index() -> int:
    from qdrant_client import models

    parents, children = build_corpus()

    # Parents: stored whole, not embedded.
    store = {
        p.id: {
            "text": p.text,
            "number": p.number,
            "title": p.title,
            "level": p.level,
            "metadata": p.metadata,
        }
        for p in parents
    }
    PARENTS_STORE.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")

    model = get_model()
    vectors = embed(model, [c.text for c in children])

    client = _client()
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=BGE_M3_DIM, distance=models.Distance.COSINE),
    )

    points = [
        models.PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                **c.metadata,
                "chunk_id": c.id,
                "text": c.text,
                "parent_id": c.metadata["parent_id"],
                "source": c.metadata["source"],
                "section_number": c.metadata.get("section_number", ""),
                "section_title": c.metadata.get("section_title", ""),
            },
        )
        for i, c in enumerate(children)
    ]
    for start in range(0, len(points), 256):
        client.upsert(COLLECTION, points=points[start : start + 256])

    return len(children)


def search(query: str, k: int = 5):
    model = get_model()
    qv = embed(model, [query])[0]
    hits = _client().query_points(COLLECTION, query=qv.tolist(), limit=k, with_payload=True).points
    return hits


def main() -> None:
    try:  # avoid GBK console crashes when printing Chinese clause ids
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        for h in search(query):
            print(f"[{h.score:.3f}] {h.payload['source']} {h.payload['section_number']} -> {h.payload['parent_id']}")
        return

    n = index()
    count = _client().count(COLLECTION).count
    print(f"indexed children={n}  qdrant vector count={count}  match={n == count}")


if __name__ == "__main__":
    main()
