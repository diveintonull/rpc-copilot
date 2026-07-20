"""Create the real Qdrant index on first container startup."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from ingest.index import COLLECTION, DEFAULT_QDRANT_URL, index
from ingest.visual import VISUAL_MANIFEST, prepare_visual_pages
from rag.visual import VISUAL_COLLECTION, build_visual_index


def ensure_real_index(
    *,
    client_factory: Callable[..., Any] = QdrantClient,
    indexer: Callable[[], int] = index,
) -> int:
    """Return zero for an existing index or build and return its point count."""
    url = os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)
    client = client_factory(url=url, timeout=5)
    try:
        if (
            client.collection_exists(COLLECTION)
            and client.count(COLLECTION, exact=False).count > 0
        ):
            return 0
    finally:
        client.close()
    return indexer()


def ensure_visual_index(
    *,
    client_factory: Callable[..., Any] = QdrantClient,
    page_preparer: Callable[[], Any] = prepare_visual_pages,
    indexer: Callable[[], int] = build_visual_index,
) -> int:
    """Build rendered pages and their dedicated visual collection when absent."""
    url = os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)
    client = client_factory(url=url, timeout=5)
    try:
        if (
            VISUAL_MANIFEST.is_file()
            and client.collection_exists(VISUAL_COLLECTION)
            and client.count(VISUAL_COLLECTION, exact=False).count > 0
        ):
            return 0
    finally:
        client.close()
    page_preparer()
    return indexer()


def main() -> None:
    load_dotenv()
    indexed = ensure_real_index()
    if indexed:
        print(f"bootstrapped {COLLECTION} with {indexed} child chunks")
    else:
        print(f"using existing non-empty {COLLECTION} collection")
    if os.environ.get("MULTIMODAL_RAG_ENABLED", "false").strip().casefold() == "true":
        visual_indexed = ensure_visual_index()
        if visual_indexed:
            print(
                f"bootstrapped {VISUAL_COLLECTION} with "
                f"{visual_indexed} rendered pages"
            )
        else:
            print(f"using existing non-empty {VISUAL_COLLECTION} collection")


if __name__ == "__main__":
    main()
