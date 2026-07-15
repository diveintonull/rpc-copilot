"""Create the real Qdrant index on first container startup."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from ingest.index import COLLECTION, DEFAULT_QDRANT_URL, index


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


def main() -> None:
    load_dotenv()
    indexed = ensure_real_index()
    if indexed:
        print(f"bootstrapped {COLLECTION} with {indexed} child chunks")
    else:
        print(f"using existing non-empty {COLLECTION} collection")


if __name__ == "__main__":
    main()
