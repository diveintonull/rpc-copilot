"""Tests for first-start Qdrant index bootstrapping."""

from __future__ import annotations

from types import SimpleNamespace

from api.bootstrap import ensure_real_index, ensure_visual_index


class FakeClient:
    def __init__(self, *, exists: bool, count: int) -> None:
        self.exists = exists
        self.point_count = count
        self.closed = False

    def collection_exists(self, _name: str) -> bool:
        return self.exists

    def count(self, _name: str, *, exact: bool):
        assert exact is False
        return SimpleNamespace(count=self.point_count)

    def close(self) -> None:
        self.closed = True


def test_bootstrap_skips_existing_non_empty_collection() -> None:
    client = FakeClient(exists=True, count=1708)

    result = ensure_real_index(
        client_factory=lambda **_kwargs: client,
        indexer=lambda: (_ for _ in ()).throw(
            AssertionError("indexer must not run")
        ),
    )

    assert result == 0
    assert client.closed is True


def test_bootstrap_indexes_an_empty_collection() -> None:
    client = FakeClient(exists=True, count=0)

    result = ensure_real_index(
        client_factory=lambda **_kwargs: client,
        indexer=lambda: 1708,
    )

    assert result == 1708
    assert client.closed is True


def test_visual_bootstrap_prepares_pages_before_indexing(
    tmp_path,
    monkeypatch,
) -> None:
    import api.bootstrap as bootstrap

    client = FakeClient(exists=False, count=0)
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(bootstrap, "VISUAL_MANIFEST", manifest)
    calls = []

    result = ensure_visual_index(
        client_factory=lambda **_kwargs: client,
        page_preparer=lambda: calls.append("prepare"),
        indexer=lambda: calls.append("index") or 88,
    )

    assert result == 88
    assert calls == ["prepare", "index"]
    assert client.closed is True
