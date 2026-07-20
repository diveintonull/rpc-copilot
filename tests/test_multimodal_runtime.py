"""Runtime configuration tests for opt-in multimodal RAG."""

from __future__ import annotations

import pytest

from agent.tools import LocalToolBackend, MultimodalLocalToolBackend
from api.real_runtime import build_real_runner


class FakeLLM:
    def entails(self, _claim, _evidence):
        return True


class FakeGraph:
    pass


def test_runtime_selects_multimodal_backend_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("MULTIMODAL_RAG_ENABLED", "true")
    captured = []

    def graph_factory(_router, tools, _llm, **_kwargs):
        captured.append(tools)
        return FakeGraph()

    build_real_runner(llm=FakeLLM(), graph_factory=graph_factory)

    assert len(captured) == 3
    assert all(isinstance(tools, MultimodalLocalToolBackend) for tools in captured)


def test_runtime_keeps_text_backend_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MULTIMODAL_RAG_ENABLED", raising=False)
    captured = []

    def graph_factory(_router, tools, _llm, **_kwargs):
        captured.append(tools)
        return FakeGraph()

    build_real_runner(llm=FakeLLM(), graph_factory=graph_factory)

    assert all(type(tools) is LocalToolBackend for tools in captured)


def test_runtime_rejects_invalid_multimodal_boolean(monkeypatch) -> None:
    monkeypatch.setenv("MULTIMODAL_RAG_ENABLED", "sometimes")

    with pytest.raises(
        RuntimeError,
        match="MULTIMODAL_RAG_ENABLED must be true or false",
    ):
        build_real_runner(llm=FakeLLM(), graph_factory=lambda *_args, **_kwargs: FakeGraph())
