"""Tests for streaming intermediate LangGraph states through the real runner."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager

from api.main import ChatRequest
from api.real_runtime import RealAgentRunner


class FakeLLM:
    @contextmanager
    def stream_to(self, _emitter):
        yield


class FakeGraph:
    def __init__(self) -> None:
        self.stream_calls: list[dict] = []

    def stream(self, initial_state: dict, *, stream_mode: str):
        self.stream_calls.append(
            {"initial_state": initial_state, "stream_mode": stream_mode}
        )
        yield dict(initial_state)
        evidence = [
            {
                "parent_id": "GBT-22239@2019#8.1.4.1",
                "source_id": "GBT-22239",
                "version": "2019",
                "section_number": "8.1.4.1",
                "text": "应采用组合鉴别技术。",
            }
        ]
        yield {
            **initial_state,
            "answer": "管理员应采用组合鉴别技术[1]。",
            "evidence": evidence,
            "trace": initial_state["trace"]
            + [{"node": "execute_regulation_qa"}],
        }
        yield {
            **initial_state,
            "answer": "管理员应采用组合鉴别技术[1]。",
            "evidence": evidence,
            "citations_valid": True,
            "final_status": "completed",
            "trace": initial_state["trace"]
            + [
                {"node": "execute_regulation_qa"},
                {"node": "verify", "validation_action": "pass"},
                {"node": "finish", "final_status": "completed"},
            ],
        }


class RecordingEmitter:
    def __init__(self) -> None:
        self.states: list[dict] = []

    def observe_state(self, state: dict) -> None:
        self.states.append(dict(state))


def test_real_runner_observes_each_graph_value_before_returning_final_state() -> None:
    graphs: list[FakeGraph] = []

    def graph_factory(*_args, **_kwargs):
        graph = FakeGraph()
        graphs.append(graph)
        return graph

    runner = RealAgentRunner(
        llm=FakeLLM(),
        tools=object(),
        graph_factory=graph_factory,
    )
    emitter = RecordingEmitter()
    request = ChatRequest(
        request_id="req-observe-values",
        mode="regulation_qa",
        query="管理员身份鉴别有什么要求？",
    )

    result = asyncio.run(runner.run_streaming(request, emitter))

    assert graphs[0].stream_calls[0]["stream_mode"] == "values"
    assert len(emitter.states) == 3
    assert emitter.states[1]["evidence"][0]["source_id"] == "GBT-22239"
    assert emitter.states[1].get("final_status") is None
    assert result["final_status"] == "completed"
