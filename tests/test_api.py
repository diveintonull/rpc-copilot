"""Contract tests for streaming API events and task cancellation."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from api.events import EVENT_TYPES, TERMINAL_EVENT_TYPES, encode_sse
from api.main import create_app
from api.task_manager import DuplicateTaskError, TaskManager


EXPECTED_EVENT_TYPES = {
    "status",
    "text",
    "reference",
    "recommendation",
    "trace",
    "done",
    "error",
}


def parse_sse(body: str) -> list[dict]:
    """Parse the JSON data field from the small SSE format used by the API."""
    events = []
    for block in body.strip().split("\n\n"):
        data_line = next(
            line for line in block.splitlines() if line.startswith("data: ")
        )
        events.append(json.loads(data_line.removeprefix("data: ")))
    return events


def successful_state(request) -> dict:
    return {
        "request_id": request.request_id,
        "query": request.query,
        "answer": "管理员账户应采用组合身份鉴别，并定期检查登录控制。",
        "evidence": [
            {
                "parent_id": "GBT-22239@2019#8.1.4.1",
                "source_id": "GBT-22239",
                "version": "2019",
                "section_number": "8.1.4.1",
                "text": "应采用两种或两种以上组合的鉴别技术。",
                "score": 0.91,
            }
        ],
        "trace": [
            {
                "node": "execute_regulation_qa",
                "tool": "search_regulation",
                "duration_ms": 12.5,
                "status": "completed",
                "reasoning": "SECRET CHAIN OF THOUGHT",
                "skill_text": "SECRET PROMPT",
                "result_count": 1,
            }
        ],
        "citations_valid": True,
        "final_status": "completed",
    }


def test_event_contract_allows_only_the_seven_stable_types() -> None:
    assert EVENT_TYPES == EXPECTED_EVENT_TYPES
    assert TERMINAL_EVENT_TYPES == {"done", "error"}

    payload = {
        "type": "status",
        "request_id": "req-event-contract",
        "data": {"status": "running"},
    }
    encoded = encode_sse(payload)

    assert encoded.startswith("event: status\n")
    assert encoded.endswith("\n\n")
    assert parse_sse(encoded) == [payload]

    with pytest.raises(ValueError, match="unknown SSE event type"):
        encode_sse(
            {
                "type": "thought",
                "request_id": "req-event-contract",
                "data": {},
            }
        )


def test_chat_streams_text_reference_safe_trace_and_terminal_done() -> None:
    async def runner(request) -> dict:
        await asyncio.sleep(0)
        return successful_state(request)

    app = create_app(agent_runner=runner, text_chunk_size=8)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "request_id": "req-success",
                "query": "管理员身份鉴别有什么要求？",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-request-id"] == "req-success"

    events = parse_sse(response.text)
    event_types = [event["type"] for event in events]
    assert set(event_types) <= EXPECTED_EVENT_TYPES
    assert event_types[0] == "status"
    assert event_types[-1] == "done"
    assert event_types.count("done") == 1
    assert "error" not in event_types
    assert event_types.count("text") > 1
    assert "".join(
        event["data"]["delta"]
        for event in events
        if event["type"] == "text"
    ) == successful_state(
        type("Request", (), {"request_id": "req-success", "query": "q"})()
    )["answer"]

    reference = next(
        event for event in events if event["type"] == "reference"
    )
    assert reference["data"]["parent_id"] == (
        "GBT-22239@2019#8.1.4.1"
    )
    assert reference["data"]["version"] == "2019"

    trace = next(event for event in events if event["type"] == "trace")
    assert trace["data"] == {
        "node": "execute_regulation_qa",
        "tool": "search_regulation",
        "duration_ms": 12.5,
        "status": "completed",
    }
    assert "SECRET" not in response.text
    assert app.state.task_manager.active_count == 0


def test_trace_exposes_safe_citation_failure_summary_without_claim_text() -> None:
    async def runner(request) -> dict:
        state = successful_state(request)
        state["trace"] = [
            {
                "node": "verify",
                "citations_valid": False,
                "validation_action": "refuse",
                "citation_failures": [
                    {
                        "code": "uncited_claim",
                        "claim": "PRIVATE CLAIM ONE",
                        "citation": None,
                    },
                    {
                        "code": "unsupported_claim",
                        "claim": "PRIVATE CLAIM TWO",
                        "citation": 1,
                    },
                ],
            }
        ]
        return state

    app = create_app(agent_runner=runner, text_chunk_delay=0)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"request_id": "req-safe-failures", "query": "test"},
        )

    trace = next(
        event for event in parse_sse(response.text) if event["type"] == "trace"
    )
    assert trace["data"] == {
        "node": "verify",
        "validation_action": "refuse",
        "failure_count": 2,
        "failure_codes": ["uncited_claim", "unsupported_claim"],
    }
    assert "PRIVATE CLAIM" not in response.text


def test_chat_forwards_runner_deltas_without_replaying_the_final_answer() -> None:
    class StreamingRunner:
        async def __call__(self, _request) -> dict:
            raise AssertionError("streaming runner should use run_streaming")

        async def run_streaming(self, request, emitter) -> dict:
            emitter.start()
            emitter.emit("第一段")
            await asyncio.sleep(0)
            emitter.emit("第二段")
            state = successful_state(request)
            state["answer"] = "第一段第二段"
            return state

    app = create_app(agent_runner=StreamingRunner(), text_chunk_delay=0)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"request_id": "req-live-stream", "query": "测试流式回答"},
        )

    events = parse_sse(response.text)
    text_events = [event for event in events if event["type"] == "text"]
    assert [event["data"] for event in text_events] == [
        {"delta": "第一段"},
        {"delta": "第二段"},
    ]
    assert events[-1]["type"] == "done"


def test_chat_resets_partial_text_when_the_graph_retries_generation() -> None:
    class RetryingStreamingRunner:
        async def __call__(self, _request) -> dict:
            raise AssertionError("streaming runner should use run_streaming")

        async def run_streaming(self, request, emitter) -> dict:
            emitter.start()
            emitter.emit("需要丢弃的草稿")
            emitter.start()
            emitter.emit("最终回答")
            state = successful_state(request)
            state["answer"] = "最终回答"
            return state

    app = create_app(agent_runner=RetryingStreamingRunner(), text_chunk_delay=0)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"request_id": "req-stream-retry", "query": "测试重试"},
        )

    rendered = ""
    text_events = [
        event for event in parse_sse(response.text) if event["type"] == "text"
    ]
    for event in text_events:
        if event["data"].get("reset") is True:
            rendered = ""
        rendered += event["data"]["delta"]
    assert rendered == "最终回答"
    assert any(event["data"].get("reset") is True for event in text_events)


def test_intermediate_state_streams_reference_and_trace_before_done() -> None:
    class ObservingRunner:
        async def __call__(self, _request) -> dict:
            raise AssertionError("streaming runner should use run_streaming")

        async def run_streaming(self, request, emitter) -> dict:
            emitter.start()
            emitter.emit("回答[1]")
            intermediate = successful_state(request)
            intermediate["answer"] = "回答[1]"
            intermediate["trace"] = [
                {"node": "execute_regulation_qa", "tool": "search_regulation"}
            ]
            intermediate.pop("final_status")
            emitter.observe_state(intermediate)
            await asyncio.sleep(0)
            final = dict(intermediate)
            final["final_status"] = "completed"
            final["trace"] = intermediate["trace"] + [
                {"node": "verify", "validation_action": "pass"},
                {"node": "finish", "final_status": "completed"},
            ]
            return final

    app = create_app(agent_runner=ObservingRunner(), text_chunk_delay=0)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"request_id": "req-live-state", "query": "测试中间状态"},
        )

    events = parse_sse(response.text)
    event_types = [event["type"] for event in events]
    assert event_types.count("reference") == 1
    assert event_types.index("reference") < event_types.index("done")
    trace_nodes = [
        event["data"]["node"] for event in events if event["type"] == "trace"
    ]
    assert trace_nodes == ["execute_regulation_qa", "verify", "finish"]


def test_reference_stream_resets_when_retry_changes_evidence() -> None:
    first_reference = {
        "parent_id": "SOURCE@2024#1",
        "source_id": "SOURCE",
        "version": "2024",
        "section_number": "1",
        "text": "first",
    }
    second_reference = {
        "parent_id": "SOURCE@2025#2",
        "source_id": "SOURCE",
        "version": "2025",
        "section_number": "2",
        "text": "second",
    }

    class ChangingEvidenceRunner:
        async def __call__(self, _request) -> dict:
            raise AssertionError("streaming runner should use run_streaming")

        async def run_streaming(self, request, emitter) -> dict:
            base = {
                "request_id": request.request_id,
                "answer": "最终回答[1]",
                "trace": [],
            }
            emitter.observe_state({**base, "evidence": [first_reference]})
            emitter.observe_state({**base, "evidence": []})
            emitter.observe_state({**base, "evidence": [second_reference]})
            await asyncio.sleep(0)
            return {
                **base,
                "evidence": [second_reference],
                "final_status": "completed",
            }

    app = create_app(agent_runner=ChangingEvidenceRunner(), text_chunk_delay=0)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"request_id": "req-reference-reset", "query": "测试证据重置"},
        )

    reference_data = [
        event["data"]
        for event in parse_sse(response.text)
        if event["type"] == "reference"
    ]
    assert reference_data[0]["parent_id"] == "SOURCE@2024#1"
    assert reference_data[1]["reset"] is True
    assert reference_data[2]["parent_id"] == "SOURCE@2025#2"
    assert reference_data[0]["generation"] < reference_data[2]["generation"]


@pytest.mark.parametrize(
    ("runner", "expected_code"),
    [
        (
            lambda _request: asyncio.sleep(
                0,
                result={
                    "answer": "",
                    "evidence": [],
                    "trace": [],
                    "final_status": "failed",
                },
            ),
            "agent_failed",
        ),
        (
            lambda _request: _raise_after_yield(
                RuntimeError("PRIVATE INTERNAL FAILURE")
            ),
            "agent_error",
        ),
    ],
)
def test_chat_failure_always_ends_with_one_safe_error(
    runner,
    expected_code: str,
) -> None:
    app = create_app(agent_runner=runner)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"request_id": "req-error", "query": "test failure"},
        )

    events = parse_sse(response.text)
    assert events[-1]["type"] == "error"
    assert [event["type"] for event in events].count("error") == 1
    assert events[-1]["data"]["code"] == expected_code
    assert "PRIVATE INTERNAL FAILURE" not in response.text
    assert app.state.task_manager.active_count == 0


async def _raise_after_yield(error: Exception):
    await asyncio.sleep(0)
    raise error


def test_stop_cancels_registered_coroutine_and_leaves_no_zombie_task() -> None:
    async def scenario() -> None:
        manager = TaskManager()
        app = create_app(task_manager=manager)
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def long_running() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        await manager.start("req-stop", long_running())
        await started.wait()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post("/tasks/req-stop/stop")

        assert response.status_code == 200
        assert response.json() == {
            "request_id": "req-stop",
            "status": "cancelled",
        }
        await asyncio.wait_for(cleaned.wait(), timeout=1)
        assert manager.active_count == 0
        assert not await manager.contains("req-stop")

    asyncio.run(scenario())


def test_stop_unknown_request_returns_404() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.post("/tasks/does-not-exist/stop")

    assert response.status_code == 404
    assert response.json()["detail"] == "task not found"


def test_task_manager_rejects_duplicate_active_request_id() -> None:
    async def scenario() -> None:
        manager = TaskManager()
        blocker = asyncio.Event()

        async def wait_forever() -> None:
            await blocker.wait()

        await manager.start("req-duplicate", wait_forever())
        duplicate = wait_forever()
        try:
            with pytest.raises(DuplicateTaskError):
                await manager.start("req-duplicate", duplicate)
        finally:
            duplicate.close()
            assert await manager.stop("req-duplicate") is True

    asyncio.run(scenario())


def test_health_reports_active_task_count() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/health")
        readiness = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "active_tasks": 0}
    assert readiness.status_code == 200
    assert readiness.json() == {"status": "ready"}


def test_chat_rejects_work_while_runtime_dependency_is_unready() -> None:
    runner_called = False

    async def runner(_request) -> dict:
        nonlocal runner_called
        runner_called = True
        return {"answer": "must not run"}

    async def unready() -> bool:
        return False

    app = create_app(agent_runner=runner, readiness_probe=unready)

    with TestClient(app) as client:
        readiness = client.get("/ready")
        response = client.post("/chat", json={"query": "test"})

    assert readiness.status_code == 503
    assert readiness.json()["detail"] == "service dependency is not ready"
    assert response.status_code == 503
    assert runner_called is False
    assert app.state.task_manager.active_count == 0


def test_chat_rejects_blank_query_before_starting_task() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"request_id": "req-blank", "query": "   "},
        )

    assert response.status_code == 422
    assert app.state.task_manager.active_count == 0


def test_chat_passes_selected_work_mode_and_control_text_to_runner() -> None:
    received = []

    async def runner(request) -> dict:
        received.append(
            {
                "mode": request.mode,
                "query": request.query,
                "control_text": request.control_text,
            }
        )
        return {
            "answer": "需要人工确认差距分析结果。",
            "evidence": [],
            "trace": [],
            "final_status": "refused",
        }

    app = create_app(agent_runner=runner)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "request_id": "req-gap-mode",
                "mode": "gap_analysis",
                "query": "检查身份鉴别控制差距",
                "control_text": "管理员当前只使用密码登录。",
            },
        )

    assert response.status_code == 200
    assert received == [
        {
            "mode": "gap_analysis",
            "query": "检查身份鉴别控制差距",
            "control_text": "管理员当前只使用密码登录。",
        }
    ]
    assert parse_sse(response.text)[-1]["data"]["status"] == "refused"


def test_frontend_static_assets_are_served_with_required_workspaces() -> None:
    app = create_app()

    with TestClient(app) as client:
        page = client.get("/")
        javascript = client.get("/app.js")
        stylesheet = client.get("/styles.css")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert javascript.status_code == 200
    assert javascript.headers["content-type"].startswith(
        ("text/javascript", "application/javascript")
    )
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")

    html = page.text
    for mode in (
        "regulation_qa",
        "clause_comparison",
        "gap_analysis",
    ):
        assert f'data-mode="{mode}"' in html
    for element_id in (
        "chat-timeline",
        "evidence-list",
        "trace-list",
        "stop-button",
    ):
        assert f'id="{element_id}"' in html

    assert "SSEFrameParser" in javascript.text
    assert "StreamSession" in javascript.text
    assert "runContractSelfTests" in javascript.text
    assert "renderMarkdownInto" in javascript.text
    assert "appendInlineMarkdown" in javascript.text
    assert "data.reset === true" in javascript.text
    assert "resetReferences" in javascript.text
    assert "referenceGeneration" in javascript.text
    assert "ANSWER_RENDER_INTERVAL_MS" in javascript.text
    assert "scheduleAnswerRender" in javascript.text
    assert "if (firstLine)" in javascript.text
    assert "event.isComposing" in javascript.text
    assert "event.shiftKey" in javascript.text
    assert 'this.elements.queryInput.value = "";' in javascript.text
    assert 'this.elements.controlInput.value = "";' in javascript.text
    assert "validation_action" in javascript.text
    assert "failure_codes" in javascript.text
    assert 'const TRACE_FIELDS = [' in (
        javascript.text
    )
