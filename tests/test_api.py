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

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "active_tasks": 0}


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
    assert 'const TRACE_FIELDS = ["node", "tool", "duration_ms", "status"]' in (
        javascript.text
    )
