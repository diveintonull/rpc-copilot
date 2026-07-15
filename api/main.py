"""FastAPI application for streaming observable Agent results."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from api.events import SSEEvent, encode_sse, make_event
from api.task_manager import DuplicateTaskError, TaskManager


DEFAULT_TEXT_CHUNK_SIZE = 24
WEB_DIRECTORY = Path(__file__).resolve().parents[1] / "web"
WorkMode = Literal[
    "regulation_qa",
    "clause_comparison",
    "gap_analysis",
]
REFERENCE_FIELDS = {
    "parent_id",
    "source_id",
    "version",
    "section_number",
    "text",
    "score",
}


class ChatRequest(BaseModel):
    """Validated input for one streamed Agent request."""

    request_id: str | None = None
    mode: WorkMode = "regulation_qa"
    query: str = Field(min_length=1)
    control_text: str = ""

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("request_id must not be blank")
        return value

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


AgentRunner = Callable[[ChatRequest], Awaitable[Mapping[str, Any]]]


async def _unconfigured_runner(_request: ChatRequest) -> Mapping[str, Any]:
    raise RuntimeError("agent runner is not configured")


def _text_chunks(text: str, chunk_size: int) -> list[str]:
    return [
        text[index : index + chunk_size]
        for index in range(0, len(text), chunk_size)
    ]


def _safe_trace(event: Any) -> dict[str, Any] | None:
    if not isinstance(event, Mapping):
        return None

    safe = {}
    for key in ("node", "tool", "duration_ms", "status"):
        value = event.get(key)
        if key == "duration_ms":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                safe[key] = value
        elif isinstance(value, str):
            safe[key] = value
    if "status" not in safe:
        final_status = event.get("final_status")
        if isinstance(final_status, str):
            safe["status"] = final_status
    return safe or None


def _reference_payload(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, Mapping) or not item.get("parent_id"):
        return None
    return {
        key: item[key]
        for key in REFERENCE_FIELDS
        if key in item
    }


def _references(evidence: Any) -> list[dict[str, Any]]:
    if not isinstance(evidence, list):
        return []

    references = []
    seen = set()
    for item in evidence:
        candidates = [item]
        if isinstance(item, Mapping) and isinstance(item.get("evidence"), list):
            candidates = item["evidence"]
        for candidate in candidates:
            payload = _reference_payload(candidate)
            if payload is None:
                continue
            parent_id = payload["parent_id"]
            if parent_id not in seen:
                seen.add(parent_id)
                references.append(payload)
    return references


async def _produce_events(
    request: ChatRequest,
    runner: AgentRunner,
    queue: asyncio.Queue[SSEEvent],
    text_chunk_size: int,
) -> None:
    request_id = request.request_id
    assert request_id is not None
    terminal_sent = False

    await queue.put(
        make_event("status", request_id, {"status": "running"})
    )
    try:
        state = await runner(request)
        if not isinstance(state, Mapping):
            raise TypeError("agent runner must return a state object")

        answer = state.get("answer", "")
        if isinstance(answer, str):
            for chunk in _text_chunks(answer, text_chunk_size):
                await queue.put(
                    make_event("text", request_id, {"delta": chunk})
                )

        for reference in _references(state.get("evidence", [])):
            await queue.put(make_event("reference", request_id, reference))

        recommendations = state.get("recommendations", [])
        if isinstance(recommendations, list):
            for recommendation in recommendations:
                if isinstance(recommendation, Mapping):
                    payload = dict(recommendation)
                elif isinstance(recommendation, str):
                    payload = {"text": recommendation}
                else:
                    continue
                await queue.put(
                    make_event("recommendation", request_id, payload)
                )

        trace = state.get("trace", [])
        if isinstance(trace, list):
            for raw_event in trace:
                safe_event = _safe_trace(raw_event)
                if safe_event is not None:
                    await queue.put(
                        make_event("trace", request_id, safe_event)
                    )

        final_status = state.get("final_status", "completed")
        if final_status in {"failed", "cancelled"}:
            code = (
                "request_cancelled"
                if final_status == "cancelled"
                else "agent_failed"
            )
            await queue.put(
                make_event(
                    "error",
                    request_id,
                    {
                        "status": final_status,
                        "code": code,
                        "message": "agent execution did not complete",
                    },
                )
            )
        else:
            await queue.put(
                make_event(
                    "done",
                    request_id,
                    {"status": final_status},
                )
            )
        terminal_sent = True
    except asyncio.CancelledError:
        await queue.put(
            make_event(
                "error",
                request_id,
                {
                    "status": "cancelled",
                    "code": "request_cancelled",
                    "message": "request cancelled",
                },
            )
        )
        terminal_sent = True
    except Exception:
        await queue.put(
            make_event(
                "error",
                request_id,
                {
                    "status": "failed",
                    "code": "agent_error",
                    "message": "agent execution failed",
                },
            )
        )
        terminal_sent = True
    finally:
        if not terminal_sent:
            await queue.put(
                make_event(
                    "error",
                    request_id,
                    {
                        "status": "failed",
                        "code": "stream_incomplete",
                        "message": "stream ended without a terminal event",
                    },
                )
            )


async def _event_stream(
    request_id: str,
    queue: asyncio.Queue[SSEEvent],
    task: asyncio.Task[Any],
    task_manager: TaskManager,
) -> AsyncIterator[str]:
    terminal_received = False
    try:
        while True:
            event = await queue.get()
            yield encode_sse(event)
            if event["type"] in {"done", "error"}:
                terminal_received = True
                break
    finally:
        if terminal_received:
            try:
                await task
            except asyncio.CancelledError:
                pass
        elif not task.done():
            await task_manager.stop(request_id)


def create_app(
    *,
    agent_runner: AgentRunner | None = None,
    task_manager: TaskManager | None = None,
    text_chunk_size: int = DEFAULT_TEXT_CHUNK_SIZE,
) -> FastAPI:
    """Create an app with explicit runner and task lifecycle dependencies."""
    if text_chunk_size < 1:
        raise ValueError("text_chunk_size must be positive")

    manager = task_manager or TaskManager()
    runner = agent_runner or _unconfigured_runner

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await manager.cancel_all()

    application = FastAPI(title="GRC Copilot API", lifespan=lifespan)
    application.state.task_manager = manager
    application.state.agent_runner = runner

    @application.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "active_tasks": manager.active_count}

    @application.post("/chat")
    async def chat(request: ChatRequest) -> StreamingResponse:
        request_id = request.request_id or str(uuid4())
        bound_request = request.model_copy(
            update={"request_id": request_id}
        )
        queue: asyncio.Queue[SSEEvent] = asyncio.Queue()

        try:
            task = await manager.start(
                request_id,
                _produce_events(
                    bound_request,
                    runner,
                    queue,
                    text_chunk_size,
                ),
            )
        except DuplicateTaskError as exc:
            raise HTTPException(
                status_code=409,
                detail="task already active",
            ) from exc

        await asyncio.sleep(0)
        return StreamingResponse(
            _event_stream(request_id, queue, task, manager),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            },
        )

    @application.post("/tasks/{request_id}/stop")
    async def stop(request_id: str) -> dict[str, str]:
        if not await manager.stop(request_id):
            raise HTTPException(status_code=404, detail="task not found")
        return {"request_id": request_id, "status": "cancelled"}

    application.mount(
        "/",
        StaticFiles(directory=WEB_DIRECTORY, html=True),
        name="web",
    )

    return application


app = create_app()
