"""Composition root that connects FastAPI requests to the real Agent graph."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from typing import Any

from agent.graph import build_graph
from agent.llm import OpenAICompatibleAgentLLM
from agent.tools import LocalToolBackend
from api.main import ChatRequest


GraphFactory = Callable[..., Any]


class RealAgentRunner:
    """Cache one mode-bound graph and invoke it for each API request."""

    def __init__(
        self,
        *,
        llm: Any,
        tools: Any,
        graph_factory: GraphFactory = build_graph,
        validate_entailment: bool = True,
    ) -> None:
        evaluator = (
            getattr(llm, "entails", None) if validate_entailment else None
        )
        self.llm = llm
        self.graphs = {
            mode: graph_factory(
                lambda _query, _control_text, selected=mode: selected,
                tools,
                llm,
                entailment_evaluator=evaluator,
            )
            for mode in (
                "regulation_qa",
                "clause_comparison",
                "gap_analysis",
            )
        }

    async def __call__(self, request: ChatRequest) -> dict:
        """Run the synchronous local retrieval/model graph off the event loop."""
        request_id = request.request_id
        if request_id is None:
            raise ValueError("real runner requires a bound request_id")
        initial_state = {
            "request_id": request_id,
            "query": request.query.strip(),
            "control_text": request.control_text.strip(),
            "tool_calls": [],
            "retry_count": 0,
            "trace": [{"node": "received"}],
        }
        graph = self.graphs[request.mode]
        return await asyncio.to_thread(graph.invoke, initial_state)

    async def run_streaming(self, request: ChatRequest, emitter: Any) -> dict:
        """Invoke the graph while forwarding answer-model token deltas."""
        request_id = request.request_id
        if request_id is None:
            raise ValueError("real runner requires a bound request_id")
        initial_state = {
            "request_id": request_id,
            "query": request.query.strip(),
            "control_text": request.control_text.strip(),
            "tool_calls": [],
            "retry_count": 0,
            "trace": [{"node": "received"}],
        }
        graph = self.graphs[request.mode]

        def invoke() -> dict:
            with self.llm.stream_to(emitter):
                final_state = None
                for state in graph.stream(
                    initial_state,
                    stream_mode="values",
                ):
                    if not isinstance(state, dict):
                        continue
                    emitter.observe_state(state)
                    final_state = state
                if final_state is None:
                    raise RuntimeError("agent graph produced no state")
                return final_state

        return await asyncio.to_thread(invoke)


def build_real_runner(
    *,
    llm: Any | None = None,
    tools: Any | None = None,
    graph_factory: GraphFactory = build_graph,
    validate_entailment: bool | None = None,
) -> RealAgentRunner:
    """Build the production runner from `.env` OpenAI-compatible settings."""
    selected_llm = llm
    if selected_llm is None:
        api_key = os.environ.get("LLM_API_KEY", "").strip()
        model = os.environ.get("LLM_MODEL", "").strip()
        base_url = os.environ.get("LLM_BASE_URL", "").strip() or None
        missing = [
            name
            for name, value in (
                ("LLM_API_KEY", api_key),
                ("LLM_MODEL", model),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "real mode requires " + ", ".join(missing)
            )
        selected_llm = OpenAICompatibleAgentLLM(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    if validate_entailment is None:
        raw = os.environ.get("LLM_VALIDATE_ENTAILMENT", "true")
        normalized = raw.strip().casefold()
        if normalized not in {"true", "false"}:
            raise RuntimeError(
                "LLM_VALIDATE_ENTAILMENT must be true or false"
            )
        validate_entailment = normalized == "true"

    return RealAgentRunner(
        llm=selected_llm,
        tools=tools or LocalToolBackend(),
        graph_factory=graph_factory,
        validate_entailment=validate_entailment,
    )
