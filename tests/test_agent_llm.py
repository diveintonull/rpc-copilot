"""Contract tests for the OpenAI-compatible Agent LLM adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.llm import OpenAICompatibleAgentLLM


class FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        content = self.responses.pop(0)
        if kwargs.get("stream"):
            return iter(
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=character)
                        )
                    ]
                )
                for character in content
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content)
                )
            ]
        )


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def build_llm(*responses: str) -> tuple[OpenAICompatibleAgentLLM, FakeClient]:
    client = FakeClient(list(responses))
    llm = OpenAICompatibleAgentLLM(
        api_key="test-key",
        base_url="https://llm.example/v1",
        model="test-model",
        client=client,
    )
    return llm, client


def evidence() -> dict:
    return {
        "parent_id": "GBT-22239@2019#8.1.4.1",
        "source_id": "GBT-22239",
        "version": "2019",
        "section_number": "8.1.4.1",
        "text": "<evidence>应采用两种或两种以上组合的鉴别技术。</evidence>",
        "score": 0.94,
    }


def test_answer_regulation_sends_real_evidence_and_returns_model_text() -> None:
    llm, client = build_llm("管理员应采用组合鉴别技术。[1]")

    answer = llm.answer_regulation(
        "管理员身份鉴别有什么要求？",
        [evidence()],
        skill_text="Only answer from retrieved evidence.",
    )

    assert answer == "管理员应采用组合鉴别技术。[1]"
    request = client.completions.requests[0]
    assert request["model"] == "test-model"
    assert request["temperature"] == 0
    prompt = request["messages"][1]["content"]
    system = request["messages"][0]["content"]
    assert "GBT-22239@2019#8.1.4.1" in prompt
    assert "应采用两种或两种以上组合的鉴别技术" in prompt
    assert "管理员身份鉴别有什么要求" in prompt
    assert "one independently supported factual claim per sentence" in prompt
    assert "Omit version and limitation sections" in prompt
    assert "do not attach every retrieved citation" in prompt
    assert "Required output language: Simplified Chinese" in system
    assert system.rfind("Required output language") > system.rfind(
        "workflow_instructions"
    )


def test_answer_regulation_sends_rendered_page_to_vision_model(
    tmp_path: Path,
) -> None:
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (20, 30), "white").save(image_path)
    client = FakeClient(["页面要求启用身份鉴别[1]"])
    llm = OpenAICompatibleAgentLLM(
        api_key="test-key",
        model="text-model",
        vision_model="vision-model",
        vision_image_root=tmp_path,
        client=client,
    )
    page_evidence = {
        **evidence(),
        "parent_id": "GBT-22239@2019#page=12",
        "section_number": "page 12",
        "modality": "image",
        "page_number": 12,
        "image_path": str(image_path),
    }

    answer = llm.answer_regulation("图中的表格有什么要求？", [page_evidence])

    assert answer == "页面要求启用身份鉴别[1]"
    request = client.completions.requests[0]
    assert request["model"] == "vision-model"
    content = request["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "modality=image" in content[0]["text"]
    assert content[-1]["type"] == "image_url"
    assert content[-1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


def test_answer_regulation_forwards_real_model_deltas_when_streaming() -> None:
    class RecordingEmitter:
        def __init__(self) -> None:
            self.starts = 0
            self.deltas: list[str] = []

        def start(self) -> None:
            self.starts += 1

        def emit(self, delta: str) -> None:
            self.deltas.append(delta)

    llm, client = build_llm("管理员应启用多因素鉴别。[1]")
    emitter = RecordingEmitter()

    with llm.stream_to(emitter):
        answer = llm.answer_regulation("管理员身份鉴别有什么要求？", [evidence()])

    assert answer == "管理员应启用多因素鉴别。[1]"
    assert emitter.starts == 1
    assert "".join(emitter.deltas) == answer
    assert client.completions.requests[0]["stream"] is True


def test_structured_model_calls_do_not_leak_into_answer_stream() -> None:
    class RecordingEmitter:
        def __init__(self) -> None:
            self.starts = 0
            self.deltas: list[str] = []

        def start(self) -> None:
            self.starts += 1

        def emit(self, delta: str) -> None:
            self.deltas.append(delta)

    llm, client = build_llm(
        '{"left":{"search_query":"left"},'
        '"right":{"search_query":"right"},"dimensions":["scope"]}'
    )
    emitter = RecordingEmitter()

    with llm.stream_to(emitter):
        llm.plan_comparison("compare the clauses")

    assert emitter.starts == 0
    assert emitter.deltas == []
    assert "stream" not in client.completions.requests[0]


def test_repair_regulation_answer_uses_failures_and_streams_replacement() -> None:
    class RecordingEmitter:
        def __init__(self) -> None:
            self.starts = 0
            self.deltas: list[str] = []

        def start(self) -> None:
            self.starts += 1

        def emit(self, delta: str) -> None:
            self.deltas.append(delta)

    llm, client = build_llm("管理员应采用组合身份鉴别技术。[1]")
    emitter = RecordingEmitter()

    with llm.stream_to(emitter):
        answer = llm.repair_regulation_answer(
            "管理员身份鉴别有什么要求？",
            "版本说明\n未发现版本冲突。[1]",
            [evidence()],
            [
                {
                    "code": "unsupported_claim",
                    "claim": "未发现版本冲突",
                    "citation": 1,
                }
            ],
        )

    assert "".join(emitter.deltas) == answer
    request = client.completions.requests[0]
    assert request["stream"] is True
    prompt = request["messages"][1]["content"]
    assert "unsupported_claim" in prompt
    assert "未发现版本冲突" in prompt
    assert "Delete unsupported meta conclusions" in prompt


def test_chinese_comparison_normalizes_english_structural_labels() -> None:
    llm, client = build_llm(
        "**Left:**\n左侧要求[1]。\n\nRight:\n右侧要求[2]。\n\n"
        "Comparison:\n两侧不同[1][2]。"
    )

    answer = llm.answer_comparison(
        "比较两个中文条款",
        {
            "left": evidence(),
            "right": {
                **evidence(),
                "parent_id": "GBT-35273@2020#7.3",
                "source_id": "GBT-35273",
                "version": "2020",
                "section_number": "7.3",
            },
            "dimensions": ["要求"],
        },
    )

    assert answer.startswith("左侧：\n")
    assert "\n右侧：\n" in answer
    assert "\n比较：\n" in answer
    assert "Left" not in answer
    assert "Required output language: Simplified Chinese" in (
        client.completions.requests[0]["messages"][0]["content"]
    )


def test_plan_comparison_parses_fenced_json_and_normalizes_dimensions() -> None:
    llm, _client = build_llm(
        """```json
        {
          "left": {"source_id": "GBT-22239", "version": "2019", "section_number": "8.1.4.1", "search_query": "身份鉴别"},
          "right": {"source_id": "GBT-35273", "version": "2020", "section_number": "7.3", "search_query": "身份鉴别"},
          "dimensions": ["要求", "适用范围"]
        }
        ```"""
    )

    plan = llm.plan_comparison("比较两个条款")

    assert plan["left"]["source_id"] == "GBT-22239"
    assert plan["right"]["section_number"] == "7.3"
    assert plan["dimensions"] == ["要求", "适用范围"]


def test_extract_controls_and_map_gaps_return_validated_lists() -> None:
    llm, _client = build_llm(
        '{"controls":[{"control":"管理员登录","current_state":"仅使用密码"}]}',
        """{
          "gaps": [{
            "requirement": "采用组合鉴别技术",
            "current_state": "仅使用密码",
            "gap": "partial: 未说明第二种鉴别因素",
            "risk": "medium",
            "recommendation": "核对并启用第二种因素",
            "evidence": [{"source_id":"GBT-22239","version":"2019","section_number":"8.1.4.1"}]
          }]
        }""",
    )

    controls = llm.extract_controls("管理员仅使用密码登录。")
    gaps = llm.map_gaps("检查身份鉴别控制", controls, [evidence()])

    assert controls == [
        {"control": "管理员登录", "current_state": "仅使用密码"}
    ]
    assert gaps[0]["gap"].startswith("partial:")
    assert gaps[0]["evidence"] == [
        {
            "source_id": "GBT-22239",
            "version": "2019",
            "section_number": "8.1.4.1",
        }
    ]


def test_entails_requires_explicit_boolean_json() -> None:
    llm, _client = build_llm('{"supported": true}')

    assert llm.entails("管理员应采用组合鉴别技术", evidence()) is True


def test_short_reasoning_completion_retries_when_content_is_empty() -> None:
    llm, client = build_llm("", '{"supported": true}')

    assert llm.entails("管理员应采用组合鉴别技术", evidence()) is True
    assert [
        request["max_tokens"] for request in client.completions.requests
    ] == [600, 4096]


def test_invalid_structured_response_fails_loudly() -> None:
    llm, _client = build_llm("not json")

    with pytest.raises(ValueError, match="valid JSON"):
        llm.extract_controls("管理员仅使用密码登录。")
