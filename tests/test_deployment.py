import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import ChatRequest


ROOT = Path(__file__).resolve().parents[1]


def _service_names(compose_text: str) -> set[str]:
    names = set()
    in_services = False
    for line in compose_text.splitlines():
        if line == "services:":
            in_services = True
            continue
        if in_services and line and not line.startswith(" "):
            break
        if in_services and line.startswith("  ") and not line.startswith("    "):
            names.add(line.strip().removesuffix(":"))
    return names


def test_compose_has_only_app_and_qdrant_with_readiness_and_volumes() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert _service_names(compose) == {"app", "qdrant"}
    assert "qdrant/qdrant:v1.18.2" in compose
    assert "condition: service_healthy" in compose
    assert "http://qdrant:6333" in compose
    assert "/readyz" in compose
    assert "http://127.0.0.1:8000/ready" in compose
    assert "qdrant_storage:/qdrant/storage" in compose
    assert "model_cache:/home/app/.cache" in compose


def test_dockerfile_uses_locked_runtime_and_non_root_user() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "ghcr.io/astral-sh/uv:0.8.22" in dockerfile
    assert "python:3.13-slim" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    assert "--only-group container" in dockerfile
    assert "container = [" in project
    assert "USER app" in dockerfile
    assert 'CMD ["python", "-m", "api.serve"]' in dockerfile


def test_dockerignore_excludes_local_state_and_secrets() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    for required in (".git", ".venv", ".env", "data/*", "docs/", "results/"):
        assert required in ignored
    assert "!data/parsed/*.md" in ignored
    assert "!data/parsed/_parents_store.json" in ignored


def test_deployment_demo_serves_grounded_question_without_external_model() -> None:
    from api.deployment import create_deployment_app

    async def ready() -> bool:
        return True

    app = create_deployment_app(readiness_probe=ready, run_mode="demo")

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "request_id": "compose-demo-qa",
                "mode": "regulation_qa",
                "query": "管理员身份鉴别有哪些要求？",
            },
        )

    assert response.status_code == 200
    assert "event: reference" in response.text
    assert "GBT-22239@2019#8.1.4.1" in response.text
    assert "event: done" in response.text


@pytest.mark.parametrize(
    ("mode", "query", "control_text"),
    [
        ("regulation_qa", "数据出境需要满足什么条件？", ""),
        ("clause_comparison", "比较两部法律的数据出境要求", ""),
        (
            "gap_analysis",
            "检查数据留存控制差距",
            "公司目前保留日志六个月。",
        ),
    ],
)
def test_deployment_demo_refuses_questions_outside_fixture_scope(
    mode: str,
    query: str,
    control_text: str,
) -> None:
    from api.deployment import create_deployment_app

    async def ready() -> bool:
        return True

    app = create_deployment_app(readiness_probe=ready, run_mode="demo")

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "request_id": f"compose-demo-out-of-scope-{mode}",
                "mode": mode,
                "query": query,
                "control_text": control_text,
            },
    )

    assert response.status_code == 200
    body = response.content.decode("utf-8")
    assert '"node":"demo_scope_guard"' in body
    assert '"tool":"fixture_catalog"' in body
    assert "event: reference" not in body
    assert "GBT-22239@2019#8.1.4.1" not in body
    assert '"status":"refused"' in body


def test_deployment_slow_demo_cooperates_with_task_cancellation() -> None:
    from api.deployment import demo_runner

    async def scenario() -> None:
        request = ChatRequest(
            request_id="compose-stop",
            mode="regulation_qa",
            query="慢速法规问答，用于停止验收",
        )
        task = asyncio.create_task(demo_runner(request))
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_deployment_real_mode_uses_injected_real_runner() -> None:
    from api.deployment import create_deployment_app

    received = []

    async def real_runner(request):
        received.append((request.mode, request.query))
        return {
            "answer": f"真实回答：{request.query}[1]",
            "evidence": [
                {
                    "parent_id": "data-security-law@2021#第二十七条",
                    "source_id": "data-security-law",
                    "version": "2021",
                    "section_number": "第二十七条",
                    "text": "应当建立健全全流程数据安全管理制度。",
                    "score": 0.91,
                }
            ],
            "trace": [],
            "final_status": "completed",
        }

    async def ready() -> bool:
        return True

    app = create_deployment_app(
        readiness_probe=ready,
        run_mode="real",
        real_runner=real_runner,
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "mode": "regulation_qa",
                "query": "数据安全管理制度有什么要求？",
            },
        )

    assert response.status_code == 200
    assert received == [
        ("regulation_qa", "数据安全管理制度有什么要求？")
    ]
    assert "真实回答" in response.text
    assert "data-security-law@2021#第二十七条" in response.text


def test_configured_llm_defaults_deployment_to_real_mode(monkeypatch) -> None:
    from api import deployment

    monkeypatch.delenv("APP_RUN_MODE", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "configured")
    monkeypatch.setenv("LLM_MODEL", "configured-model")

    assert deployment.resolve_run_mode() == "real"
