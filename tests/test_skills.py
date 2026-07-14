"""Contract tests for progressive Agent Skill discovery and loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.nodes import load_skill_node, match_skill_node
from agent.skills import (
    SkillFormatError,
    SkillPathError,
    UnknownSkillError,
    discover_skills,
    load_skill,
    match_skill,
)


class SpyTokenizer:
    """Make Token accounting deterministic and observable in tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, text: str) -> list[str]:
        self.calls.append(text)
        return text.split()


def write_skill(
    root: Path,
    name: str,
    *,
    description: str,
    body: str,
    resources: list[str] | None = None,
    extra_frontmatter: str = "",
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    resource_lines = ""
    if resources is not None:
        resource_lines = "resources:\n" + "".join(
            f"  - {resource}\n" for resource in resources
        )
    path = skill_dir / "SKILL.md"
    path.write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"{resource_lines}"
        f"{extra_frontmatter}"
        "---\n"
        f"{body}",
        encoding="utf-8",
    )
    return path


def test_discovery_keeps_body_out_of_catalog_and_tokenizer(tmp_path: Path) -> None:
    secret_body = "FULL PRIVATE WORKFLOW BODY"
    write_skill(
        tmp_path,
        "regulation-qa",
        description="Answer regulation questions",
        body=secret_body,
    )
    tokenizer = SpyTokenizer()

    catalog = discover_skills(tmp_path, tokenizer=tokenizer)

    entry = catalog.entries["regulation-qa"]
    assert entry.name == "regulation-qa"
    assert entry.description == "Answer regulation questions"
    assert entry.resources == ()
    assert not hasattr(entry, "body")
    assert all(secret_body not in text for text in tokenizer.calls)
    assert catalog.catalog_tokens == sum(
        len(text.split()) for text in tokenizer.calls
    )


def test_frontmatter_rejects_unknown_keys(tmp_path: Path) -> None:
    write_skill(
        tmp_path,
        "regulation-qa",
        description="Answer regulation questions",
        body="workflow",
        extra_frontmatter="priority: high\n",
    )

    with pytest.raises(SkillFormatError, match="frontmatter keys"):
        discover_skills(tmp_path)


def test_match_skill_returns_at_most_one_canonical_name(tmp_path: Path) -> None:
    for name, description in (
        ("regulation-qa", "Answer regulation questions"),
        ("clause-comparison", "Compare two clauses"),
        ("gap-analysis", "Analyse control gaps"),
    ):
        write_skill(
            tmp_path,
            name,
            description=description,
            body=f"{name} workflow",
        )
    catalog = discover_skills(tmp_path)

    assert match_skill("regulation_qa", catalog) == "regulation-qa"
    assert match_skill("clause_comparison", catalog) == "clause-comparison"
    assert match_skill("gap_analysis", catalog) == "gap-analysis"
    assert match_skill("unsupported", catalog) is None


def test_unknown_skill_is_rejected(tmp_path: Path) -> None:
    catalog = discover_skills(tmp_path)

    with pytest.raises(UnknownSkillError, match="unknown skill"):
        load_skill("missing-skill", catalog)


def test_body_and_resources_load_only_after_selection(tmp_path: Path) -> None:
    body = "step one then step two"
    resource_text = "fixed output template"
    write_skill(
        tmp_path,
        "regulation-qa",
        description="Answer regulation questions",
        body=body,
        resources=["resources/template.md"],
    )
    resource = tmp_path / "regulation-qa" / "resources" / "template.md"
    resource.parent.mkdir()
    resource.write_text(resource_text, encoding="utf-8")
    tokenizer = SpyTokenizer()
    catalog = discover_skills(tmp_path, tokenizer=tokenizer)
    discovery_calls = list(tokenizer.calls)

    loaded = load_skill("regulation-qa", catalog)

    assert tokenizer.calls[: len(discovery_calls)] == discovery_calls
    assert body not in discovery_calls
    assert resource_text not in discovery_calls
    assert loaded.name == "regulation-qa"
    assert loaded.text == body
    assert loaded.resources == {"resources/template.md": resource_text}
    assert loaded.token_usage == {
        "catalog": catalog.entries["regulation-qa"].catalog_tokens,
        "body": len(body.split()),
        "resources": len(resource_text.split()),
        "total": (
            catalog.entries["regulation-qa"].catalog_tokens
            + len(body.split())
            + len(resource_text.split())
        ),
    }


@pytest.mark.parametrize(
    "resource_path",
    ["../secret.md", "resources/../../secret.md"],
)
def test_resource_cannot_escape_skill_directory(
    tmp_path: Path,
    resource_path: str,
) -> None:
    write_skill(
        tmp_path,
        "regulation-qa",
        description="Answer regulation questions",
        body="workflow",
        resources=[resource_path],
    )
    (tmp_path / "secret.md").write_text("secret", encoding="utf-8")
    catalog = discover_skills(tmp_path)

    with pytest.raises(SkillPathError, match="outside skill directory"):
        load_skill("regulation-qa", catalog)


def test_declared_resource_must_exist(tmp_path: Path) -> None:
    write_skill(
        tmp_path,
        "regulation-qa",
        description="Answer regulation questions",
        body="workflow",
        resources=["resources/missing.md"],
    )
    catalog = discover_skills(tmp_path)

    with pytest.raises(SkillPathError, match="resource does not exist"):
        load_skill("regulation-qa", catalog)


@pytest.mark.parametrize(
    ("intent", "skill_name"),
    [
        ("regulation_qa", "regulation-qa"),
        ("clause_comparison", "clause-comparison"),
        ("gap_analysis", "gap-analysis"),
    ],
)
def test_skill_nodes_load_each_supported_intent_with_token_trace(
    tmp_path: Path,
    intent: str,
    skill_name: str,
) -> None:
    body = f"{skill_name} runtime instructions"
    write_skill(
        tmp_path,
        skill_name,
        description=f"Instructions for {skill_name}",
        body=body,
    )
    tokenizer = SpyTokenizer()
    catalog = discover_skills(tmp_path, tokenizer=tokenizer)
    state = {
        "intent": intent,
        "trace": [{"node": "route_intent", "intent": intent}],
    }

    match_update = match_skill_node(state, catalog)
    state.update(match_update)
    load_update = load_skill_node(state, catalog)

    assert match_update["active_skill"] == skill_name
    assert match_update["trace"][-1] == {
        "node": "match_skill",
        "intent": intent,
        "matched_skill": skill_name,
        "catalog_tokens": catalog.catalog_tokens,
    }
    assert load_update["skill_text"] == body
    assert load_update["trace"][-1] == {
        "node": "load_skill",
        "matched_skill": skill_name,
        "loaded": True,
        "body_tokens": len(body.split()),
        "resource_tokens": 0,
    }


@pytest.mark.parametrize(
    ("trigger_intent", "skill_name", "non_trigger_intent"),
    [
        ("regulation_qa", "regulation-qa", "clause_comparison"),
        (
            "clause_comparison",
            "clause-comparison",
            "gap_analysis",
        ),
        ("gap_analysis", "gap-analysis", "regulation_qa"),
    ],
)
def test_real_domain_skill_trigger_and_non_trigger_contracts(
    trigger_intent: str,
    skill_name: str,
    non_trigger_intent: str,
) -> None:
    catalog = discover_skills(Path("skills"))

    assert match_skill(trigger_intent, catalog) == skill_name
    assert match_skill(non_trigger_intent, catalog) != skill_name
    description = catalog.entries[skill_name].description.casefold()
    assert "use for" in description
    assert "do not use" in description


@pytest.mark.parametrize(
    ("skill_name", "required_boundary_rules"),
    [
        (
            "regulation-qa",
            ("cite every factual claim", "version conflict", "refuse"),
        ),
        (
            "clause-comparison",
            ("both sides", "left", "right", "refuse"),
        ),
        (
            "gap-analysis",
            (
                "aligned",
                "partial",
                "unknown",
                "human confirmation",
                "do not declare",
            ),
        ),
    ],
)
def test_real_domain_skill_documents_boundary_rules(
    skill_name: str,
    required_boundary_rules: tuple[str, ...],
) -> None:
    catalog = discover_skills(Path("skills"))

    body = load_skill(skill_name, catalog).text.casefold()

    for rule in required_boundary_rules:
        assert rule in body
