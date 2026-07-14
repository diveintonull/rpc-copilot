"""Progressive Skill catalog and loading contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class SkillError(ValueError):
    """Base error for invalid Skill metadata or loading requests."""


class SkillFormatError(SkillError):
    """A SKILL.md frontmatter block does not follow the contract."""


class UnknownSkillError(SkillError):
    """A caller requested a Skill absent from the discovered catalog."""


class SkillPathError(SkillError):
    """A declared Skill resource is missing or escapes its directory."""


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[object]: ...


class FixedTokenizer:
    """Provide deterministic fixed tokenization without model downloads."""

    _TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]|[^\s]")

    def encode(self, text: str) -> list[str]:
        return self._TOKEN.findall(text)


@dataclass(frozen=True)
class SkillEntry:
    name: str
    description: str
    resources: tuple[str, ...]
    directory: Path
    catalog_tokens: int


@dataclass(frozen=True)
class SkillCatalog:
    root: Path
    entries: dict[str, SkillEntry]
    tokenizer: Tokenizer
    catalog_tokens: int


@dataclass(frozen=True)
class LoadedSkill:
    name: str
    description: str
    text: str
    resources: dict[str, str]
    token_usage: dict[str, int]


def _read_frontmatter(path: Path) -> list[str]:
    """Read only the frontmatter lines, stopping before the Skill body."""
    with path.open("r", encoding="utf-8") as handle:
        if handle.readline().rstrip("\r\n") != "---":
            raise SkillFormatError(f"missing frontmatter start: {path}")
        lines = []
        for line in handle:
            stripped = line.rstrip("\r\n")
            if stripped == "---":
                return lines
            lines.append(stripped)
    raise SkillFormatError(f"missing frontmatter end: {path}")


def _read_body(path: Path) -> str:
    """Read the body only after a Skill has been selected."""
    with path.open("r", encoding="utf-8") as handle:
        delimiter_count = 0
        body_lines = []
        for line in handle:
            if line.rstrip("\r\n") == "---" and delimiter_count < 2:
                delimiter_count += 1
                continue
            if delimiter_count == 2:
                body_lines.append(line)
    if delimiter_count != 2:
        raise SkillFormatError(f"invalid frontmatter delimiters: {path}")
    return "".join(body_lines)


def _parse_frontmatter(lines: list[str]) -> dict[str, object]:
    """Parse the small scalar-and-list YAML subset used by Skill metadata."""
    payload: dict[str, object] = {}
    active_list: list[str] | None = None

    for line in lines:
        if line.startswith("  - "):
            if active_list is None:
                raise SkillFormatError("list item without a list key")
            value = line[4:].strip()
            if not value:
                raise SkillFormatError("empty list item")
            active_list.append(value)
            continue

        if ":" not in line or line.startswith((" ", "\t")):
            raise SkillFormatError(f"invalid frontmatter line: {line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in payload:
            raise SkillFormatError(f"invalid or duplicate key: {key}")
        if value:
            payload[key] = value
            active_list = None
        else:
            active_list = []
            payload[key] = active_list

    return payload


def discover_skills(
    path: Path,
    *,
    tokenizer: Tokenizer | None = None,
) -> SkillCatalog:
    """Return the lightweight metadata catalog under one Skill root."""
    selected_tokenizer = tokenizer or FixedTokenizer()
    root = Path(path).resolve()

    entries: dict[str, SkillEntry] = {}
    total_tokens = 0

    for skill_file in sorted(root.glob("*/SKILL.md")):
        lines = _read_frontmatter(skill_file)
        metadata = _parse_frontmatter(lines)
        allowed_keys = {"name", "description", "resources"}
        unknown_keys = set(metadata) - allowed_keys

        if unknown_keys:
            raise SkillFormatError(f"invalid frontmatter keys: {sorted(unknown_keys)}")

        name = metadata.get("name")
        description = metadata.get("description")
        resources = metadata.get("resources", [])

        if not isinstance(name, str) or not name.strip():
            raise SkillFormatError("skill name must be a non-empty string")

        if not isinstance(description, str) or not description.strip():
            raise SkillFormatError("skill description must be a non-empty string")

        if not isinstance(resources, list) or not all(
            isinstance(resource, str) and bool(resource.strip())
            for resource in resources
        ):
            raise SkillFormatError("skill resources must be a list of non-empty strings")

        directory_name = skill_file.parent.name

        if name != directory_name:
            raise SkillFormatError(
                f"skill name does not match directory: "
                f"{name!r} != {directory_name!r}"
            )

        if name in entries:
            raise SkillFormatError(f"duplicate skill name: {name}")

        catalog_text = (
            f"name: {name}\n"
            f"description: {description}\n"
            f"resources: {', '.join(resources)}"
        )
        catalog_tokens = len(selected_tokenizer.encode(catalog_text))

        entries[name] = SkillEntry(
            name=name,
            description=description,
            resources=tuple(resources),
            directory=skill_file.parent.resolve(),
            catalog_tokens=catalog_tokens,
        )
        total_tokens += catalog_tokens

    return SkillCatalog(
        root=root,
        entries=entries,
        tokenizer=selected_tokenizer,
        catalog_tokens=total_tokens,
    )


def match_skill(intent: str, catalog: SkillCatalog) -> str | None:
    """Return the one canonical Skill name matching an intent."""
    intent_to_skill = {
        "regulation_qa": "regulation-qa",
        "clause_comparison": "clause-comparison",
        "gap_analysis": "gap-analysis",
    }

    skill_name = intent_to_skill.get(intent)

    if skill_name is None:
        return None

    if skill_name not in catalog.entries:
        return None

    return skill_name


def load_skill(name: str, catalog: SkillCatalog) -> LoadedSkill:
    """Load one selected Skill body and its in-directory resources."""
    entry = catalog.entries.get(name)

    if entry is None:
        raise UnknownSkillError(f"unknown skill: {name}")

    skill_file = entry.directory / "SKILL.md"
    body = _read_body(skill_file)
    body_tokens = len(catalog.tokenizer.encode(body))

    loaded_resources: dict[str, str] = {}
    resource_tokens = 0
    skill_directory = entry.directory.resolve()

    for resource_name in entry.resources:
        resource_path = (skill_directory / resource_name).resolve()

        if not resource_path.is_relative_to(skill_directory):
            raise SkillPathError(f"resource outside skill directory: {resource_name}")

        if not resource_path.is_file():
            raise SkillPathError(f"resource does not exist: {resource_name}")

        resource_text = resource_path.read_text(encoding="utf-8")
        loaded_resources[resource_name] = resource_text
        resource_tokens += len(catalog.tokenizer.encode(resource_text))

    token_usage = {
        "catalog": entry.catalog_tokens,
        "body": body_tokens,
        "resources": resource_tokens,
        "total": (entry.catalog_tokens + body_tokens + resource_tokens),
    }

    return LoadedSkill(
        name=entry.name,
        description=entry.description,
        text=body,
        resources=loaded_resources,
        token_usage=token_usage,
    )
