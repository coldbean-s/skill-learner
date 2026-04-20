from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import yaml


@dataclass
class SkillMeta:
    name: str
    description: str
    version: str = "1.0.0"
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    author: str = "auto-generated"


@dataclass
class SkillPatch:
    description: str | None = None
    content: str | None = None
    tags: list[str] | None = None
    bump_version: bool = True


@dataclass
class Skill:
    meta: SkillMeta
    content: str

    MAX_SIZE = 100_000

    def apply_patch(self, patch: SkillPatch) -> Skill:
        new_meta = SkillMeta(
            name=self.meta.name,
            description=patch.description if patch.description is not None else self.meta.description,
            version=_bump_minor(self.meta.version) if patch.bump_version else self.meta.version,
            category=self.meta.category,
            tags=patch.tags if patch.tags is not None else list(self.meta.tags),
            created_at=self.meta.created_at,
            updated_at=datetime.now(),
            author=self.meta.author,
        )
        new_content = patch.content if patch.content is not None else self.content
        return Skill(meta=new_meta, content=new_content)


class SkillLearnerError(Exception):
    pass


class SkillParseError(SkillLearnerError, ValueError):
    pass


class SkillSizeError(SkillLearnerError, ValueError):
    pass


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)
_REQUIRED_FIELDS = {"name", "description"}


def parse_skill_md(text: str) -> Skill:
    m = _FRONTMATTER_RE.match(text.strip())
    if not m:
        raise SkillParseError("Invalid SKILL.md: missing frontmatter delimiters (---)")

    raw_fm = yaml.safe_load(m.group(1))
    if not isinstance(raw_fm, dict):
        raise SkillParseError("Invalid frontmatter: not a mapping")

    missing = _REQUIRED_FIELDS - set(raw_fm.keys())
    if missing:
        raise SkillParseError(f"Missing required frontmatter fields: {', '.join(sorted(missing))}")

    def _parse_dt(val: Any) -> datetime:
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            return datetime.fromisoformat(val)
        return datetime.now()

    meta = SkillMeta(
        name=str(raw_fm["name"]),
        description=str(raw_fm["description"]),
        version=str(raw_fm.get("version", "1.0.0")),
        category=str(raw_fm.get("category", "general")),
        tags=raw_fm.get("tags") or [],
        created_at=_parse_dt(raw_fm.get("created_at")),
        updated_at=_parse_dt(raw_fm.get("updated_at")),
        author=str(raw_fm.get("author", "auto-generated")),
    )
    return Skill(meta=meta, content=m.group(2).strip())


def serialize_skill_md(skill: Skill) -> str:
    total = len(skill.content)
    if total > Skill.MAX_SIZE:
        raise SkillSizeError(f"Skill content exceeds max size: {total} > {Skill.MAX_SIZE}")

    fm = {
        "name": skill.meta.name,
        "description": skill.meta.description,
        "version": skill.meta.version,
        "category": skill.meta.category,
        "tags": skill.meta.tags,
        "created_at": skill.meta.created_at.isoformat(),
        "updated_at": skill.meta.updated_at.isoformat(),
        "author": skill.meta.author,
    }
    fm_str = yaml.dump(fm, allow_unicode=True, default_flow_style=False, sort_keys=False).strip()
    return f"---\n{fm_str}\n---\n\n{skill.content}\n"


def _bump_minor(version: str) -> str:
    parts = version.split(".")
    if len(parts) >= 2:
        parts[1] = str(int(parts[1]) + 1)
        if len(parts) >= 3:
            parts[2] = "0"
    return ".".join(parts)
