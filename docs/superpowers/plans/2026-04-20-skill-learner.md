# skill-learner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an open-source Python package that gives any AI agent the ability to automatically accumulate, review, and reuse skills across sessions.

**Architecture:** Event-driven engine core (`SkillEngine`) with convenience wrapper (`SkillLoop`). File+SQLite storage, Protocol-based LLM abstraction, multi-platform integrations. Background daemon thread for best-effort skill review.

**Tech Stack:** Python 3.10+, PyYAML, SQLite3, click (CLI), pytest

**Spec:** `docs/superpowers/specs/2026-04-20-skill-learner-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/skill_learner/__init__.py` | Public API: SkillEngine, SkillLoop, config, protocols |
| `src/skill_learner/protocols.py` | LLMProvider, LLMResponse, ToolCall, StorageBackend, EmbeddingProvider protocols |
| `src/skill_learner/skill.py` | Skill, SkillMeta, SkillPatch dataclasses + SKILL.md parse/serialize |
| `src/skill_learner/config.py` | SkillLearnerConfig dataclass + YAML/dict/env loading |
| `src/skill_learner/trigger.py` | ReviewTrigger dual counter |
| `src/skill_learner/prompts.py` | SKILLS_GUIDANCE, REVIEW_PROMPT, tool schemas |
| `src/skill_learner/storage/__init__.py` | Re-export FileStorage, SQLiteStorage |
| `src/skill_learner/storage/filesystem.py` | FileStorage: SKILL.md CRUD + .index.json |
| `src/skill_learner/storage/sqlite.py` | SQLiteStorage: wraps FileStorage + .index.db |
| `src/skill_learner/reviewer.py` | BackgroundReviewer: daemon thread + lock + tool call processing |
| `src/skill_learner/semantic.py` | cosine_similarity, skill_to_text, embedding encode/decode helpers |
| `src/skill_learner/injector.py` | PromptInjector: build_skills_prompt + keyword search + semantic search |
| `src/skill_learner/engine.py` | SkillEngine: event coordinator |
| `src/skill_learner/loop.py` | SkillLoop: wrap() + inject_prompt() convenience |
| `src/skill_learner/integrations/__init__.py` | Re-exports |
| `src/skill_learner/integrations/claude_code.py` | init_claude_code(): hooks + CLAUDE.md |
| `src/skill_learner/integrations/cursor.py` | init_cursor(): .cursorrules |
| `src/skill_learner/integrations/generic_agent.py` | Helper utilities for generic Python agents |
| `src/skill_learner/cli.py` | CLI entry point: init, list, show, stats, review, export, import |
| `tests/helpers.py` | Shared mock classes: MockLLMProvider, MockLLMResponse, MockToolCall |
| `tests/conftest.py` | Shared fixtures: tmp_skills_dir, mock_llm |
| `tests/test_skill.py` | Skill parsing/serialization |
| `tests/test_trigger.py` | ReviewTrigger counter logic |
| `tests/test_storage_fs.py` | FileStorage CRUD |
| `tests/test_storage_sqlite.py` | SQLiteStorage CRUD + stats |
| `tests/test_reviewer.py` | BackgroundReviewer (mocked LLM) |
| `tests/test_semantic.py` | cosine_similarity, encode/decode, skill_to_text |
| `tests/test_injector.py` | PromptInjector output format + semantic search |
| `tests/test_loop.py` | SkillLoop wrap/inject tests |
| `tests/test_engine.py` | Full engine cycle integration |
| `examples/anthropic_agent.py` | Anthropic SDK integration example |
| `examples/openai_agent.py` | OpenAI SDK integration example |
| `examples/custom_agent.py` | Custom agent with manual event API |

---

### Task 1: Project Scaffolding + pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `src/skill_learner/__init__.py`
- Create: `src/skill_learner/storage/__init__.py`
- Create: `src/skill_learner/integrations/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `LICENSE`
- Create: `.gitignore`

- [ ] **Step 1: Initialize git repo**

```bash
cd /d/AI_Tech/skill-learner
git init
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "skill-learner"
version = "0.1.0"
description = "Self-accumulating skill system for AI agents — learn, review, and reuse skills across sessions"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [{ name = "coldbean-s" }]
keywords = ["ai", "agent", "skills", "learning", "llm"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "pyyaml>=6.0",
]

[project.optional-dependencies]
cli = ["click>=8.0"]
dev = [
    "pytest>=7.0",
    "pytest-mock>=3.0",
]
all = ["skill-learner[cli,dev]"]

[project.scripts]
skill-learner = "skill_learner.cli:main"

[project.urls]
Homepage = "https://github.com/coldbean-s/skill-learner"
Repository = "https://github.com/coldbean-s/skill-learner"

[tool.hatch.build.targets.wheel]
packages = ["src/skill_learner"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 3: Create __init__.py files (empty stubs for now)**

`src/skill_learner/__init__.py`:
```python
"""skill-learner: Self-accumulating skill system for AI agents."""

__version__ = "0.1.0"
```

`src/skill_learner/storage/__init__.py`:
```python
```

`src/skill_learner/integrations/__init__.py`:
```python
```

`tests/__init__.py`:
```python
```

- [ ] **Step 4: Create tests/helpers.py with shared mock classes**

`tests/helpers.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class MockLLMResponse:
    content: str = ""
    tool_calls: list[MockToolCall] = field(default_factory=list)


class MockLLMProvider:
    """Mock LLM that returns pre-configured responses."""

    def __init__(self, responses: list[MockLLMResponse] | None = None):
        self._responses = list(responses or [])
        self._call_count = 0
        self.calls: list[dict] = []

    def add_response(self, response: MockLLMResponse) -> None:
        self._responses.append(response)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
    ) -> MockLLMResponse:
        self.calls.append({"messages": messages, "tools": tools, "max_tokens": max_tokens})
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return MockLLMResponse(content="Nothing to save.")
```

- [ ] **Step 5: Create conftest.py with fixtures**

`tests/conftest.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import MockLLMProvider


@pytest.fixture
def tmp_skills_dir(tmp_path: Path) -> Path:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    return skills_dir


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider()
```

- [ ] **Step 5: Create LICENSE (MIT) and .gitignore**

`LICENSE`:
```
MIT License

Copyright (c) 2026 coldbean-s

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

`.gitignore`:
```
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
*.egg
.pytest_cache/
.venv/
venv/
.index.db
.index.json
*.db-journal
```

- [ ] **Step 6: Install in dev mode and verify**

```bash
cd /d/AI_Tech/skill-learner
pip install -e ".[dev,cli]"
pytest --co  # should collect 0 tests, no errors
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: scaffold project structure with pyproject.toml"
```

---

### Task 2: Protocols + Skill Data Model

**Files:**
- Create: `src/skill_learner/protocols.py`
- Create: `src/skill_learner/skill.py`
- Create: `tests/test_skill.py`

- [ ] **Step 1: Write tests for Skill parsing and serialization**

`tests/test_skill.py`:
```python
from __future__ import annotations

from datetime import datetime

import pytest

from skill_learner.skill import Skill, SkillMeta, SkillPatch, parse_skill_md, serialize_skill_md


SAMPLE_SKILL_MD = """---
name: competitor-analysis
description: SaaS competitor analysis report workflow
version: 1.0.0
category: business
tags: [analysis, research]
created_at: '2026-04-20T10:00:00'
updated_at: '2026-04-20T10:00:00'
author: auto-generated
---

# Competitor Analysis

## Workflow
1. Search the web
2. Build report
"""


class TestParseSkillMd:
    def test_parse_valid_skill(self):
        skill = parse_skill_md(SAMPLE_SKILL_MD)
        assert skill.meta.name == "competitor-analysis"
        assert skill.meta.description == "SaaS competitor analysis report workflow"
        assert skill.meta.version == "1.0.0"
        assert skill.meta.category == "business"
        assert skill.meta.tags == ["analysis", "research"]
        assert "# Competitor Analysis" in skill.content

    def test_parse_missing_frontmatter_raises(self):
        with pytest.raises(ValueError, match="frontmatter"):
            parse_skill_md("no frontmatter here")

    def test_parse_missing_required_field_raises(self):
        bad_md = "---\nname: test\n---\ncontent"
        with pytest.raises(ValueError, match="description"):
            parse_skill_md(bad_md)

    def test_parse_empty_tags_defaults_to_list(self):
        md = "---\nname: t\ndescription: d\nversion: 1.0.0\ncategory: c\n---\ncontent"
        skill = parse_skill_md(md)
        assert skill.meta.tags == []


class TestSerializeSkillMd:
    def test_roundtrip(self):
        skill = parse_skill_md(SAMPLE_SKILL_MD)
        serialized = serialize_skill_md(skill)
        reparsed = parse_skill_md(serialized)
        assert reparsed.meta.name == skill.meta.name
        assert reparsed.content.strip() == skill.content.strip()

    def test_serialize_respects_max_size(self):
        meta = SkillMeta(
            name="big",
            description="too big",
            version="1.0.0",
            category="test",
            tags=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        skill = Skill(meta=meta, content="x" * (Skill.MAX_SIZE + 1))
        with pytest.raises(ValueError, match="size"):
            serialize_skill_md(skill)


class TestSkillPatch:
    def test_apply_patch_updates_fields(self):
        skill = parse_skill_md(SAMPLE_SKILL_MD)
        patch = SkillPatch(description="Updated desc", tags=["new-tag"])
        patched = skill.apply_patch(patch)
        assert patched.meta.description == "Updated desc"
        assert patched.meta.tags == ["new-tag"]
        assert patched.meta.version == "1.1.0"  # bump_version=True by default

    def test_apply_patch_no_bump(self):
        skill = parse_skill_md(SAMPLE_SKILL_MD)
        patch = SkillPatch(description="Updated", bump_version=False)
        patched = skill.apply_patch(patch)
        assert patched.meta.version == "1.0.0"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_skill.py -v
```
Expected: FAIL — `skill_learner.skill` module does not exist.

- [ ] **Step 3: Implement protocols.py**

`src/skill_learner/protocols.py`:
```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from skill_learner.skill import Skill, SkillMeta, SkillPatch


@runtime_checkable
class ToolCall(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def arguments(self) -> dict[str, Any]: ...


@runtime_checkable
class LLMResponse(Protocol):
    @property
    def content(self) -> str: ...
    @property
    def tool_calls(self) -> list[ToolCall]: ...


@runtime_checkable
class LLMProvider(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse: ...


class StorageBackend(Protocol):
    def save_skill(self, skill: Skill) -> None: ...
    def load_skill(self, name: str) -> Skill | None: ...
    def list_skills(self) -> list[SkillMeta]: ...
    def update_skill(self, name: str, patch: SkillPatch) -> None: ...
    def delete_skill(self, name: str) -> None: ...
    def record_usage(self, name: str) -> None: ...
    def get_stats(self) -> dict[str, Any]: ...
```

- [ ] **Step 4: Implement skill.py**

`src/skill_learner/skill.py`:
```python
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


class SkillParseError(SkillLearnerError):
    pass


class SkillSizeError(SkillLearnerError):
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_skill.py -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/skill_learner/protocols.py src/skill_learner/skill.py tests/test_skill.py
git commit -m "feat: add Protocol definitions and Skill data model with parse/serialize"
```

---

### Task 3: Config + Prompts

**Files:**
- Create: `src/skill_learner/config.py`
- Create: `src/skill_learner/prompts.py`

- [ ] **Step 1: Implement config.py**

`src/skill_learner/config.py`:
```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SkillLearnerConfig:
    skills_dir: str = "~/.skill_learner/skills"
    nudge_interval: int = 10
    max_skill_size: int = 100_000
    storage_backend: str = "sqlite"
    review_max_tokens: int = 2048
    review_model_hint: str = ""
    review_timeout: int = 30
    categories: list[str] = field(default_factory=lambda: [
        "business", "coding", "research", "ops", "general",
    ])

    @property
    def skills_path(self) -> Path:
        return Path(self.skills_dir).expanduser().resolve()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillLearnerConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SkillLearnerConfig:
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        return cls.from_dict(data)

    @classmethod
    def from_env(cls, prefix: str = "SKILL_LEARNER_") -> SkillLearnerConfig:
        data: dict[str, Any] = {}
        for key, val in os.environ.items():
            if key.startswith(prefix):
                field_name = key[len(prefix):].lower()
                data[field_name] = val
        if "nudge_interval" in data:
            data["nudge_interval"] = int(data["nudge_interval"])
        if "max_skill_size" in data:
            data["max_skill_size"] = int(data["max_skill_size"])
        if "review_max_tokens" in data:
            data["review_max_tokens"] = int(data["review_max_tokens"])
        if "review_timeout" in data:
            data["review_timeout"] = int(data["review_timeout"])
        return cls.from_dict(data)
```

- [ ] **Step 2: Implement prompts.py**

`src/skill_learner/prompts.py`:
```python
SKILLS_GUIDANCE = """## Skill Accumulation

After completing a **complex task** (5+ tool calls), fixing a tricky bug, or discovering a non-obvious workflow, use the save_skill tool to save the approach as a reusable skill.

When using an existing skill and finding it **outdated, incomplete, or wrong**, immediately use update_skill to patch it — don't wait to be told.

Skills that are not maintained become liabilities."""

REVIEW_PROMPT = """Review the conversation above and determine if any reusable skill should be saved or updated.

Focus on:
- Was a non-obvious method used?
- Were there trial-and-error moments, detours, or strategy adjustments based on actual findings?
- Did the user expect or require an approach different from the initial one?

If a relevant skill already exists, **update it** with what you learned.
Otherwise, if the approach is reusable, **create a new skill**.
If there is nothing worth saving, call nothing_to_save() and stop."""

SKILL_MANAGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_skill",
            "description": "Save a new reusable skill. If a skill with this name already exists, it will be updated instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name (lowercase, hyphens, e.g. 'competitor-analysis')",
                    },
                    "category": {
                        "type": "string",
                        "description": "Category (business, coding, research, ops, general)",
                    },
                    "description": {
                        "type": "string",
                        "description": "One-line description of what this skill does",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full skill content in Markdown (prerequisites, workflow, gotchas)",
                    },
                },
                "required": ["name", "category", "description", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_skill",
            "description": "Update an existing skill with new or corrected content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the skill to update",
                    },
                    "patch_content": {
                        "type": "string",
                        "description": "Updated full content (replaces existing content)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Updated description (optional)",
                    },
                },
                "required": ["name", "patch_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_skill",
            "description": "Delete a skill that is obsolete or incorrect.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the skill to delete",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this skill should be deleted",
                    },
                },
                "required": ["name", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nothing_to_save",
            "description": "Signal that there is nothing worth saving from this conversation.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]
```

- [ ] **Step 3: Verify imports work**

```bash
python -c "from skill_learner.config import SkillLearnerConfig; from skill_learner.prompts import SKILLS_GUIDANCE; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/skill_learner/config.py src/skill_learner/prompts.py
git commit -m "feat: add configuration and prompt templates"
```

---

### Task 4: ReviewTrigger

**Files:**
- Create: `src/skill_learner/trigger.py`
- Create: `tests/test_trigger.py`

- [ ] **Step 1: Write tests**

`tests/test_trigger.py`:
```python
from skill_learner.trigger import ReviewTrigger


class TestReviewTrigger:
    def test_initial_state_not_ready(self):
        t = ReviewTrigger(nudge_interval=10)
        assert not t.should_review()

    def test_iteration_threshold_triggers(self):
        t = ReviewTrigger(nudge_interval=5)
        for _ in range(5):
            t.tick_iteration()
        assert t.should_review()

    def test_turn_threshold_triggers(self):
        t = ReviewTrigger(nudge_interval=3)
        for _ in range(3):
            t.tick_turn()
        assert t.should_review()

    def test_mixed_counters_below_threshold(self):
        t = ReviewTrigger(nudge_interval=10)
        for _ in range(4):
            t.tick_iteration()
        for _ in range(4):
            t.tick_turn()
        assert not t.should_review()

    def test_reset_clears_both(self):
        t = ReviewTrigger(nudge_interval=3)
        for _ in range(3):
            t.tick_iteration()
        assert t.should_review()
        t.reset()
        assert not t.should_review()

    def test_zero_interval_never_triggers(self):
        t = ReviewTrigger(nudge_interval=0)
        for _ in range(100):
            t.tick_iteration()
        assert not t.should_review()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_trigger.py -v
```

- [ ] **Step 3: Implement trigger.py**

`src/skill_learner/trigger.py`:
```python
from __future__ import annotations


class ReviewTrigger:
    def __init__(self, nudge_interval: int = 10):
        self._iters = 0
        self._turns = 0
        self._nudge_interval = nudge_interval

    def tick_iteration(self) -> None:
        self._iters += 1

    def tick_turn(self) -> None:
        self._turns += 1

    def should_review(self) -> bool:
        if self._nudge_interval <= 0:
            return False
        return (self._iters >= self._nudge_interval
                or self._turns >= self._nudge_interval)

    def reset(self) -> None:
        self._iters = 0
        self._turns = 0
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_trigger.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/skill_learner/trigger.py tests/test_trigger.py
git commit -m "feat: add ReviewTrigger dual counter"
```

---

### Task 5: FileStorage

**Files:**
- Create: `src/skill_learner/storage/filesystem.py`
- Create: `tests/test_storage_fs.py`
- Modify: `src/skill_learner/storage/__init__.py`

- [ ] **Step 1: Write tests**

`tests/test_storage_fs.py`:
```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from skill_learner.skill import Skill, SkillMeta, SkillPatch
from skill_learner.storage.filesystem import FileStorage


def _make_skill(name: str = "test-skill", category: str = "general") -> Skill:
    return Skill(
        meta=SkillMeta(
            name=name,
            description=f"Test skill {name}",
            version="1.0.0",
            category=category,
            tags=["test"],
            created_at=datetime(2026, 4, 20),
            updated_at=datetime(2026, 4, 20),
        ),
        content="# Test\n\nSome content.",
    )


class TestFileStorageSave:
    def test_save_creates_skill_md(self, tmp_skills_dir: Path):
        fs = FileStorage(tmp_skills_dir)
        fs.save_skill(_make_skill())
        skill_file = tmp_skills_dir / "general" / "test-skill" / "SKILL.md"
        assert skill_file.exists()
        assert "test-skill" in skill_file.read_text(encoding="utf-8")

    def test_save_updates_index(self, tmp_skills_dir: Path):
        fs = FileStorage(tmp_skills_dir)
        fs.save_skill(_make_skill())
        skills = fs.list_skills()
        assert len(skills) == 1
        assert skills[0].name == "test-skill"


class TestFileStorageLoad:
    def test_load_existing(self, tmp_skills_dir: Path):
        fs = FileStorage(tmp_skills_dir)
        fs.save_skill(_make_skill())
        loaded = fs.load_skill("test-skill")
        assert loaded is not None
        assert loaded.meta.name == "test-skill"
        assert "Some content" in loaded.content

    def test_load_nonexistent_returns_none(self, tmp_skills_dir: Path):
        fs = FileStorage(tmp_skills_dir)
        assert fs.load_skill("no-such-skill") is None


class TestFileStorageUpdate:
    def test_update_changes_content(self, tmp_skills_dir: Path):
        fs = FileStorage(tmp_skills_dir)
        fs.save_skill(_make_skill())
        fs.update_skill("test-skill", SkillPatch(content="# Updated\n\nNew content."))
        loaded = fs.load_skill("test-skill")
        assert loaded is not None
        assert "New content" in loaded.content
        assert loaded.meta.version == "1.1.0"


class TestFileStorageDelete:
    def test_delete_removes_files(self, tmp_skills_dir: Path):
        fs = FileStorage(tmp_skills_dir)
        fs.save_skill(_make_skill())
        fs.delete_skill("test-skill")
        assert fs.load_skill("test-skill") is None
        assert len(fs.list_skills()) == 0


class TestFileStorageList:
    def test_list_multiple_categories(self, tmp_skills_dir: Path):
        fs = FileStorage(tmp_skills_dir)
        fs.save_skill(_make_skill("skill-a", "coding"))
        fs.save_skill(_make_skill("skill-b", "business"))
        skills = fs.list_skills()
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"skill-a", "skill-b"}
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_storage_fs.py -v
```

- [ ] **Step 3: Implement filesystem.py**

`src/skill_learner/storage/filesystem.py`:
```python
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from skill_learner.skill import (
    Skill,
    SkillMeta,
    SkillPatch,
    SkillLearnerError,
    parse_skill_md,
    serialize_skill_md,
)

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$")


class FileStorage:
    def __init__(self, skills_dir: Path | str):
        self._dir = Path(skills_dir).expanduser().resolve()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / ".index.json"

    def save_skill(self, skill: Skill) -> None:
        if not _NAME_RE.match(skill.meta.name):
            raise SkillLearnerError(
                f"Invalid skill name '{skill.meta.name}': must be lowercase alphanumeric with hyphens"
            )
        skill_dir = self._dir / skill.meta.category / skill.meta.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        content = serialize_skill_md(skill)
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        self._rebuild_index()

    def load_skill(self, name: str) -> Skill | None:
        for skill_file in self._dir.rglob(f"{name}/SKILL.md"):
            try:
                text = skill_file.read_text(encoding="utf-8")
                return parse_skill_md(text)
            except Exception:
                logger.warning("Failed to parse %s", skill_file)
                return None
        return None

    def list_skills(self) -> list[SkillMeta]:
        if self._index_path.exists():
            try:
                index = json.loads(self._index_path.read_text(encoding="utf-8"))
                if self._index_is_fresh(index):
                    return [self._meta_from_dict(s) for s in index.get("skills", [])]
            except Exception:
                pass
        return self._rebuild_index()

    def update_skill(self, name: str, patch: SkillPatch) -> None:
        existing = self.load_skill(name)
        if existing is None:
            raise SkillLearnerError(f"Skill '{name}' not found")
        updated = existing.apply_patch(patch)
        self.save_skill(updated)

    def delete_skill(self, name: str) -> None:
        for skill_dir in self._dir.rglob(name):
            if (skill_dir / "SKILL.md").exists():
                shutil.rmtree(skill_dir)
                break
        self._rebuild_index()

    def record_usage(self, name: str) -> None:
        pass

    def get_stats(self) -> dict[str, Any]:
        skills = self.list_skills()
        categories: dict[str, int] = {}
        for s in skills:
            categories[s.category] = categories.get(s.category, 0) + 1
        return {"total_skills": len(skills), "by_category": categories}

    def _rebuild_index(self) -> list[SkillMeta]:
        skills: list[SkillMeta] = []
        for skill_file in self._dir.rglob("SKILL.md"):
            try:
                text = skill_file.read_text(encoding="utf-8")
                skill = parse_skill_md(text)
                skills.append(skill.meta)
            except Exception:
                logger.warning("Skipping invalid skill file: %s", skill_file)
        index = {
            "skills": [self._meta_to_dict(s) for s in skills],
            "mtimes": {str(f): f.stat().st_mtime for f in self._dir.rglob("SKILL.md")},
        }
        try:
            self._index_path.write_text(
                json.dumps(index, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.warning("Failed to write index file")
        return skills

    def _index_is_fresh(self, index: dict) -> bool:
        stored_mtimes = index.get("mtimes", {})
        current_files = list(self._dir.rglob("SKILL.md"))
        if len(current_files) != len(stored_mtimes):
            return False
        for f in current_files:
            key = str(f)
            if key not in stored_mtimes:
                return False
            if abs(f.stat().st_mtime - stored_mtimes[key]) > 0.01:
                return False
        return True

    @staticmethod
    def _meta_to_dict(meta: SkillMeta) -> dict:
        return {
            "name": meta.name,
            "description": meta.description,
            "version": meta.version,
            "category": meta.category,
            "tags": meta.tags,
            "created_at": meta.created_at.isoformat(),
            "updated_at": meta.updated_at.isoformat(),
        }

    @staticmethod
    def _meta_from_dict(d: dict) -> SkillMeta:
        from datetime import datetime
        return SkillMeta(
            name=d["name"],
            description=d["description"],
            version=d.get("version", "1.0.0"),
            category=d.get("category", "general"),
            tags=d.get("tags", []),
            created_at=datetime.fromisoformat(d["created_at"]) if "created_at" in d else datetime.now(),
            updated_at=datetime.fromisoformat(d["updated_at"]) if "updated_at" in d else datetime.now(),
        )
```

- [ ] **Step 4: Update storage/__init__.py**

```python
from skill_learner.storage.filesystem import FileStorage

__all__ = ["FileStorage"]
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_storage_fs.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/skill_learner/storage/ tests/test_storage_fs.py
git commit -m "feat: add FileStorage with SKILL.md CRUD and JSON index"
```

---

### Task 6: SQLiteStorage

**Files:**
- Create: `src/skill_learner/storage/sqlite.py`
- Create: `tests/test_storage_sqlite.py`
- Modify: `src/skill_learner/storage/__init__.py`

- [ ] **Step 1: Write tests**

`tests/test_storage_sqlite.py`:
```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from skill_learner.skill import Skill, SkillMeta, SkillPatch
from skill_learner.storage.sqlite import SQLiteStorage


def _make_skill(name: str = "test-skill", category: str = "general") -> Skill:
    return Skill(
        meta=SkillMeta(
            name=name,
            description=f"Test skill {name}",
            version="1.0.0",
            category=category,
            tags=["test"],
            created_at=datetime(2026, 4, 20),
            updated_at=datetime(2026, 4, 20),
        ),
        content="# Test\n\nSome content.",
    )


class TestSQLiteStorageCRUD:
    def test_save_and_load(self, tmp_skills_dir: Path):
        st = SQLiteStorage(tmp_skills_dir)
        st.save_skill(_make_skill())
        loaded = st.load_skill("test-skill")
        assert loaded is not None
        assert loaded.meta.name == "test-skill"

    def test_list_returns_from_sqlite(self, tmp_skills_dir: Path):
        st = SQLiteStorage(tmp_skills_dir)
        st.save_skill(_make_skill("a", "coding"))
        st.save_skill(_make_skill("b", "business"))
        skills = st.list_skills()
        assert len(skills) == 2

    def test_delete(self, tmp_skills_dir: Path):
        st = SQLiteStorage(tmp_skills_dir)
        st.save_skill(_make_skill())
        st.delete_skill("test-skill")
        assert st.load_skill("test-skill") is None
        assert len(st.list_skills()) == 0


class TestSQLiteStorageStats:
    def test_record_usage_increments(self, tmp_skills_dir: Path):
        st = SQLiteStorage(tmp_skills_dir)
        st.save_skill(_make_skill())
        st.record_usage("test-skill")
        st.record_usage("test-skill")
        stats = st.get_stats()
        assert stats["total_skills"] == 1
        assert stats["skills"][0]["use_count"] == 2

    def test_stats_empty(self, tmp_skills_dir: Path):
        st = SQLiteStorage(tmp_skills_dir)
        stats = st.get_stats()
        assert stats["total_skills"] == 0


class TestSQLiteStorageSyncRecovery:
    def test_sqlite_rebuilds_from_disk_if_db_missing(self, tmp_skills_dir: Path):
        st = SQLiteStorage(tmp_skills_dir)
        st.save_skill(_make_skill())
        db_path = tmp_skills_dir / ".index.db"
        db_path.unlink()
        st2 = SQLiteStorage(tmp_skills_dir)
        skills = st2.list_skills()
        assert len(skills) == 1
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_storage_sqlite.py -v
```

- [ ] **Step 3: Implement sqlite.py**

`src/skill_learner/storage/sqlite.py`:
```python
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from skill_learner.skill import Skill, SkillMeta, SkillPatch
from skill_learner.storage.filesystem import FileStorage

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS skills (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    tags TEXT NOT NULL DEFAULT '[]',
    version TEXT NOT NULL DEFAULT '1.0.0',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    use_count INTEGER NOT NULL DEFAULT 0,
    last_used_at TEXT
);
"""


class SQLiteStorage:
    def __init__(self, skills_dir: Path | str):
        self._dir = Path(skills_dir).expanduser().resolve()
        self._fs = FileStorage(self._dir)
        self._db_path = self._dir / ".index.db"
        self._ensure_schema()
        self._sync_from_disk()

    def save_skill(self, skill: Skill) -> None:
        self._fs.save_skill(skill)
        self._upsert_index(skill.meta)

    def load_skill(self, name: str) -> Skill | None:
        return self._fs.load_skill(name)

    def list_skills(self) -> list[SkillMeta]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT name, description, category, tags, version, created_at, updated_at FROM skills"
            ).fetchall()
            return [self._row_to_meta(r) for r in rows]
        finally:
            conn.close()

    def update_skill(self, name: str, patch: SkillPatch) -> None:
        self._fs.update_skill(name, patch)
        updated = self._fs.load_skill(name)
        if updated:
            self._upsert_index(updated.meta)

    def delete_skill(self, name: str) -> None:
        self._fs.delete_skill(name)
        conn = self._connect()
        try:
            conn.execute("DELETE FROM skills WHERE name = ?", (name,))
            conn.commit()
        finally:
            conn.close()

    def record_usage(self, name: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE skills SET use_count = use_count + 1, last_used_at = ? WHERE name = ?",
                (datetime.now().isoformat(), name),
            )
            conn.commit()
        finally:
            conn.close()

    def get_stats(self) -> dict[str, Any]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT name, category, use_count, last_used_at FROM skills ORDER BY use_count DESC"
            ).fetchall()
            return {
                "total_skills": len(rows),
                "skills": [
                    {"name": r[0], "category": r[1], "use_count": r[2], "last_used_at": r[3]}
                    for r in rows
                ],
            }
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version < _SCHEMA_VERSION:
                conn.execute(_CREATE_TABLE)
                conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                conn.commit()
        finally:
            conn.close()

    def _sync_from_disk(self) -> None:
        disk_skills = self._fs.list_skills()
        conn = self._connect()
        try:
            db_names = {r[0] for r in conn.execute("SELECT name FROM skills").fetchall()}
            disk_names = {s.name for s in disk_skills}
            for meta in disk_skills:
                if meta.name not in db_names:
                    self._upsert_index(meta, conn=conn)
            for name in db_names - disk_names:
                conn.execute("DELETE FROM skills WHERE name = ?", (name,))
            conn.commit()
        finally:
            conn.close()

    def _upsert_index(self, meta: SkillMeta, conn: sqlite3.Connection | None = None) -> None:
        should_close = conn is None
        if conn is None:
            conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO skills (name, description, category, tags, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description=excluded.description,
                    category=excluded.category,
                    tags=excluded.tags,
                    version=excluded.version,
                    updated_at=excluded.updated_at
                """,
                (
                    meta.name,
                    meta.description,
                    meta.category,
                    json.dumps(meta.tags),
                    meta.version,
                    meta.created_at.isoformat(),
                    meta.updated_at.isoformat(),
                ),
            )
            if should_close:
                conn.commit()
        except Exception:
            logger.warning("SQLite sync failed for skill '%s'", meta.name)
        finally:
            if should_close:
                conn.close()

    @staticmethod
    def _row_to_meta(row: tuple) -> SkillMeta:
        return SkillMeta(
            name=row[0],
            description=row[1],
            category=row[2],
            tags=json.loads(row[3]) if row[3] else [],
            version=row[4],
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
        )
```

- [ ] **Step 4: Update storage/__init__.py**

```python
from skill_learner.storage.filesystem import FileStorage
from skill_learner.storage.sqlite import SQLiteStorage

__all__ = ["FileStorage", "SQLiteStorage"]
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_storage_sqlite.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/skill_learner/storage/ tests/test_storage_sqlite.py
git commit -m "feat: add SQLiteStorage with index, stats, and disk sync recovery"
```

---

### Task 7: BackgroundReviewer

**Files:**
- Create: `src/skill_learner/reviewer.py`
- Create: `tests/test_reviewer.py`

- [ ] **Step 1: Write tests**

`tests/test_reviewer.py`:
```python
from __future__ import annotations

import time
from pathlib import Path

import pytest

from skill_learner.reviewer import BackgroundReviewer
from skill_learner.storage.filesystem import FileStorage
from tests.helpers import MockLLMProvider, MockLLMResponse, MockToolCall


class TestBackgroundReviewer:
    def test_submit_creates_skill_on_save_tool_call(self, tmp_skills_dir: Path):
        mock_llm = MockLLMProvider([
            MockLLMResponse(
                content="",
                tool_calls=[MockToolCall(
                    name="save_skill",
                    arguments={
                        "name": "debug-memory",
                        "category": "coding",
                        "description": "Memory leak debug workflow",
                        "content": "# Debug Memory Leaks\n\n1. Use tracemalloc\n2. Check refs",
                    },
                )],
            )
        ])
        storage = FileStorage(tmp_skills_dir)
        reviewer = BackgroundReviewer(mock_llm, storage)

        messages = [{"role": "user", "content": "help me debug memory"}]
        reviewer.submit(messages)
        time.sleep(1)

        skill = storage.load_skill("debug-memory")
        assert skill is not None
        assert "tracemalloc" in skill.content

    def test_submit_nothing_to_save(self, tmp_skills_dir: Path):
        mock_llm = MockLLMProvider([
            MockLLMResponse(
                content="Nothing to save.",
                tool_calls=[MockToolCall(name="nothing_to_save", arguments={})],
            )
        ])
        storage = FileStorage(tmp_skills_dir)
        reviewer = BackgroundReviewer(mock_llm, storage)

        reviewer.submit([{"role": "user", "content": "hello"}])
        time.sleep(0.5)

        assert len(storage.list_skills()) == 0

    def test_concurrent_submit_only_runs_one(self, tmp_skills_dir: Path):
        call_count = 0

        class SlowLLM:
            def complete(self, messages, tools=None, max_tokens=2048):
                nonlocal call_count
                call_count += 1
                time.sleep(0.5)
                return MockLLMResponse(
                    content="",
                    tool_calls=[MockToolCall(name="nothing_to_save", arguments={})],
                )

        storage = FileStorage(tmp_skills_dir)
        reviewer = BackgroundReviewer(SlowLLM(), storage)

        reviewer.submit([{"role": "user", "content": "a"}])
        time.sleep(0.05)
        reviewer.submit([{"role": "user", "content": "b"}])
        time.sleep(1)

        assert call_count == 1

    def test_submit_updates_existing_skill(self, tmp_skills_dir: Path):
        storage = FileStorage(tmp_skills_dir)
        from skill_learner.skill import Skill, SkillMeta
        storage.save_skill(Skill(
            meta=SkillMeta(name="old-skill", description="Old", category="general"),
            content="# Old\n\nOld content.",
        ))
        mock_llm = MockLLMProvider([
            MockLLMResponse(
                tool_calls=[MockToolCall(
                    name="update_skill",
                    arguments={"name": "old-skill", "patch_content": "# Updated\n\nNew content."},
                )],
            )
        ])
        reviewer = BackgroundReviewer(mock_llm, storage)
        reviewer.submit_and_wait([{"role": "user", "content": "test"}], timeout=5)
        loaded = storage.load_skill("old-skill")
        assert loaded is not None
        assert "New content" in loaded.content

    def test_submit_deletes_skill(self, tmp_skills_dir: Path):
        storage = FileStorage(tmp_skills_dir)
        from skill_learner.skill import Skill, SkillMeta
        storage.save_skill(Skill(
            meta=SkillMeta(name="bad-skill", description="Bad", category="general"),
            content="# Bad",
        ))
        mock_llm = MockLLMProvider([
            MockLLMResponse(
                tool_calls=[MockToolCall(
                    name="delete_skill",
                    arguments={"name": "bad-skill", "reason": "obsolete"},
                )],
            )
        ])
        reviewer = BackgroundReviewer(mock_llm, storage)
        reviewer.submit_and_wait([{"role": "user", "content": "test"}], timeout=5)
        assert storage.load_skill("bad-skill") is None

    def test_submit_llm_error_releases_lock(self, tmp_skills_dir: Path):
        class ErrorLLM:
            def complete(self, messages, tools=None, max_tokens=2048):
                raise RuntimeError("LLM crashed")

        storage = FileStorage(tmp_skills_dir)
        reviewer = BackgroundReviewer(ErrorLLM(), storage)
        reviewer.submit_and_wait([{"role": "user", "content": "test"}], timeout=2)
        # Lock should be released — second submit should work
        reviewer.submit_and_wait([{"role": "user", "content": "test2"}], timeout=2)

    def test_submit_with_timeout(self, tmp_skills_dir: Path):
        mock_llm = MockLLMProvider([
            MockLLMResponse(
                content="Nothing to save.",
                tool_calls=[MockToolCall(name="nothing_to_save", arguments={})],
            )
        ])
        storage = FileStorage(tmp_skills_dir)
        reviewer = BackgroundReviewer(mock_llm, storage)
        reviewer.submit_and_wait([{"role": "user", "content": "test"}], timeout=5)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_reviewer.py -v
```

- [ ] **Step 3: Implement reviewer.py**

`src/skill_learner/reviewer.py`:
```python
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Any

from skill_learner.prompts import REVIEW_PROMPT, SKILL_MANAGE_TOOLS
from skill_learner.skill import Skill, SkillMeta, SkillPatch

if TYPE_CHECKING:
    from skill_learner.protocols import LLMProvider, StorageBackend

logger = logging.getLogger(__name__)


class BackgroundReviewer:
    def __init__(self, llm: LLMProvider, storage: StorageBackend):
        self._llm = llm
        self._storage = storage
        self._lock = threading.Lock()

    def submit(self, messages: list[dict]) -> None:
        if not self._lock.acquire(blocking=False):
            logger.debug("Review already in flight, skipping")
            return

        snapshot = list(messages)
        t = threading.Thread(
            target=self._run_review_guarded,
            args=(snapshot,),
            daemon=True,
            name="skill-review",
        )
        t.start()

    def submit_and_wait(self, messages: list[dict], timeout: float = 30) -> None:
        if not self._lock.acquire(blocking=False):
            logger.debug("Review already in flight, skipping")
            return

        snapshot = list(messages)
        t = threading.Thread(
            target=self._run_review_guarded,
            args=(snapshot,),
            daemon=True,
            name="skill-review-wait",
        )
        t.start()
        t.join(timeout=timeout)

    def _run_review_guarded(self, messages: list[dict]) -> None:
        try:
            self._run_review(messages)
        except Exception:
            logger.debug("Review failed (best-effort)", exc_info=True)
        finally:
            self._lock.release()

    def _run_review(self, messages: list[dict]) -> None:
        review_messages = messages + [{"role": "user", "content": REVIEW_PROMPT}]
        response = self._llm.complete(
            messages=review_messages,
            tools=SKILL_MANAGE_TOOLS,
            max_tokens=2048,
        )

        for tc in response.tool_calls:
            self._handle_tool_call(tc.name, tc.arguments)

    def _handle_tool_call(self, name: str, args: dict[str, Any]) -> None:
        if name == "save_skill":
            skill_name = args["name"]
            existing = self._storage.load_skill(skill_name)
            if existing:
                self._storage.update_skill(
                    skill_name,
                    SkillPatch(
                        description=args.get("description"),
                        content=args.get("content"),
                    ),
                )
                logger.info("Updated existing skill: %s", skill_name)
            else:
                skill = Skill(
                    meta=SkillMeta(
                        name=skill_name,
                        description=args["description"],
                        category=args.get("category", "general"),
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                    ),
                    content=args["content"],
                )
                self._storage.save_skill(skill)
                logger.info("Created new skill: %s", skill_name)

        elif name == "update_skill":
            self._storage.update_skill(
                args["name"],
                SkillPatch(
                    content=args.get("patch_content"),
                    description=args.get("description"),
                ),
            )
            logger.info("Patched skill: %s", args["name"])

        elif name == "delete_skill":
            self._storage.delete_skill(args["name"])
            logger.info("Deleted skill: %s (reason: %s)", args["name"], args.get("reason", ""))

        elif name == "nothing_to_save":
            logger.debug("Review decided nothing to save")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_reviewer.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/skill_learner/reviewer.py tests/test_reviewer.py
git commit -m "feat: add BackgroundReviewer with concurrency guard and tool call processing"
```

---

### Task 8: PromptInjector

**Files:**
- Create: `src/skill_learner/injector.py`
- Create: `tests/test_injector.py`

- [ ] **Step 1: Write tests**

`tests/test_injector.py`:
```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from skill_learner.injector import PromptInjector
from skill_learner.skill import Skill, SkillMeta
from skill_learner.storage.filesystem import FileStorage


def _save_sample_skills(storage: FileStorage) -> None:
    for name, cat, desc in [
        ("competitor-analysis", "business", "SaaS competitor analysis workflow"),
        ("debug-memory", "coding", "Memory leak debugging steps"),
        ("arxiv-search", "research", "Academic paper search flow"),
    ]:
        storage.save_skill(Skill(
            meta=SkillMeta(
                name=name, description=desc, category=cat,
                tags=["test"], created_at=datetime(2026, 4, 20),
                updated_at=datetime(2026, 4, 20),
            ),
            content=f"# {name}\n\nContent for {name}.",
        ))


class TestPromptInjector:
    def test_build_empty_returns_guidance_only(self, tmp_skills_dir: Path):
        storage = FileStorage(tmp_skills_dir)
        injector = PromptInjector(storage)
        prompt = injector.build_skills_prompt()
        assert "Skill Accumulation" in prompt
        assert "<available_skills>" not in prompt

    def test_build_with_skills_includes_index(self, tmp_skills_dir: Path):
        storage = FileStorage(tmp_skills_dir)
        _save_sample_skills(storage)
        injector = PromptInjector(storage)
        prompt = injector.build_skills_prompt()
        assert "competitor-analysis" in prompt
        assert "debug-memory" in prompt
        assert "<available_skills>" in prompt

    def test_search_by_keyword(self, tmp_skills_dir: Path):
        storage = FileStorage(tmp_skills_dir)
        _save_sample_skills(storage)
        injector = PromptInjector(storage)
        results = injector.search("memory debug")
        assert any(s.meta.name == "debug-memory" for s in results)

    def test_search_no_match(self, tmp_skills_dir: Path):
        storage = FileStorage(tmp_skills_dir)
        _save_sample_skills(storage)
        injector = PromptInjector(storage)
        results = injector.search("quantum physics")
        assert len(results) == 0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_injector.py -v
```

- [ ] **Step 3: Implement injector.py**

`src/skill_learner/injector.py`:
```python
from __future__ import annotations

from typing import Any

from skill_learner.prompts import SKILLS_GUIDANCE
from skill_learner.skill import Skill


class PromptInjector:
    def __init__(self, storage: Any):
        self._storage = storage

    def build_skills_prompt(self) -> str:
        skills = self._storage.list_skills()

        if not skills:
            return SKILLS_GUIDANCE

        by_category: dict[str, list[tuple[str, str]]] = {}
        for s in skills:
            by_category.setdefault(s.category, []).append((s.name, s.description))

        lines = []
        for cat in sorted(by_category):
            lines.append(f"  {cat}:")
            for name, desc in by_category[cat]:
                lines.append(f"    - {name}: {desc}")

        index = "\n".join(lines)
        return f"""## Available Skills (scan before responding)

Before answering, scan the skill list below.
If a skill **matches or relates to your task**, load it with load_skill(name) and follow its guidance.

<available_skills>
{index}
</available_skills>

Only proceed without a skill if none are relevant.

{SKILLS_GUIDANCE}"""

    def search(self, query: str) -> list[Skill]:
        query_terms = set(query.lower().split())
        results: list[tuple[int, Skill]] = []

        skills = self._storage.list_skills()
        for meta in skills:
            searchable = f"{meta.name} {meta.description} {' '.join(meta.tags)}".lower()
            score = sum(1 for term in query_terms if term in searchable)
            if score > 0:
                skill = self._storage.load_skill(meta.name)
                if skill:
                    results.append((score, skill))

        results.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in results]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_injector.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/skill_learner/injector.py tests/test_injector.py
git commit -m "feat: add PromptInjector with skill index building and keyword search"
```

---

### Task 9: SkillEngine

**Files:**
- Create: `src/skill_learner/engine.py`
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write tests**

`tests/test_engine.py`:
```python
from __future__ import annotations

import time
from pathlib import Path

import pytest

from skill_learner.config import SkillLearnerConfig
from skill_learner.engine import SkillEngine
from tests.helpers import MockLLMProvider, MockLLMResponse, MockToolCall


class TestSkillEngine:
    def _make_engine(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider, nudge: int = 3) -> SkillEngine:
        config = SkillLearnerConfig(
            skills_dir=str(tmp_skills_dir),
            nudge_interval=nudge,
            storage_backend="filesystem",
        )
        return SkillEngine(config=config, llm=mock_llm)

    def test_on_session_start_returns_prompt(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider):
        engine = self._make_engine(tmp_skills_dir, mock_llm)
        prompt = engine.on_session_start()
        assert "Skill Accumulation" in prompt

    def test_tool_calls_trigger_review(self, tmp_skills_dir: Path):
        mock_llm = MockLLMProvider([
            MockLLMResponse(
                tool_calls=[MockToolCall(
                    name="save_skill",
                    arguments={
                        "name": "auto-skill",
                        "category": "general",
                        "description": "Auto generated",
                        "content": "# Auto\n\nContent",
                    },
                )],
            ),
        ])
        engine = self._make_engine(tmp_skills_dir, mock_llm, nudge=3)

        for _ in range(3):
            engine.on_tool_call("web_search")

        messages = [{"role": "user", "content": "do something complex"}]
        engine.on_turn_complete(messages)
        time.sleep(1)

        skill = engine.storage.load_skill("auto-skill")
        assert skill is not None

    def test_below_threshold_no_review(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider):
        engine = self._make_engine(tmp_skills_dir, mock_llm, nudge=10)
        engine.on_tool_call("test")
        engine.on_turn_complete([{"role": "user", "content": "hi"}])
        assert len(mock_llm.calls) == 0

    def test_on_session_end_forces_review(self, tmp_skills_dir: Path):
        mock_llm = MockLLMProvider([
            MockLLMResponse(
                tool_calls=[MockToolCall(name="nothing_to_save", arguments={})],
            ),
        ])
        engine = self._make_engine(tmp_skills_dir, mock_llm, nudge=999)
        engine.on_session_end([{"role": "user", "content": "done"}])
        assert len(mock_llm.calls) == 1

    def test_get_stats(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider):
        engine = self._make_engine(tmp_skills_dir, mock_llm)
        stats = engine.get_stats()
        assert "total_skills" in stats
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_engine.py -v
```

- [ ] **Step 3: Implement engine.py**

`src/skill_learner/engine.py`:
```python
from __future__ import annotations

import logging
from typing import Any

from skill_learner.config import SkillLearnerConfig
from skill_learner.injector import PromptInjector
from skill_learner.reviewer import BackgroundReviewer
from skill_learner.skill import Skill
from skill_learner.storage.filesystem import FileStorage
from skill_learner.storage.sqlite import SQLiteStorage
from skill_learner.trigger import ReviewTrigger

logger = logging.getLogger(__name__)


class SkillEngine:
    def __init__(self, config: SkillLearnerConfig, llm: Any):
        self.config = config
        config.skills_path.mkdir(parents=True, exist_ok=True)

        if config.storage_backend == "sqlite":
            self.storage = SQLiteStorage(config.skills_path)
        else:
            self.storage = FileStorage(config.skills_path)

        self.trigger = ReviewTrigger(config.nudge_interval)
        self.reviewer = BackgroundReviewer(llm, self.storage)
        self.injector = PromptInjector(self.storage)

    def on_tool_call(self, tool_name: str, result: Any = None) -> None:
        try:
            self.trigger.tick_iteration()
        except Exception:
            logger.debug("on_tool_call failed", exc_info=True)

    def on_turn_complete(self, messages: list[dict]) -> None:
        try:
            self.trigger.tick_turn()
            if self.trigger.should_review():
                self.reviewer.submit(messages)
                self.trigger.reset()
        except Exception:
            logger.debug("on_turn_complete failed", exc_info=True)

    def on_session_start(self) -> str:
        try:
            return self.injector.build_skills_prompt()
        except Exception:
            logger.debug("on_session_start failed", exc_info=True)
            return ""

    def on_session_end(self, messages: list[dict]) -> None:
        try:
            self.reviewer.submit_and_wait(messages, timeout=self.config.review_timeout)
        except Exception:
            logger.debug("on_session_end failed", exc_info=True)

    def get_relevant_skills(self, query: str) -> list[Skill]:
        return self.injector.search(query)

    def get_stats(self) -> dict[str, Any]:
        return self.storage.get_stats()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_engine.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/skill_learner/engine.py tests/test_engine.py
git commit -m "feat: add SkillEngine event coordinator"
```

---

### Task 10: SkillLoop Convenience Wrapper

**Files:**
- Create: `src/skill_learner/loop.py`

- [ ] **Step 1: Implement loop.py**

`src/skill_learner/loop.py`:
```python
from __future__ import annotations

import functools
from typing import Any, Callable

from skill_learner.engine import SkillEngine


class SkillLoop:
    def __init__(self, engine: SkillEngine):
        self._engine = engine

    def wrap(self, agent_fn: Callable) -> Callable:
        @functools.wraps(agent_fn)
        def wrapper(messages: list[dict], **kwargs: Any) -> Any:
            skills_prompt = self._engine.on_session_start()
            if skills_prompt:
                has_system = messages and messages[0].get("role") == "system"
                if has_system:
                    messages = list(messages)
                    messages[0] = {
                        **messages[0],
                        "content": messages[0]["content"] + "\n\n" + skills_prompt,
                    }
                else:
                    messages = [{"role": "system", "content": skills_prompt}] + list(messages)

            response = agent_fn(messages, **kwargs)

            self._engine.on_turn_complete(messages)
            return response

        return wrapper

    def inject_prompt(self, base_prompt: str) -> str:
        skills_prompt = self._engine.on_session_start()
        if skills_prompt:
            return base_prompt + "\n\n" + skills_prompt
        return base_prompt

    @property
    def engine(self) -> SkillEngine:
        return self._engine
```

- [ ] **Step 2: Write tests**

`tests/test_loop.py`:
```python
from __future__ import annotations

from pathlib import Path

from skill_learner.config import SkillLearnerConfig
from skill_learner.engine import SkillEngine
from skill_learner.loop import SkillLoop
from tests.helpers import MockLLMProvider


class TestSkillLoop:
    def _make_loop(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider) -> SkillLoop:
        config = SkillLearnerConfig(skills_dir=str(tmp_skills_dir), nudge_interval=99, storage_backend="filesystem")
        engine = SkillEngine(config=config, llm=mock_llm)
        return SkillLoop(engine)

    def test_wrap_injects_system_message_when_none_exists(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider):
        loop = self._make_loop(tmp_skills_dir, mock_llm)
        captured = {}

        def agent_fn(messages, **kwargs):
            captured["messages"] = messages
            return "ok"

        wrapped = loop.wrap(agent_fn)
        wrapped([{"role": "user", "content": "hello"}])
        assert captured["messages"][0]["role"] == "system"
        assert "Skill Accumulation" in captured["messages"][0]["content"]

    def test_wrap_appends_to_existing_system_message(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider):
        loop = self._make_loop(tmp_skills_dir, mock_llm)
        captured = {}

        def agent_fn(messages, **kwargs):
            captured["messages"] = messages
            return "ok"

        wrapped = loop.wrap(agent_fn)
        wrapped([{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}])
        assert "You are helpful." in captured["messages"][0]["content"]
        assert "Skill Accumulation" in captured["messages"][0]["content"]

    def test_inject_prompt_appends_skills(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider):
        loop = self._make_loop(tmp_skills_dir, mock_llm)
        result = loop.inject_prompt("Base prompt here.")
        assert "Base prompt here." in result
        assert "Skill Accumulation" in result

    def test_inject_prompt_returns_base_when_empty(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider):
        loop = self._make_loop(tmp_skills_dir, mock_llm)
        result = loop.inject_prompt("")
        assert "Skill Accumulation" in result
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_loop.py -v
```

- [ ] **Step 4: Verify tests pass after implementation**

```bash
pytest tests/test_loop.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/skill_learner/loop.py tests/test_loop.py
git commit -m "feat: add SkillLoop convenience wrapper with tests"
```

---

### Task 11: Public API (__init__.py)

**Files:**
- Modify: `src/skill_learner/__init__.py`

- [ ] **Step 1: Update __init__.py with all exports**

```python
"""skill-learner: Self-accumulating skill system for AI agents."""

__version__ = "0.1.0"

from skill_learner.config import SkillLearnerConfig
from skill_learner.engine import SkillEngine
from skill_learner.injector import PromptInjector
from skill_learner.loop import SkillLoop
from skill_learner.protocols import LLMProvider, LLMResponse, StorageBackend, ToolCall
from skill_learner.reviewer import BackgroundReviewer
from skill_learner.skill import Skill, SkillLearnerError, SkillMeta, SkillPatch
from skill_learner.storage import FileStorage, SQLiteStorage
from skill_learner.trigger import ReviewTrigger

__all__ = [
    "BackgroundReviewer",
    "FileStorage",
    "LLMProvider",
    "LLMResponse",
    "PromptInjector",
    "ReviewTrigger",
    "Skill",
    "SkillEngine",
    "SkillLearnerConfig",
    "SkillLearnerError",
    "SkillLoop",
    "SkillMeta",
    "SkillPatch",
    "SQLiteStorage",
    "StorageBackend",
    "ToolCall",
]
```

- [ ] **Step 2: Verify full import works**

```bash
python -c "from skill_learner import SkillEngine, SkillLoop, SkillLearnerConfig; print('All exports OK')"
```

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v
```

- [ ] **Step 4: Commit**

```bash
git add src/skill_learner/__init__.py
git commit -m "feat: expose full public API from __init__.py"
```

---

### Task 12: Integrations — Claude Code

**Files:**
- Create: `src/skill_learner/integrations/claude_code.py`

- [ ] **Step 1: Implement claude_code.py**

`src/skill_learner/integrations/claude_code.py`:
```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skill_learner.prompts import SKILLS_GUIDANCE
from skill_learner.storage.filesystem import FileStorage


def init_claude_code(
    skills_dir: str = "~/.skill_learner/skills",
    project_dir: str = ".",
) -> dict[str, str]:
    """Initialize skill-learner for Claude Code.

    Returns dict of files created/modified.
    """
    project = Path(project_dir).resolve()
    skills_path = Path(skills_dir).expanduser().resolve()
    created: dict[str, str] = {}

    claude_dir = project / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    sl_dir = claude_dir / "skill-learner"
    sl_dir.mkdir(parents=True, exist_ok=True)

    counter_sh = sl_dir / "counter.sh"
    counter_sh.write_text(
        '#!/bin/bash\n'
        'COUNTER_FILE="/tmp/skill_learner_counter"\n'
        'if [ -f "$COUNTER_FILE" ]; then\n'
        '  count=$(cat "$COUNTER_FILE")\n'
        '  echo $((count + 1)) > "$COUNTER_FILE"\n'
        'else\n'
        '  echo 1 > "$COUNTER_FILE"\n'
        'fi\n',
        encoding="utf-8",
    )
    created["counter.sh"] = str(counter_sh)

    review_py = sl_dir / "review.py"
    review_py.write_text(
        '"""Trigger skill review from Claude Code hook."""\n'
        'import sys\n'
        'from pathlib import Path\n\n'
        'COUNTER_FILE = Path("/tmp/skill_learner_counter")\n'
        'THRESHOLD = 10\n\n'
        'def check_and_review():\n'
        '    if not COUNTER_FILE.exists():\n'
        '        return\n'
        '    count = int(COUNTER_FILE.read_text().strip())\n'
        '    if count >= THRESHOLD:\n'
        '        COUNTER_FILE.write_text("0")\n'
        '        print(f"[skill-learner] Threshold reached ({count}), review triggered")\n'
        '        # In a real setup, this would call the review via API\n\n'
        'if __name__ == "__main__":\n'
        '    check_and_review()\n',
        encoding="utf-8",
    )
    created["review.py"] = str(review_py)

    storage = FileStorage(skills_path)
    skills = storage.list_skills()
    skills_index = ""
    if skills:
        lines = []
        for s in skills:
            lines.append(f"- **{s.name}**: {s.description}")
        skills_index = "\n".join(lines)

    claude_md = project / "CLAUDE.md"
    skill_block = _build_claude_md_block(skills_index)

    if claude_md.exists():
        existing = claude_md.read_text(encoding="utf-8")
        marker_start = "<!-- skill-learner:start -->"
        marker_end = "<!-- skill-learner:end -->"
        if marker_start in existing:
            import re
            existing = re.sub(
                f"{re.escape(marker_start)}.*?{re.escape(marker_end)}",
                skill_block,
                existing,
                flags=re.DOTALL,
            )
            claude_md.write_text(existing, encoding="utf-8")
        else:
            claude_md.write_text(existing + "\n\n" + skill_block, encoding="utf-8")
    else:
        claude_md.write_text(skill_block, encoding="utf-8")

    created["CLAUDE.md"] = str(claude_md)

    settings_path = claude_dir / "settings.json"
    settings: dict[str, Any] = {}
    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))

    hooks = settings.setdefault("hooks", {})
    hooks["PostToolCall"] = [{"command": f"bash {counter_sh}"}]

    settings_path.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    created["settings.json"] = str(settings_path)

    return created


def _build_claude_md_block(skills_index: str) -> str:
    index_section = ""
    if skills_index:
        index_section = f"""
## Available Skills

{skills_index}

When a task relates to a listed skill, load and follow it before proceeding.
"""

    return f"""<!-- skill-learner:start -->
## Skill Learning System

{SKILLS_GUIDANCE}
{index_section}
<!-- skill-learner:end -->"""
```

- [ ] **Step 2: Verify import**

```bash
python -c "from skill_learner.integrations.claude_code import init_claude_code; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/skill_learner/integrations/claude_code.py
git commit -m "feat: add Claude Code integration (hooks + CLAUDE.md generation)"
```

---

### Task 13: Integrations — Cursor + Generic

**Files:**
- Create: `src/skill_learner/integrations/cursor.py`
- Create: `src/skill_learner/integrations/generic_agent.py`
- Modify: `src/skill_learner/integrations/__init__.py`

- [ ] **Step 1: Implement cursor.py**

`src/skill_learner/integrations/cursor.py`:
```python
from __future__ import annotations

from pathlib import Path

from skill_learner.prompts import SKILLS_GUIDANCE
from skill_learner.storage.filesystem import FileStorage


def init_cursor(
    skills_dir: str = "~/.skill_learner/skills",
    project_dir: str = ".",
) -> dict[str, str]:
    project = Path(project_dir).resolve()
    skills_path = Path(skills_dir).expanduser().resolve()
    created: dict[str, str] = {}

    storage = FileStorage(skills_path)
    skills = storage.list_skills()

    lines = [SKILLS_GUIDANCE, ""]
    if skills:
        lines.append("## Available Skills\n")
        for s in skills:
            lines.append(f"- **{s.name}**: {s.description}")
        lines.append("")
        lines.append("Load relevant skills before proceeding with tasks.")
        lines.append("")

    lines.append("To manually trigger skill review: `skill-learner review --messages <file>`")

    rules_file = project / ".cursorrules"
    marker_start = "# skill-learner:start"
    marker_end = "# skill-learner:end"
    block = f"{marker_start}\n" + "\n".join(lines) + f"\n{marker_end}"

    if rules_file.exists():
        existing = rules_file.read_text(encoding="utf-8")
        if marker_start in existing:
            import re
            existing = re.sub(
                f"{re.escape(marker_start)}.*?{re.escape(marker_end)}",
                block,
                existing,
                flags=re.DOTALL,
            )
            rules_file.write_text(existing, encoding="utf-8")
        else:
            rules_file.write_text(existing + "\n\n" + block, encoding="utf-8")
    else:
        rules_file.write_text(block, encoding="utf-8")

    created[".cursorrules"] = str(rules_file)
    return created
```

- [ ] **Step 2: Implement generic_agent.py**

`src/skill_learner/integrations/generic_agent.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from skill_learner.config import SkillLearnerConfig
from skill_learner.engine import SkillEngine
from skill_learner.loop import SkillLoop


def create_skill_loop(
    llm: Any,
    skills_dir: str = "~/.skill_learner/skills",
    nudge_interval: int = 10,
    storage_backend: str = "sqlite",
) -> SkillLoop:
    config = SkillLearnerConfig(
        skills_dir=skills_dir,
        nudge_interval=nudge_interval,
        storage_backend=storage_backend,
    )
    engine = SkillEngine(config=config, llm=llm)
    return SkillLoop(engine)
```

- [ ] **Step 3: Update integrations/__init__.py**

```python
from skill_learner.integrations.claude_code import init_claude_code
from skill_learner.integrations.cursor import init_cursor
from skill_learner.integrations.generic_agent import create_skill_loop

__all__ = ["init_claude_code", "init_cursor", "create_skill_loop"]
```

- [ ] **Step 4: Commit**

```bash
git add src/skill_learner/integrations/
git commit -m "feat: add Cursor and generic agent integrations"
```

---

### Task 14: CLI

**Files:**
- Create: `src/skill_learner/cli.py`

- [ ] **Step 1: Implement cli.py**

`src/skill_learner/cli.py`:
```python
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import click
    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

from skill_learner.config import SkillLearnerConfig
from skill_learner.storage.filesystem import FileStorage
from skill_learner.storage.sqlite import SQLiteStorage


def _get_storage(skills_dir: str, backend: str = "sqlite"):
    path = Path(skills_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    if backend == "sqlite":
        return SQLiteStorage(path)
    return FileStorage(path)


if HAS_CLICK:
    @click.group()
    @click.option("--skills-dir", default="~/.skill_learner/skills", help="Skills directory")
    @click.pass_context
    def cli(ctx: click.Context, skills_dir: str) -> None:
        """skill-learner: Self-accumulating skill system for AI agents."""
        ctx.ensure_object(dict)
        ctx.obj["skills_dir"] = skills_dir

    @cli.command()
    @click.option("--target", type=click.Choice(["claude-code", "cursor", "generic"]), required=True)
    @click.option("--project-dir", default=".", help="Project directory")
    @click.pass_context
    def init(ctx: click.Context, target: str, project_dir: str) -> None:
        """Initialize skill-learner for a platform."""
        skills_dir = ctx.obj["skills_dir"]
        if target == "claude-code":
            from skill_learner.integrations.claude_code import init_claude_code
            created = init_claude_code(skills_dir, project_dir)
        elif target == "cursor":
            from skill_learner.integrations.cursor import init_cursor
            created = init_cursor(skills_dir, project_dir)
        else:
            click.echo("Generic: use SkillLoop in your Python code. See examples/.")
            return
        for name, path in created.items():
            click.echo(f"  Created: {name} -> {path}")

    @cli.command("list")
    @click.option("--category", default=None, help="Filter by category")
    @click.pass_context
    def list_skills(ctx: click.Context, category: str | None) -> None:
        """List all saved skills."""
        storage = _get_storage(ctx.obj["skills_dir"])
        skills = storage.list_skills()
        if category:
            skills = [s for s in skills if s.category == category]
        if not skills:
            click.echo("No skills found.")
            return
        for s in skills:
            click.echo(f"  [{s.category}] {s.name} (v{s.version}) — {s.description}")

    @cli.command()
    @click.argument("name")
    @click.pass_context
    def show(ctx: click.Context, name: str) -> None:
        """Show full skill content."""
        storage = _get_storage(ctx.obj["skills_dir"])
        skill = storage.load_skill(name)
        if not skill:
            click.echo(f"Skill '{name}' not found.")
            return
        click.echo(f"--- {skill.meta.name} v{skill.meta.version} [{skill.meta.category}] ---")
        click.echo(skill.content)

    @cli.command()
    @click.pass_context
    def stats(ctx: click.Context) -> None:
        """Show usage statistics."""
        storage = _get_storage(ctx.obj["skills_dir"])
        data = storage.get_stats()
        click.echo(f"Total skills: {data['total_skills']}")
        if "skills" in data:
            for s in data["skills"]:
                click.echo(f"  {s['name']}: {s['use_count']} uses")
        elif "by_category" in data:
            for cat, count in data["by_category"].items():
                click.echo(f"  {cat}: {count} skills")

    @cli.command()
    @click.option("--messages", "messages_file", required=True, type=click.Path(exists=True),
                  help="JSON file with messages array [{role, content}, ...]")
    @click.pass_context
    def review(ctx: click.Context, messages_file: str) -> None:
        """Manually trigger a skill review from a conversation JSON.

        Note: The CLI review command saves the validated messages for use with
        the Python API. For full automated review, use the Python API directly:

            from skill_learner import BackgroundReviewer
            reviewer = BackgroundReviewer(my_llm, storage)
            reviewer.submit_and_wait(messages)
        """
        messages = json.loads(Path(messages_file).read_text(encoding="utf-8"))
        if not isinstance(messages, list):
            click.echo("Error: messages file must contain a JSON array of {role, content} objects", err=True)
            return
        for i, msg in enumerate(messages):
            if "role" not in msg or "content" not in msg:
                click.echo(f"Error: message at index {i} missing 'role' or 'content'", err=True)
                return
        click.echo(f"Validated {len(messages)} messages.")
        click.echo("To run the review, use the Python API with your LLM provider:")
        click.echo("  from skill_learner import BackgroundReviewer, FileStorage")
        click.echo(f"  storage = FileStorage('{ctx.obj['skills_dir']}')")
        click.echo("  reviewer = BackgroundReviewer(your_llm, storage)")
        click.echo(f"  reviewer.submit_and_wait(messages, timeout=60)")

    @cli.command("export")
    @click.option("--format", "fmt", type=click.Choice(["json", "markdown"]), default="json")
    @click.option("--output", default=None, help="Output file path")
    @click.pass_context
    def export_skills(ctx: click.Context, fmt: str, output: str | None) -> None:
        """Export skill library."""
        storage = _get_storage(ctx.obj["skills_dir"])
        skills_meta = storage.list_skills()
        skills = [storage.load_skill(s.name) for s in skills_meta]
        skills = [s for s in skills if s is not None]

        if fmt == "json":
            data = [{"name": s.meta.name, "description": s.meta.description,
                      "category": s.meta.category, "version": s.meta.version,
                      "tags": s.meta.tags, "content": s.content} for s in skills]
            text = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            parts = []
            for s in skills:
                parts.append(f"# {s.meta.name}\n\n**{s.meta.description}**\n\n{s.content}\n\n---\n")
            text = "\n".join(parts)

        if output:
            Path(output).write_text(text, encoding="utf-8")
            click.echo(f"Exported {len(skills)} skills to {output}")
        else:
            click.echo(text)

    @cli.command("import")
    @click.argument("file", type=click.Path(exists=True))
    @click.pass_context
    def import_skills(ctx: click.Context, file: str) -> None:
        """Import skills from a JSON export file."""
        from datetime import datetime
        from skill_learner.skill import Skill, SkillMeta

        storage = _get_storage(ctx.obj["skills_dir"])
        data = json.loads(Path(file).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            click.echo("Error: import file must contain a JSON array of skill objects", err=True)
            return

        count = 0
        for item in data:
            try:
                skill = Skill(
                    meta=SkillMeta(
                        name=item["name"],
                        description=item["description"],
                        category=item.get("category", "general"),
                        version=item.get("version", "1.0.0"),
                        tags=item.get("tags", []),
                        created_at=datetime.fromisoformat(item["created_at"]) if "created_at" in item else datetime.now(),
                        updated_at=datetime.now(),
                    ),
                    content=item["content"],
                )
                storage.save_skill(skill)
                count += 1
            except (KeyError, ValueError) as e:
                click.echo(f"  Skipping invalid entry: {e}", err=True)
        click.echo(f"Imported {count} skills.")

    def main():
        cli()

else:
    def main():
        print("CLI requires click: pip install skill-learner[cli]")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add __main__.py for `python -m skill_learner`**

`src/skill_learner/__main__.py`:
```python
from skill_learner.cli import main

main()
```

- [ ] **Step 3: Verify CLI works**

```bash
skill-learner --help
skill-learner list
```

- [ ] **Step 4: Commit**

```bash
git add src/skill_learner/cli.py src/skill_learner/__main__.py
git commit -m "feat: add CLI with init, list, show, stats, export commands"
```

---

### Task 15: Examples

**Files:**
- Create: `examples/anthropic_agent.py`
- Create: `examples/openai_agent.py`
- Create: `examples/custom_agent.py`

- [ ] **Step 1: Create example files**

`examples/anthropic_agent.py`:
```python
"""Example: Using skill-learner with the Anthropic SDK."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# pip install anthropic skill-learner
from anthropic import Anthropic

from skill_learner import SkillEngine, SkillLearnerConfig


@dataclass
class AnthropicToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class AnthropicResponse:
    content: str = ""
    tool_calls: list[AnthropicToolCall] = field(default_factory=list)


class AnthropicLLMProvider:
    """Adapter: Anthropic SDK -> skill-learner LLMProvider protocol."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self._client = Anthropic()
        self._model = model

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
    ) -> AnthropicResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]

        response = self._client.messages.create(**kwargs)

        content = ""
        tc_list: list[AnthropicToolCall] = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tc_list.append(AnthropicToolCall(name=block.name, arguments=block.input))

        return AnthropicResponse(content=content, tool_calls=tc_list)

    @staticmethod
    def _convert_tool(tool: dict) -> dict:
        func = tool.get("function", tool)
        return {
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {}),
        }


def main():
    llm = AnthropicLLMProvider()
    config = SkillLearnerConfig(skills_dir="./my_skills", nudge_interval=5)
    engine = SkillEngine(config=config, llm=llm)

    # On session start: inject skills into your system prompt
    skills_prompt = engine.on_session_start()
    print(f"Skills prompt ({len(skills_prompt)} chars) injected")

    # During agent loop: report tool calls
    engine.on_tool_call("web_search")
    engine.on_tool_call("file_write")

    # After responding to user: check if review should trigger
    messages = [
        {"role": "user", "content": "Help me analyze competitor pricing"},
        {"role": "assistant", "content": "I'll research competitor pricing..."},
    ]
    engine.on_turn_complete(messages)

    # On session end: force a review
    engine.on_session_end(messages)

    print(f"Stats: {engine.get_stats()}")


if __name__ == "__main__":
    main()
```

`examples/openai_agent.py`:
```python
"""Example: Using skill-learner with the OpenAI SDK."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# pip install openai skill-learner
from openai import OpenAI

from skill_learner import SkillEngine, SkillLearnerConfig


@dataclass
class OpenAIToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class OpenAIResponse:
    content: str = ""
    tool_calls: list[OpenAIToolCall] = field(default_factory=list)


class OpenAILLMProvider:
    """Adapter: OpenAI SDK -> skill-learner LLMProvider protocol."""

    def __init__(self, model: str = "gpt-4o"):
        self._client = OpenAI()
        self._model = model

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
    ) -> OpenAIResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        tc_list: list[OpenAIToolCall] = []
        if msg.tool_calls:
            import json
            for tc in msg.tool_calls:
                tc_list.append(OpenAIToolCall(
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return OpenAIResponse(content=msg.content or "", tool_calls=tc_list)


def main():
    llm = OpenAILLMProvider()
    config = SkillLearnerConfig(skills_dir="./my_skills", nudge_interval=5)
    engine = SkillEngine(config=config, llm=llm)

    skills_prompt = engine.on_session_start()
    print(f"Skills prompt injected ({len(skills_prompt)} chars)")

    # ... your agent loop here ...

    print(f"Stats: {engine.get_stats()}")


if __name__ == "__main__":
    main()
```

`examples/custom_agent.py`:
```python
"""Example: Quick start with SkillLoop wrapper."""

from __future__ import annotations
from typing import Any

from skill_learner.integrations.generic_agent import create_skill_loop


class MySimpleLLM:
    """Minimal LLM provider for demonstration."""

    def complete(self, messages, tools=None, max_tokens=2048):
        from dataclasses import dataclass, field

        @dataclass
        class R:
            content: str = "Nothing to save."
            tool_calls: list = field(default_factory=list)

        return R()


def my_agent(messages: list[dict], **kwargs: Any) -> str:
    """Your existing agent function."""
    return "Here's your answer!"


def main():
    loop = create_skill_loop(llm=MySimpleLLM(), skills_dir="./my_skills", nudge_interval=5)
    wrapped = loop.wrap(my_agent)

    response = wrapped([{"role": "user", "content": "Help me with something complex"}])
    print(f"Agent response: {response}")
    print(f"Stats: {loop.engine.get_stats()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add examples/
git commit -m "docs: add integration examples for Anthropic, OpenAI, and custom agents"
```

---

### Task 16: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

`README.md` should contain:
- One-line description + badges placeholder
- Quick Install (`pip install skill-learner`)
- 30-second Quick Start code block (SkillLoop wrap pattern)
- How It Works (4-layer diagram in text)
- Event API usage
- CLI commands table
- Platform integrations (Claude Code / Cursor / Generic)
- Configuration reference
- Contributing section
- License (MIT)

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with quick start, architecture, and integration guides"
```

---

### Task 17: Full Test Suite Pass + Final Verification

**Files:** All test files

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: all tests PASS.

- [ ] **Step 2: Verify CLI end-to-end**

```bash
skill-learner list --skills-dir /tmp/test-skills
skill-learner stats --skills-dir /tmp/test-skills
```

- [ ] **Step 3: Verify package can be built**

```bash
pip install build
python -m build
ls dist/
```
Expected: `skill_learner-0.1.0.tar.gz` and `skill_learner-0.1.0-py3-none-any.whl`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: verify full test suite and package build"
```

---

### Task 18: Semantic Search (optional embedding-based retrieval)

**Goal:** Add embedding-based semantic search as an optional upgrade to the keyword search. When an `EmbeddingProvider` is supplied, skills are auto-embedded on save and `search()` uses cosine similarity. Falls back to keyword search when no embedder is configured. Zero new required dependencies.

**Design:**
- `EmbeddingProvider` protocol: user provides their own embed function (Anthropic, OpenAI, sentence-transformers, etc.)
- Embeddings stored as BLOBs in SQLite `skill_embeddings` table
- Pure-Python cosine similarity (no numpy needed at this scale)
- `SQLiteStorage` auto-embeds on `save_skill` when embedder is present
- `PromptInjector.search()` tries semantic first → falls back to keyword

**Files:**
- Create: `src/skill_learner/semantic.py`
- Create: `tests/test_semantic.py`
- Modify: `src/skill_learner/protocols.py`
- Modify: `src/skill_learner/config.py`
- Modify: `src/skill_learner/storage/sqlite.py`
- Modify: `src/skill_learner/injector.py`
- Modify: `src/skill_learner/engine.py`
- Modify: `src/skill_learner/__init__.py`
- Modify: `tests/helpers.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_injector.py`
- Modify: `tests/test_storage_sqlite.py`
- Modify: `pyproject.toml`
- Modify: `examples/anthropic_agent.py`

- [ ] **Step 1: Write tests for semantic.py**

`tests/test_semantic.py`:
```python
from __future__ import annotations

import math

from skill_learner.semantic import (
    cosine_similarity,
    decode_embedding,
    encode_embedding,
    skill_to_text,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert cosine_similarity(a, b) == 0.0

    def test_similar_vectors_high_score(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.9, 0.1]
        assert cosine_similarity(a, b) > 0.95


class TestEncoding:
    def test_roundtrip(self):
        original = [0.1, 0.2, 0.3, -0.5, 1.0]
        encoded = encode_embedding(original)
        decoded = decode_embedding(encoded)
        assert len(decoded) == len(original)
        for a, b in zip(original, decoded):
            assert abs(a - b) < 1e-6

    def test_empty(self):
        assert decode_embedding(encode_embedding([])) == []

    def test_blob_is_bytes(self):
        blob = encode_embedding([1.0, 2.0])
        assert isinstance(blob, bytes)
        assert len(blob) == 8  # 2 floats × 4 bytes


class TestSkillToText:
    def test_basic(self):
        text = skill_to_text("debug-memory", "Memory leak debugging", ["coding", "debug"], "# Steps\n1. Use tracemalloc")
        assert "debug memory" in text  # hyphens become spaces
        assert "Memory leak debugging" in text
        assert "coding" in text
        assert "tracemalloc" in text

    def test_content_truncated(self):
        long_content = "x" * 2000
        text = skill_to_text("t", "d", [], long_content, max_content=100)
        assert len(text) < 200

    def test_empty_tags(self):
        text = skill_to_text("name", "desc", [], "content")
        assert "name" in text
        assert "desc" in text
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_semantic.py -v
```

- [ ] **Step 3: Implement semantic.py**

`src/skill_learner/semantic.py`:
```python
from __future__ import annotations

import math
import struct


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def skill_to_text(
    name: str,
    description: str,
    tags: list[str],
    content: str,
    max_content: int = 500,
) -> str:
    parts = [name.replace("-", " "), description]
    if tags:
        parts.append(" ".join(tags))
    if content:
        parts.append(content[:max_content])
    return " ".join(parts)


def encode_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def decode_embedding(data: bytes) -> list[float]:
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_semantic.py -v
```

- [ ] **Step 5: Add EmbeddingProvider to protocols.py**

Append after the `LLMProvider` protocol:

```python
@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

- [ ] **Step 6: Add config fields to config.py**

Add to `SkillLearnerConfig`:

```python
    semantic_top_k: int = 5
    semantic_threshold: float = 0.3
```

And add to `from_env` int parsing:

```python
        if "semantic_top_k" in data:
            data["semantic_top_k"] = int(data["semantic_top_k"])
```

- [ ] **Step 7: Add MockEmbeddingProvider to tests/helpers.py**

Append to `tests/helpers.py`:

```python
class MockEmbeddingProvider:
    """Mock embedder that produces bag-of-words style embeddings.

    Uses a fixed vocabulary so similar texts yield similar vectors,
    making semantic search testable without a real model.
    """

    VOCAB = [
        "debug", "memory", "leak", "competitor", "analysis",
        "search", "paper", "code", "test", "deploy",
        "api", "database", "web", "file", "error",
        "config", "build", "docker", "git", "review",
    ]

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            lower = text.lower()
            emb = [1.0 if word in lower else 0.0 for word in self.VOCAB]
            norm = sum(x * x for x in emb) ** 0.5
            if norm > 0:
                emb = [x / norm for x in emb]
            results.append(emb)
        return results
```

- [ ] **Step 8: Add mock_embedder fixture to tests/conftest.py**

```python
from tests.helpers import MockEmbeddingProvider

@pytest.fixture
def mock_embedder() -> MockEmbeddingProvider:
    return MockEmbeddingProvider()
```

- [ ] **Step 9: Add embeddings table + methods to SQLiteStorage**

Modify `src/skill_learner/storage/sqlite.py`:

Add new table constant:

```python
_CREATE_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS skill_embeddings (
    name TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    dimension INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
"""
```

Modify `__init__` to accept optional embedder:

```python
class SQLiteStorage:
    def __init__(self, skills_dir: Path | str, embedding_provider: Any = None):
        self._dir = Path(skills_dir).expanduser().resolve()
        self._fs = FileStorage(self._dir)
        self._db_path = self._dir / ".index.db"
        self._embedder = embedding_provider
        self._ensure_schema()
        self._sync_from_disk()
```

Add to `_ensure_schema`:

```python
    conn.execute(_CREATE_EMBEDDINGS_TABLE)
```

Modify `save_skill` to auto-embed:

```python
    def save_skill(self, skill: Skill) -> None:
        self._fs.save_skill(skill)
        self._upsert_index(skill.meta)
        if self._embedder:
            self._auto_embed(skill)
```

Add embedding methods:

```python
    def _auto_embed(self, skill: Skill) -> None:
        from skill_learner.semantic import encode_embedding, skill_to_text
        try:
            text = skill_to_text(
                skill.meta.name, skill.meta.description,
                skill.meta.tags, skill.content,
            )
            vectors = self._embedder.embed([text])
            if vectors and vectors[0]:
                self.save_embedding(skill.meta.name, vectors[0])
        except Exception:
            logger.debug("Auto-embed failed for '%s'", skill.meta.name, exc_info=True)

    def save_embedding(self, name: str, embedding: list[float]) -> None:
        from skill_learner.semantic import encode_embedding
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO skill_embeddings (name, embedding, dimension, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    embedding=excluded.embedding,
                    dimension=excluded.dimension,
                    updated_at=excluded.updated_at
                """,
                (name, encode_embedding(embedding), len(embedding), datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def get_embedding(self, name: str) -> list[float] | None:
        from skill_learner.semantic import decode_embedding
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT embedding FROM skill_embeddings WHERE name = ?", (name,)
            ).fetchone()
            return decode_embedding(row[0]) if row else None
        finally:
            conn.close()

    def get_all_embeddings(self) -> list[tuple[str, list[float]]]:
        from skill_learner.semantic import decode_embedding
        conn = self._connect()
        try:
            rows = conn.execute("SELECT name, embedding FROM skill_embeddings").fetchall()
            return [(r[0], decode_embedding(r[1])) for r in rows]
        finally:
            conn.close()

    def delete_skill(self, name: str) -> None:
        self._fs.delete_skill(name)
        conn = self._connect()
        try:
            conn.execute("DELETE FROM skills WHERE name = ?", (name,))
            conn.execute("DELETE FROM skill_embeddings WHERE name = ?", (name,))
            conn.commit()
        finally:
            conn.close()
```

- [ ] **Step 10: Add SQLiteStorage embedding tests**

Append to `tests/test_storage_sqlite.py`:

```python
class TestSQLiteStorageEmbeddings:
    def test_save_auto_embeds_when_provider_given(self, tmp_skills_dir: Path):
        from tests.helpers import MockEmbeddingProvider
        embedder = MockEmbeddingProvider()
        st = SQLiteStorage(tmp_skills_dir, embedding_provider=embedder)
        st.save_skill(_make_skill("code-review", "coding"))
        emb = st.get_embedding("code-review")
        assert emb is not None
        assert len(emb) == len(MockEmbeddingProvider.VOCAB)

    def test_no_embeddings_without_provider(self, tmp_skills_dir: Path):
        st = SQLiteStorage(tmp_skills_dir)
        st.save_skill(_make_skill())
        emb = st.get_embedding("test-skill")
        assert emb is None

    def test_get_all_embeddings(self, tmp_skills_dir: Path):
        from tests.helpers import MockEmbeddingProvider
        st = SQLiteStorage(tmp_skills_dir, embedding_provider=MockEmbeddingProvider())
        st.save_skill(_make_skill("skill-a", "coding"))
        st.save_skill(_make_skill("skill-b", "general"))
        all_embs = st.get_all_embeddings()
        assert len(all_embs) == 2
        names = {name for name, _ in all_embs}
        assert names == {"skill-a", "skill-b"}

    def test_delete_removes_embedding(self, tmp_skills_dir: Path):
        from tests.helpers import MockEmbeddingProvider
        st = SQLiteStorage(tmp_skills_dir, embedding_provider=MockEmbeddingProvider())
        st.save_skill(_make_skill("to-delete", "general"))
        assert st.get_embedding("to-delete") is not None
        st.delete_skill("to-delete")
        assert st.get_embedding("to-delete") is None
```

- [ ] **Step 11: Enhance PromptInjector with semantic search**

Modify `src/skill_learner/injector.py`:

```python
from __future__ import annotations

from typing import Any

from skill_learner.prompts import SKILLS_GUIDANCE
from skill_learner.skill import Skill


class PromptInjector:
    def __init__(
        self,
        storage: Any,
        embedding_provider: Any = None,
        semantic_top_k: int = 5,
        semantic_threshold: float = 0.3,
    ):
        self._storage = storage
        self._embedder = embedding_provider
        self._top_k = semantic_top_k
        self._threshold = semantic_threshold

    def build_skills_prompt(self) -> str:
        skills = self._storage.list_skills()

        if not skills:
            return SKILLS_GUIDANCE

        by_category: dict[str, list[tuple[str, str]]] = {}
        for s in skills:
            by_category.setdefault(s.category, []).append((s.name, s.description))

        lines = []
        for cat in sorted(by_category):
            lines.append(f"  {cat}:")
            for name, desc in by_category[cat]:
                lines.append(f"    - {name}: {desc}")

        index = "\n".join(lines)
        return f"""## Available Skills (scan before responding)

Before answering, scan the skill list below.
If a skill **matches or relates to your task**, load it with load_skill(name) and follow its guidance.

<available_skills>
{index}
</available_skills>

Only proceed without a skill if none are relevant.

{SKILLS_GUIDANCE}"""

    def search(self, query: str) -> list[Skill]:
        if self._embedder and hasattr(self._storage, "get_all_embeddings"):
            results = self._semantic_search(query)
            if results:
                return results
        return self._keyword_search(query)

    def _semantic_search(self, query: str) -> list[Skill]:
        from skill_learner.semantic import cosine_similarity

        try:
            query_emb = self._embedder.embed([query])[0]
        except Exception:
            return []

        all_embs = self._storage.get_all_embeddings()
        if not all_embs:
            return []

        scored: list[tuple[float, str]] = []
        for name, emb in all_embs:
            sim = cosine_similarity(query_emb, emb)
            if sim >= self._threshold:
                scored.append((sim, name))

        scored.sort(reverse=True)
        results: list[Skill] = []
        for _, name in scored[: self._top_k]:
            skill = self._storage.load_skill(name)
            if skill:
                results.append(skill)
        return results

    def _keyword_search(self, query: str) -> list[Skill]:
        query_terms = set(query.lower().split())
        results: list[tuple[int, Skill]] = []

        skills = self._storage.list_skills()
        for meta in skills:
            searchable = f"{meta.name} {meta.description} {' '.join(meta.tags)}".lower()
            score = sum(1 for term in query_terms if term in searchable)
            if score > 0:
                skill = self._storage.load_skill(meta.name)
                if skill:
                    results.append((score, skill))

        results.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in results]
```

- [ ] **Step 12: Add semantic search tests to test_injector.py**

Append to `tests/test_injector.py`:

```python
from skill_learner.storage.sqlite import SQLiteStorage
from tests.helpers import MockEmbeddingProvider


class TestPromptInjectorSemantic:
    def _setup(self, tmp_skills_dir: Path) -> tuple:
        embedder = MockEmbeddingProvider()
        storage = SQLiteStorage(tmp_skills_dir, embedding_provider=embedder)
        _save_sample_skills_to_storage(storage)
        injector = PromptInjector(storage, embedding_provider=embedder)
        return storage, injector

    def test_semantic_search_finds_related_skill(self, tmp_skills_dir: Path):
        _, injector = self._setup(tmp_skills_dir)
        results = injector.search("how to find memory leaks in code")
        assert len(results) > 0
        assert any(s.meta.name == "debug-memory" for s in results)

    def test_semantic_search_ranks_by_similarity(self, tmp_skills_dir: Path):
        _, injector = self._setup(tmp_skills_dir)
        results = injector.search("debugging code memory issues")
        if len(results) >= 2:
            assert results[0].meta.name == "debug-memory"

    def test_semantic_fallback_to_keyword(self, tmp_skills_dir: Path):
        storage = FileStorage(tmp_skills_dir)
        _save_sample_skills(storage)
        injector = PromptInjector(storage)  # no embedder
        results = injector.search("memory debug")
        assert any(s.meta.name == "debug-memory" for s in results)

    def test_semantic_threshold_filters(self, tmp_skills_dir: Path):
        embedder = MockEmbeddingProvider()
        storage = SQLiteStorage(tmp_skills_dir, embedding_provider=embedder)
        _save_sample_skills_to_storage(storage)
        injector = PromptInjector(storage, embedding_provider=embedder, semantic_threshold=0.99)
        results = injector.search("completely unrelated quantum physics topic")
        assert len(results) == 0


def _save_sample_skills_to_storage(storage) -> None:
    """Save sample skills using storage.save_skill (triggers auto-embed)."""
    from datetime import datetime
    from skill_learner.skill import Skill, SkillMeta
    for name, cat, desc in [
        ("competitor-analysis", "business", "SaaS competitor analysis workflow"),
        ("debug-memory", "coding", "Memory leak debugging steps for code"),
        ("arxiv-search", "research", "Academic paper search flow"),
    ]:
        storage.save_skill(Skill(
            meta=SkillMeta(
                name=name, description=desc, category=cat,
                tags=["test"], created_at=datetime(2026, 4, 20),
                updated_at=datetime(2026, 4, 20),
            ),
            content=f"# {name}\n\nContent for {name}.",
        ))
```

- [ ] **Step 13: Update SkillEngine to accept embedding_provider**

Modify `src/skill_learner/engine.py` constructor:

```python
class SkillEngine:
    def __init__(self, config: SkillLearnerConfig, llm: Any, embedding_provider: Any = None):
        self.config = config
        config.skills_path.mkdir(parents=True, exist_ok=True)

        if config.storage_backend == "sqlite":
            self.storage = SQLiteStorage(config.skills_path, embedding_provider=embedding_provider)
        else:
            self.storage = FileStorage(config.skills_path)

        self.trigger = ReviewTrigger(config.nudge_interval)
        self.reviewer = BackgroundReviewer(llm, self.storage)
        self.injector = PromptInjector(
            self.storage,
            embedding_provider=embedding_provider,
            semantic_top_k=config.semantic_top_k,
            semantic_threshold=config.semantic_threshold,
        )
```

- [ ] **Step 14: Update __init__.py exports**

Add to imports:

```python
from skill_learner.protocols import EmbeddingProvider
```

Add `"EmbeddingProvider"` to `__all__`.

- [ ] **Step 15: Add `embedding` optional dependency to pyproject.toml**

No required dependency — the user brings their own embedder. But add a convenience extra for common providers:

```toml
[project.optional-dependencies]
cli = ["click>=8.0"]
embedding = ["sentence-transformers>=2.0"]
dev = [
    "pytest>=7.0",
    "pytest-mock>=3.0",
]
all = ["skill-learner[cli,embedding,dev]"]
```

- [ ] **Step 16: Add embedding example to examples/anthropic_agent.py**

Append after the existing `AnthropicLLMProvider` class:

```python
class AnthropicEmbeddingProvider:
    """Adapter: Anthropic Voyager embeddings -> skill-learner EmbeddingProvider."""

    def __init__(self, model: str = "voyage-3"):
        try:
            import voyageai
            self._client = voyageai.Client()
        except ImportError:
            self._client = None
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            raise RuntimeError("pip install voyageai")
        result = self._client.embed(texts, model=self._model)
        return result.embeddings
```

And update main():

```python
def main():
    llm = AnthropicLLMProvider()
    # Optional: add embedding for semantic search
    # embedder = AnthropicEmbeddingProvider()
    embedder = None  # set to AnthropicEmbeddingProvider() if voyageai is installed

    config = SkillLearnerConfig(skills_dir="./my_skills", nudge_interval=5)
    engine = SkillEngine(config=config, llm=llm, embedding_provider=embedder)
    # ... rest unchanged
```

- [ ] **Step 17: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASS, including new semantic tests.

- [ ] **Step 18: Commit**

```bash
git add src/skill_learner/semantic.py src/skill_learner/protocols.py src/skill_learner/config.py \
        src/skill_learner/storage/sqlite.py src/skill_learner/injector.py src/skill_learner/engine.py \
        src/skill_learner/__init__.py tests/test_semantic.py tests/test_injector.py \
        tests/test_storage_sqlite.py tests/helpers.py tests/conftest.py \
        pyproject.toml examples/anthropic_agent.py
git commit -m "feat: add optional semantic search with EmbeddingProvider protocol

Embedding-based skill retrieval using cosine similarity.
Auto-embeds skills on save when an EmbeddingProvider is configured.
Falls back to keyword search when no embedder is available.
Zero new required dependencies — user brings their own embedder."
```
