# skill-learner Design Spec

**Date:** 2026-04-20
**Status:** Approved
**Scope:** Open-source Python package for agent skill self-accumulation

## Overview

A reusable Python library that implements the Hermes Agent "skill self-accumulation" pattern: AI agents automatically review their work, extract reusable skills, and inject them into future sessions. Inspired by Nous Research's Hermes Agent closed-loop learning mechanism.

**Target:** Open-source on GitHub (PyPI publication). Pure interface abstractions for LLM backend. File + SQLite storage. Multi-platform integrations (Claude Code, Cursor, generic Python agent).

## Architecture

**Pattern: Event-driven engine (B) + Middleware wrapper (A)**

The core is an event-driven `SkillEngine` that receives signals from the host agent. On top sits `SkillLoop`, a convenience wrapper that wraps an agent function in one line. Platform-specific integrations bridge each environment's native mechanism (hooks, rules files, etc.) to the engine's event API.

### Four-Layer Model (from Hermes)

| Layer | Component | Responsibility |
|-------|-----------|----------------|
| L1 Prompt Guidance | `PromptInjector` | Inject skill index + guidance into system prompt |
| L2 Counter Trigger | `ReviewTrigger` | Dual counters (turns / iterations), fire when threshold reached |
| L3 Background Review | `BackgroundReviewer` | Daemon thread, independent review agent, best-effort |
| L4 Skill Injection | `PromptInjector` + `StorageBackend` | Scan skill files, build index, inject on session start |

## Package Structure

```
skill-learner/
├── src/skill_learner/
│   ├── __init__.py              # Public API exports
│   ├── engine.py                # SkillEngine - event-driven core
│   ├── loop.py                  # SkillLoop - convenience wrapper
│   ├── trigger.py               # ReviewTrigger - dual counter logic
│   ├── reviewer.py              # BackgroundReviewer - daemon thread review
│   ├── injector.py              # PromptInjector - system prompt builder
│   ├── skill.py                 # Skill data model (parse/serialize SKILL.md)
│   ├── protocols.py             # LLMProvider / StorageBackend protocols
│   ├── config.py                # Configuration (YAML / dict)
│   ├── prompts.py               # Review prompt templates
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── filesystem.py        # File-based storage (SKILL.md + JSON)
│   │   └── sqlite.py            # SQLite index + metadata + stats
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── claude_code.py       # Claude Code hooks + CLAUDE.md generation
│   │   ├── cursor.py            # Cursor .cursorrules generation
│   │   └── generic_agent.py     # Generic Python agent integration helpers
│   └── cli.py                   # CLI entry point
├── tests/
│   ├── test_engine.py
│   ├── test_trigger.py
│   ├── test_reviewer.py
│   ├── test_injector.py
│   ├── test_skill.py
│   ├── test_storage_fs.py
│   └── test_storage_sqlite.py
├── examples/
│   ├── anthropic_agent.py
│   ├── openai_agent.py
│   └── custom_agent.py
├── pyproject.toml
├── LICENSE                      # MIT
└── README.md
```

## Core Protocols (protocols.py)

All abstractions use `typing.Protocol` — no base class inheritance required.

### LLMProvider

```python
class LLMProvider(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse: ...
```

### LLMResponse / ToolCall

```python
class ToolCall(Protocol):
    name: str
    arguments: dict[str, Any]

class LLMResponse(Protocol):
    content: str
    tool_calls: list[ToolCall]
```

### StorageBackend

```python
class StorageBackend(Protocol):
    def save_skill(self, skill: Skill) -> None: ...
    def load_skill(self, name: str) -> Skill | None: ...
    def list_skills(self) -> list[SkillMeta]: ...
    def update_skill(self, name: str, patch: SkillPatch) -> None: ...
    def delete_skill(self, name: str) -> None: ...
    def record_usage(self, name: str) -> None: ...  # increment use stats
    def get_stats(self) -> dict[str, Any]: ...
```

## Skill Data Model (skill.py)

### SKILL.md Format

```yaml
---
name: competitor-analysis
description: SaaS competitor analysis report workflow
version: 1.0.0
category: business
tags: [analysis, research]
created_at: 2026-04-20T10:00:00
updated_at: 2026-04-20T10:00:00
author: auto-generated
---

# Competitor Analysis Report

## Prerequisites
...
## Workflow
...
## Gotchas
...
```

### Python Model

```python
@dataclass
class SkillMeta:
    name: str
    description: str
    version: str
    category: str
    tags: list[str]
    created_at: datetime
    updated_at: datetime

@dataclass
class Skill:
    meta: SkillMeta
    content: str  # full markdown body below frontmatter

    MAX_SIZE = 100_000  # characters

@dataclass
class SkillPatch:
    description: str | None = None
    content: str | None = None
    tags: list[str] | None = None
    bump_version: bool = True
```

Frontmatter parsed with PyYAML (required dependency — single pure-Python package, no transitive deps). Content is everything below the second `---`.

## Engine (engine.py)

Central coordinator. Holds references to trigger, reviewer, injector, storage.

```python
class SkillEngine:
    def __init__(self, config: SkillLearnerConfig):
        self.storage: StorageBackend
        self.llm: LLMProvider
        self.trigger: ReviewTrigger
        self.reviewer: BackgroundReviewer
        self.injector: PromptInjector

    # Event API
    def on_tool_call(self, tool_name: str, result: Any = None) -> None
    def on_turn_complete(self, messages: list[dict]) -> None
        # After should_review() returns True, immediately calls:
        #   self.reviewer.submit(messages)
        #   self.trigger.reset()
        # Reset is always called right after submit, making lifecycle unambiguous.
    def on_session_start(self) -> str  # returns prompt fragment (sync, blocks until index loaded)
    def on_session_end(self, messages: list[dict]) -> None
        # Force-submits a review and waits with configurable timeout (default 30s).
        # If review does not complete within timeout, it is abandoned.

    # Query API
    def get_relevant_skills(self, query: str) -> list[Skill]
    def get_stats(self) -> dict[str, Any]
```

## Trigger (trigger.py)

Dual counter with configurable threshold.

```python
class ReviewTrigger:
    def __init__(self, nudge_interval: int = 10):
        self._iters: int = 0
        self._turns: int = 0
        self._nudge_interval: int = nudge_interval

    def tick_iteration(self) -> None
    def tick_turn(self) -> None
    def should_review(self) -> bool
    def reset(self) -> None
```

`should_review()` returns True when either counter >= threshold.

## Reviewer (reviewer.py)

Spawns daemon thread, sends conversation snapshot + review prompt to LLM, processes tool_call responses to save/update skills.

**Concurrency guard:** A `threading.Lock` ensures only one review runs at a time. If `submit()` is called while a review is in flight, the call is silently dropped (best-effort: missing one review is acceptable).

```python
class BackgroundReviewer:
    REVIEW_PROMPT: str  # from prompts.py

    def __init__(self, llm: LLMProvider, storage: StorageBackend):
        self._lock = threading.Lock()
        ...

    def submit(self, messages: list[dict]) -> None
        # Acquire lock non-blocking; if already held, return (skip)
        # list() snapshot, spawn daemon thread
        # Thread releases lock in finally block

    def _run_review(self, messages: list[dict]) -> None
        # try/except pass — best-effort, lock released in finally
        # Append REVIEW_PROMPT to messages
        # Call llm.complete() with skill_manage tool schemas
        # Parse tool_calls, execute save/update/patch via storage
```

**Name collision policy:** `save_skill` checks if a skill with the same name already exists. If it does, the operation is automatically delegated to `update_skill` (merge semantics). The tool schema description tells the review LLM about this behavior so it can choose the right tool intentionally.

Tool schemas exposed to the review LLM:
- `save_skill(name, category, description, content)` → create new (auto-merges if exists)
- `update_skill(name, patch_content)` → patch existing
- `delete_skill(name, reason)` → remove obsolete/wrong skill
- `nothing_to_save()` → no-op signal

## Injector (injector.py)

Builds the prompt fragment injected at session start.

```python
class PromptInjector:
    SKILLS_GUIDANCE: str  # from prompts.py

    def __init__(self, storage: StorageBackend):
        self._cache: dict[str, Skill] = {}  # LRU-like

    def build_skills_prompt(self) -> str
        # List all skills from storage (rebuilds cache on each call)
        # Cache is per-call, no invalidation logic needed
        # Format as indexed prompt block
        # Append SKILLS_GUIDANCE

    def search(self, query: str) -> list[Skill]
        # Keyword match against name + description + tags
```

## Storage

### FileStorage (storage/filesystem.py)

```
{skills_dir}/
├── {category}/
│   └── {skill_name}/
│       └── SKILL.md
└── .index.json          # cached index for fast listing
```

- `save_skill()`: create directory + write SKILL.md + update .index.json
- `load_skill()`: parse SKILL.md frontmatter + body
- `list_skills()`: read .index.json (rebuild from disk if stale via mtime check)
- Validate: name format, size limit (100k chars), required frontmatter fields

### SQLiteStorage (storage/sqlite.py)

Wraps FileStorage, adds SQLite index at `{skills_dir}/.index.db`.

Tables:
```sql
CREATE TABLE skills (
    name TEXT PRIMARY KEY,
    description TEXT,
    category TEXT,
    tags TEXT,          -- JSON array
    version TEXT,
    created_at TEXT,
    updated_at TEXT,
    use_count INTEGER DEFAULT 0,
    last_used_at TEXT
);
```

- All writes go through FileStorage first, then sync to SQLite
- If SQLite sync fails after file write succeeds, log a warning and trigger index rebuild on next `list_skills()` call (consistent with best-effort philosophy)
- `get_stats()`: query use_count, last_used_at, popular skills, etc.
- `record_usage(name)`: explicitly increment use_count — only called when a skill is actually injected into a live session prompt (NOT on `load_skill()` reads)
- Schema versioning via `PRAGMA user_version`, sequential migration files for future schema changes

## Integrations

### Claude Code (integrations/claude_code.py)

`init_claude_code(skills_dir, project_dir)`:
1. Generate `.claude/settings.json` hooks:
   - `PostToolCall`: shell script that increments counter in temp file
   - Session end hook: Python script that reads counter, triggers review if threshold reached
2. Append to CLAUDE.md:
   - Skills guidance prompt
   - Skill index (auto-generated)
3. Generate helper scripts in `.claude/skill-learner/`:
   - `counter.sh`: increment counter
   - `review.py`: trigger review

### Cursor (integrations/cursor.py)

`init_cursor(skills_dir, project_dir)`:
1. Generate/append `.cursorrules` with:
   - Skills guidance prompt
   - Skill index
   - Manual review instruction ("after complex tasks, run `skill-learner review`")

### Generic Agent (integrations/generic_agent.py)

Provides `SkillLoop` wrapper:

```python
class SkillLoop:
    def __init__(self, engine: SkillEngine): ...

    def wrap(self, agent_fn: Callable) -> Callable:
        """Wrap an agent function with automatic skill learning.

        Contract: agent_fn must accept (messages: list[dict], **kwargs) as arguments
        and return a response (str or dict with 'content' key).
        The wrapper:
        1. Calls engine.on_session_start() and prepends skills prompt as system message
        2. Calls agent_fn(messages, **kwargs)
        3. Calls engine.on_turn_complete(messages) after each call
        """

    def inject_prompt(self, base_prompt: str) -> str:
        """Convenience: inject skills into a prompt string.
        Primary API for users who prefer manual integration over wrap()."""
```

## CLI (cli.py)

Entry point: `skill-learner` or `python -m skill_learner`

Commands:
- `init --target {claude-code|cursor|generic} [--skills-dir PATH]` — initialize for a platform
- `list [--category CAT]` — list all skills
- `show NAME` — display full skill content
- `stats` — usage statistics
- `review --messages FILE` — manually trigger review from a conversation JSON (file must contain a JSON array of message objects with `role` and `content` fields, matching LLMProvider.complete() format)
- `export --format {json|markdown} [--output PATH]` — export skill library
- `import FILE` — import skills from export

## Configuration (config.py)

```python
@dataclass
class SkillLearnerConfig:
    skills_dir: str = "~/.skill_learner/skills"
    nudge_interval: int = 10
    max_skill_size: int = 100_000
    storage_backend: str = "sqlite"  # "filesystem" | "sqlite"
    review_max_tokens: int = 2048
    review_model_hint: str = ""  # passed to LLMProvider, usage is provider-specific
    review_timeout: int = 30  # seconds to wait for review on session_end
    categories: list[str] = field(default_factory=lambda: [
        "business", "coding", "research", "ops", "general"
    ])
```

Loadable from: Python dict, YAML file (`skill_learner.yaml`), or environment variables (`SKILL_LEARNER_*`).

## Dependencies

**Core:**
- `typing`, `dataclasses`, `threading`, `sqlite3`, `json`, `pathlib`, `re` — all stdlib
- `pyyaml` — required, for YAML config and frontmatter parsing (pure Python, no transitive deps)

**Optional:**
- `click` — for CLI (fallback: `argparse`)

**Dev:**
- `pytest`, `pytest-mock`

## Error Handling

- All review operations are best-effort: `try/except` with silent failure
- Storage operations raise `SkillLearnerError` subclasses for caller visibility
- CLI surfaces errors with clear messages
- Engine never crashes the host agent — all event handlers are wrapped in try/except

## Testing Strategy

- Unit tests for each module (trigger, skill parsing, storage CRUD)
- Integration test: full engine cycle (tool calls → trigger → review → skill saved → injected)
- Mock LLMProvider for tests (returns predefined tool_calls)
- Temp directories for storage tests
