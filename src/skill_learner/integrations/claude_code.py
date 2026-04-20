from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from skill_learner.storage.filesystem import FileStorage

CLAUDE_CODE_SKILLS_GUIDANCE = """\
## Skill Accumulation

You have access to a **self-accumulating skill library** at `{skills_dir}`.

### Reading Skills
At the start of any non-trivial task, scan the skills directory for relevant SKILL.md files.
Each skill has YAML frontmatter (name, description, category, tags) followed by markdown content.
If a matching skill exists, read it and follow its workflow before improvising.

### Saving New Skills
After completing a **complex task** (5+ tool calls), fixing a tricky bug, or discovering
a non-obvious workflow, save the approach as a reusable skill:

1. Choose a kebab-case name (e.g. `memory-leak-debugging`)
2. Pick a category: `coding`, `ops`, `research`, `business`, or `general`
3. Create the file at `{skills_dir}/<category>/<name>/SKILL.md`
4. Use this format:

```
---
name: <name>
description: <one-line description>
version: 1.0.0
category: <category>
tags: [<tag1>, <tag2>]
created_at: <ISO timestamp>
updated_at: <ISO timestamp>
author: auto-generated
---

<Markdown content: prerequisites, workflow steps, gotchas>
```

### Updating Skills
When using an existing skill and finding it **outdated, incomplete, or wrong**,
update the SKILL.md file directly. Bump the version and updated_at timestamp.

### What to Save
- Non-obvious debugging approaches (the fix that took 10 minutes to find)
- Multi-step workflows with specific ordering or gotchas
- API quirks, library-specific workarounds
- Project-specific patterns that would trip up a fresh session

### What NOT to Save
- Trivial operations (standard CRUD, obvious config)
- One-off fixes unlikely to recur
- Information already in project docs or CLAUDE.md"""


def init_claude_code(
    skills_dir: str = "~/.skill_learner/skills",
    project_dir: str = ".",
) -> dict[str, str]:
    project = Path(project_dir).resolve()
    skills_path = Path(skills_dir).expanduser().resolve()
    created: dict[str, str] = {}

    claude_dir = project / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)

    skills_path.mkdir(parents=True, exist_ok=True)
    for cat in ("coding", "ops", "research", "business", "general"):
        (skills_path / cat).mkdir(exist_ok=True)

    storage = FileStorage(skills_path)
    skills = storage.list_skills()
    skills_index = ""
    if skills:
        lines = []
        for s in skills:
            lines.append(f"- **{s.name}** (`{s.category}`): {s.description}")
        skills_index = "\n".join(lines)

    skills_dir_display = str(skills_path).replace("\\", "/")
    guidance = CLAUDE_CODE_SKILLS_GUIDANCE.replace("{skills_dir}", skills_dir_display)
    skill_block = _build_claude_md_block(guidance, skills_index)

    claude_md = project / "CLAUDE.md"
    if claude_md.exists():
        existing = claude_md.read_text(encoding="utf-8")
        marker_start = "<!-- skill-learner:start -->"
        marker_end = "<!-- skill-learner:end -->"
        if marker_start in existing:
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

    return created


def _build_claude_md_block(guidance: str, skills_index: str) -> str:
    index_section = ""
    if skills_index:
        index_section = f"""
## Available Skills

{skills_index}

When a task relates to a listed skill, read and follow it before proceeding.
"""

    return f"""<!-- skill-learner:start -->
## Skill Learning System

{guidance}
{index_section}
<!-- skill-learner:end -->"""
