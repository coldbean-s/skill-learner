from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from skill_learner.prompts import SKILLS_GUIDANCE
from skill_learner.storage.filesystem import FileStorage


def init_claude_code(
    skills_dir: str = "~/.skill_learner/skills",
    project_dir: str = ".",
) -> dict[str, str]:
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
        'from pathlib import Path\n\n'
        'COUNTER_FILE = Path("/tmp/skill_learner_counter")\n'
        'THRESHOLD = 10\n\n'
        'def check_and_review():\n'
        '    if not COUNTER_FILE.exists():\n'
        '        return\n'
        '    count = int(COUNTER_FILE.read_text().strip())\n'
        '    if count >= THRESHOLD:\n'
        '        COUNTER_FILE.write_text("0")\n'
        '        print(f"[skill-learner] Threshold reached ({count}), review triggered")\n\n'
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
