from __future__ import annotations

import re
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
