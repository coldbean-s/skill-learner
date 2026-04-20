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

_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9\-]*[a-z0-9])?$")


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
