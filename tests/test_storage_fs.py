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
