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
