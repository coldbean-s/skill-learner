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
