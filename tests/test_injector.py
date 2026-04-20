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


# --- Semantic search tests ---

from skill_learner.storage.sqlite import SQLiteStorage
from tests.helpers import MockEmbeddingProvider


_SAMPLE_SKILLS = [
    ("competitor-analysis", "business", "SaaS competitor analysis workflow"),
    ("debug-memory", "coding", "Memory leak debugging steps for code"),
    ("arxiv-search", "research", "Academic paper search flow"),
]


def _save_sample_skills_to_storage(storage) -> None:
    for name, cat, desc in _SAMPLE_SKILLS:
        storage.save_skill(Skill(
            meta=SkillMeta(
                name=name, description=desc, category=cat,
                tags=["test"], created_at=datetime(2026, 4, 20),
                updated_at=datetime(2026, 4, 20),
            ),
            content=f"# {name}\n\nContent for {name}.",
        ))


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
        injector = PromptInjector(storage)
        results = injector.search("memory debug")
        assert any(s.meta.name == "debug-memory" for s in results)

    def test_semantic_threshold_filters(self, tmp_skills_dir: Path):
        embedder = MockEmbeddingProvider()
        storage = SQLiteStorage(tmp_skills_dir, embedding_provider=embedder)
        _save_sample_skills_to_storage(storage)
        injector = PromptInjector(storage, embedding_provider=embedder, semantic_threshold=0.99)
        results = injector.search("completely unrelated quantum physics topic")
        assert len(results) == 0
