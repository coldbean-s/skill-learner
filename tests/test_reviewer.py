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
