from __future__ import annotations

import time
from pathlib import Path

import pytest

from skill_learner.config import SkillLearnerConfig
from skill_learner.engine import SkillEngine
from tests.helpers import MockLLMProvider, MockLLMResponse, MockToolCall


class TestSkillEngine:
    def _make_engine(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider, nudge: int = 3) -> SkillEngine:
        config = SkillLearnerConfig(
            skills_dir=str(tmp_skills_dir),
            nudge_interval=nudge,
            storage_backend="filesystem",
        )
        return SkillEngine(config=config, llm=mock_llm)

    def test_on_session_start_returns_prompt(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider):
        engine = self._make_engine(tmp_skills_dir, mock_llm)
        prompt = engine.on_session_start()
        assert "Skill Accumulation" in prompt

    def test_tool_calls_trigger_review(self, tmp_skills_dir: Path):
        mock_llm = MockLLMProvider([
            MockLLMResponse(
                tool_calls=[MockToolCall(
                    name="save_skill",
                    arguments={
                        "name": "auto-skill",
                        "category": "general",
                        "description": "Auto generated",
                        "content": "# Auto\n\nContent",
                    },
                )],
            ),
        ])
        engine = self._make_engine(tmp_skills_dir, mock_llm, nudge=3)

        for _ in range(3):
            engine.on_tool_call("web_search")

        messages = [{"role": "user", "content": "do something complex"}]
        engine.on_turn_complete(messages)
        time.sleep(1)

        skill = engine.storage.load_skill("auto-skill")
        assert skill is not None

    def test_below_threshold_no_review(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider):
        engine = self._make_engine(tmp_skills_dir, mock_llm, nudge=10)
        engine.on_tool_call("test")
        engine.on_turn_complete([{"role": "user", "content": "hi"}])
        assert len(mock_llm.calls) == 0

    def test_on_session_end_forces_review(self, tmp_skills_dir: Path):
        mock_llm = MockLLMProvider([
            MockLLMResponse(
                tool_calls=[MockToolCall(name="nothing_to_save", arguments={})],
            ),
        ])
        engine = self._make_engine(tmp_skills_dir, mock_llm, nudge=999)
        engine.on_session_end([{"role": "user", "content": "done"}])
        assert len(mock_llm.calls) == 1

    def test_get_stats(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider):
        engine = self._make_engine(tmp_skills_dir, mock_llm)
        stats = engine.get_stats()
        assert "total_skills" in stats
