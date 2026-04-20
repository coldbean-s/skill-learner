from __future__ import annotations

from pathlib import Path

from skill_learner.config import SkillLearnerConfig
from skill_learner.engine import SkillEngine
from skill_learner.loop import SkillLoop
from tests.helpers import MockLLMProvider


class TestSkillLoop:
    def _make_loop(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider) -> SkillLoop:
        config = SkillLearnerConfig(skills_dir=str(tmp_skills_dir), nudge_interval=99, storage_backend="filesystem")
        engine = SkillEngine(config=config, llm=mock_llm)
        return SkillLoop(engine)

    def test_wrap_injects_system_message_when_none_exists(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider):
        loop = self._make_loop(tmp_skills_dir, mock_llm)
        captured = {}

        def agent_fn(messages, **kwargs):
            captured["messages"] = messages
            return "ok"

        wrapped = loop.wrap(agent_fn)
        wrapped([{"role": "user", "content": "hello"}])
        assert captured["messages"][0]["role"] == "system"
        assert "Skill Accumulation" in captured["messages"][0]["content"]

    def test_wrap_appends_to_existing_system_message(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider):
        loop = self._make_loop(tmp_skills_dir, mock_llm)
        captured = {}

        def agent_fn(messages, **kwargs):
            captured["messages"] = messages
            return "ok"

        wrapped = loop.wrap(agent_fn)
        wrapped([{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}])
        assert "You are helpful." in captured["messages"][0]["content"]
        assert "Skill Accumulation" in captured["messages"][0]["content"]

    def test_inject_prompt_appends_skills(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider):
        loop = self._make_loop(tmp_skills_dir, mock_llm)
        result = loop.inject_prompt("Base prompt here.")
        assert "Base prompt here." in result
        assert "Skill Accumulation" in result

    def test_inject_prompt_returns_base_when_empty(self, tmp_skills_dir: Path, mock_llm: MockLLMProvider):
        loop = self._make_loop(tmp_skills_dir, mock_llm)
        result = loop.inject_prompt("")
        assert "Skill Accumulation" in result
