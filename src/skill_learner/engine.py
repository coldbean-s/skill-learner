from __future__ import annotations

import logging
from typing import Any

from skill_learner.config import SkillLearnerConfig
from skill_learner.injector import PromptInjector
from skill_learner.reviewer import BackgroundReviewer
from skill_learner.skill import Skill
from skill_learner.storage.filesystem import FileStorage
from skill_learner.storage.sqlite import SQLiteStorage
from skill_learner.trigger import ReviewTrigger

logger = logging.getLogger(__name__)


class SkillEngine:
    def __init__(self, config: SkillLearnerConfig, llm: Any, embedding_provider: Any = None):
        self.config = config
        config.skills_path.mkdir(parents=True, exist_ok=True)

        if config.storage_backend == "sqlite":
            self.storage = SQLiteStorage(config.skills_path, embedding_provider=embedding_provider)
        else:
            self.storage = FileStorage(config.skills_path)

        self.trigger = ReviewTrigger(config.nudge_interval)
        self.reviewer = BackgroundReviewer(llm, self.storage)
        self.injector = PromptInjector(
            self.storage,
            embedding_provider=embedding_provider,
            semantic_top_k=config.semantic_top_k,
            semantic_threshold=config.semantic_threshold,
        )

    def on_tool_call(self, tool_name: str, result: Any = None) -> None:
        try:
            self.trigger.tick_iteration()
        except Exception:
            logger.debug("on_tool_call failed", exc_info=True)

    def on_turn_complete(self, messages: list[dict]) -> None:
        try:
            self.trigger.tick_turn()
            if self.trigger.should_review():
                self.reviewer.submit(messages)
                self.trigger.reset()
        except Exception:
            logger.debug("on_turn_complete failed", exc_info=True)

    def on_session_start(self) -> str:
        try:
            return self.injector.build_skills_prompt()
        except Exception:
            logger.debug("on_session_start failed", exc_info=True)
            return ""

    def on_session_end(self, messages: list[dict]) -> None:
        try:
            self.reviewer.submit_and_wait(messages, timeout=self.config.review_timeout)
        except Exception:
            logger.debug("on_session_end failed", exc_info=True)

    def get_relevant_skills(self, query: str) -> list[Skill]:
        return self.injector.search(query)

    def get_stats(self) -> dict[str, Any]:
        return self.storage.get_stats()
