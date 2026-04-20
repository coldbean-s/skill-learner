from __future__ import annotations

from typing import Any

from skill_learner.config import SkillLearnerConfig
from skill_learner.engine import SkillEngine
from skill_learner.loop import SkillLoop


def create_skill_loop(
    llm: Any,
    skills_dir: str = "~/.skill_learner/skills",
    nudge_interval: int = 10,
    storage_backend: str = "sqlite",
    embedding_provider: Any = None,
) -> SkillLoop:
    config = SkillLearnerConfig(
        skills_dir=skills_dir,
        nudge_interval=nudge_interval,
        storage_backend=storage_backend,
    )
    engine = SkillEngine(config=config, llm=llm, embedding_provider=embedding_provider)
    return SkillLoop(engine)
