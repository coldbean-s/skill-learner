from __future__ import annotations

import functools
from typing import Any, Callable

from skill_learner.engine import SkillEngine


class SkillLoop:
    def __init__(self, engine: SkillEngine):
        self._engine = engine

    def wrap(self, agent_fn: Callable) -> Callable:
        @functools.wraps(agent_fn)
        def wrapper(messages: list[dict], **kwargs: Any) -> Any:
            skills_prompt = self._engine.on_session_start()
            if skills_prompt:
                has_system = messages and messages[0].get("role") == "system"
                if has_system:
                    messages = list(messages)
                    messages[0] = {
                        **messages[0],
                        "content": messages[0]["content"] + "\n\n" + skills_prompt,
                    }
                else:
                    messages = [{"role": "system", "content": skills_prompt}] + list(messages)

            response = agent_fn(messages, **kwargs)

            self._engine.on_turn_complete(messages)
            return response

        return wrapper

    def inject_prompt(self, base_prompt: str) -> str:
        skills_prompt = self._engine.on_session_start()
        if skills_prompt:
            return base_prompt + "\n\n" + skills_prompt
        return base_prompt

    @property
    def engine(self) -> SkillEngine:
        return self._engine
