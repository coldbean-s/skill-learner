"""Example: Quick start with SkillLoop wrapper."""

from __future__ import annotations
from typing import Any

from skill_learner.integrations.generic_agent import create_skill_loop


class MySimpleLLM:
    """Minimal LLM provider for demonstration."""

    def complete(self, messages, tools=None, max_tokens=2048):
        from dataclasses import dataclass, field

        @dataclass
        class R:
            content: str = "Nothing to save."
            tool_calls: list = field(default_factory=list)

        return R()


def my_agent(messages: list[dict], **kwargs: Any) -> str:
    """Your existing agent function."""
    return "Here's your answer!"


def main():
    loop = create_skill_loop(llm=MySimpleLLM(), skills_dir="./my_skills", nudge_interval=5)
    wrapped = loop.wrap(my_agent)

    response = wrapped([{"role": "user", "content": "Help me with something complex"}])
    print(f"Agent response: {response}")
    print(f"Stats: {loop.engine.get_stats()}")


if __name__ == "__main__":
    main()
