"""Example: Using skill-learner with the OpenAI SDK."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# pip install openai skill-learner
from openai import OpenAI

from skill_learner import SkillEngine, SkillLearnerConfig


@dataclass
class OpenAIToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class OpenAIResponse:
    content: str = ""
    tool_calls: list[OpenAIToolCall] = field(default_factory=list)


class OpenAILLMProvider:
    """Adapter: OpenAI SDK -> skill-learner LLMProvider protocol."""

    def __init__(self, model: str = "gpt-4o"):
        self._client = OpenAI()
        self._model = model

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
    ) -> OpenAIResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        tc_list: list[OpenAIToolCall] = []
        if msg.tool_calls:
            import json
            for tc in msg.tool_calls:
                tc_list.append(OpenAIToolCall(
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return OpenAIResponse(content=msg.content or "", tool_calls=tc_list)


def main():
    llm = OpenAILLMProvider()
    config = SkillLearnerConfig(skills_dir="./my_skills", nudge_interval=5)
    engine = SkillEngine(config=config, llm=llm)

    skills_prompt = engine.on_session_start()
    print(f"Skills prompt injected ({len(skills_prompt)} chars)")

    print(f"Stats: {engine.get_stats()}")


if __name__ == "__main__":
    main()
