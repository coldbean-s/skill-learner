"""Example: Using skill-learner with the Anthropic SDK."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# pip install anthropic skill-learner
from anthropic import Anthropic

from skill_learner import SkillEngine, SkillLearnerConfig


@dataclass
class AnthropicToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class AnthropicResponse:
    content: str = ""
    tool_calls: list[AnthropicToolCall] = field(default_factory=list)


class AnthropicLLMProvider:
    """Adapter: Anthropic SDK -> skill-learner LLMProvider protocol."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self._client = Anthropic()
        self._model = model

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
    ) -> AnthropicResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]

        response = self._client.messages.create(**kwargs)

        content = ""
        tc_list: list[AnthropicToolCall] = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tc_list.append(AnthropicToolCall(name=block.name, arguments=block.input))

        return AnthropicResponse(content=content, tool_calls=tc_list)

    @staticmethod
    def _convert_tool(tool: dict) -> dict:
        func = tool.get("function", tool)
        return {
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {}),
        }


class AnthropicEmbeddingProvider:
    """Adapter: Voyage AI embeddings -> skill-learner EmbeddingProvider.

    pip install voyageai
    """

    def __init__(self, model: str = "voyage-3"):
        import voyageai
        self._client = voyageai.Client()
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(texts, model=self._model)
        return result.embeddings


def main():
    llm = AnthropicLLMProvider()
    # Optional: add embedding provider for semantic skill search
    # embedder = AnthropicEmbeddingProvider()
    embedder = None

    config = SkillLearnerConfig(skills_dir="./my_skills", nudge_interval=5)
    engine = SkillEngine(config=config, llm=llm, embedding_provider=embedder)

    skills_prompt = engine.on_session_start()
    print(f"Skills prompt ({len(skills_prompt)} chars) injected")

    engine.on_tool_call("web_search")
    engine.on_tool_call("file_write")

    messages = [
        {"role": "user", "content": "Help me analyze competitor pricing"},
        {"role": "assistant", "content": "I'll research competitor pricing..."},
    ]
    engine.on_turn_complete(messages)

    engine.on_session_end(messages)

    print(f"Stats: {engine.get_stats()}")


if __name__ == "__main__":
    main()
