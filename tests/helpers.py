from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class MockLLMResponse:
    content: str = ""
    tool_calls: list[MockToolCall] = field(default_factory=list)


class MockLLMProvider:
    """Mock LLM that returns pre-configured responses."""

    def __init__(self, responses: list[MockLLMResponse] | None = None):
        self._responses = list(responses or [])
        self._call_count = 0
        self.calls: list[dict] = []

    def add_response(self, response: MockLLMResponse) -> None:
        self._responses.append(response)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
    ) -> MockLLMResponse:
        self.calls.append({"messages": messages, "tools": tools, "max_tokens": max_tokens})
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return MockLLMResponse(content="Nothing to save.")
