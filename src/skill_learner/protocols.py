from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from skill_learner.skill import Skill, SkillMeta, SkillPatch


@runtime_checkable
class ToolCall(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def arguments(self) -> dict[str, Any]: ...


@runtime_checkable
class LLMResponse(Protocol):
    @property
    def content(self) -> str: ...
    @property
    def tool_calls(self) -> list[ToolCall]: ...


@runtime_checkable
class LLMProvider(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse: ...


class StorageBackend(Protocol):
    def save_skill(self, skill: Skill) -> None: ...
    def load_skill(self, name: str) -> Skill | None: ...
    def list_skills(self) -> list[SkillMeta]: ...
    def update_skill(self, name: str, patch: SkillPatch) -> None: ...
    def delete_skill(self, name: str) -> None: ...
    def record_usage(self, name: str) -> None: ...
    def get_stats(self) -> dict[str, Any]: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
