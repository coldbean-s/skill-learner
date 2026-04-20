from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SkillLearnerConfig:
    skills_dir: str = "~/.skill_learner/skills"
    nudge_interval: int = 10
    max_skill_size: int = 100_000
    storage_backend: str = "sqlite"
    review_max_tokens: int = 2048
    review_model_hint: str = ""
    review_timeout: int = 30
    semantic_top_k: int = 5
    semantic_threshold: float = 0.3
    categories: list[str] = field(default_factory=lambda: [
        "business", "coding", "research", "ops", "general",
    ])

    @property
    def skills_path(self) -> Path:
        return Path(self.skills_dir).expanduser().resolve()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillLearnerConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SkillLearnerConfig:
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        return cls.from_dict(data)

    @classmethod
    def from_env(cls, prefix: str = "SKILL_LEARNER_") -> SkillLearnerConfig:
        data: dict[str, Any] = {}
        for key, val in os.environ.items():
            if key.startswith(prefix):
                field_name = key[len(prefix):].lower()
                data[field_name] = val
        for int_field in ("nudge_interval", "max_skill_size", "review_max_tokens", "review_timeout", "semantic_top_k"):
            if int_field in data:
                data[int_field] = int(data[int_field])
        return cls.from_dict(data)
