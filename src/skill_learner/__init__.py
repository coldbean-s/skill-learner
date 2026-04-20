"""skill-learner: Self-accumulating skill system for AI agents."""

__version__ = "0.1.0"

from skill_learner.config import SkillLearnerConfig
from skill_learner.engine import SkillEngine
from skill_learner.injector import PromptInjector
from skill_learner.loop import SkillLoop
from skill_learner.protocols import EmbeddingProvider, LLMProvider, LLMResponse, StorageBackend, ToolCall
from skill_learner.reviewer import BackgroundReviewer
from skill_learner.skill import Skill, SkillLearnerError, SkillMeta, SkillPatch
from skill_learner.storage import FileStorage, SQLiteStorage
from skill_learner.trigger import ReviewTrigger

__all__ = [
    "BackgroundReviewer",
    "EmbeddingProvider",
    "FileStorage",
    "LLMProvider",
    "LLMResponse",
    "PromptInjector",
    "ReviewTrigger",
    "Skill",
    "SkillEngine",
    "SkillLearnerConfig",
    "SkillLearnerError",
    "SkillLoop",
    "SkillMeta",
    "SkillPatch",
    "SQLiteStorage",
    "StorageBackend",
    "ToolCall",
]
