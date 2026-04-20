from __future__ import annotations

import logging
from typing import Any

from skill_learner.prompts import SKILLS_GUIDANCE
from skill_learner.skill import Skill

logger = logging.getLogger(__name__)


class PromptInjector:
    def __init__(
        self,
        storage: Any,
        embedding_provider: Any = None,
        semantic_top_k: int = 5,
        semantic_threshold: float = 0.3,
    ):
        self._storage = storage
        self._embedder = embedding_provider
        self._top_k = semantic_top_k
        self._threshold = semantic_threshold

    def build_skills_prompt(self) -> str:
        skills = self._storage.list_skills()

        if not skills:
            return SKILLS_GUIDANCE

        by_category: dict[str, list[tuple[str, str]]] = {}
        for s in skills:
            by_category.setdefault(s.category, []).append((s.name, s.description))

        lines = []
        for cat in sorted(by_category):
            lines.append(f"  {cat}:")
            for name, desc in by_category[cat]:
                lines.append(f"    - {name}: {desc}")

        index = "\n".join(lines)
        return f"""## Available Skills (scan before responding)

Before answering, scan the skill list below.
If a skill **matches or relates to your task**, load it with load_skill(name) and follow its guidance.

<available_skills>
{index}
</available_skills>

Only proceed without a skill if none are relevant.

{SKILLS_GUIDANCE}"""

    def search(self, query: str) -> list[Skill]:
        if self._embedder and hasattr(self._storage, "get_all_embeddings"):
            results = self._semantic_search(query)
            if results:
                return results
        return self._keyword_search(query)

    def _semantic_search(self, query: str) -> list[Skill]:
        from skill_learner.semantic import cosine_similarity

        try:
            query_emb = self._embedder.embed([query])[0]
        except Exception:
            logger.debug("Embedding query failed, falling back to keyword", exc_info=True)
            return []

        all_embs = self._storage.get_all_embeddings()
        if not all_embs:
            return []

        scored: list[tuple[float, str]] = []
        for name, emb in all_embs:
            sim = cosine_similarity(query_emb, emb)
            if sim >= self._threshold:
                scored.append((sim, name))

        scored.sort(reverse=True)
        results: list[Skill] = []
        for _, name in scored[: self._top_k]:
            skill = self._storage.load_skill(name)
            if skill:
                results.append(skill)
        return results

    def _keyword_search(self, query: str) -> list[Skill]:
        query_terms = set(query.lower().split())
        results: list[tuple[int, Skill]] = []

        skills = self._storage.list_skills()
        for meta in skills:
            searchable = f"{meta.name} {meta.description} {' '.join(meta.tags)}".lower()
            score = sum(1 for term in query_terms if term in searchable)
            if score > 0:
                skill = self._storage.load_skill(meta.name)
                if skill:
                    results.append((score, skill))

        results.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in results]
