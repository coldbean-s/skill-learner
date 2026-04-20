from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Any

from skill_learner.prompts import REVIEW_PROMPT, SKILL_MANAGE_TOOLS
from skill_learner.skill import Skill, SkillMeta, SkillPatch

if TYPE_CHECKING:
    from skill_learner.protocols import LLMProvider, StorageBackend

logger = logging.getLogger(__name__)


class BackgroundReviewer:
    def __init__(self, llm: LLMProvider, storage: StorageBackend):
        self._llm = llm
        self._storage = storage
        self._lock = threading.Lock()

    def submit(self, messages: list[dict]) -> None:
        if not self._lock.acquire(blocking=False):
            logger.debug("Review already in flight, skipping")
            return

        snapshot = list(messages)
        t = threading.Thread(
            target=self._run_review_guarded,
            args=(snapshot,),
            daemon=True,
            name="skill-review",
        )
        t.start()

    def submit_and_wait(self, messages: list[dict], timeout: float = 30) -> None:
        if not self._lock.acquire(blocking=False):
            logger.debug("Review already in flight, skipping")
            return

        snapshot = list(messages)
        t = threading.Thread(
            target=self._run_review_guarded,
            args=(snapshot,),
            daemon=True,
            name="skill-review-wait",
        )
        t.start()
        t.join(timeout=timeout)

    def _run_review_guarded(self, messages: list[dict]) -> None:
        try:
            self._run_review(messages)
        except Exception:
            logger.debug("Review failed (best-effort)", exc_info=True)
        finally:
            self._lock.release()

    def _run_review(self, messages: list[dict]) -> None:
        review_messages = messages + [{"role": "user", "content": REVIEW_PROMPT}]
        response = self._llm.complete(
            messages=review_messages,
            tools=SKILL_MANAGE_TOOLS,
            max_tokens=2048,
        )

        for tc in response.tool_calls:
            self._handle_tool_call(tc.name, tc.arguments)

    def _handle_tool_call(self, name: str, args: dict[str, Any]) -> None:
        if name == "save_skill":
            skill_name = args["name"]
            existing = self._storage.load_skill(skill_name)
            if existing:
                self._storage.update_skill(
                    skill_name,
                    SkillPatch(
                        description=args.get("description"),
                        content=args.get("content"),
                    ),
                )
                logger.info("Updated existing skill: %s", skill_name)
            else:
                skill = Skill(
                    meta=SkillMeta(
                        name=skill_name,
                        description=args["description"],
                        category=args.get("category", "general"),
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                    ),
                    content=args["content"],
                )
                self._storage.save_skill(skill)
                logger.info("Created new skill: %s", skill_name)

        elif name == "update_skill":
            self._storage.update_skill(
                args["name"],
                SkillPatch(
                    content=args.get("patch_content"),
                    description=args.get("description"),
                ),
            )
            logger.info("Patched skill: %s", args["name"])

        elif name == "delete_skill":
            self._storage.delete_skill(args["name"])
            logger.info("Deleted skill: %s (reason: %s)", args["name"], args.get("reason", ""))

        elif name == "nothing_to_save":
            logger.debug("Review decided nothing to save")
