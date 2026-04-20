from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from skill_learner.skill import Skill, SkillMeta, SkillPatch
from skill_learner.storage.filesystem import FileStorage

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 2

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS skills (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    tags TEXT NOT NULL DEFAULT '[]',
    version TEXT NOT NULL DEFAULT '1.0.0',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    use_count INTEGER NOT NULL DEFAULT 0,
    last_used_at TEXT
);
"""

_CREATE_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS skill_embeddings (
    name TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    dimension INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class SQLiteStorage:
    def __init__(self, skills_dir: Path | str, embedding_provider: Any = None):
        self._dir = Path(skills_dir).expanduser().resolve()
        self._fs = FileStorage(self._dir)
        self._db_path = self._dir / ".index.db"
        self._embedder = embedding_provider
        self._ensure_schema()
        self._sync_from_disk()

    def save_skill(self, skill: Skill) -> None:
        self._fs.save_skill(skill)
        self._upsert_index(skill.meta)
        if self._embedder:
            self._auto_embed(skill)

    def load_skill(self, name: str) -> Skill | None:
        return self._fs.load_skill(name)

    def list_skills(self) -> list[SkillMeta]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT name, description, category, tags, version, created_at, updated_at FROM skills"
            ).fetchall()
            return [self._row_to_meta(r) for r in rows]
        finally:
            conn.close()

    def update_skill(self, name: str, patch: SkillPatch) -> None:
        self._fs.update_skill(name, patch)
        updated = self._fs.load_skill(name)
        if updated:
            self._upsert_index(updated.meta)
            if self._embedder:
                self._auto_embed(updated)

    def delete_skill(self, name: str) -> None:
        self._fs.delete_skill(name)
        conn = self._connect()
        try:
            conn.execute("DELETE FROM skills WHERE name = ?", (name,))
            conn.execute("DELETE FROM skill_embeddings WHERE name = ?", (name,))
            conn.commit()
        finally:
            conn.close()

    def record_usage(self, name: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE skills SET use_count = use_count + 1, last_used_at = ? WHERE name = ?",
                (datetime.now().isoformat(), name),
            )
            conn.commit()
        finally:
            conn.close()

    def get_stats(self) -> dict[str, Any]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT name, category, use_count, last_used_at FROM skills ORDER BY use_count DESC"
            ).fetchall()
            return {
                "total_skills": len(rows),
                "skills": [
                    {"name": r[0], "category": r[1], "use_count": r[2], "last_used_at": r[3]}
                    for r in rows
                ],
            }
        finally:
            conn.close()

    # --- embedding methods ---

    def _auto_embed(self, skill: Skill) -> None:
        from skill_learner.semantic import encode_embedding, skill_to_text
        try:
            text = skill_to_text(
                skill.meta.name, skill.meta.description,
                skill.meta.tags, skill.content,
            )
            vectors = self._embedder.embed([text])
            if vectors and vectors[0]:
                self.save_embedding(skill.meta.name, vectors[0])
        except Exception:
            logger.debug("Auto-embed failed for '%s'", skill.meta.name, exc_info=True)

    def save_embedding(self, name: str, embedding: list[float]) -> None:
        from skill_learner.semantic import encode_embedding
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO skill_embeddings (name, embedding, dimension, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    embedding=excluded.embedding,
                    dimension=excluded.dimension,
                    updated_at=excluded.updated_at
                """,
                (name, encode_embedding(embedding), len(embedding), datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def get_embedding(self, name: str) -> list[float] | None:
        from skill_learner.semantic import decode_embedding
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT embedding FROM skill_embeddings WHERE name = ?", (name,)
            ).fetchone()
            return decode_embedding(row[0]) if row else None
        finally:
            conn.close()

    def get_all_embeddings(self) -> list[tuple[str, list[float]]]:
        from skill_learner.semantic import decode_embedding
        conn = self._connect()
        try:
            rows = conn.execute("SELECT name, embedding FROM skill_embeddings").fetchall()
            return [(r[0], decode_embedding(r[1])) for r in rows]
        finally:
            conn.close()

    # --- internals ---

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version < _SCHEMA_VERSION:
                conn.execute(_CREATE_TABLE)
                conn.execute(_CREATE_EMBEDDINGS_TABLE)
                conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                conn.commit()
        finally:
            conn.close()

    def _sync_from_disk(self) -> None:
        disk_skills = self._fs.list_skills()
        conn = self._connect()
        try:
            db_names = {r[0] for r in conn.execute("SELECT name FROM skills").fetchall()}
            disk_names = {s.name for s in disk_skills}
            for meta in disk_skills:
                if meta.name not in db_names:
                    self._upsert_index(meta, conn=conn)
            for name in db_names - disk_names:
                conn.execute("DELETE FROM skills WHERE name = ?", (name,))
            conn.commit()
        finally:
            conn.close()

    def _upsert_index(self, meta: SkillMeta, conn: sqlite3.Connection | None = None) -> None:
        should_close = conn is None
        if conn is None:
            conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO skills (name, description, category, tags, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description=excluded.description,
                    category=excluded.category,
                    tags=excluded.tags,
                    version=excluded.version,
                    updated_at=excluded.updated_at
                """,
                (
                    meta.name,
                    meta.description,
                    meta.category,
                    json.dumps(meta.tags),
                    meta.version,
                    meta.created_at.isoformat(),
                    meta.updated_at.isoformat(),
                ),
            )
            if should_close:
                conn.commit()
        except Exception:
            logger.warning("SQLite sync failed for skill '%s'", meta.name)
        finally:
            if should_close:
                conn.close()

    @staticmethod
    def _row_to_meta(row: tuple) -> SkillMeta:
        return SkillMeta(
            name=row[0],
            description=row[1],
            category=row[2],
            tags=json.loads(row[3]) if row[3] else [],
            version=row[4],
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
        )
