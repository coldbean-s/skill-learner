from __future__ import annotations

import math
import struct


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def skill_to_text(
    name: str,
    description: str,
    tags: list[str],
    content: str,
    max_content: int = 500,
) -> str:
    parts = [name.replace("-", " "), description]
    if tags:
        parts.append(" ".join(tags))
    if content:
        parts.append(content[:max_content])
    return " ".join(parts)


def encode_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def decode_embedding(data: bytes) -> list[float]:
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))
