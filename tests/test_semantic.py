from __future__ import annotations

from skill_learner.semantic import (
    cosine_similarity,
    decode_embedding,
    encode_embedding,
    skill_to_text,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert cosine_similarity(a, b) == 0.0

    def test_similar_vectors_high_score(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.9, 0.1]
        assert cosine_similarity(a, b) > 0.95


class TestEncoding:
    def test_roundtrip(self):
        original = [0.1, 0.2, 0.3, -0.5, 1.0]
        encoded = encode_embedding(original)
        decoded = decode_embedding(encoded)
        assert len(decoded) == len(original)
        for a, b in zip(original, decoded):
            assert abs(a - b) < 1e-6

    def test_empty(self):
        assert decode_embedding(encode_embedding([])) == []

    def test_blob_is_bytes(self):
        blob = encode_embedding([1.0, 2.0])
        assert isinstance(blob, bytes)
        assert len(blob) == 8


class TestSkillToText:
    def test_basic(self):
        text = skill_to_text("debug-memory", "Memory leak debugging", ["coding", "debug"], "# Steps\n1. Use tracemalloc")
        assert "debug memory" in text
        assert "Memory leak debugging" in text
        assert "coding" in text
        assert "tracemalloc" in text

    def test_content_truncated(self):
        long_content = "x" * 2000
        text = skill_to_text("t", "d", [], long_content, max_content=100)
        assert len(text) < 200

    def test_empty_tags(self):
        text = skill_to_text("name", "desc", [], "content")
        assert "name" in text
        assert "desc" in text
