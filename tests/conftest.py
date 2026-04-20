from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import MockLLMProvider


@pytest.fixture
def tmp_skills_dir(tmp_path: Path) -> Path:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    return skills_dir


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider()
