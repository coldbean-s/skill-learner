from __future__ import annotations

from datetime import datetime

import pytest

from skill_learner.skill import Skill, SkillMeta, SkillPatch, parse_skill_md, serialize_skill_md


SAMPLE_SKILL_MD = """---
name: competitor-analysis
description: SaaS competitor analysis report workflow
version: 1.0.0
category: business
tags: [analysis, research]
created_at: '2026-04-20T10:00:00'
updated_at: '2026-04-20T10:00:00'
author: auto-generated
---

# Competitor Analysis

## Workflow
1. Search the web
2. Build report
"""


class TestParseSkillMd:
    def test_parse_valid_skill(self):
        skill = parse_skill_md(SAMPLE_SKILL_MD)
        assert skill.meta.name == "competitor-analysis"
        assert skill.meta.description == "SaaS competitor analysis report workflow"
        assert skill.meta.version == "1.0.0"
        assert skill.meta.category == "business"
        assert skill.meta.tags == ["analysis", "research"]
        assert "# Competitor Analysis" in skill.content

    def test_parse_missing_frontmatter_raises(self):
        with pytest.raises(ValueError, match="frontmatter"):
            parse_skill_md("no frontmatter here")

    def test_parse_missing_required_field_raises(self):
        bad_md = "---\nname: test\n---\ncontent"
        with pytest.raises(ValueError, match="description"):
            parse_skill_md(bad_md)

    def test_parse_empty_tags_defaults_to_list(self):
        md = "---\nname: t\ndescription: d\nversion: 1.0.0\ncategory: c\n---\ncontent"
        skill = parse_skill_md(md)
        assert skill.meta.tags == []


class TestSerializeSkillMd:
    def test_roundtrip(self):
        skill = parse_skill_md(SAMPLE_SKILL_MD)
        serialized = serialize_skill_md(skill)
        reparsed = parse_skill_md(serialized)
        assert reparsed.meta.name == skill.meta.name
        assert reparsed.content.strip() == skill.content.strip()

    def test_serialize_respects_max_size(self):
        meta = SkillMeta(
            name="big",
            description="too big",
            version="1.0.0",
            category="test",
            tags=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        skill = Skill(meta=meta, content="x" * (Skill.MAX_SIZE + 1))
        with pytest.raises(ValueError, match="size"):
            serialize_skill_md(skill)


class TestSkillPatch:
    def test_apply_patch_updates_fields(self):
        skill = parse_skill_md(SAMPLE_SKILL_MD)
        patch = SkillPatch(description="Updated desc", tags=["new-tag"])
        patched = skill.apply_patch(patch)
        assert patched.meta.description == "Updated desc"
        assert patched.meta.tags == ["new-tag"]
        assert patched.meta.version == "1.1.0"

    def test_apply_patch_no_bump(self):
        skill = parse_skill_md(SAMPLE_SKILL_MD)
        patch = SkillPatch(description="Updated", bump_version=False)
        patched = skill.apply_patch(patch)
        assert patched.meta.version == "1.0.0"
