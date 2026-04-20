SKILLS_GUIDANCE = """## Skill Accumulation

After completing a **complex task** (5+ tool calls), fixing a tricky bug, or discovering a non-obvious workflow, use the save_skill tool to save the approach as a reusable skill.

When using an existing skill and finding it **outdated, incomplete, or wrong**, immediately use update_skill to patch it — don't wait to be told.

Skills that are not maintained become liabilities."""

REVIEW_PROMPT = """Review the conversation above and determine if any reusable skill should be saved or updated.

Focus on:
- Was a non-obvious method used?
- Were there trial-and-error moments, detours, or strategy adjustments based on actual findings?
- Did the user expect or require an approach different from the initial one?

If a relevant skill already exists, **update it** with what you learned.
Otherwise, if the approach is reusable, **create a new skill**.
If there is nothing worth saving, call nothing_to_save() and stop."""

SKILL_MANAGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_skill",
            "description": "Save a new reusable skill. If a skill with this name already exists, it will be updated instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name (lowercase, hyphens, e.g. 'competitor-analysis')",
                    },
                    "category": {
                        "type": "string",
                        "description": "Category (business, coding, research, ops, general)",
                    },
                    "description": {
                        "type": "string",
                        "description": "One-line description of what this skill does",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full skill content in Markdown (prerequisites, workflow, gotchas)",
                    },
                },
                "required": ["name", "category", "description", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_skill",
            "description": "Update an existing skill with new or corrected content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the skill to update",
                    },
                    "patch_content": {
                        "type": "string",
                        "description": "Updated full content (replaces existing content)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Updated description (optional)",
                    },
                },
                "required": ["name", "patch_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_skill",
            "description": "Delete a skill that is obsolete or incorrect.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the skill to delete",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this skill should be deleted",
                    },
                },
                "required": ["name", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nothing_to_save",
            "description": "Signal that there is nothing worth saving from this conversation.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]
