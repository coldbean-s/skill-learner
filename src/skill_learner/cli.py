from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import click
    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

from skill_learner.config import SkillLearnerConfig
from skill_learner.storage.filesystem import FileStorage
from skill_learner.storage.sqlite import SQLiteStorage


def _get_storage(skills_dir: str, backend: str = "sqlite"):
    path = Path(skills_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    if backend == "sqlite":
        return SQLiteStorage(path)
    return FileStorage(path)


if HAS_CLICK:
    @click.group()
    @click.option("--skills-dir", default="~/.skill_learner/skills", help="Skills directory")
    @click.pass_context
    def cli(ctx: click.Context, skills_dir: str) -> None:
        """skill-learner: Self-accumulating skill system for AI agents."""
        ctx.ensure_object(dict)
        ctx.obj["skills_dir"] = skills_dir

    @cli.command()
    @click.option("--target", type=click.Choice(["claude-code", "cursor", "generic"]), required=True)
    @click.option("--project-dir", default=".", help="Project directory")
    @click.pass_context
    def init(ctx: click.Context, target: str, project_dir: str) -> None:
        """Initialize skill-learner for a platform."""
        skills_dir = ctx.obj["skills_dir"]
        if target == "claude-code":
            from skill_learner.integrations.claude_code import init_claude_code
            created = init_claude_code(skills_dir, project_dir)
        elif target == "cursor":
            from skill_learner.integrations.cursor import init_cursor
            created = init_cursor(skills_dir, project_dir)
        else:
            click.echo("Generic: use SkillLoop in your Python code. See examples/.")
            return
        for name, path in created.items():
            click.echo(f"  Created: {name} -> {path}")

    @cli.command("list")
    @click.option("--category", default=None, help="Filter by category")
    @click.pass_context
    def list_skills(ctx: click.Context, category: str | None) -> None:
        """List all saved skills."""
        storage = _get_storage(ctx.obj["skills_dir"])
        skills = storage.list_skills()
        if category:
            skills = [s for s in skills if s.category == category]
        if not skills:
            click.echo("No skills found.")
            return
        for s in skills:
            click.echo(f"  [{s.category}] {s.name} (v{s.version}) — {s.description}")

    @cli.command()
    @click.argument("name")
    @click.pass_context
    def show(ctx: click.Context, name: str) -> None:
        """Show full skill content."""
        storage = _get_storage(ctx.obj["skills_dir"])
        skill = storage.load_skill(name)
        if not skill:
            click.echo(f"Skill '{name}' not found.")
            return
        click.echo(f"--- {skill.meta.name} v{skill.meta.version} [{skill.meta.category}] ---")
        click.echo(skill.content)

    @cli.command()
    @click.pass_context
    def stats(ctx: click.Context) -> None:
        """Show usage statistics."""
        storage = _get_storage(ctx.obj["skills_dir"])
        data = storage.get_stats()
        click.echo(f"Total skills: {data['total_skills']}")
        if "skills" in data:
            for s in data["skills"]:
                click.echo(f"  {s['name']}: {s['use_count']} uses")
        elif "by_category" in data:
            for cat, count in data["by_category"].items():
                click.echo(f"  {cat}: {count} skills")

    @cli.command()
    @click.option("--messages", "messages_file", required=True, type=click.Path(exists=True),
                  help="JSON file with messages array [{role, content}, ...]")
    @click.pass_context
    def review(ctx: click.Context, messages_file: str) -> None:
        """Manually trigger a skill review from a conversation JSON."""
        messages = json.loads(Path(messages_file).read_text(encoding="utf-8"))
        if not isinstance(messages, list):
            click.echo("Error: messages file must contain a JSON array of {role, content} objects", err=True)
            return
        for i, msg in enumerate(messages):
            if "role" not in msg or "content" not in msg:
                click.echo(f"Error: message at index {i} missing 'role' or 'content'", err=True)
                return
        click.echo(f"Validated {len(messages)} messages.")
        click.echo("To run the review, use the Python API with your LLM provider:")
        click.echo("  from skill_learner import BackgroundReviewer, FileStorage")
        click.echo(f"  storage = FileStorage('{ctx.obj['skills_dir']}')")
        click.echo("  reviewer = BackgroundReviewer(your_llm, storage)")
        click.echo("  reviewer.submit_and_wait(messages, timeout=60)")

    @cli.command("export")
    @click.option("--format", "fmt", type=click.Choice(["json", "markdown"]), default="json")
    @click.option("--output", default=None, help="Output file path")
    @click.pass_context
    def export_skills(ctx: click.Context, fmt: str, output: str | None) -> None:
        """Export skill library."""
        storage = _get_storage(ctx.obj["skills_dir"])
        skills_meta = storage.list_skills()
        skills = [storage.load_skill(s.name) for s in skills_meta]
        skills = [s for s in skills if s is not None]

        if fmt == "json":
            data = [{"name": s.meta.name, "description": s.meta.description,
                      "category": s.meta.category, "version": s.meta.version,
                      "tags": s.meta.tags, "content": s.content} for s in skills]
            text = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            parts = []
            for s in skills:
                parts.append(f"# {s.meta.name}\n\n**{s.meta.description}**\n\n{s.content}\n\n---\n")
            text = "\n".join(parts)

        if output:
            Path(output).write_text(text, encoding="utf-8")
            click.echo(f"Exported {len(skills)} skills to {output}")
        else:
            click.echo(text)

    @cli.command("import")
    @click.argument("file", type=click.Path(exists=True))
    @click.pass_context
    def import_skills(ctx: click.Context, file: str) -> None:
        """Import skills from a JSON export file."""
        from datetime import datetime
        from skill_learner.skill import Skill, SkillMeta

        storage = _get_storage(ctx.obj["skills_dir"])
        data = json.loads(Path(file).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            click.echo("Error: import file must contain a JSON array of skill objects", err=True)
            return

        count = 0
        for item in data:
            try:
                skill = Skill(
                    meta=SkillMeta(
                        name=item["name"],
                        description=item["description"],
                        category=item.get("category", "general"),
                        version=item.get("version", "1.0.0"),
                        tags=item.get("tags", []),
                        created_at=datetime.fromisoformat(item["created_at"]) if "created_at" in item else datetime.now(),
                        updated_at=datetime.now(),
                    ),
                    content=item["content"],
                )
                storage.save_skill(skill)
                count += 1
            except (KeyError, ValueError) as e:
                click.echo(f"  Skipping invalid entry: {e}", err=True)
        click.echo(f"Imported {count} skills.")

    def main():
        cli()

else:
    def main():
        print("CLI requires click: pip install skill-learner[cli]")
        sys.exit(1)


if __name__ == "__main__":
    main()
