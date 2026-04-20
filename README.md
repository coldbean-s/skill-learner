# skill-learner

Self-accumulating skill system for AI agents — learn, review, and reuse skills across sessions.

Give any AI agent the ability to automatically save non-obvious workflows as reusable skills, review them in the background, and inject relevant ones into future conversations.

## Quick Install

```bash
pip install skill-learner
```

## 30-Second Quick Start

```python
from skill_learner.integrations.generic_agent import create_skill_loop

loop = create_skill_loop(llm=your_llm, skills_dir="./skills", nudge_interval=5)
wrapped = loop.wrap(your_agent_function)

response = wrapped([{"role": "user", "content": "Help me debug this memory leak"}])
```

That's it. Skills are automatically extracted from complex conversations and injected into future ones.

## How It Works

```
┌─────────────────────────────────────────────┐
│  Your Agent (Anthropic, OpenAI, custom)     │
├─────────────────────────────────────────────┤
│  SkillLoop.wrap()  ──  convenience layer    │
├─────────────────────────────────────────────┤
│  SkillEngine       ──  event coordinator    │
│  ├── PromptInjector   (skill → prompt)      │
│  ├── ReviewTrigger    (when to review)      │
│  └── BackgroundReviewer (LLM → skill CRUD)  │
├─────────────────────────────────────────────┤
│  Storage: File (SKILL.md) + SQLite index    │
│  Optional: Embedding-based semantic search  │
└─────────────────────────────────────────────┘
```

1. **Inject** — On session start, existing skills are injected into the system prompt
2. **Track** — Tool calls and turns are counted toward a review threshold
3. **Review** — When threshold is reached (or session ends), a background LLM call extracts reusable skills
4. **Search** — Keyword or semantic search finds relevant skills for new tasks

## Event API

For full control, use `SkillEngine` directly:

```python
from skill_learner import SkillEngine, SkillLearnerConfig

config = SkillLearnerConfig(skills_dir="./skills", nudge_interval=10)
engine = SkillEngine(config=config, llm=your_llm)

# Session lifecycle
prompt = engine.on_session_start()       # Get skills prompt to inject
engine.on_tool_call("web_search")        # Count tool usage
engine.on_turn_complete(messages)         # Check if review should trigger
engine.on_session_end(messages)           # Force final review

# Query
skills = engine.get_relevant_skills("memory debugging")
stats = engine.get_stats()
```

## Semantic Search (Optional)

Add an `EmbeddingProvider` for embedding-based skill retrieval:

```python
class MyEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        # Your embedding logic (sentence-transformers, Voyage, OpenAI, etc.)
        ...

engine = SkillEngine(config=config, llm=llm, embedding_provider=MyEmbedder())
```

Skills are auto-embedded on save. Search uses cosine similarity with keyword fallback.

## CLI

```
skill-learner list                     # List all skills
skill-learner show <name>              # Show skill content
skill-learner stats                    # Usage statistics
skill-learner export --format json     # Export library
skill-learner import skills.json       # Import skills
skill-learner init --target claude-code  # Set up for Claude Code
skill-learner init --target cursor       # Set up for Cursor
```

## Platform Integrations

### Claude Code
```bash
skill-learner init --target claude-code --project-dir .
```
Creates hooks in `.claude/settings.json` and injects skill index into `CLAUDE.md`.

### Cursor
```bash
skill-learner init --target cursor --project-dir .
```
Injects skill index into `.cursorrules`.

### Generic Python Agent
```python
from skill_learner.integrations.generic_agent import create_skill_loop
loop = create_skill_loop(llm=your_llm)
```

## Configuration

```yaml
# skill-learner.yaml
skills_dir: ~/.skill_learner/skills
nudge_interval: 10          # Review after N tool calls or turns
storage_backend: sqlite     # "sqlite" or "filesystem"
review_timeout: 30          # Seconds to wait for review
semantic_top_k: 5           # Max semantic search results
semantic_threshold: 0.3     # Min cosine similarity
```

Load via `SkillLearnerConfig.from_yaml("skill-learner.yaml")`, `from_dict({...})`, or `from_env(prefix="SKILL_LEARNER_")`.

## LLM Provider Protocol

Any class with this signature works:

```python
class YourLLMProvider:
    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
    ) -> YourResponse:
        ...

class YourResponse:
    content: str
    tool_calls: list[YourToolCall]

class YourToolCall:
    name: str
    arguments: dict
```

See `examples/` for Anthropic and OpenAI adapters.

## License

MIT
