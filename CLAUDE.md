# kd2

Symbolic regression platform for PDE discovery.

## Status

**Phase 0**: Environment configured, workflow ready.

## Structure

```
src/kd2/          # Source code (TBD)
tests/            # Tests
ref_libs/         # Reference implementations (git-ignored)
SPEC.md           # Project specification
NOTES/            # Knowledge base
  architecture/   # Architecture docs (arc_*.md)
  decisions/      # Design decision records
  concepts/       # Concept explanations
  explorations/   # Research notes
TASKS/            # Task documents
.claude/          # Claude Code config
  agents/         # 6 agent prompts
  commands/       # 10 slash commands
  rules/          # Coding standards
```

## Commands

```bash
# Environment
conda activate kd2-env   # Python 3.11, required before all commands

# Tests
pytest tests/ -v
pytest -m smoke        # Quick validation
mypy src/
ruff check src/

# Slash commands
/start-task TASKS/xxx.md   # Start a task (with Teacher concept briefing)
/wrap-up TASKS/xxx.md      # Finish a task (with Teacher code walkthrough)
/record [title]            # Record a design decision
/tdd [description]         # TDD workflow
/explain [topic]           # Concept explanation
/code-review               # Code quality review
```

## Rules

- Type hints required
- 使用中文和用户对话
- Numerical safety: use safe_div, safe_exp, safe_log
- Per-layer test coverage targets (see `.claude/rules/testing.md`)
- See `NOTES/AGENTS.md` for multi-agent setup (按需查阅), `.claude/rules/` for coding standards

## Workflow

1. Read `SPEC.md` for architecture
2. Read `NOTES/decisions/` for prior decisions
3. Check `TASKS/` for current task
4. `/start-task` to begin (Teacher explains concepts first)
5. TDD: write test → implement → verify
6. `/wrap-up` to finish (Teacher reviews code)
7. Small commits after each task
