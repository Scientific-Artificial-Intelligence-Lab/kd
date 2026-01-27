# kd2

Symbolic regression platform for PDE discovery.

## Status

**Phase 0**: Environment not configured yet.

## Structure

```
src/kd2/          # Source code (TBD)
tests/            # Tests
ref_libs/         # Reference implementations (git-ignored)
SPEC.md           # Project specification
TASKS/            # Task documents
.claude/          # Claude Code config
```

## Commands

```bash
# Tests (when ready)
pytest tests/ -v
mypy src/
ruff check src/
```

## Rules

- Type hints required
- Numerical safety: use safe_div, safe_exp, safe_log
- 80%+ test coverage
- See `.claude/rules/` for details

## Workflow

1. Read `SPEC.md` for architecture
2. Check `TASKS/` for current task
3. TDD: write test → implement → verify
4. Small commits after each task
