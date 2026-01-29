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
  agents/         # 10 agents (teacher, tester, test-reviewer, dev, code-reviewer, porter, researcher, architect, codex-prep, verifier)
  commands/       # Slash commands
  rules/          # Coding standards
.session/         # Temporary session state (gitignored)
.codex/           # Codex collaboration directory
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
/start-task TASKS/xxx.md   # Read task + spawn teacher for concepts
/tdd [description]         # Spawn tester + auto test-reviewer (RED)
/dev [description]         # Spawn dev + auto code-reviewer (GREEN)
/codex-review [files]      # Spawn codex-prep for external Codex review
/wrap-up TASKS/xxx.md      # Spawn verifier + teacher, then archive
/handoff                   # Generate handoff doc for session transition
/record [title]            # Record a design decision
/explain [topic]           # Spawn teacher for concept explanation
```

## Rules

- Type hints required
- 使用中文和用户对话
- Numerical safety: use safe_div, safe_exp, safe_log
- Per-layer test coverage targets (see `.claude/rules/testing.md`)
- See `.claude/rules/` for coding standards

## Multi-Agent Workflow

主 agent 作为**协调者**（不写代码、不跑测试、不大段讲解），重活全部 spawn：

```
/start-task        → spawn teacher 讲解概念
/tdd               → spawn tester 写测试 → 自动 spawn test-reviewer 审查 (RED)
/dev               → spawn dev 实现 → 自动 spawn code-reviewer 审查 (GREEN)
/codex-review      → spawn codex-prep 准备文件 → 用户操作 Codex
/wrap-up           → spawn verifier 验证 → spawn teacher 讲解 → 归档
```

会话过长时用 `/handoff` 生成交接文档，开新对话继续。

## Workflow

1. Read `SPEC.md` for architecture
2. Read `NOTES/decisions/` for prior decisions
3. Check `TASKS/` for current task
4. `/start-task` to begin
5. `/tdd` (RED) → `/dev` (GREEN)（各阶段自动 review）
6. (Optional) `/codex-review` for external review
7. `/wrap-up` to finish
8. Small commits after each task
