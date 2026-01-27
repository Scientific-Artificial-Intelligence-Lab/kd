# kd2 - Agent Onboarding Guide

Symbolic regression platform for PDE discovery.

## Quick Start

1. Read `SPEC.md` for architecture and design decisions
2. Read `NOTES/` for detailed implementation specs (Chinese)
3. Check `TASKS/` for current task, use `TASKS/TEMPLATE.md` for format
4. TDD: write test → implement → verify
5. Small commits after each task

## Language

使用中文和用户对话。代码、注释、文档用英文。

## Status

**Phase 0**: Environment not configured yet.

## Project Structure

```
src/kd2/          # Source code
tests/            # Tests (pytest)
ref_libs/         # Reference implementations (git-ignored)
SPEC.md           # Architecture specification
NOTES/            # Detailed design docs (3.6K lines, Chinese)
TASKS/            # Task documents (active tasks)
TASKS/done/       # Completed tasks
.claude/          # Claude Code-specific config
```

## Critical Rules

### Type Hints
All public functions must have complete type annotations.

### Numerical Safety (CRITICAL)
Never use raw arithmetic on tensors. Always use protected operations:

```python
# Division: use safe_div (never a / b)
result = safe_div(a, b, eps=1e-10)

# Exponential: clamp before exp (never torch.exp(x))
result = torch.exp(torch.clamp(x, max=50))

# Logarithm: clamp abs value (never torch.log(x))
result = torch.log(torch.clamp(x.abs(), min=1e-10))
```

### Memory Management
- Detach tensors before storing: `losses.append(loss.detach().item())`
- Pre-allocate buffers, don't create tensors in loops
- Device-aware: use `device=device`, never hardcode `.cuda()`

### Code Organization
- Files < 500 lines, functions < 50 lines
- High cohesion, low coupling
- No print() — use logging
- No magic numbers — use constants

## Test Coverage (per-layer)

| Layer | Target |
|-------|--------|
| Numerical functions (safe_div, etc.) | ~100% |
| IR system + Converters | ~90% |
| Executor + Evaluator | ~80% |
| Constraints + Solver | ~70% |
| Plugins | ~60% |
| Experiment/Viz | smoke tests |

Focus on testing invariants and contracts over line coverage.

## Test Commands

```bash
pytest tests/ -v              # Full suite
pytest -m smoke               # Quick validation
pytest -m "not slow"          # Daily development
mypy src/                     # Type check
ruff check src/               # Lint
```

## Key References

| What | Where |
|------|-------|
| Architecture | `SPEC.md` |
| Detailed design | `NOTES/arc_plan.md`, `NOTES/arc_core.md` |
| IR & Execution ref | `ref_libs/DISCOVER/dso/dso/program.py` |
| Tree structure ref | `ref_libs/sga/sgapde/pde.py` |
| STRidge ref | `ref_libs/DISCOVER/dso/dso/stridge.py` |

## Workflow

- **Task authoring**: AI drafts from SPEC.md + NOTES/, user reviews
- **Review**: User reviews every implementation step within each task
- **Completed tasks**: Move to `TASKS/done/`
- **Ref code**: Understand ref_libs/ logic, implement clean (not direct port)
