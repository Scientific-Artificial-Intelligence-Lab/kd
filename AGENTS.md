# kd2 - Agent Onboarding Guide

Symbolic regression platform for PDE discovery.

## Quick Start

1. Read `SPEC.md` for architecture
2. Read `NOTES/decisions/` for prior design decisions
3. Read `NOTES/architecture/` for detailed specs (Chinese)
4. Check `TASKS/` for current task
5. `/start-task` to begin → TDD → `/wrap-up` to finish

## Language

使用中文和用户对话。代码、注释、文档用英文。

## Agent Roles

### Resident Agents (4)

| Agent | Prompt | Responsibilities |
|-------|--------|-----------------|
| **Architect** | `.claude/agents/architect.md` | Design decisions, task breakdown, architecture review. Does NOT write src/ code |
| **Tester** | `.claude/agents/tester.md` | TDD, unit/integration/equivalence tests, coverage. Pragmatic TDD (can read src/) |
| **Dev** | `.claude/agents/dev.md` | Write functional code in src/, implement features, fix bugs |
| **Porter** | `.claude/agents/porter.md` | Port algorithms from ref_libs/ to src/kd2/. Equivalence verification |

### On-Demand Agents (3)

| Agent | Prompt | Responsibilities |
|-------|--------|-----------------|
| **Teacher** | `.claude/agents/teacher.md` | Concept explanation, code walkthrough. Auto-triggered by `/start-task` and `/wrap-up` |
| **Researcher** | `.claude/agents/researcher.md` | Deep analysis of ref_libs/, papers, algorithms |
| **MentorDev** | `.claude/agents/mentor_dev.md` | Hand-holding coding mentor. Writes one function at a time, explains everything including syntax. For learning |

## Starting Sessions

### Option A: tmux (one command)
```bash
./scripts/start-dev.sh
```
Opens 4 panes: Architect, Tester, Dev, Porter.

### Option B: Manual
```bash
# Each in a separate terminal
claude --prompt "You are the Architect agent. Read .claude/agents/architect.md for your role. 使用中文对话。"
claude --prompt "You are the Tester agent. Read .claude/agents/tester.md for your role. 使用中文对话。"
claude --prompt "You are the Dev agent. Read .claude/agents/dev.md for your role. 使用中文对话。"
claude --prompt "You are the Porter agent. Read .claude/agents/porter.md for your role. 使用中文对话。"
```

## Slash Commands

| Command | Purpose |
|---------|---------|
| `/start-task` | Start task (Teacher concept briefing → plan) |
| `/wrap-up` | Finish task (verify → Teacher code walkthrough → record) |
| `/record` | Record design decision to NOTES/decisions/ |
| `/tdd` | TDD workflow (RED → GREEN → REFACTOR) |
| `/explain` | Teacher explains a concept |
| `/code-review` | Code quality and security review |
| `/integration` | Integration tests |
| `/test-coverage` | Coverage analysis |
| `/learn` | Extract reusable patterns |
| `/port` | Algorithm porting workflow |

## Agent Coordination

Agents communicate through the file system:
- `TASKS/` — Task documents define what to build
- `NOTES/decisions/` — Design decisions (read before starting any work)
- `NOTES/concepts/` — Concept explanations from Teacher
- `tests/` — Tester writes tests, Dev/Porter make them pass

### Typical Task Flow

```
Architect: /start-task → Teacher explains → plan steps
Tester:    Write failing tests (RED)
Dev:       Implement to pass tests (GREEN)
Dev:       Refactor (REFACTOR)
Tester:    Verify coverage
Architect: /wrap-up → Teacher reviews → record decisions
```

## Critical Rules

### Numerical Safety (CRITICAL)
```python
result = safe_div(a, b, eps=1e-10)              # Never a / b
result = torch.exp(torch.clamp(x, max=50))      # Never torch.exp(x)
result = torch.log(torch.clamp(x.abs(), min=1e-10))  # Never torch.log(x)
```

### Type Hints
All public functions must have complete type annotations.

### Memory
- `losses.append(loss.detach().item())` — always detach
- Pre-allocate buffers, don't create tensors in loops
- `device=device` — never hardcode `.cuda()`

### Code Organization
- Files < 500 lines, functions < 50 lines
- No `print()` — use `logging`
- No magic numbers — use constants

## Key References

| What | Where |
|------|-------|
| Architecture | `SPEC.md` |
| Design docs | `NOTES/architecture/arc_plan.md`, `arc_core.md` |
| Decisions | `NOTES/decisions/` |
| IR & Execution ref | `ref_libs/DISCOVER/dso/dso/program.py` |
| Tree structure ref | `ref_libs/sga/sgapde/pde.py` |
| STRidge ref | `ref_libs/DISCOVER/dso/dso/stridge.py` |
