# Task: 000 - Project Environment Setup

> Status: `ready`
> Parent: SPEC.md Phase 1
> Assignee: claude

## Goal

Set up the Python development environment so that `pytest`, `mypy`, and `ruff` all pass on an empty package.

## Non-goals

- [ ] No feature code (IR, Executor, etc.)
- [ ] No complex project configuration (Hydra, WandB)
- [ ] No data loading or processing

## Context

### Relevant files

```
pyproject.toml          # To create
src/kd2/__init__.py     # To create
src/kd2/core/__init__.py
src/kd2/core/safety.py  # safe_div, safe_exp, safe_log
tests/conftest.py       # To create
```

### Current behavior

N/A - new project, no files exist.

### Dependencies

- Requires: nothing
- Blocks: all Phase 1 tasks (001-010)

## Design

### Approach

Create minimal project skeleton with pyproject.toml, package structure, and the safety utilities that are referenced throughout the codebase rules.

### Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Build system | pyproject.toml (setuptools) | Standard, simple |
| Python version | 3.11+ | As specified in SPEC.md |
| Dev dependencies | pytest, mypy, ruff, pytest-cov | Minimum viable toolchain |
| Core dependencies | torch, numpy, sympy | As specified in SPEC.md |

## Implementation steps

1. [ ] **Step 1**: Create pyproject.toml with package metadata, dependencies, and tool config
   - Files: `pyproject.toml`
   - Test: `pip install -e ".[dev]"` succeeds

2. [ ] **Step 2**: Create package skeleton
   - Files: `src/kd2/__init__.py`, `src/kd2/core/__init__.py`
   - Test: `python -c "import kd2"` succeeds

3. [ ] **Step 3**: Implement safety utilities
   - Files: `src/kd2/core/safety.py`, `tests/unit/test_safety.py`
   - Test: `pytest tests/unit/test_safety.py -v`

4. [ ] **Step 4**: Create test infrastructure
   - Files: `tests/__init__.py`, `tests/conftest.py`, `tests/unit/__init__.py`
   - Test: `pytest tests/ -v && mypy src/ && ruff check src/`

## Acceptance criteria

### Functional

- [ ] `pip install -e ".[dev]"` succeeds
- [ ] `import kd2` works
- [ ] safe_div, safe_exp, safe_log handle edge cases (zero, NaN, Inf, extreme values)

### Tests

- [ ] `pytest tests/ -v` passes
- [ ] safety.py has ~100% coverage

### Quality

- [ ] `mypy src/kd2` passes with no errors
- [ ] `ruff check src/kd2` passes with no errors

## Validation

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=src/kd2
mypy src/kd2
ruff check src/kd2
```

## Notes

This task establishes the foundation that all subsequent tasks build on. The safety utilities (safe_div, safe_exp, safe_log) are referenced in `.claude/rules/coding-style.md` and `.claude/rules/patterns.md` as required for all numerical operations.
