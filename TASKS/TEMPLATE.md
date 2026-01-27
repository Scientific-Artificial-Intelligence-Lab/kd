# Task: [TASK_ID] - [Short Title]

> Status: `draft` | `ready` | `in_progress` | `blocked` | `done`
> Parent: SPEC.md / [Parent Task ID]
> Assignee: human | claude

## Goal

What user-visible behavior or capability are we trying to achieve?

_Keep this to 1-2 sentences. If it takes more, the task is too big._

## Non-goals

What are we explicitly NOT doing in this task?

- [ ] ...
- [ ] ...

## Context

### Relevant files

```
src/kd2/core/...
tests/...
```

### Current behavior

What happens today? (or "N/A - new feature")

### Dependencies

- Requires: [Task ID] (if any)
- Blocks: [Task ID] (if any)

## Design

### Approach

Brief description of the implementation approach.

### Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ... | ... | ... |

## Implementation steps

_Small, verifiable steps. Each should be completable in one session._

1. [ ] **Step 1**: [Description]
   - Files: `path/to/file.py`
   - Test: `pytest tests/... -k "test_name"`

2. [ ] **Step 2**: [Description]
   - Files: `path/to/file.py`
   - Test: `pytest tests/... -k "test_name"`

3. [ ] **Step 3**: [Description]
   - Files: `path/to/file.py`
   - Test: `pytest tests/... -k "test_name"`

## Acceptance criteria

### Functional

- [ ] [Criterion 1]
- [ ] [Criterion 2]

### Tests

- [ ] Unit tests pass: `pytest tests/unit/test_xxx.py`
- [ ] Integration tests pass: `pytest tests/integration/test_xxx.py`
- [ ] Coverage >= 80%: `pytest --cov=src/kd2/xxx`

### Quality

- [ ] Type hints complete: `mypy src/kd2/xxx`
- [ ] No lint errors: `ruff check src/kd2/xxx`
- [ ] Numerical operations protected (safe_div, safe_exp)

## Validation

```bash
# Minimal validation (must pass)
pytest tests/unit/test_xxx.py -v

# Full validation (recommended)
pytest tests/ -v --cov=src/kd2
mypy src/kd2
ruff check src/kd2
```

## Constraints

- [ ] No breaking changes to existing APIs
- [ ] Memory usage: [constraint if any]
- [ ] Performance: [constraint if any]

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ... | Low/Med/High | Low/Med/High | ... |

## Rollback plan

How to revert if the change causes regressions?

```bash
git revert <commit-hash>
# or
git checkout HEAD~1 -- path/to/files
```

## Notes

_Session notes, learnings, decisions made during implementation._

---

## Completion checklist

Before marking as `done`:

- [ ] All implementation steps completed
- [ ] All acceptance criteria met
- [ ] Tests written and passing
- [ ] Code reviewed (self or peer)
- [ ] No TODOs left in code
- [ ] SPEC.md updated if needed
