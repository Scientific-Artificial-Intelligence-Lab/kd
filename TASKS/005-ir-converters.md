# Task: 005 - IR Converters

> Status: `ready`
> Parent: SPEC.md (Phase 1)
> Assignee: claude

## Goal

实现 GenIR ↔ AnalysisIR 双向转换，确保 round-trip 不变量。

## Non-goals

- [ ] 不实现表达式优化/简化
- [ ] 不实现 ExecIR 转换（Phase 5）
- [ ] 不实现并行转换

## Context

### Relevant files

```
src/kd2/core/ir/converters.py    # 新建
tests/unit/test_converters.py    # 新建
```

### Current behavior

N/A - 新模块

### Dependencies

- Requires: 002 (Token, Library), 003 (GenIR), 004 (AnalysisIR)
- Blocks: 008 (TreeExecutor), 010 (Evaluator)

## Design

### Approach

1. `to_analysis_ir`: 前缀序列递归建树
2. `to_gen_ir`: 前序遍历扁平化
3. `to_canonical_gen_ir`: 扁平化时可交换子节点排序

### Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| canonical 排序 | 子节点按 canonical_hash 排序 | 确保 a+b 和 b+a 转为相同 GenIR |
| 一元 diff | 正常处理 | diff_x(u) 是一元算子，children 长度为 1 |

### API 设计

```python
def to_analysis_ir(gen_ir: GenIR, library: Library) -> ASTNode:
    """
    前缀 token 序列 → AST

    算法：递归下降解析
    """

def to_gen_ir(node: ASTNode, library: Library) -> GenIR:
    """
    AST → 前缀 token 序列

    算法：前序遍历
    """

def to_canonical_gen_ir(node: ASTNode, library: Library) -> GenIR:
    """
    AST → canonical 前缀序列

    对可交换算子的子节点按 hash 排序后再扁平化
    """
```

### 转换算法

```python
# GenIR → AST
def to_analysis_ir(gen_ir, library):
    tokens = list(gen_ir.tokens)
    idx = 0

    def parse():
        nonlocal idx
        token_name = tokens[idx]
        idx += 1
        token = library.get(token_name)
        children = tuple(parse() for _ in range(token.arity))
        return ASTNode(token_name, children)

    return parse()

# AST → GenIR (canonical)
def to_canonical_gen_ir(node, library):
    def flatten(n):
        token = library.get(n.token)
        children = list(n.children)
        if token.is_commutative:
            children.sort(key=lambda c: c.canonical_hash(library))
        result = [n.token]
        for child in children:
            result.extend(flatten(child))
        return result

    return GenIR(tuple(flatten(node)))
```

## Implementation steps

1. [ ] **Step 1**: 实现 to_analysis_ir
   - Files: `src/kd2/core/ir/converters.py`
   - Test: `pytest tests/unit/test_converters.py -k "test_to_analysis_ir"`
   - Agent: Dev

2. [ ] **Step 2**: 实现 to_gen_ir
   - Files: `src/kd2/core/ir/converters.py`
   - Test: `pytest tests/unit/test_converters.py -k "test_to_gen_ir"`
   - Agent: Dev

3. [ ] **Step 3**: 实现 to_canonical_gen_ir
   - Files: `src/kd2/core/ir/converters.py`
   - Test: `pytest tests/unit/test_converters.py -k "test_canonical"`
   - Agent: Dev

4. [ ] **Step 4**: 验证 round-trip 不变量
   - Files: `tests/unit/test_converters.py`
   - Test: `pytest tests/unit/test_converters.py -k "test_roundtrip"`
   - Agent: Tester

## Acceptance criteria

### Functional

- [ ] `to_analysis_ir` 正确构建 AST
- [ ] `to_gen_ir` 正确扁平化
- [ ] Round-trip 不变量：`to_canonical_gen_ir(to_analysis_ir(g)) == canonical(g)`
- [ ] 可交换排序：`to_canonical_gen_ir(add(b, a)) == to_canonical_gen_ir(add(a, b))`
- [ ] 一元 diff token 正确处理

### Tests

- [ ] Unit tests pass: `pytest tests/unit/test_converters.py`
- [ ] Coverage >= 95%

### Quality

- [ ] Type hints complete: `mypy src/kd2/core/ir/converters.py`
- [ ] No lint errors: `ruff check src/kd2/core/ir/`

## Validation

```bash
# Minimal validation
pytest tests/unit/test_converters.py -v

# Full validation
pytest tests/unit/ -v --cov=src/kd2/core/ir
mypy src/kd2/core/ir
ruff check src/kd2/core/ir
```

## Constraints

- [ ] 不修改输入的 GenIR 或 ASTNode（不可变原则）

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 不完整 GenIR | Med | Low | 转换前检查 is_complete() |

## Rollback plan

```bash
git revert <commit-hash>
```

## Notes

_测试用例应覆盖：_
- 单节点：`"u"` → `ASTNode("u")` → `"u"`
- 一元：`"sin,u"` → `ASTNode("sin", (ASTNode("u"),))` → `"sin,u"`
- 二元：`"add,u,v"` → ...
- 嵌套：`"add,mul,u,v,w"`
- diff：`"diff_x,u"` → `ASTNode("diff_x", (ASTNode("u"),))`
- 可交换：`"add,v,u"` canonical → `"add,u,v"`（假设 u < v）

---

## Completion checklist

Before marking as `done`:

- [ ] All implementation steps completed
- [ ] All acceptance criteria met
- [ ] Tests written and passing
- [ ] Code reviewed (`/code-review`)
- [ ] No TODOs left in code
- [ ] `/wrap-up` completed
