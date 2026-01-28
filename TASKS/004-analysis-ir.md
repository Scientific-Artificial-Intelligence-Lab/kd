# Task: 004 - AnalysisIR (ASTNode)

> Status: `ready`
> Parent: SPEC.md (Phase 1)
> Assignee: claude

## Goal

实现 AST 树表示（AnalysisIR），支持结构分析、term 拆分和 canonical hash。

## Non-goals

- [ ] 不实现树变换/重写操作
- [ ] 不实现符号简化
- [ ] 不实现 LaTeX 输出（Phase 5 可视化）

## Context

### Relevant files

```
src/kd2/core/ir/analysis_ir.py    # 新建
tests/unit/test_analysis_ir.py    # 新建
```

### Current behavior

N/A - 新模块

### Dependencies

- Requires: 002 (Token)
- Blocks: 005 (IR Converters), 010 (Evaluator - split_terms)

## Design

### Approach

ASTNode 是不可变的树节点，核心职责：
1. 存储树结构（token + children）
2. 提供结构分析方法（depth, length, complexity）
3. 拆分为 term 列表（`split_terms`）—— Decision 003 的 IR 层工具
4. 计算 canonical hash（可交换子节点排序）

### Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| children 类型 | `tuple[ASTNode, ...]` | Decision 002: 不可变 |
| frozen | True | 支持 hash，防止意外修改 |
| split_terms | IR 层工具函数 | Decision 003: 非 Evaluator 职责 |
| sub 处理 | 转为 neg term | `a - b` 拆为 `[a, neg(b)]` |

### API 设计

```python
@dataclass(frozen=True)
class ASTNode:
    token: str
    children: tuple["ASTNode", ...] = ()

    @property
    def is_terminal(self) -> bool:
        """是否为叶节点（arity=0）"""

    def depth(self) -> int:
        """树深度（叶节点为 1）"""

    def length(self) -> int:
        """节点总数"""

    def complexity(self) -> int:
        """复杂度指标（可配置权重）"""

    def canonical_hash(self, library: Library) -> str:
        """SHA256 hash，可交换子节点先排序"""

    def __hash__(self) -> int
    def __eq__(self, other: object) -> bool


def split_terms(node: ASTNode, library: Library) -> list[ASTNode]:
    """
    沿顶层 add/sub 拆分为 term 列表

    例如：add(a, sub(b, c)) → [a, b, neg(c)]
    """
```

### split_terms 算法

```
def split_terms(node, library):
    if node.token == "add":
        return split_terms(node.children[0]) + split_terms(node.children[1])
    elif node.token == "sub":
        left_terms = split_terms(node.children[0])
        right_terms = [negate(t) for t in split_terms(node.children[1])]
        return left_terms + right_terms
    else:
        return [node]  # 单个 term

def negate(node):
    if node.token == "neg":
        return node.children[0]  # 双重否定消除
    return ASTNode("neg", (node,))
```

## Implementation steps

1. [ ] **Step 1**: 实现 ASTNode 基础结构
   - Files: `src/kd2/core/ir/analysis_ir.py`
   - Test: `pytest tests/unit/test_analysis_ir.py -k "test_astnode_basic"`
   - Agent: Dev

2. [ ] **Step 2**: 实现树属性方法
   - Files: `src/kd2/core/ir/analysis_ir.py`
   - Test: `pytest tests/unit/test_analysis_ir.py -k "test_tree_properties"`
   - Agent: Dev

3. [ ] **Step 3**: 实现 canonical_hash
   - Files: `src/kd2/core/ir/analysis_ir.py`
   - Test: `pytest tests/unit/test_analysis_ir.py -k "test_canonical_hash"`
   - Agent: Dev

4. [ ] **Step 4**: 实现 split_terms
   - Files: `src/kd2/core/ir/analysis_ir.py`
   - Test: `pytest tests/unit/test_analysis_ir.py -k "test_split_terms"`
   - Agent: Dev

## Acceptance criteria

### Functional

- [ ] ASTNode 创建正确，frozen 不可修改
- [ ] `depth()`, `length()`, `complexity()` 计算正确
- [ ] `canonical_hash()` 对 `a+b` 和 `b+a` 返回相同 hash
- [ ] `split_terms()` 正确拆分：
  - 单 term: `mul(u, v)` → `[mul(u, v)]`
  - 多 term: `add(a, b)` → `[a, b]`
  - 嵌套: `add(a, add(b, c))` → `[a, b, c]`
  - sub: `sub(a, b)` → `[a, neg(b)]`
  - 复合: `add(mul(u, ux), uxx)` → `[mul(u, ux), uxx]`

### Tests

- [ ] Unit tests pass: `pytest tests/unit/test_analysis_ir.py`
- [ ] Coverage >= 95%

### Quality

- [ ] Type hints complete: `mypy src/kd2/core/ir/analysis_ir.py`
- [ ] No lint errors: `ruff check src/kd2/core/ir/`

## Validation

```bash
# Minimal validation
pytest tests/unit/test_analysis_ir.py -v

# Full validation
pytest tests/unit/ -v --cov=src/kd2/core/ir
mypy src/kd2/core/ir
ruff check src/kd2/core/ir
```

## Constraints

- [ ] ASTNode 必须 frozen 和 hashable
- [ ] split_terms 必须处理嵌套 add/sub

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 深递归栈溢出 | Low | Med | 表达式深度通常 <50，可加深度限制 |
| neg 累积 | Low | Low | 双重 neg 消除 |

## Rollback plan

```bash
git revert <commit-hash>
```

## Notes

_split_terms 是 Evaluator LINEAR 模式的关键依赖（Decision 003）_

---

## Completion checklist

Before marking as `done`:

- [ ] All implementation steps completed
- [ ] All acceptance criteria met
- [ ] Tests written and passing
- [ ] Code reviewed (`/code-review`)
- [ ] No TODOs left in code
- [ ] `/wrap-up` completed
