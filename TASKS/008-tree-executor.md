# Task: 008 - ExecutionContext & TreeExecutor

> Status: `ready`
> Parent: SPEC.md (Phase 1)
> Assignee: claude

## Goal

实现表达式执行引擎，递归遍历 AST 计算数值结果。

## Non-goals

- [ ] 不实现批量执行优化（Phase 5 CSE）
- [ ] 不实现 open-form diff 算子（Phase 2）
- [ ] 不实现执行缓存（Phase 3）
- [ ] 不实现并行执行

## Context

### Relevant files

```
src/kd2/core/executor/__init__.py     # 新建
src/kd2/core/executor/context.py      # 新建
src/kd2/core/executor/tree_executor.py # 新建
tests/unit/test_executor.py           # 新建
```

### Current behavior

N/A - 新模块

### Dependencies

- Requires: 002 (Library), 004 (ASTNode), 005 (Converters), 006 (PDEDataset), 007 (DerivativeProvider)
- Blocks: 010 (Evaluator)

## Design

### Approach

1. `ExecutionContext`：封装执行所需的所有数据（fields, coords, derivatives, constants）
2. `TreeExecutor`：递归遍历 AST，按 Decision 002 实现
3. 数值安全检查：NaN/Inf 检测

### Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| 执行方式 | 递归 AST 遍历 | Decision 002: 直观，便于调试 |
| diff terminal | 通过 DerivativeProvider | `u_x` 查找预计算导数 |
| 常数处理 | context.constants 字典 | 支持命名常数 |
| 无效结果 | ExecutionResult.is_valid=False | 不抛异常，返回标记 |

### API 设计

```python
@dataclass
class ExecutionContext:
    """执行上下文"""
    dataset: PDEDataset
    derivative_provider: DerivativeProvider
    library: Library
    constants: dict[str, float] = field(default_factory=dict)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def get_variable(self, name: str) -> torch.Tensor:
        """获取变量值（field 或 coord）"""

    def get_derivative(self, field: str, axis: str, order: int) -> torch.Tensor:
        """获取预计算导数"""


class ErrorType(Enum):
    NONE = "none"
    NAN = "nan"
    INF = "inf"
    UNKNOWN_TOKEN = "unknown_token"
    SHAPE_MISMATCH = "shape_mismatch"


@dataclass
class ExecutionResult:
    """执行结果"""
    value: torch.Tensor | None
    is_valid: bool
    error_type: ErrorType = ErrorType.NONE
    error_message: str = ""


class TreeExecutor:
    """递归 AST 执行器"""

    def __init__(self, context: ExecutionContext):
        self.context = context

    def execute(self, node: ASTNode) -> ExecutionResult:
        """执行 AST，返回结果张量"""

    def execute_batch(self, nodes: list[ASTNode]) -> list[ExecutionResult]:
        """批量执行多个 AST"""
```

### 执行算法

```python
def execute(self, node: ASTNode) -> ExecutionResult:
    token = self.context.library.get(node.token)

    # Terminal: 变量或导数
    if token.arity == 0:
        return self._execute_terminal(node)

    # 递归执行子节点
    child_results = [self.execute(c) for c in node.children]

    # 检查子节点有效性
    if any(not r.is_valid for r in child_results):
        return ExecutionResult(None, False, ErrorType.NAN)

    # 应用函数
    child_values = [r.value for r in child_results]
    result = token.function(*child_values)

    # 检查结果有效性
    if torch.isnan(result).any():
        return ExecutionResult(result, False, ErrorType.NAN)
    if torch.isinf(result).any():
        return ExecutionResult(result, False, ErrorType.INF)

    return ExecutionResult(result, True)

def _execute_terminal(self, node: ASTNode) -> ExecutionResult:
    name = node.token

    # 检查是否是导数 terminal（如 u_x, u_xx）
    if "_" in name:
        field, axis_part = name.split("_", 1)
        axis = axis_part[0]
        order = len(axis_part)
        value = self.context.get_derivative(field, axis, order)
    else:
        value = self.context.get_variable(name)

    return ExecutionResult(value, True)
```

## Implementation steps

1. [ ] **Step 1**: 实现 ExecutionContext
   - Files: `src/kd2/core/executor/context.py`
   - Test: `pytest tests/unit/test_executor.py -k "test_context"`
   - Agent: Dev

2. [ ] **Step 2**: 实现 ExecutionResult 和 ErrorType
   - Files: `src/kd2/core/executor/context.py`
   - Test: `pytest tests/unit/test_executor.py -k "test_result"`
   - Agent: Dev

3. [ ] **Step 3**: 实现 TreeExecutor 核心逻辑
   - Files: `src/kd2/core/executor/tree_executor.py`
   - Test: `pytest tests/unit/test_executor.py -k "test_execute_basic"`
   - Agent: Dev

4. [ ] **Step 4**: 实现 terminal 解析（变量 + 导数）
   - Files: `src/kd2/core/executor/tree_executor.py`
   - Test: `pytest tests/unit/test_executor.py -k "test_terminal"`
   - Agent: Dev

5. [ ] **Step 5**: 实现数值安全检查
   - Files: `src/kd2/core/executor/tree_executor.py`
   - Test: `pytest tests/unit/test_executor.py -k "test_nan_inf"`
   - Agent: Dev

6. [ ] **Step 6**: 端到端测试（Burgers 表达式）
   - Files: `tests/unit/test_executor.py`
   - Test: `pytest tests/unit/test_executor.py -k "test_burgers"`
   - Agent: Tester

## Acceptance criteria

### Functional

- [ ] 简单表达式执行正确：`add(u, v)`
- [ ] 嵌套表达式执行正确：`sin(mul(u, u_x))`
- [ ] 导数 terminal 正确查找：`u_x`, `u_xx`, `u_t`
- [ ] 常数 terminal 正确查找
- [ ] NaN/Inf 检测并标记 `is_valid=False`
- [ ] Burgers 表达式 `add(mul(u, u_x), u_xx)` 执行正确

### Tests

- [ ] Unit tests pass: `pytest tests/unit/test_executor.py`
- [ ] Coverage >= 95%

### Quality

- [ ] Type hints complete: `mypy src/kd2/core/executor/`
- [ ] No lint errors: `ruff check src/kd2/core/`
- [ ] device-aware 张量操作

## Validation

```bash
# Minimal validation
pytest tests/unit/test_executor.py -v

# Full validation
pytest tests/unit/ -v --cov=src/kd2/core/executor
mypy src/kd2/core/executor
ruff check src/kd2/core/executor
```

## Constraints

- [ ] 执行结果形状必须与数据集形状一致
- [ ] 不修改输入 ASTNode

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 深递归栈溢出 | Low | Med | 表达式深度通常 <50 |
| 形状不匹配 | Med | Med | 广播或报错 |

## Rollback plan

```bash
git revert <commit-hash>
```

## Notes

_测试用例应覆盖：_
- 常量：`1.0`
- 变量：`u`
- 一元：`sin(u)`, `diff_x(u)` (一元 diff token)
- 二元：`add(u, v)`, `mul(u, u_x)`
- 嵌套：`add(mul(u, u_x), u_xx)`
- 无效：产生 NaN 的表达式

---

## Completion checklist

Before marking as `done`:

- [ ] All implementation steps completed
- [ ] All acceptance criteria met
- [ ] Tests written and passing
- [ ] Code reviewed (`/code-review`)
- [ ] No TODOs left in code
- [ ] `/wrap-up` completed
