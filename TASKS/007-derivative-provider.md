# Task: 007 - DerivativeProvider & FiniteDiffProvider

> Status: `ready`
> Parent: SPEC.md (Phase 1)
> Assignee: claude

## Goal

实现导数提供者接口和有限差分实现，为 Executor 提供预计算的导数数据。

## Non-goals

- [ ] 不实现 AutogradProvider（Phase 2）
- [ ] 不实现 open-form diff（Phase 2）
- [ ] 不实现混合导数（Phase 5）
- [ ] 不实现 Scattered 数据支持（Phase 5）

## Context

### Relevant files

```
src/kd2/data/derivatives/__init__.py   # 新建
src/kd2/data/derivatives/base.py       # 新建
src/kd2/data/derivatives/finite_diff.py # 新建
src/kd2/data/derivatives/naming.py     # 新建
tests/unit/test_finite_diff.py         # 新建
```

### Current behavior

N/A - 新模块

### Dependencies

- Requires: 006 (PDEDataset)
- Blocks: 008 (ExecutionContext)

## Design

### Approach

1. 定义 `DerivativeProvider` 抽象接口
2. 实现 `FiniteDiffProvider`：中心差分，预计算并缓存
3. 导数命名工具：display ↔ canonical 双向转换

### Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| 差分方法 | 中心差分 | 2 阶精度，简单可靠 |
| 预计算 | 初始化时计算所有导数 | 避免重复计算 |
| diff() | 暂抛 NotImplementedError | Phase 2 autograd 再实现 |
| Grid 检查 | 初始化时断言 | 早期失败 |

### API 设计

```python
class DerivativeProvider(ABC):
    """导数提供者抽象基类"""

    @abstractmethod
    def get_derivative(
        self,
        field: str,
        axis: str,
        order: int
    ) -> torch.Tensor:
        """获取预计算的导数"""

    @abstractmethod
    def diff(
        self,
        expression: torch.Tensor,
        axis: str,
        order: int
    ) -> torch.Tensor:
        """对表达式求导（open-form）"""

    @abstractmethod
    def available_derivatives(self) -> list[tuple[str, str, int]]:
        """返回可用的预计算导数列表 [(field, axis, order), ...]"""


class FiniteDiffProvider(DerivativeProvider):
    """有限差分导数提供者"""

    def __init__(
        self,
        dataset: PDEDataset,
        max_order: int = 3,
        method: str = "central",
        accuracy: int = 2
    ):
        """
        Args:
            dataset: PDE 数据集（必须是 Grid 拓扑）
            max_order: 最大导数阶数
            method: 差分方法（"central", "forward", "backward"）
            accuracy: 精度阶数（2, 4, 6）
        """

    def get_derivative(self, field: str, axis: str, order: int) -> torch.Tensor:
        """返回预计算的导数"""

    def diff(self, expression: torch.Tensor, axis: str, order: int) -> torch.Tensor:
        """Phase 1 不实现，抛出 NotImplementedError"""
        raise NotImplementedError("open-form diff requires AutogradProvider")


# 命名工具
def to_canonical_name(display: str) -> str:
    """u_xx -> deriv:u:x:2"""

def to_display_name(canonical: str) -> str:
    """deriv:u:x:2 -> u_xx"""

def parse_derivative_name(name: str) -> tuple[str, str, int]:
    """解析导数名称，返回 (field, axis, order)"""
```

### 有限差分算法

```python
def central_diff(f: Tensor, dx: float, axis: int, order: int) -> Tensor:
    """
    中心差分

    1阶: (f[i+1] - f[i-1]) / (2*dx)
    2阶: (f[i+1] - 2*f[i] + f[i-1]) / dx^2
    3阶: (f[i+2] - 2*f[i+1] + 2*f[i-1] - f[i-2]) / (2*dx^3)
    """
```

## Implementation steps

1. [ ] **Step 1**: 实现 DerivativeProvider ABC
   - Files: `src/kd2/data/derivatives/base.py`
   - Test: `pytest tests/unit/test_finite_diff.py -k "test_interface"`
   - Agent: Dev

2. [ ] **Step 2**: 实现导数命名工具
   - Files: `src/kd2/data/derivatives/naming.py`
   - Test: `pytest tests/unit/test_finite_diff.py -k "test_naming"`
   - Agent: Dev

3. [ ] **Step 3**: 实现中心差分核心算法
   - Files: `src/kd2/data/derivatives/finite_diff.py`
   - Test: `pytest tests/unit/test_finite_diff.py -k "test_central_diff"`
   - Agent: Dev

4. [ ] **Step 4**: 实现 FiniteDiffProvider
   - Files: `src/kd2/data/derivatives/finite_diff.py`
   - Test: `pytest tests/unit/test_finite_diff.py -k "test_provider"`
   - Agent: Dev

5. [ ] **Step 5**: 验证导数精度
   - Files: `tests/unit/test_finite_diff.py`
   - Test: `pytest tests/unit/test_finite_diff.py -k "test_accuracy"`
   - Agent: Tester

## Acceptance criteria

### Functional

- [ ] FiniteDiffProvider 初始化时检查 Grid 拓扑
- [ ] `get_derivative()` 返回正确形状的张量
- [ ] 1/2/3 阶导数精度符合预期（相对误差 < 1e-3 对光滑函数）
- [ ] 命名转换：`u_xx` ↔ `deriv:u:x:2`
- [ ] `diff()` 抛出 NotImplementedError

### Tests

- [ ] Unit tests pass: `pytest tests/unit/test_finite_diff.py`
- [ ] Coverage >= 95%

### Quality

- [ ] Type hints complete: `mypy src/kd2/data/derivatives/`
- [ ] No lint errors: `ruff check src/kd2/data/`
- [ ] torch.Tensor 全程

## Validation

```bash
# Minimal validation
pytest tests/unit/test_finite_diff.py -v

# Full validation
pytest tests/unit/ -v --cov=src/kd2/data/derivatives
mypy src/kd2/data/derivatives
ruff check src/kd2/data/derivatives
```

## Constraints

- [ ] 必须检查 Grid 拓扑
- [ ] 边界处理：减小有效区域或使用单边差分

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 边界误差 | Med | Low | 文档说明边界点精度降低 |
| 高阶导数噪声放大 | Med | Med | 限制 max_order=3 |

## Rollback plan

```bash
git revert <commit-hash>
```

## Notes

_精度验证测试用例：_
- `sin(x)` → 导数 `cos(x)`，验证精度
- `x^2` → 导数 `2x`
- `exp(x)` → 导数 `exp(x)`

---

## Completion checklist

Before marking as `done`:

- [ ] All implementation steps completed
- [ ] All acceptance criteria met
- [ ] Tests written and passing
- [ ] Code reviewed (`/code-review`)
- [ ] No TODOs left in code
- [ ] `/wrap-up` completed
