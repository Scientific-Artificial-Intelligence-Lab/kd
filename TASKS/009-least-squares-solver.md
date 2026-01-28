# Task: 009 - LeastSquaresSolver

> Status: `ready`
> Parent: SPEC.md (Phase 1)
> Assignee: claude

## Goal

实现最小二乘稀疏求解器，为 Evaluator LINEAR 模式提供系数优化能力。

## Non-goals

- [ ] 不实现 STRidgeSolver（Phase 2）
- [ ] 不实现 LassoSolver（Phase 2）
- [ ] 不实现正则化选项
- [ ] 不实现特征选择

## Context

### Relevant files

```
src/kd2/core/linear_solve/__init__.py    # 新建
src/kd2/core/linear_solve/base.py        # 新建
src/kd2/core/linear_solve/least_squares.py # 新建
tests/unit/test_solver.py                # 新建
```

### Current behavior

N/A - 新模块

### Dependencies

- Requires: 无（纯 torch 数值计算）
- Blocks: 010 (Evaluator)

## Design

### Approach

1. 定义 `SparseSolver` 抽象接口
2. 实现 `LeastSquaresSolver`：使用 `torch.linalg.lstsq`
3. 返回结构化结果，包含系数、残差、R²

### Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| 求解方法 | torch.linalg.lstsq | 标准库，稳定可靠 |
| 稀疏性 | Phase 1 不实现 | STRidge/Lasso 在 Phase 2 |
| 返回值 | SolveResult dataclass | 结构化，便于扩展 |

### API 设计

```python
@dataclass
class SolveResult:
    """求解结果"""
    coefficients: torch.Tensor      # 系数向量 (n_terms,)
    residual: float                  # 残差 ||y - Θξ||²
    r2: float                        # R² 决定系数
    condition_number: float          # 条件数（可选，用于诊断）
    selected_indices: list[int] | None = None  # 非零系数索引（稀疏求解器用）


class SparseSolver(ABC):
    """稀疏求解器抽象基类"""

    @abstractmethod
    def solve(
        self,
        theta: torch.Tensor,
        y: torch.Tensor
    ) -> SolveResult:
        """
        求解 y = Θξ

        Args:
            theta: 特征矩阵 (n_samples, n_terms)
            y: 目标向量 (n_samples,) 或 (n_samples, 1)

        Returns:
            SolveResult: 求解结果
        """


class LeastSquaresSolver(SparseSolver):
    """最小二乘求解器"""

    def __init__(self, rcond: float | None = None):
        """
        Args:
            rcond: 奇异值截断阈值（None 使用默认值）
        """
        self.rcond = rcond

    def solve(self, theta: torch.Tensor, y: torch.Tensor) -> SolveResult:
        """使用 torch.linalg.lstsq 求解"""
```

### 求解算法

```python
def solve(self, theta: torch.Tensor, y: torch.Tensor) -> SolveResult:
    # 确保 y 是 2D
    if y.dim() == 1:
        y = y.unsqueeze(1)

    # 求解
    result = torch.linalg.lstsq(theta, y, rcond=self.rcond)
    coefficients = result.solution.squeeze()

    # 计算残差
    y_pred = theta @ coefficients
    residual = ((y.squeeze() - y_pred) ** 2).sum().item()

    # 计算 R²
    ss_tot = ((y.squeeze() - y.mean()) ** 2).sum().item()
    r2 = 1.0 - residual / ss_tot if ss_tot > 0 else 0.0

    # 条件数（可选）
    try:
        cond = torch.linalg.cond(theta).item()
    except:
        cond = float("inf")

    return SolveResult(
        coefficients=coefficients,
        residual=residual,
        r2=r2,
        condition_number=cond
    )
```

## Implementation steps

1. [ ] **Step 1**: 实现 SolveResult 和 SparseSolver ABC
   - Files: `src/kd2/core/linear_solve/base.py`
   - Test: `pytest tests/unit/test_solver.py -k "test_interface"`
   - Agent: Dev

2. [ ] **Step 2**: 实现 LeastSquaresSolver 核心逻辑
   - Files: `src/kd2/core/linear_solve/least_squares.py`
   - Test: `pytest tests/unit/test_solver.py -k "test_basic"`
   - Agent: Dev

3. [ ] **Step 3**: 实现 R² 和残差计算
   - Files: `src/kd2/core/linear_solve/least_squares.py`
   - Test: `pytest tests/unit/test_solver.py -k "test_metrics"`
   - Agent: Dev

4. [ ] **Step 4**: 测试数值稳定性
   - Files: `tests/unit/test_solver.py`
   - Test: `pytest tests/unit/test_solver.py -k "test_stability"`
   - Agent: Tester

## Acceptance criteria

### Functional

- [ ] 已知系数恢复：`y = 2*x1 + 3*x2` → `[2, 3]`
- [ ] 超定系统（n > m）正确求解
- [ ] 欠定系统（n < m）返回最小范数解
- [ ] R² 计算正确（完美拟合接近 1.0）
- [ ] 残差计算正确

### Tests

- [ ] Unit tests pass: `pytest tests/unit/test_solver.py`
- [ ] Coverage >= 95%

### Quality

- [ ] Type hints complete: `mypy src/kd2/core/linear_solve/`
- [ ] No lint errors: `ruff check src/kd2/core/`
- [ ] torch.Tensor 全程

## Validation

```bash
# Minimal validation
pytest tests/unit/test_solver.py -v

# Full validation
pytest tests/unit/ -v --cov=src/kd2/core/linear_solve
mypy src/kd2/core/linear_solve
ruff check src/kd2/core/linear_solve
```

## Constraints

- [ ] 输入输出均为 torch.Tensor
- [ ] 处理接近奇异的矩阵（使用 rcond）

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 病态矩阵 | Med | Med | rcond 截断，返回 condition_number |
| 数值精度 | Low | Low | 使用 float64 如需要 |

## Rollback plan

```bash
git revert <commit-hash>
```

## Notes

_测试用例：_
- 精确解：`y = Θξ` 完美恢复
- 噪声数据：验证 R² < 1
- 病态矩阵：共线特征，检查 condition_number
- Burgers 系数：`[-1.0, 0.1]` 恢复测试（Task 011 集成）

---

## Completion checklist

Before marking as `done`:

- [ ] All implementation steps completed
- [ ] All acceptance criteria met
- [ ] Tests written and passing
- [ ] Code reviewed (`/code-review`)
- [ ] No TODOs left in code
- [ ] `/wrap-up` completed
