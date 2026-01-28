# Task: 010 - Evaluator

> Status: `ready`
> Parent: SPEC.md (Phase 1)
> Assignee: claude

## Goal

实现评估器，支持 DIRECT 和 LINEAR 两种评估模式，计算表达式的拟合质量。

## Non-goals

- [ ] 不实现评估缓存（Phase 3）
- [ ] 不实现自定义指标插件
- [ ] 不实现批量评估优化
- [ ] 不实现常数优化（Phase 4）

## Context

### Relevant files

```
src/kd2/core/evaluator/__init__.py    # 新建
src/kd2/core/evaluator/evaluator.py   # 新建
src/kd2/core/evaluator/metrics.py     # 新建
tests/unit/test_evaluator.py          # 新建
```

### Current behavior

N/A - 新模块

### Dependencies

- Requires: 004 (split_terms), 008 (Executor), 009 (Solver)
- Blocks: 011 (Phase 1 Integration)

## Design

### Approach

1. DIRECT 模式：整棵 AST 执行，直接计算 MSE/NMSE/R²
2. LINEAR 模式（核心）：拆分 terms → 执行各 term → 拼 Θ 矩阵 → 最小二乘 → 系数 + 指标
3. Decision 003：`evaluate_terms()` 是核心 API

### Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| 核心 API | `evaluate_terms(terms: list[ASTNode])` | Decision 003 |
| 默认模式 | LINEAR | PDE 发现主流方法 |
| 指标 | MSE, NMSE, R², AIC | 内置常用指标 |
| 无效处理 | 返回 penalty 分数 | 不中断评估流程 |

### API 设计

```python
class EvaluationMode(Enum):
    DIRECT = "direct"    # 整棵树直接计算
    LINEAR = "linear"    # term 列表 + 最小二乘


@dataclass
class EvaluationResult:
    """评估结果"""
    mse: float
    nmse: float
    r2: float
    aic: float | None = None
    complexity: int = 0
    coefficients: torch.Tensor | None = None
    is_valid: bool = True
    error_message: str = ""


class Evaluator:
    """表达式评估器"""

    def __init__(
        self,
        executor: TreeExecutor,
        solver: SparseSolver,
        y_true: torch.Tensor,
        penalty_value: float = 1e10
    ):
        """
        Args:
            executor: 表达式执行器
            solver: 线性求解器
            y_true: 目标值（LHS，如 u_t）
            penalty_value: 无效表达式的惩罚分数
        """

    def evaluate_terms(
        self,
        terms: list[ASTNode],
        mode: EvaluationMode = EvaluationMode.LINEAR
    ) -> EvaluationResult:
        """
        评估 term 列表（核心 API - Decision 003）

        LINEAR 模式：
            1. 执行各 term 得到 Θ 矩阵列
            2. 最小二乘求解 y = Θξ
            3. 计算指标

        DIRECT 模式：
            需要先将 terms 组合为单棵树再执行
        """

    def evaluate_expression(
        self,
        expression: ASTNode,
        mode: EvaluationMode = EvaluationMode.LINEAR
    ) -> EvaluationResult:
        """
        评估单个表达式（便捷 API）

        如果 mode=LINEAR，先调用 split_terms 拆分
        """


# 指标计算
def compute_mse(y_pred: Tensor, y_true: Tensor) -> float:
    """均方误差"""

def compute_nmse(y_pred: Tensor, y_true: Tensor) -> float:
    """归一化均方误差 = MSE / Var(y_true)"""

def compute_r2(y_pred: Tensor, y_true: Tensor) -> float:
    """R² 决定系数"""

def compute_aic(mse: float, n_samples: int, n_params: int) -> float:
    """AIC = n*log(MSE) + 2*k"""
```

### LINEAR 模式流程

```python
def evaluate_terms(self, terms, mode=LINEAR):
    if mode == LINEAR:
        # 1. 执行各 term，构建 Θ 矩阵
        theta_columns = []
        for term in terms:
            result = self.executor.execute(term)
            if not result.is_valid:
                return self._invalid_result(f"Term {term} is invalid")
            theta_columns.append(result.value.flatten())

        theta = torch.stack(theta_columns, dim=1)  # (n_samples, n_terms)
        y = self.y_true.flatten()

        # 2. 最小二乘求解
        solve_result = self.solver.solve(theta, y)

        # 3. 计算指标
        y_pred = theta @ solve_result.coefficients
        mse = compute_mse(y_pred, y)
        nmse = compute_nmse(y_pred, y)
        r2 = solve_result.r2
        aic = compute_aic(mse, len(y), len(terms))

        return EvaluationResult(
            mse=mse,
            nmse=nmse,
            r2=r2,
            aic=aic,
            coefficients=solve_result.coefficients,
            complexity=sum(t.length() for t in terms)
        )
```

## Implementation steps

1. [ ] **Step 1**: 实现指标计算函数
   - Files: `src/kd2/core/evaluator/metrics.py`
   - Test: `pytest tests/unit/test_evaluator.py -k "test_metrics"`
   - Agent: Dev

2. [ ] **Step 2**: 实现 EvaluationResult 和 EvaluationMode
   - Files: `src/kd2/core/evaluator/evaluator.py`
   - Test: `pytest tests/unit/test_evaluator.py -k "test_result"`
   - Agent: Dev

3. [ ] **Step 3**: 实现 Evaluator.evaluate_terms (LINEAR)
   - Files: `src/kd2/core/evaluator/evaluator.py`
   - Test: `pytest tests/unit/test_evaluator.py -k "test_linear"`
   - Agent: Dev

4. [ ] **Step 4**: 实现 Evaluator.evaluate_terms (DIRECT)
   - Files: `src/kd2/core/evaluator/evaluator.py`
   - Test: `pytest tests/unit/test_evaluator.py -k "test_direct"`
   - Agent: Dev

5. [ ] **Step 5**: 实现 evaluate_expression 便捷 API
   - Files: `src/kd2/core/evaluator/evaluator.py`
   - Test: `pytest tests/unit/test_evaluator.py -k "test_expression"`
   - Agent: Dev

6. [ ] **Step 6**: 无效表达式处理
   - Files: `src/kd2/core/evaluator/evaluator.py`
   - Test: `pytest tests/unit/test_evaluator.py -k "test_invalid"`
   - Agent: Dev

## Acceptance criteria

### Functional

- [ ] LINEAR 模式正确构建 Θ 矩阵并求解
- [ ] DIRECT 模式直接计算指标
- [ ] MSE, NMSE, R² 计算正确
- [ ] AIC 计算正确（k = 非零系数数）
- [ ] 无效表达式返回 penalty 分数
- [ ] `evaluate_expression` 自动调用 `split_terms`

### Tests

- [ ] Unit tests pass: `pytest tests/unit/test_evaluator.py`
- [ ] Coverage >= 95%

### Quality

- [ ] Type hints complete: `mypy src/kd2/core/evaluator/`
- [ ] No lint errors: `ruff check src/kd2/core/`
- [ ] detach tensors when storing metrics

## Validation

```bash
# Minimal validation
pytest tests/unit/test_evaluator.py -v

# Full validation
pytest tests/unit/ -v --cov=src/kd2/core/evaluator
mypy src/kd2/core/evaluator
ruff check src/kd2/core/evaluator
```

## Constraints

- [ ] 返回的 coefficients 必须 detach
- [ ] 指标为 float，不是 Tensor

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| term 执行失败 | Med | Low | 返回 invalid result |
| Θ 矩阵病态 | Med | Med | solver 处理 + condition_number 警告 |

## Rollback plan

```bash
git revert <commit-hash>
```

## Notes

_Decision 003 要点：_
- `evaluate_terms()` 是核心 API
- split_terms 是 IR 层工具，不是 Evaluator 职责
- LINEAR 模式是 PDE 发现的主流方法

---

## Completion checklist

Before marking as `done`:

- [ ] All implementation steps completed
- [ ] All acceptance criteria met
- [ ] Tests written and passing
- [ ] Code reviewed (`/code-review`)
- [ ] No TODOs left in code
- [ ] `/wrap-up` completed
