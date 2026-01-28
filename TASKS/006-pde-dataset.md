# Task: 006 - PDEDataset & Synthetic Data

> Status: `ready`
> Parent: SPEC.md (Phase 1)
> Assignee: claude

## Goal

实现 PDE 数据规范（PDEDataset）和合成 Burgers 数据生成器，支撑后续 Executor 和 Evaluator 测试。

## Non-goals

- [ ] 不实现 Scattered 拓扑支持（Phase 5）
- [ ] 不实现真实 .mat 数据加载（按需后续添加）
- [ ] 不实现 ScaleHandler 归一化（Phase 2）

## Context

### Relevant files

```
src/kd2/data/schema.py        # 新建
src/kd2/data/synthetic.py     # 新建
src/kd2/data/__init__.py      # 新建
tests/unit/test_schema.py     # 新建
tests/unit/test_synthetic.py  # 新建
```

### Current behavior

N/A - 新模块

### Dependencies

- Requires: 无
- Blocks: 007 (DerivativeProvider)

## Design

### Approach

1. 定义 PDEDataset 数据结构（遵循 arc_data.md 规范）
2. 实现数据集指纹计算（用于缓存隔离）
3. 生成合成 Burgers 数据（精确解或简单数值解）

### Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Grid 优先 | MVP 只实现 Grid | Phase 1-4 策略 |
| 合成数据 | 数值求解 Burgers | 比 Cole-Hopf 更通用 |
| 坐标硬编码 | 禁止 | n 维支持原则 |

### API 设计

```python
class TaskType(Enum):
    PDE = "pde"
    ODE = "ode"
    REGRESSION = "regression"

class DataTopology(Enum):
    GRID = "grid"
    SCATTERED = "scattered"

@dataclass
class AxisInfo:
    name: str                    # "x", "t", ...
    values: torch.Tensor         # 1D 坐标值
    is_periodic: bool = False

@dataclass
class FieldData:
    name: str                    # "u", "v", ...
    values: torch.Tensor         # nD 张量

@dataclass
class PDEDataset:
    name: str
    task_type: TaskType
    topology: DataTopology = DataTopology.GRID

    # Grid 模式
    axes: dict[str, AxisInfo] | None = None
    axis_order: list[str] | None = None
    fields: dict[str, FieldData] | None = None

    # LHS 定义
    lhs_field: str = ""
    lhs_axis: str = ""

    # 元数据
    noise_level: float = 0.0
    ground_truth: str | None = None

    def get_shape(self) -> tuple[int, ...]:
        """返回数据形状"""

    def get_coords(self, axis: str) -> torch.Tensor:
        """获取指定轴坐标"""

    def get_field(self, name: str) -> torch.Tensor:
        """获取指定场数据"""


def compute_dataset_fingerprint(dataset: PDEDataset) -> str:
    """计算数据集指纹用于缓存隔离"""


def generate_burgers_data(
    nx: int = 256,
    nt: int = 101,
    nu: float = 0.1,
    noise_level: float = 0.0,
    device: torch.device = torch.device("cpu")
) -> PDEDataset:
    """生成 Burgers 方程数据：u_t + u*u_x = nu*u_xx"""
```

### Burgers 数据生成

使用简单的数值求解（有限差分 + Euler 时间积分）或解析解（Cole-Hopf 对特定初始条件）：

```
Burgers 方程: u_t + u * u_x = nu * u_xx
初始条件: u(x, 0) = -sin(pi * x)
边界条件: 周期性
```

## Implementation steps

1. [ ] **Step 1**: 实现数据 schema（枚举 + dataclass）
   - Files: `src/kd2/data/schema.py`
   - Test: `pytest tests/unit/test_schema.py -k "test_schema_basic"`
   - Agent: Dev

2. [ ] **Step 2**: 实现 PDEDataset 辅助方法
   - Files: `src/kd2/data/schema.py`
   - Test: `pytest tests/unit/test_schema.py -k "test_dataset_methods"`
   - Agent: Dev

3. [ ] **Step 3**: 实现 dataset fingerprint
   - Files: `src/kd2/data/schema.py`
   - Test: `pytest tests/unit/test_schema.py -k "test_fingerprint"`
   - Agent: Dev

4. [ ] **Step 4**: 实现 Burgers 数据生成器
   - Files: `src/kd2/data/synthetic.py`
   - Test: `pytest tests/unit/test_synthetic.py -k "test_burgers"`
   - Agent: Dev

## Acceptance criteria

### Functional

- [ ] PDEDataset 创建正确，字段访问正常
- [ ] `get_shape()`, `get_coords()`, `get_field()` 正常工作
- [ ] fingerprint 对相同数据稳定，不同数据不同
- [ ] Burgers 数据形状正确：`(nx, nt)`
- [ ] Burgers 数据满足方程（残差小）

### Tests

- [ ] Unit tests pass: `pytest tests/unit/test_schema.py tests/unit/test_synthetic.py`
- [ ] Coverage >= 90%

### Quality

- [ ] Type hints complete: `mypy src/kd2/data/`
- [ ] No lint errors: `ruff check src/kd2/data/`
- [ ] torch.Tensor 全程，device-aware

## Validation

```bash
# Minimal validation
pytest tests/unit/test_schema.py tests/unit/test_synthetic.py -v

# Full validation
pytest tests/unit/ -v --cov=src/kd2/data
mypy src/kd2/data
ruff check src/kd2/data
```

## Constraints

- [ ] 禁止硬编码坐标名（"x", "t" 等）
- [ ] torch.Tensor 全程（不用 numpy）

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Burgers 数值不稳定 | Med | Low | 小 dt，高粘性 nu |
| fingerprint 碰撞 | Low | Low | 含 shape + 采样 hash |

## Rollback plan

```bash
git revert <commit-hash>
```

## Notes

_Burgers 方程是 Phase 1-3 的标准测试用例：_
- 真实方程：`u_t = -u * u_x + 0.1 * u_xx`
- 系数：`[-1.0, 0.1]`

---

## Completion checklist

Before marking as `done`:

- [ ] All implementation steps completed
- [ ] All acceptance criteria met
- [ ] Tests written and passing
- [ ] Code reviewed (`/code-review`)
- [ ] No TODOs left in code
- [ ] `/wrap-up` completed
