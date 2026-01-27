# kd2 数据层设计

> **关联文档**：[arc_final.md](arc_final.md) - 主文档

---

## 一、PDEDataset 规范

```python
@dataclass
class AxisInfo:
    name: str                    # 用户自定义名称 "x", "y", "t"
    values: torch.Tensor         # 1D 坐标值
    is_periodic: bool = False

@dataclass
class FieldData:
    name: str                    # "u", "v"
    values: torch.Tensor         # nD 张量 [n_x, n_y, n_t, ...]

class DataTopology(Enum):
    GRID = "grid"           # 规则网格，支持有限差分
    SCATTERED = "scattered" # 散点，只能用 NN 求导 (Phase 5)

@dataclass
class PDEDataset:
    name: str
    task_type: TaskType          # PDE, ODE, REGRESSION
    topology: DataTopology = DataTopology.GRID

    # Grid 模式 (MVP)
    axes: Optional[Dict[str, AxisInfo]] = None
    axis_order: Optional[List[str]] = None
    fields: Optional[Dict[str, FieldData]] = None

    # Scattered 模式 (Phase 5)
    coords: Optional[torch.Tensor] = None         # (N, d)
    field_values: Optional[Dict[str, torch.Tensor]] = None

    # LHS 定义
    lhs_field: str = ""          # "u"
    lhs_axis: str = ""           # "t" → 表示 u_t = RHS

    # 元数据
    noise_level: float = 0.0
    ground_truth: Optional[str] = None
```

### 1.1 数据拓扑

| 拓扑 | 数据形状 | 有限差分 | NN 求导 | 典型来源 |
|------|---------|---------|---------|---------|
| **Grid** | `(n_x, n_y, n_t)` 规则张量 | ✅ | ✅ | 数值模拟 |
| **Scattered** | `(N, d)` 散点矩阵 | ❌ | ✅ | 实验测量 |

**MVP 策略**：Phase 1-4 只实现 Grid，Scattered 留给 Phase 5。

### 1.2 n 维支持原则

- 代码中**禁止**硬编码 `x`, `y`, `t` 等变量名
- 使用 `axes[axis_name]` 或 `coords[:, i]` 索引
- 配置文件定义坐标映射关系

---

## 二、数据集指纹（dataset_fp）

**已确定**：元信息 + 采样 hash，用于缓存隔离。

```python
def compute_dataset_fingerprint(dataset: PDEDataset) -> str:
    """
    计算数据集指纹

    纳入: name, topology, lhs_field, lhs_axis, shape, data_hash
    """
    # 1. 元信息
    meta = f"{dataset.name}:{dataset.topology.value}"
    meta += f":{dataset.lhs_field}:{dataset.lhs_axis}"

    # 2. 形状信息
    shapes = "_".join(f"{k}{v.values.shape}" for k, v in sorted(dataset.fields.items()))

    # 3. 数据 hash（大数据集采样）
    content_hash = hashlib.sha256()
    for field in sorted(dataset.fields.values(), key=lambda f: f.name):
        data = field.values.numpy()
        if data.nbytes > 10_000_000:  # > 10MB: 采样
            content_hash.update(data.ravel()[::1000].tobytes())
        else:
            content_hash.update(data.tobytes())

    return f"{meta}_{shapes}_{content_hash.hexdigest()[:8]}"
```

<!--
## 实现参考：dataset_fp 细节

Codex 建议补强（可选）：
- shapes 不一定能唯一决定 axis_order（多个轴长度相同时）
- 可显式纳入 axis_order 字符串
- 坐标 axes[*].values 也会改变导数，可纳入 axis 值的采样 hash

实际遇到缓存碰撞问题时再考虑。
-->

---

## 三、导数提供者（两层设计）

**设计原则**：Surrogate 网络封装在 `DerivativeProvider` 内部，通过依赖注入配置。

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
        """
        获取预计算的导数 (高吞吐路径)
        例如: get_derivative("u", "x", 2) → u_xx
        """

    @abstractmethod
    def diff(
        self,
        expression: torch.Tensor,
        axis: str,
        order: int
    ) -> torch.Tensor:
        """
        开放形式导数 (open-form diff)
        对任意子表达式求导
        """
```

### 3.1 FiniteDiffProvider

```python
class FiniteDiffProvider(DerivativeProvider):
    """有限差分导数提供者"""

    def __init__(
        self,
        dataset: PDEDataset,
        method: str = "central",
        accuracy: int = 2
    ):
        assert dataset.topology == DataTopology.GRID, "FD 需要 Grid 数据"
        self._cache: Dict[Tuple, torch.Tensor] = {}
        self._precompute(dataset)

    def get_derivative(self, field, axis, order) -> torch.Tensor:
        """返回预计算的导数张量"""
        return self._cache[(field, axis, order)]

    def diff(self, expression, axis, order) -> torch.Tensor:
        """
        对表达式结果做有限差分
        注意：这是对数值结果再做 FD，不是符号求导
        """
        # 实现时再决定是否支持
        raise NotImplementedError("待实现")
```

### 3.2 AutogradProvider

```python
class AutogradProvider(DerivativeProvider):
    """
    自动微分导数提供者
    内部封装 Surrogate 网络
    """

    def __init__(
        self,
        dataset: PDEDataset,
        surrogate: nn.Module,     # 依赖注入: PINN, MetaNet...
        scale_handler: ScaleHandler
    ):
        self.surrogate = surrogate
        self.scale_handler = scale_handler

    def get_derivative(self, field, axis, order) -> torch.Tensor:
        """通过 surrogate + autograd 获取导数"""

    def diff(self, expression, axis, order) -> torch.Tensor:
        """
        真正的 open-form diff: 对任意计算图节点求导
        使用 torch.autograd.grad
        """
        coords = self.dataset.get_coords_with_grad()
        result = expression
        for _ in range(order):
            result = torch.autograd.grad(
                result,
                coords[axis],
                grad_outputs=torch.ones_like(result),
                create_graph=True  # 需要时开启
            )[0]
        return result
```

<!--
## 实现参考：导数提供者细节

### SGA vs DISCOVER 对 diff 的使用
调研结果：
- SGA：不支持 open-form diff，只用预计算 terminals（工程优化）
- DISCOVER：完全支持 open-form diff，可以对任何中间表达式求导

kd2 策略：两者都提供，让插件自己选择。

### FiniteDiffProvider.diff() 实现选项
1. 不实现，约束禁用 diff token（当 provider=FD 时）
2. 实现对数值结果再做 FD（DISCOVER NumPy 版本就是这么做的）

DISCOVER 的 NumPy 版本支持对中间结果做 FD，所以方案 2 是可行的。
具体实现时根据需求决定。

### valid_mask
Codex 建议：FD 边界点可能不准确，可输出 valid_mask。
遇到回归质量问题时再考虑。
-->

---

## 四、导数命名规范

**已确定**：显示格式 + canonical 格式双层命名。

```python
# 显示格式（human-readable）
DISPLAY_FORMAT = "{field}_{axis*order}"  # u_x, u_xx, u_t

# 内部规范格式（canonical）
CANONICAL_FORMAT = "deriv:{field}:{axis}:{order}"  # deriv:u:x:2

def to_canonical(display: str) -> str:
    """u_xx -> deriv:u:x:2"""
    field, rest = display.split("_", 1)
    axis = rest[0]
    order = len(rest)
    return f"deriv:{field}:{axis}:{order}"

def to_display(canonical: str) -> str:
    """deriv:u:x:2 -> u_xx"""
    _, field, axis, order = canonical.split(":")
    return f"{field}_{axis * int(order)}"
```

**MVP 策略**：Phase 1-4 仅单轴导数 `u_x`, `u_xx`；Phase 5 扩展混合导数 `u_xy`。

---

## 五、ScaleHandler（Phase 2+）

**已确定**：Phase 1 不归一化，Phase 2 引入 ScaleHandler。

```python
class ScaleMode(Enum):
    NONE = "none"           # 不归一化（Phase 1 默认）
    MINMAX = "minmax"       # [0, 1]
    ZSCORE = "zscore"       # (x - μ) / σ

@dataclass
class ScaleInfo:
    mode: ScaleMode
    shift: float      # μ or min
    scale: float      # σ or (max - min)

class ScaleHandler:
    """
    尺度处理器：确保预计算导数和 open-form diff 尺度一致

    关键：d^n(u)/dx^n 的尺度 = field_scale / coord_scale^n
    """

    def __init__(self, dataset: PDEDataset, mode: ScaleMode = ScaleMode.NONE):
        self.mode = mode
        self.field_scales: Dict[str, ScaleInfo] = {}
        self.coord_scales: Dict[str, ScaleInfo] = {}

    def get_derivative_scale(self, field: str, axis: str, order: int) -> float:
        """导数的尺度因子"""
        if self.mode == ScaleMode.NONE:
            return 1.0
        f_scale = self.field_scales[field].scale
        c_scale = self.coord_scales[axis].scale
        return f_scale / (c_scale ** order)
```

<!--
## 实现参考：尺度一致性

### 关键约束
预计算 u_x 和 open-form diff(expr, x) 必须在同一尺度体系。
ScaleHandler 的职责就是确保这一点。

### 链式法则校正
如果数据被归一化到 [0, 1]：
    x_norm = (x - x_min) / (x_max - x_min)
则导数需要校正：
    du/dx = du/dx_norm * dx_norm/dx = du/dx_norm * 1/(x_max - x_min)

通用公式：scale = field_scale / (coord_scale ** order)
-->

---

## 六、数据加载器

```python
def load_dataset(name: str, **kwargs) -> PDEDataset:
    """加载内置数据集"""

def load_from_mat(path: Path, config: Dict) -> PDEDataset:
    """从 .mat 文件加载"""

def load_from_numpy(path: Path, config: Dict) -> PDEDataset:
    """从 .npy/.npz 文件加载"""
```

**内置数据集**：Burgers, KdV, Chafee-Infante 等。

<!--
## 实现参考：数据布局

### 标准 layout
Codex 建议：平台内部统一一个 point-layout（N = Π n_axis），
所有 Executor/Evaluator/Solver 都以 flat 向量工作。

可选策略：
- 方案 A：内部始终 flat，需要时 reshape
- 方案 B：保持 grid shape，需要时 flatten

实现时根据实际需求选择。FD 需要 grid shape，回归需要 flat。
-->
