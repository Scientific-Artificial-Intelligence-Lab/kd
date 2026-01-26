# kd2 架构设计文档（最终版）

> **状态**：正式版 v1.0
> **定位**：符号回归领域（特别是 PDE 发现）的通用实验平台
> **设计原则**：String as Interface, Tensor as Data

---

## 一、设计目标与约束

### 1.1 核心目标

1. **统一平台**：整合 DISCOVER、SGA、DLGA、PySR 等算法为可插拔插件
2. **n 维支持**：代码中不硬编码 x、y、t，统一使用坐标配置
3. **可微执行**：全程使用 `torch.Tensor`，保留计算图用于未来端到端优化
4. **高效评估**：支持万级候选的批量评估，带缓存和数值护栏
5. **可扩展性**：未来可接入 Agent 自动实验、LLM 搜索等

### 1.2 设计决策总结

| 决策点 | 选择 | 理由 |
|--------|------|------|
| **数据类型** | `torch.Tensor` 全程 | 保留计算图，未来可端到端优化 |
| **IR 层数** | 两层 (GenIR + AnalysisIR) | ExecIR (DAG+CSE) 作为 Phase 5 优化 |
| **首个插件** | SGA | 代码简单，GA 逻辑清晰，适合验证核心架构 |
| **导数模式** | 有限差分 + 自动微分并行 | 灵活适配不同场景 |
| **插件接口** | `propose()/update()` | 平台统一控制评估，便于缓存和并行 |
| **缓存方案** | DiskCache | 无需运维，单机科研友好 |
| **哈希策略** | 可交换算子子节点排序后哈希 | `a+b` 与 `b+a` 共享缓存 |
| **实验追踪** | WandB + Hydra | 科研标配，配置管理优雅 |
| **DISCOVER 集成** | PyTorch 重写 | TF1 不兼容，保留算法逻辑 |
| **技术栈** | Python 3.11+, PyTorch 2.x | 未来可 C/C++ 优化瓶颈 |
| **数据拓扑** | Grid 优先，Scattered 延后 | Phase 1 定义字段，MVP 只实现 Grid |
| **Checkpointing** | Phase 3 纳入 | 插件 `get_state/set_state` + ExperimentManager |
| **结果档案** | 简化版 ResultArchive | 收集结果 + 后处理提取 Pareto 前沿 |

---

## 二、系统架构

### 2.1 五层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                            │
│   CLI Tool  │  Web UI  │  Agent API  │  Notebooks                │
├─────────────────────────────────────────────────────────────────┤
│                     Experiment Layer                             │
│   ExperimentManager  │  ResultStore  │  Visualization            │
│   Hydra Config       │  WandB Logger                             │
├─────────────────────────────────────────────────────────────────┤
│                      Plugin Layer                                │
│   DISCOVER  │  SGA  │  DLGA  │  PySR  │  Custom                  │
│              ↓ AlgorithmPlugin Interface (propose/update)        │
├─────────────────────────────────────────────────────────────────┤
│                       Core Layer                                 │
│   IR System │ Executor │ Evaluator │ Constraints │ LinearSolve  │
│   Library   │ Cache(DiskCache)                                   │
├─────────────────────────────────────────────────────────────────┤
│                    Foundation Layer                              │
│   Data Layer │ DerivativeProvider │ PDEDataset                   │
│   PyTorch    │ NumPy              │ SymPy (管理层)                │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
kd2/
├── src/kd2/
│   ├── core/                    # 核心模块
│   │   ├── ir/                  # 两层 IR 系统
│   │   │   ├── token.py        # Token, TokenType, Arity
│   │   │   ├── gen_ir.py       # GenIR (前缀 token 序列, canonical)
│   │   │   ├── analysis_ir.py  # AnalysisIR (AST 树, view)
│   │   │   └── converters.py   # IR 转换器
│   │   │
│   │   ├── library/             # Token/算子库
│   │   │   ├── operators.py    # 内置算子 (add, mul, sin, cos, ...)
│   │   │   ├── diff_ops.py     # 微分算子 (diff, diff2, ...)
│   │   │   └── registry.py     # 算子注册表
│   │   │
│   │   ├── executor/            # 执行引擎
│   │   │   ├── base.py         # Executor ABC
│   │   │   ├── stack_exec.py   # NumPy 栈执行器
│   │   │   ├── torch_exec.py   # PyTorch 可微执行器
│   │   │   └── safety.py       # 数值稳定性 (safe_div, NaN/Inf guard)
│   │   │
│   │   ├── evaluator/           # 评估器
│   │   │   ├── base.py         # Evaluator ABC (返回 Tensor)
│   │   │   ├── metrics.py      # MSE, NMSE, R2, AIC
│   │   │   ├── complexity.py   # 复杂度评估
│   │   │   └── reward.py       # 奖励函数 (DISCOVER 风格)
│   │   │
│   │   ├── constraints/         # 约束系统 (两阶段)
│   │   │   ├── base.py         # Constraint ABC
│   │   │   ├── sampling/       # 采样期约束 (mask/logit调整)
│   │   │   │   ├── length.py   # 长度约束
│   │   │   │   ├── trig.py     # 三角函数嵌套约束
│   │   │   │   └── relational.py # 父子/兄弟关系约束
│   │   │   ├── evaluation/     # 评估期约束 (fail-fast)
│   │   │   │   ├── diff.py     # 微分约束 (lhs_axis 禁止)
│   │   │   │   └── physical.py # 物理约束
│   │   │   └── joint_prior.py  # 联合先验
│   │   │
│   │   ├── linear_solve/        # 线性求解器
│   │   │   ├── base.py         # SparseSolver ABC
│   │   │   ├── lstsq.py        # 最小二乘
│   │   │   ├── stridge.py      # STRidge 稀疏回归
│   │   │   └── lasso.py        # Lasso
│   │   │
│   │   └── cache/               # 缓存系统
│   │       ├── disk_cache.py   # DiskCache 封装
│   │       └── hash.py         # Canonical hash (排序子节点)
│   │
│   ├── data/                    # 数据层
│   │   ├── schema.py           # PDEDataset 规范
│   │   ├── loaders.py          # 数据加载器 (numpy, mat)
│   │   ├── generators.py       # 合成数据生成
│   │   └── derivatives/         # 导数提供者 (两层设计)
│   │       ├── base.py         # DerivativeProvider ABC
│   │       ├── finite_diff.py  # 有限差分
│   │       ├── autograd.py     # 自动微分 (surrogate 封装)
│   │       └── scale.py        # 尺度处理 (链式法则校正)
│   │
│   ├── plugins/                 # 插件系统
│   │   ├── base.py             # AlgorithmPlugin (propose/update)
│   │   ├── registry.py         # 插件注册
│   │   ├── sga/                # SGA 插件 (首个)
│   │   ├── discover/           # DISCOVER 插件 (PyTorch 重写)
│   │   ├── dlga/               # DLGA 插件
│   │   └── pysr/               # PySR 插件
│   │
│   ├── experiment/              # 实验管理
│   │   ├── manager.py          # ExperimentManager
│   │   ├── runner.py           # 运行器 (统一 propose/eval/update 循环)
│   │   └── results.py          # 结果存储
│   │
│   ├── visualization/           # 可视化 (接口优先)
│   │   ├── tree_plot.py        # 表达式树
│   │   ├── field_plot.py       # 场数据
│   │   ├── pareto.py           # Pareto Front
│   │   ├── residual.py         # 残差图
│   │   └── latex.py            # LaTeX 渲染
│   │
│   └── config/                  # 配置管理 (Hydra)
│       ├── schema.py           # 配置 schema
│       └── defaults/           # 默认配置
│
├── tests/
├── examples/
├── data/                        # 数据集
└── pyproject.toml
```

---

## 三、核心设计

### 3.1 两层 IR 设计

**设计原则**：GenIR 是「唯一真相」(Canonical)，AnalysisIR 是「视图」(View)。

| IR 层 | 表示形式 | 职责 | 特性 |
|-------|---------|------|------|
| **GenIR** | 前缀 Token 序列 | 缓存键、日志、复现、去重 | Hashable, Immutable |
| **AnalysisIR** | AST 树 | 结构分析、约束检查、执行、可视化 | 可变、可遍历 |

**GenIR 规范**：

```python
@dataclass(frozen=True)
class GenIR:
    """前缀 Token 序列，作为表达式的 Canonical 表示"""
    tokens: Tuple[str, ...]  # 不可变

    @staticmethod
    def from_string(s: str, library: Library) -> "GenIR":
        """从逗号分隔字符串解析"""

    def to_canonical_string(self) -> str:
        """转为可哈希字符串（用于缓存键）"""

    def is_complete(self, library: Library) -> bool:
        """检查是否为完整表达式（dangling == 0）"""

    def dangling(self, library: Library) -> int:
        """计算 dangling 状态: 1 + cumsum(arity - 1)"""

    def finish_tokens(self, library: Library) -> List[str]:
        """返回可以结束当前不完整表达式的 terminal tokens"""

    def __hash__(self) -> int:
        """基于 tokens 的稳定哈希"""
```

**AnalysisIR 规范**：

```python
@dataclass
class ASTNode:
    """AST 树节点"""
    token: Token
    children: List["ASTNode"]

    def to_canonical_gen_ir(self, library: Library) -> GenIR:
        """转回 GenIR（可交换算子子节点排序后）"""

    def split_terms(self) -> List["ASTNode"]:
        """按 +/- 拆分为多个 term（用于 STRidge）"""

    def depth(self) -> int
    def length(self) -> int
    def complexity(self) -> float  # Token 加权复杂度
```

**Canonical Hash 策略**：

```python
def canonical_hash(node: ASTNode) -> str:
    """可交换算子子节点排序后哈希"""
    if node.token.is_commutative:  # add, mul
        children_hashes = sorted([canonical_hash(c) for c in node.children])
    else:
        children_hashes = [canonical_hash(c) for c in node.children]
    return hash((node.token.name, tuple(children_hashes)))
```

**Dangling 逻辑**：

```python
def calculate_dangling(tokens: List[str], library: Library) -> List[int]:
    """
    计算每个位置的 dangling 状态
    dangling[i] = 还需要多少个 token 才能完成表达式

    规则: dangling[0] = 1
          dangling[i+1] = dangling[i] + arity(token[i]) - 1

    完整表达式: dangling[-1] + arity(tokens[-1]) - 1 == 0
    """
    dangling = [1]
    for tok in tokens:
        arity = library.get(tok).arity
        dangling.append(dangling[-1] + arity - 1)
    return dangling
```

**finish_tokens 设计**（fail-fast 策略）：

```python
def finish_tokens(partial_ir: GenIR, library: Library) -> List[str]:
    """
    返回可以「合法结束」当前不完整表达式的终端 token 列表

    Fail-fast 策略: 如果 dangling > 可用终端数量，直接返回空列表，
    由插件负责处理（重新生成或丢弃）
    """
    d = partial_ir.dangling(library)
    terminals = library.get_terminals()

    if d > len(terminals):
        return []  # 无法完成，fail-fast

    return terminals[:d]  # 返回前 d 个终端
```

### 3.2 Token 与算子库

```python
class TokenType(Enum):
    TERMINAL = 0      # 变量、常数 (arity=0)
    UNARY = 1         # sin, cos, exp, n2, n3 (arity=1)
    BINARY = 2        # add, sub, mul, div (arity=2)
    DIFF = 3          # diff, diff2 (特殊处理, arity=2)

@dataclass(frozen=True)
class Token:
    name: str                    # "add", "mul", "diff", "u", "x"
    arity: int                   # 参数数量
    complexity: float            # 复杂度权重
    token_type: TokenType
    is_commutative: bool = False # 是否可交换 (add, mul)
    function: Optional[Callable] = None  # 实际计算函数
```

**默认算子集**（MVP）：

| 类型 | 算子 | Arity | 复杂度 |
|------|------|-------|--------|
| Binary | `add`, `sub`, `mul`, `div` | 2 | 1.0 |
| Unary | `sin`, `cos`, `exp` | 1 | 2.0 |
| Unary | `n2` (x²), `n3` (x³) | 1 | 1.5 |
| Diff | `diff`, `diff2` | 2 | 2.0 |
| Terminal | `u`, coords, `C` (常数) | 0 | 1.0 |

### 3.3 数据层设计

**PDEDataset 规范**：

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
    topology: DataTopology = DataTopology.GRID  # 数据拓扑

    # Grid 模式 (MVP)
    axes: Optional[Dict[str, AxisInfo]] = None    # {"x": ..., "y": ..., "t": ...}
    axis_order: Optional[List[str]] = None        # ["x", "y", "t"] 定义张量维度顺序
    fields: Optional[Dict[str, FieldData]] = None # {"u": ..., "v": ...}

    # Scattered 模式 (Phase 5)
    coords: Optional[torch.Tensor] = None         # (N, d) 散点坐标
    field_values: Optional[Dict[str, torch.Tensor]] = None  # {"u": (N,), ...}

    # LHS 定义
    lhs_field: str = ""          # "u"
    lhs_axis: str = ""           # "t" → 表示 u_t = RHS

    # 元数据
    noise_level: float = 0.0
    ground_truth: Optional[str] = None  # 真实方程（验证用）

    def validate(self):
        """验证数据完整性"""
        if self.topology == DataTopology.GRID:
            assert self.axes is not None, "Grid 模式需要 axes"
            assert self.fields is not None, "Grid 模式需要 fields"
        else:
            assert self.coords is not None, "Scattered 模式需要 coords"
            assert self.field_values is not None, "Scattered 模式需要 field_values"

    def get_meshgrid(self) -> Dict[str, torch.Tensor]:
        """获取 meshgrid 形式的坐标 (仅 Grid 模式)"""
        assert self.topology == DataTopology.GRID
```

**数据拓扑说明**：

| 拓扑 | 数据形状 | 有限差分 | NN 求导 | 典型来源 |
|------|---------|---------|---------|---------|
| **Grid** | `(n_x, n_y, n_t)` 规则张量 | ✅ | ✅ | 数值模拟 |
| **Scattered** | `(N, d)` 散点矩阵 | ❌ | ✅ | 实验测量 |

**MVP 策略**：Phase 1-4 只实现 Grid，Scattered 留给 Phase 5。

**n 维支持原则**：

- 代码中**禁止**硬编码 `x`, `y`, `t` 等变量名
- 使用 `axes[axis_name]` 或 `coords[:, i]` 索引
- 配置文件定义坐标映射关系

### 3.4 导数提供者（两层设计）

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
        expression: torch.Tensor,  # 计算图节点
        axis: str,
        order: int
    ) -> torch.Tensor:
        """
        开放形式导数 (open-form diff)
        对任意子表达式求导
        例如: diff(u * u_x, "x", 1) → (u * u_x)_x
        """
```

**有限差分提供者**：

```python
class FiniteDiffProvider(DerivativeProvider):
    """有限差分导数提供者"""

    def __init__(
        self,
        dataset: PDEDataset,
        method: str = "central",  # "central", "forward", "backward"
        accuracy: int = 2         # 精度阶数
    ):
        self._cache: Dict[Tuple, torch.Tensor] = {}
        self._precompute(dataset)

    def get_derivative(self, field, axis, order) -> torch.Tensor:
        """返回预计算的导数张量"""
        return self._cache[(field, axis, order)]

    def diff(self, expression, axis, order) -> torch.Tensor:
        """
        有限差分不支持真正的 open-form diff
        只能对已知场变量求导
        """
        raise NotImplementedError(
            "FiniteDiff 不支持 open-form diff，请使用 AutogradProvider"
        )
```

**自动微分提供者**（Surrogate 封装）：

```python
class AutogradProvider(DerivativeProvider):
    """
    自动微分导数提供者
    内部封装 Surrogate 网络（PINN/MetaNet）
    """

    def __init__(
        self,
        dataset: PDEDataset,
        surrogate: nn.Module,     # 依赖注入: PINN, MetaNet, DeepONet...
        scale_handler: ScaleHandler
    ):
        self.surrogate = surrogate
        self.scale_handler = scale_handler
        self._derivative_cache: Dict[Tuple, torch.Tensor] = {}

    def get_derivative(self, field, axis, order) -> torch.Tensor:
        """
        通过 surrogate + autograd 获取导数
        结果缓存以支持高吞吐
        """
        key = (field, axis, order)
        if key not in self._derivative_cache:
            self._derivative_cache[key] = self._compute_derivative(field, axis, order)
        return self._derivative_cache[key]

    def diff(self, expression, axis, order) -> torch.Tensor:
        """
        真正的 open-form diff: 对任意计算图节点求导
        使用 torch.autograd.grad
        """
        coords = self.dataset.get_coords_with_grad()  # requires_grad=True

        result = expression
        for _ in range(order):
            result = torch.autograd.grad(
                result,
                coords[axis],
                grad_outputs=torch.ones_like(result),
                create_graph=True
            )[0]

        # 尺度校正 (链式法则)
        result = self.scale_handler.apply(result, axis, order)
        return result
```

**尺度处理**（链式法则校正）：

```python
class ScaleHandler:
    """
    处理归一化坐标的链式法则校正

    如果数据被归一化到 [0, 1]:
        x_norm = (x - x_min) / (x_max - x_min)

    则导数需要校正:
        du/dx = du/dx_norm * dx_norm/dx
              = du/dx_norm * 1/(x_max - x_min)

    通用公式:
        scale = y_std / (x_std ** order)
    """

    def __init__(self, dataset: PDEDataset):
        self.scales: Dict[str, float] = {}
        for axis_name, axis_info in dataset.axes.items():
            self.scales[axis_name] = axis_info.values.std().item()

    def apply(
        self,
        derivative: torch.Tensor,
        axis: str,
        order: int
    ) -> torch.Tensor:
        scale = 1.0 / (self.scales[axis] ** order)
        return derivative * scale
```

### 3.5 执行器设计

**执行器接口**：

```python
class Executor(ABC):
    @abstractmethod
    def execute(
        self,
        ir: AnalysisIR,
        context: ExecutionContext
    ) -> ExecutionResult:
        """执行单个表达式"""

    def execute_batch(
        self,
        irs: List[AnalysisIR],
        context: ExecutionContext
    ) -> List[ExecutionResult]:
        """
        批量执行（接口优先）
        Phase 1: List comprehension 实现
        未来: torch.vmap 优化
        """
        return [self.execute(ir, context) for ir in irs]

@dataclass
class ExecutionContext:
    fields: Dict[str, torch.Tensor]      # {"u": tensor, "v": tensor}
    coords: Dict[str, torch.Tensor]      # {"x": tensor, "t": tensor}
    derivative_provider: DerivativeProvider
    constants: Dict[str, torch.Tensor]   # {"C1": tensor, ...}

@dataclass
class ExecutionResult:
    value: torch.Tensor      # 计算结果（保留计算图！）
    is_valid: bool           # 是否有效
    error_type: Optional[str]  # "nan", "inf", "domain_error"
```

**数值稳定性（safety.py）**：

```python
def safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """保护版除法，避免除零"""
    return a / (b + eps * torch.sign(b + 1e-20))

def check_validity(result: torch.Tensor) -> Tuple[bool, Optional[str]]:
    """检查结果有效性"""
    if torch.isnan(result).any():
        return False, "nan"
    if torch.isinf(result).any():
        return False, "inf"
    return True, None
```

### 3.6 评估器设计

**设计决策**：Evaluator 返回 `torch.Tensor`，保留计算图用于端到端优化。

```python
class Evaluator:
    """统一评估器"""

    def __init__(
        self,
        metrics: List[Metric],
        linear_solver: SparseSolver,
        complexity_weights: Dict[str, float]
    ):
        self.metrics = metrics
        self.linear_solver = linear_solver

    def evaluate(
        self,
        ir: AnalysisIR,
        execution_result: ExecutionResult,
        target: torch.Tensor
    ) -> EvaluationResult:
        """完整评估"""

    def evaluate_with_regression(
        self,
        ir: AnalysisIR,
        context: ExecutionContext,
        target: torch.Tensor
    ) -> EvaluationResult:
        """
        拆项 + 线性回归评估（SGA/DISCOVER 核心路径）

        1. 将表达式按 +/- 拆分为 terms
        2. 对每个 term 执行得到数值列
        3. 组成特征矩阵 Θ = [term1, term2, ...]
        4. 用 STRidge/Lasso 求解 target = Θξ
        5. 计算残差和各项指标
        """

@dataclass
class EvaluationResult:
    # 准确度 (torch.Tensor 保留计算图)
    mse: torch.Tensor
    nmse: torch.Tensor
    r2: torch.Tensor
    residual: torch.Tensor

    # 复杂度
    complexity: float
    term_count: int
    depth: int
    length: int

    # 综合分数
    reward: float    # DISCOVER 风格
    aic: float       # AIC = 2k + 2ln(MSE)

    # 系数
    coefficients: List[float]
    selected_terms: List[ASTNode]

    # 有效性
    is_valid: bool
    invalid_reason: Optional[str]
```

### 3.7 约束系统（两阶段设计）

**设计原则**：约束分为「采样期」和「评估期」两阶段。

#### 采样期约束（Sampling Phase）

在生成/变异时调整采样概率，防止生成非法结构。

```python
class SamplingConstraint(ABC):
    """采样期约束：调整 token 采样概率"""

    @abstractmethod
    def compute_mask(
        self,
        partial_ir: GenIR,
        parent: Optional[str],
        sibling: Optional[str],
        dangling: int,
        library: Library
    ) -> torch.Tensor:
        """
        返回 logit mask: 0 表示允许，-inf 表示禁止
        shape: [n_tokens]
        """
```

**采样期约束实现**：

```python
class LengthConstraint(SamplingConstraint):
    """限制表达式最大长度"""

class TrigConstraint(SamplingConstraint):
    """限制三角函数嵌套: sin(sin(x)) 禁止"""

class RelationalConstraint(SamplingConstraint):
    """禁止特定父子/兄弟关系: div 的右子节点不能是 div"""
```

#### 评估期约束（Evaluation Phase）

在评估时进行 fail-fast 检查，拒绝物理/数值非法表达式。

```python
class EvaluationConstraint(ABC):
    """评估期约束：fail-fast 检查"""

    @abstractmethod
    def check(self, ir: AnalysisIR, context: ExecutionContext) -> ConstraintResult:
        """
        检查表达式是否满足约束
        返回: (is_valid, reason)
        """

@dataclass
class ConstraintResult:
    is_valid: bool
    reason: Optional[str] = None
```

**评估期约束实现**：

```python
class LHSAxisConstraint(EvaluationConstraint):
    """
    关键约束: RHS 不能对 LHS 轴求导

    例如: 对于 u_t = RHS，RHS 中禁止出现 u_t, u_tt 等
    防止退化解: u_t = u_t
    """

class NumericalConstraint(EvaluationConstraint):
    """数值有效性检查: NaN, Inf, 极值"""
```

**联合先验**：

```python
class JointPrior:
    """组合多个约束"""

    def __init__(
        self,
        sampling_constraints: List[SamplingConstraint],
        evaluation_constraints: List[EvaluationConstraint]
    ):
        self.sampling = sampling_constraints
        self.evaluation = evaluation_constraints

    def get_sampling_mask(self, ...):
        """合并所有采样期约束的 mask"""

    def check_evaluation(self, ir, context):
        """顺序检查所有评估期约束，fail-fast"""
```

### 3.8 线性求解器

```python
class SparseSolver(ABC):
    """稀疏回归求解器抽象基类"""

    @abstractmethod
    def solve(
        self,
        Theta: torch.Tensor,  # [n_samples, n_terms]
        y: torch.Tensor,      # [n_samples]
    ) -> SolveResult:
        """求解 y = Θξ"""

@dataclass
class SolveResult:
    coefficients: torch.Tensor  # ξ
    selected_indices: List[int] # 非零系数的索引
    residual: torch.Tensor
    aic: float
    r2: float
```

**STRidge 实现**：

```python
class STRidgeSolver(SparseSolver):
    """
    Sequential Threshold Ridge Regression

    算法:
    1. 初始 Ridge: ξ = (Θ'Θ + αI)^{-1} Θ'y
    2. 迭代阈值化:
       - 将 |ξ_i| < threshold 的系数置零
       - 在非零系数上重新 Ridge
    3. 返回稀疏解
    """

    def __init__(
        self,
        threshold: float = 0.1,
        alpha: float = 1e-5,      # Ridge 正则化系数
        max_iter: int = 10
    ):
        ...
```

### 3.9 缓存系统

```python
class ExpressionCache:
    """表达式评估缓存（DiskCache 封装）"""

    def __init__(self, cache_dir: str = ".kd2_cache"):
        self.cache = diskcache.Cache(cache_dir)

    def get_key(self, ir: AnalysisIR) -> str:
        """获取 canonical hash 作为缓存键"""
        return canonical_hash(ir)

    def get(self, ir: AnalysisIR) -> Optional[EvaluationResult]:
        key = self.get_key(ir)
        return self.cache.get(key)

    def set(self, ir: AnalysisIR, result: EvaluationResult):
        key = self.get_key(ir)
        self.cache.set(key, result)
```

### 3.10 算法插件接口

**设计决策**：采用 `propose()/update()` 模式，而非 `search()`。

```python
class AlgorithmPlugin(ABC):
    """算法插件抽象基类"""

    name: str
    version: str
    algorithm_type: AlgorithmType  # RL, GA, HYBRID

    @abstractmethod
    def configure(self, config: Dict) -> None:
        """设置算法参数"""

    @abstractmethod
    def setup(
        self,
        dataset: PDEDataset,
        library: Library,
        constraints: JointPrior,
    ) -> None:
        """初始化搜索环境"""

    @abstractmethod
    def propose(self, k: int) -> List[Candidate]:
        """
        生成 k 个候选表达式
        返回: GenIR 列表
        """

    @abstractmethod
    def update(self, feedback: List[Feedback]) -> None:
        """
        根据评估反馈更新内部状态
        feedback: [(candidate_id, EvaluationResult), ...]
        """

    @abstractmethod
    def get_best(self) -> SearchResult:
        """获取当前最优结果"""

@dataclass
class Candidate:
    id: str
    gen_ir: GenIR

@dataclass
class Feedback:
    candidate_id: str
    result: EvaluationResult
```

**ExperimentManager 统一调度**：

```python
class ExperimentManager:
    """实验管理器：统一调度 propose/eval/update 循环"""

    def __init__(
        self,
        plugin: AlgorithmPlugin,
        executor: Executor,
        evaluator: Evaluator,
        cache: ExpressionCache,
        logger: WandBLogger,
        result_archive: "ResultArchive",
        callback: Optional[ProgressCallback] = None
    ):
        self.plugin = plugin
        self.result_archive = result_archive
        # ...

    def run(self, max_iterations: int, batch_size: int = 100, checkpoint_every: int = 10):
        """
        主循环:
        1. plugin.propose(batch_size) → 候选列表
        2. 批量执行 + 评估（带缓存）
        3. plugin.update(feedback)
        4. result_archive.add(results)  # 收集结果
        5. 记录日志
        6. 每 checkpoint_every 代保存检查点
        7. 重复直到收敛或达到 max_iterations
        """

    def save_checkpoint(self, path: Path) -> None:
        """保存检查点"""

    def load_checkpoint(self, path: Path) -> None:
        """从检查点恢复"""
```

### 3.11 断点续训 (Checkpointing)

```python
@dataclass
class Checkpoint:
    """检查点"""
    generation: int
    plugin_state: bytes          # 插件内部状态 (population/controller weights)
    result_archive: "ResultArchive"
    rng_state: Dict              # 随机数状态（可复现）
    config: Dict                 # 实验配置
    timestamp: float

class AlgorithmPlugin(ABC):
    # ... 现有接口 ...

    def get_state(self) -> bytes:
        """序列化内部状态（用于检查点）"""

    def set_state(self, state: bytes) -> None:
        """恢复内部状态"""
```

**使用示例**：

```python
# 保存检查点
manager.save_checkpoint("checkpoints/exp_001_gen_50.ckpt")

# 从检查点恢复
manager.load_checkpoint("checkpoints/exp_001_gen_50.ckpt")
manager.run(max_iterations=100)  # 从 gen 50 继续
```

### 3.12 结果档案 (ResultArchive)

```python
@dataclass
class ArchivedResult:
    """档案中的结果"""
    gen_ir: GenIR
    expression: str
    latex: str
    nmse: float
    complexity: float
    aic: float
    coefficients: List[float]
    generation_found: int

class ResultArchive:
    """
    结果档案（简化版 Pareto）

    功能:
    1. 收集搜索过程中的所有"好"结果
    2. 去重（基于 canonical hash）
    3. 后处理提取 Pareto 前沿
    """

    def __init__(self, max_size: int = 500):
        self.results: Dict[str, ArchivedResult] = {}  # hash -> result
        self.max_size = max_size

    def add(self, result: EvaluationResult, gen_ir: GenIR) -> bool:
        """
        添加结果，返回是否为新结果
        自动去重和修剪（保留最优）
        """

    def get_pareto_front(self, objectives: List[str] = ["nmse", "complexity"]) -> List[ArchivedResult]:
        """
        后处理提取 Pareto 前沿

        返回在 (nmse, complexity) 空间中的非支配解集
        """

    def get_best_by(self, metric: str) -> ArchivedResult:
        """按指定指标获取最优"""

    def get_top_k(self, metric: str, k: int) -> List[ArchivedResult]:
        """按指定指标获取 Top-K"""

    def to_dataframe(self) -> pd.DataFrame:
        """导出为 DataFrame（分析用）"""

    def to_dto(self) -> "ResultArchiveDTO":
        """可序列化（Web 兼容）"""
```

**与可视化集成**：

```python
class ParetoVisualizer(Visualizer):
    def render(self, archive: ResultArchive) -> VisualizationResult:
        front = archive.get_pareto_front()
        # 绘制 Pareto 前沿图
```

---

## 四、算法插件集成方案

### 4.1 SGA 插件（首个实现）

```python
class SGAPlugin(AlgorithmPlugin):
    """
    遗传算法插件

    搜索空间: term forest（若干项相加）
    评估: AIC = 2k + 2ln(MSE)
    """

    def __init__(self):
        self.population: List[Individual] = []
        self.generation: int = 0

    def configure(self, config):
        self.pop_size = config.get("population_size", 30)
        self.n_generations = config.get("generations", 100)
        self.crossover_rate = config.get("crossover_rate", 0.8)
        self.mutation_rate = config.get("mutation_rate", 0.2)

    def propose(self, k: int) -> List[Candidate]:
        """
        生成策略:
        - 选择（锦标赛选择）
        - 交叉（子树交叉）
        - 变异（点变异、子树变异）
        """

    def update(self, feedback: List[Feedback]):
        """
        更新策略:
        - 根据 AIC 分数排序
        - 保留精英个体
        - 更新 generation 计数
        """
```

### 4.2 DISCOVER 插件（PyTorch 重写）

```python
class DISCOVERPlugin(AlgorithmPlugin):
    """
    Deep Symbolic Regression 插件

    搜索: LSTM Controller + risk-seeking policy gradient
    评估: reward = (1 - penalty*complexity) / (1 + sqrt(NMSE))
    """

    def __init__(self):
        self.controller: nn.Module = None  # LSTM
        self.optimizer: torch.optim.Optimizer = None
        self.priority_queue: PriorityQueue = None

    def configure(self, config):
        self.hidden_size = config.get("hidden_size", 64)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.epsilon = config.get("epsilon", 0.05)  # top-ε
        self.entropy_weight = config.get("entropy_weight", 0.01)

    def propose(self, k: int) -> List[Candidate]:
        """
        用 LSTM 逐 token 采样生成表达式
        使用 dangling 状态跟踪完整性
        """

    def update(self, feedback: List[Feedback]):
        """
        Risk-seeking policy gradient:
        - 只用 top-ε% 的样本计算梯度
        - 包含 baseline 和 entropy bonus
        """
```

### 4.3 DLGA 插件

```python
class DLGAPlugin(AlgorithmPlugin):
    """
    Deep Learning Genetic Algorithm 插件

    基因编码: 模块化基因 → 乘积的和 (SOP)
    本质: 选择 Θ 矩阵的哪些列相乘
    """
```

### 4.4 PySR 插件

```python
class PySRPlugin(AlgorithmPlugin):
    """
    PySR 外部库包装
    用于普通符号回归（非 PDE）
    """
```

---

## 五、分阶段实现计划

### Phase 1: 核心基础

**目标**：手动构建表达式、执行并评估

- [ ] IR 系统 (GenIR, AnalysisIR, 转换器)
- [ ] Token 和 Library（默认算子集）
- [ ] StackExecutor (NumPy)
- [ ] PDEDataset 数据规范
- [ ] FiniteDiffProvider
- [ ] 基本评估器 (MSE, NMSE, R2)
- [ ] LeastSquaresSolver

**验证代码**：
```python
from kd2.core.ir import GenIR, Converter
from kd2.core.executor import StackExecutor
from kd2.core.evaluator import Evaluator

# 构建 u * u_x + u_xx
gen_ir = GenIR.from_string("add,mul,u,ux,uxx", library)
ast = Converter.to_analysis_ir(gen_ir)
result = executor.execute(ast, context)
eval_result = evaluator.evaluate(ast, result.value, y_true)
```

### Phase 2: 约束与线性求解

**目标**：评估包含微分的复杂表达式

- [ ] 约束系统 (采样期 + 评估期)
- [ ] JointPrior
- [ ] STRidgeSolver
- [ ] LassoSolver
- [ ] AutogradProvider（基础版）
- [ ] 常数优化（内嵌 BFGS）

### Phase 3: SGA 插件

**目标**：使用 kd2 运行 SGA 发现 Burgers 方程

- [ ] AlgorithmPlugin 基类 (propose/update)
- [ ] 插件注册机制
- [ ] 配置系统 (Hydra)
- [ ] SGA 适配器
- [ ] ExperimentManager
- [ ] 缓存系统 (DiskCache)
- [ ] WandB 集成

**验证代码**：
```python
from kd2.plugins import SGAPlugin
from kd2.data import load_dataset

dataset = load_dataset("burgers")
plugin = SGAPlugin()
plugin.configure({"population_size": 30, "generations": 100})

manager = ExperimentManager(plugin, executor, evaluator, cache, logger)
result = manager.run(max_iterations=100)
print(f"发现的方程: {result.best_expression}")
# 预期: u_t = -u * u_x + nu * u_xx
```

### Phase 4: DISCOVER 插件

**目标**：使用 kd2 运行 DISCOVER

- [ ] TorchExecutor
- [ ] AutogradProvider（完整版 + open-form diff）
- [ ] DISCOVER Controller (LSTM, PyTorch)
- [ ] Risk-seeking policy gradient
- [ ] Priority Queue Training
- [ ] MetaNet 支持

### Phase 5: 完善与扩展

- [ ] ExecIR (DAG + CSE 优化)
- [ ] 可视化模块 (Pareto, 残差图, LaTeX)
- [ ] DLGA 插件
- [ ] PySR 插件
- [ ] 辅助网络 (AutoKE N2 网络)
- [ ] 文档与示例

### Phase 6: Agent 接口（未来）

- [ ] 自动实验调度
- [ ] 超参数搜索
- [ ] 结果分析
- [ ] LLM 搜索插件

---

## 六、验证方程

| 方程 | 表达式 | 难度 |
|------|--------|------|
| Burgers | `u_t = -u * u_x + 0.1 * u_xx` | 入门 |
| KdV | `u_t = -6 * u * u_x - u_xxx` | 中等 |
| Chafee-Infante | `u_t = u_xx + u - u^3` | 中等 |
| 2D Navier-Stokes | `u_t = -u*u_x - v*u_y + nu*lap(u) - p_x` | 高级 |

---

## 七、关键参考文件

| 功能 | 参考文件 |
|------|---------|
| IR 与执行 | `ref_libs/DISCOVER/dso/dso/program.py` |
| 约束系统 | `ref_libs/DISCOVER/dso/dso/prior.py` |
| 树结构与评估 | `ref_libs/sga/sgapde/pde.py`, `tree.py` |
| 自动微分 | `ref_libs/DISCOVER/dso/dso/task/pde/utils_nn.py` |
| 计算图解析 | `ref_libs/AutoKE/parse/compute.py` |
| STRidge | `ref_libs/DISCOVER/dso/dso/stridge.py` |
| 尺度处理 | `ref_libs/sga/sgapde/context.py` |

---

## 八、风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| open-form diff 性能 | 先用缓存 + 批量化；Phase 5 用 CSE 优化 |
| DISCOVER TF1 依赖 | PyTorch 重写，保留算法逻辑 |
| 万级候选评估瓶颈 | DiskCache + canonical hash 去重 |
| 数值不稳定 | safety.py 统一护栏 (safe_div, NaN/Inf 检测) |
| 配置管理复杂 | Hydra 优雅管理 |

---

## 九、设计原则总结

1. **String as Interface, Tensor as Data**
2. **GenIR 是唯一真相，AnalysisIR 是视图**
3. **全程 torch.Tensor，保留计算图**
4. **约束两阶段：采样期 mask + 评估期 fail-fast**
5. **插件只做 propose/update，平台统一评估**
6. **n 维支持，禁止硬编码坐标名**
7. **接口优先，实现渐进（批量化、CSE 留给后续）**
8. **核心零 UI 依赖，结果可序列化**

---

## 十、Web 兼容性预留

### 10.1 设计目标

为未来 Web 界面做好架构准备，包括：
- Web 端绘图渲染
- Web 端参数输入与配置
- 实时进度推送

### 10.2 核心原则：零 UI 依赖

```
kd2/
├── src/kd2/              # 核心包：绝对不依赖 CLI/Web
│   ├── core/
│   ├── data/
│   ├── plugins/
│   └── experiment/
│
├── src/kd2_cli/          # CLI 入口（可选包）
│   └── main.py
│
└── src/kd2_api/          # API 入口（未来 Web 用）
    ├── routes.py         # REST API
    └── websocket.py      # 实时推送
```

**关键约束**：`kd2/` 核心包中禁止 import：
- `click`, `typer`, `argparse` (CLI 框架)
- `fastapi`, `flask`, `gradio` (Web 框架)
- `matplotlib.pyplot` (直接绘图)

### 10.3 结果可序列化（Pydantic）

所有结果类使用 Pydantic BaseModel，支持 JSON 序列化：

```python
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class EvaluationResultDTO(BaseModel):
    """评估结果数据传输对象"""
    mse: float
    nmse: float
    r2: float
    complexity: float
    term_count: int
    depth: int
    length: int
    reward: float
    aic: float
    coefficients: List[float]
    is_valid: bool
    invalid_reason: Optional[str] = None

    class Config:
        # 支持 torch.Tensor -> float 自动转换
        arbitrary_types_allowed = True

class SearchResultDTO(BaseModel):
    """搜索结果数据传输对象"""
    best_expression: str
    best_latex: str              # LaTeX 渲染用
    evaluation: EvaluationResultDTO
    generation: int
    total_evaluated: int
    runtime_seconds: float
    history: List[Dict[str, Any]]  # 可序列化历史
```

**内部 Tensor 结果 → DTO 转换**：

```python
class EvaluationResult:
    """内部结果（带 Tensor）"""
    mse: torch.Tensor
    # ...

    def to_dto(self) -> EvaluationResultDTO:
        """转换为可序列化 DTO"""
        return EvaluationResultDTO(
            mse=self.mse.item(),
            nmse=self.nmse.item(),
            # ...
        )
```

### 10.4 可视化双模式输出

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import matplotlib.figure

@dataclass
class VisualizationResult:
    """可视化结果（双模式）"""
    figure: Optional[matplotlib.figure.Figure]  # CLI 用
    data: Dict[str, Any]                        # Web 用 (JSON)
    html: Optional[str] = None                  # 可选 HTML 片段

class Visualizer(ABC):
    """可视化器抽象基类"""

    @abstractmethod
    def render(self, result: SearchResultDTO) -> VisualizationResult:
        pass

class ParetoVisualizer(Visualizer):
    """Pareto Front 可视化"""

    def render(self, result: SearchResultDTO) -> VisualizationResult:
        # 1. 准备数据
        data = {
            "points": [...],  # (complexity, accuracy) 点
            "pareto_front": [...],
            "best_point": {...}
        }

        # 2. 生成 matplotlib figure（CLI 用）
        fig = self._create_matplotlib_figure(data)

        # 3. 返回双模式结果
        return VisualizationResult(
            figure=fig,
            data=data,  # Web 端可用此数据自行渲染
            html=None
        )
```

### 10.5 Callback 机制预留

为实时进度推送预留 callback 接口：

```python
from typing import Protocol, Optional
from dataclasses import dataclass

@dataclass
class ProgressUpdate:
    """进度更新"""
    generation: int
    total_generations: int
    best_expression: str
    best_score: float
    evaluated_count: int
    timestamp: float

class ProgressCallback(Protocol):
    """进度回调协议"""

    def on_generation_start(self, generation: int) -> None:
        ...

    def on_generation_end(self, update: ProgressUpdate) -> None:
        ...

    def on_new_best(self, update: ProgressUpdate) -> None:
        ...

    def on_search_complete(self, result: SearchResultDTO) -> None:
        ...

class ExperimentManager:
    def __init__(
        self,
        plugin: AlgorithmPlugin,
        # ...
        callback: Optional[ProgressCallback] = None  # 可选回调
    ):
        self.callback = callback

    def run(self, max_iterations: int):
        for gen in range(max_iterations):
            if self.callback:
                self.callback.on_generation_start(gen)

            # ... 搜索逻辑 ...

            if self.callback:
                self.callback.on_generation_end(update)
```

**CLI 实现示例**：

```python
class CLIProgressCallback:
    """CLI 进度条回调"""

    def on_generation_end(self, update: ProgressUpdate):
        print(f"Gen {update.generation}: Best = {update.best_expression}, "
              f"Score = {update.best_score:.4f}")
```

**未来 WebSocket 实现示例**：

```python
class WebSocketProgressCallback:
    """WebSocket 实时推送回调"""

    def __init__(self, websocket):
        self.ws = websocket

    async def on_generation_end(self, update: ProgressUpdate):
        await self.ws.send_json(update.__dict__)
```

### 10.6 配置 Schema（Pydantic）

使用 Pydantic 定义配置 schema，支持 JSON Schema 导出（Web 表单生成）：

```python
from pydantic import BaseModel, Field

class SGAConfig(BaseModel):
    """SGA 算法配置"""
    population_size: int = Field(30, ge=10, le=500, description="种群大小")
    generations: int = Field(100, ge=10, le=1000, description="迭代代数")
    crossover_rate: float = Field(0.8, ge=0.0, le=1.0, description="交叉率")
    mutation_rate: float = Field(0.2, ge=0.0, le=1.0, description="变异率")

    class Config:
        json_schema_extra = {
            "example": {
                "population_size": 50,
                "generations": 200
            }
        }

# 导出 JSON Schema（Web 表单可用）
schema = SGAConfig.model_json_schema()
```

### 10.7 延后项

以下内容**不在 MVP 范围内**，待 Web 需求明确后实现：

| 延后项 | 说明 |
|--------|------|
| Web 框架选择 | FastAPI / Gradio / Streamlit 待定 |
| WebSocket 协议 | 具体消息格式待定 |
| 前端实现 | 可独立 repo，React/Vue 待定 |
| 认证鉴权 | 多用户场景再考虑 |

### 10.8 目录结构更新

```
kd2/
├── src/kd2/
│   ├── core/
│   ├── data/
│   ├── plugins/
│   ├── experiment/
│   ├── visualization/
│   │   ├── base.py          # Visualizer ABC, VisualizationResult
│   │   ├── pareto.py
│   │   ├── residual.py
│   │   └── latex.py
│   └── dto/                  # 数据传输对象（可序列化）
│       ├── results.py       # EvaluationResultDTO, SearchResultDTO
│       └── config.py        # 配置 schema
│
├── src/kd2_cli/              # CLI 包（可选安装）
│   ├── __init__.py
│   ├── main.py              # typer/click 入口
│   └── callbacks.py         # CLI 进度回调
│
└── pyproject.toml           # 可选依赖: kd2[cli], kd2[api]
```

**pyproject.toml 可选依赖**：

```toml
[project.optional-dependencies]
cli = ["typer", "rich"]
api = ["fastapi", "uvicorn", "websockets"]
all = ["kd2[cli]", "kd2[api]"]
```
