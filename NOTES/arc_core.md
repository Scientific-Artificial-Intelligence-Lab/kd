# kd2 核心层设计

> **关联文档**：[arc_final.md](arc_final.md) - 主文档

---

## 一、两层 IR 设计

**设计原则**：GenIR 是「唯一真相」(Canonical)，AnalysisIR 是「视图」(View)。

| IR 层 | 表示形式 | 职责 | 特性 |
|-------|---------|------|------|
| **GenIR** | 前缀 Token 序列 | 缓存键、日志、复现、去重 | Hashable, Immutable |
| **AnalysisIR** | AST 树 | 结构分析、约束检查、执行、可视化 | 可变、可遍历 |

### 1.1 GenIR 规范

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
```

### 1.2 AnalysisIR 规范

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

### 1.3 Canonical Hash

**已确定**：可交换算子子节点排序后哈希，使用稳定哈希算法（sha256）。

```python
def canonical_hash(node: ASTNode) -> str:
    """可交换算子子节点排序后哈希"""
    if node.token.is_commutative:  # add, mul
        children_hashes = sorted([canonical_hash(c) for c in node.children])
    else:
        children_hashes = [canonical_hash(c) for c in node.children]

    # 使用稳定哈希，不用 Python hash()
    payload = f"{node.token.name}:{','.join(children_hashes)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
```

<!--
## 实现参考：Codex Review 建议

### canonical hash 必须稳定
- Python 的 hash() 跨进程/跨运行不稳定（PYTHONHASHSEED 随机化）
- 建议：sha256(canonical_string)
- 需要版本化 canonicalization 规则（commutative 排序规则版本）

### finish_tokens() 语义
DISCOVER 的实现：
1. 计算 dangling[-1]（还需要多少个 terminal）
2. 从 input_tokens 中随机等概率选择足够数量的 terminal 补全

可选策略：
- 方案 A：返回"所有允许的 terminal 集合"（交给约束/采样器过滤）
- 方案 B：返回"具体的补全序列"（像 DISCOVER 一样）

具体实现时根据需求选择。
-->

---

## 二、Token 与算子库

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
| **Derivative Terminal** | `u_x`, `u_xx`, `u_t`, ... | 0 | 1.5 |

### 2.1 导数表示（混合方案 + 约束消别名）

| 场景 | Token | Arity | 执行方式 | 搜索空间 |
|------|-------|-------|---------|---------|
| 预计算导数 | `u_x`, `u_xx` | 0 (terminal) | 查表 | ✅ 允许 |
| Open-form diff (复合) | `diff(u*u_x, x)` | 2 (operator) | autograd/FD | ✅ 允许 |
| Open-form diff (简单) | `diff(u, x)` | 2 (operator) | - | ❌ 禁止 |

**关键点**：`u_x` terminals 和 `diff` operator 都是可用工具，算法插件自己选择用哪个。

---

## 三、执行器设计

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
        """批量执行（接口优先，实现渐进）"""
        return [self.execute(ir, context) for ir in irs]

@dataclass
class ExecutionContext:
    fields: Dict[str, torch.Tensor]      # {"u": tensor, "v": tensor}
    coords: Dict[str, torch.Tensor]      # {"x": tensor, "t": tensor}
    derivative_provider: DerivativeProvider
    constants: Dict[str, torch.Tensor]   # {"C1": tensor, ...}

@dataclass
class ExecutionResult:
    value: torch.Tensor      # 计算结果
    is_valid: bool           # 是否有效
    error_type: Optional[str]  # "nan", "inf", "domain_error"
```

### 3.1 数值稳定性（safety.py）

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

---

## 四、评估器设计

**已确定**：评估模式 LINEAR 默认，DIRECT 可选。

```python
class EvaluationMode(Enum):
    LINEAR = "linear"   # split_terms → Θ 矩阵 → STRidge/Lasso（默认）
    DIRECT = "direct"   # 整棵表达式直接评估（PySR 等）

class Evaluator:
    def evaluate(
        self,
        ir: AnalysisIR,
        context: ExecutionContext,
        target: torch.Tensor,
        mode: EvaluationMode = EvaluationMode.LINEAR
    ) -> EvaluationResult:
        if mode == EvaluationMode.LINEAR:
            return self._evaluate_linear(ir, context, target)
        else:
            return self._evaluate_direct(ir, context, target)
```

### 4.1 EvaluationResult

```python
@dataclass
class EvaluationResult:
    # 准确度
    mse: float
    nmse: float
    r2: float

    # 复杂度
    complexity: float
    term_count: int
    depth: int
    length: int

    # 综合分数
    aic: float       # AIC = n * ln(RSS/n) + 2k

    # 系数
    coefficients: List[float]
    selected_terms: List[str]

    # 有效性
    is_valid: bool
    invalid_reason: Optional[str] = None
```

<!--
## 实现参考：评估优化

### EvalSummary vs EvalArtifacts 拆分
Codex 建议：DiskCache 不应存大 Tensor。可考虑拆分：
- EvalSummary（可缓存）：标量指标、系数、有效性
- EvalArtifacts（按需）：residual 向量、Theta 矩阵、调试数据

是否拆分取决于实际性能瓶颈，先不做过早优化。

### LINEAR 评估链路细节
可能需要的步骤（遇到问题时再加）：
- valid_mask：边界点/NaN/Inf 处理
- Theta 标准化：对每列做标准化（记录 scale，最后还原系数）
- diagnostics：条件数、被丢弃列比例、有效样本数

### AIC 标准定义
建议统一口径：AIC = n * ln(RSS/n) + 2k
- n: 有效样本数
- k: 非零系数数（不含截距）
- RSS: 残差平方和
-->

---

## 五、约束系统（两阶段设计）

**设计原则**：约束分为「采样期」和「评估期」两阶段。

### 5.1 采样期约束

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
        """返回 logit mask: 0 表示允许，-inf 表示禁止"""
```

**内置约束**：
- `LengthConstraint`：限制表达式最大长度
- `TrigConstraint`：限制三角函数嵌套
- `DiffArgumentConstraint`：diff 的第一个子节点必须是复合表达式

### 5.2 评估期约束

在评估时进行 fail-fast 检查。

```python
class EvaluationConstraint(ABC):
    """评估期约束：fail-fast 检查"""

    @abstractmethod
    def check(self, ir: AnalysisIR, context: ExecutionContext) -> ConstraintResult:
        """返回: (is_valid, reason)"""
```

**内置约束**：
- `LHSAxisConstraint`：RHS 不能对 LHS 轴求导（防止 u_t = u_t）
- `NumericalConstraint`：NaN, Inf, 极值检查

### 5.3 别名消除约束

```python
class DiffArgumentConstraint(SamplingConstraint):
    """
    约束：diff 的第一个子节点必须是复合表达式，不能是单一 terminal

    禁止: diff(u, x)     → 应该用 u_x
    允许: diff(u*u_x, x) → open-form diff
    """

    def compute_mask(self, partial_ir, parent, sibling, dangling, library):
        if parent in ("diff", "diff2") and sibling is None:
            return mask_all_terminals(library)
        return allow_all(library)
```

---

## 六、线性求解器

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
    r2: float
```

**已确定**：内置 LeastSquares、STRidge、Lasso。

---

## 七、缓存系统

**已确定**：DiskCache + 分桶目录隔离。

```python
@dataclass
class CacheContext:
    """缓存上下文"""
    dataset_fingerprint: str    # dataset 指纹
    derivative_config: str      # "finite_diff" / "autograd"

class ExpressionCache:
    """表达式评估缓存"""

    def __init__(self, base_dir: Path, context: CacheContext):
        # 目录结构: .kd2_cache/{dataset_fp}/{deriv_config}/
        self.cache_dir = base_dir / context.dataset_fingerprint / context.derivative_config
        self.cache = diskcache.Cache(str(self.cache_dir))

    def get_key(self, ir: AnalysisIR) -> str:
        """缓存键 = canonical hash"""
        return canonical_hash(ir)
```

**目录结构**：

```
.kd2_cache/
├── burgers_100x100_a3f2b1c8/
│   ├── finite_diff/
│   │   └── cache.db
│   └── autograd/
│       └── cache.db
```

<!--
## 实现参考：缓存优化

### Term-level 缓存
Codex 建议：表达式级缓存不够，LINEAR 模式可考虑 term/subtree 级缓存。
- TermValueCache（内存 LRU，实验内）：缓存 term 的数值列
- 是否需要取决于实际瓶颈，先不做

### 缓存键完整性
缓存键可能需要包含更多上下文：
- eval_mode（LINEAR/DIRECT）
- solver 配置（STRidge 阈值等）
遇到缓存污染问题时再细化。
-->

---

## 八、指标系统

**已确定**：可插拔 Metric，内置 AIC/BIC，支持自定义。

```python
class Metric(Protocol):
    name: str
    direction: Literal["min", "max"]  # 用于 Pareto/排序

    def compute(self, result: EvaluationResult) -> float:
        ...
```

**内置指标**：MSE, NMSE, R2, AIC, BIC, Complexity。
