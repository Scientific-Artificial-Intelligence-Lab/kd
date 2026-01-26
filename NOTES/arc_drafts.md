# kd2 新架构设计方案

> 本文档基于教授的原始想法，经过对 DISCOVER、SGA、DLGA、AutoKE 四个参考实现的深入分析后细化而成。

---

## 一、教授原始想法回顾

kd2 将作为 kd 的新版本，成为符号回归领域（特别是 PDE 发现）的通用实验平台：
- 数据层处理数据
- 通用 IR 进行表示
- 算法作为插件运行
- 灵活的评估模块
- 结果可视化
- 未来接入 agent 自动实验

核心洞察：**无论是符号树还是字符串，本质是一个计算顺序。通过 NN 自动微分 + 链式法则 + PyTorch 计算图，可以生成任意计算顺序。**

---

## 二、参考实现分析总结

### 2.1 四个参考算法对比

| 维度 | DISCOVER | SGA | DLGA | AutoKE |
|------|----------|-----|------|--------|
| **表示** | 前缀 Token 序列 | Tree (森林结构) | 模块化基因 | 字符串 → AST |
| **搜索** | RL (LSTM) + GA | 遗传算法 | 遗传算法 | 无（手工配置） |
| **导数** | FD / PINN+Autograd | FD / Autograd | 强制 Autograd | Autograd |
| **系数** | STRidge 稀疏回归 | STRidge | SVD 最小二乘 | PINN 训练 |
| **评估** | reward = f(NMSE, complexity) | AIC = 2k + 2ln(MSE) | MSE + 长度惩罚 | L2 loss |

### 2.2 共同的核心依赖

1. **可微执行引擎**：能对任意子表达式求导
2. **线性回归模块**：STRidge/Lasso 求系数
3. **复杂度度量**：长度/深度/项数
4. **约束系统**：禁止非法表达式

### 2.3 关键技术点

1. **开放形式导数**：必须支持 `diff(u * u_x, x)` 而不仅是 `diff(u, x)`
2. **线性库搜索**：STRidge 是 SGA/DISCOVER 的性能关键
3. **数值稳定性**：除零保护、NaN 检测、domain scaling
4. **子表达式缓存**：避免重复计算，CSE 优化

---

## 三、系统架构设计

### 3.1 层次结构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                            │
│   CLI Tool  │  Web UI  │  Agent API  │  Notebooks                │
├─────────────────────────────────────────────────────────────────┤
│                     Experiment Layer                             │
│   ExperimentManager  │  ResultStore  │  Visualization            │
├─────────────────────────────────────────────────────────────────┤
│                      Plugin Layer                                │
│   DISCOVER  │  SGA  │  DLGA  │  PySR  │  Custom                  │
│                  ↓ AlgorithmPlugin Interface                     │
├─────────────────────────────────────────────────────────────────┤
│                       Core Layer                                 │
│   DataLayer │ ProgramIR │ Executor │ Evaluator │ Constraints    │
│   Library   │ LinearSolve (STRidge/Lasso)                        │
├─────────────────────────────────────────────────────────────────┤
│                    Foundation Layer                              │
│   NumPy │ PyTorch │ SymPy │ Config (YAML)                        │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 目录结构

```
kd2/
├── src/kd2/
│   ├── core/                    # 核心模块
│   │   ├── ir/                  # 两层 IR 系统
│   │   │   ├── base.py         # Token, TokenType
│   │   │   ├── gen_ir.py       # GenIR (前缀 token 序列)
│   │   │   ├── analysis_ir.py  # AnalysisIR (AST 树)
│   │   │   └── converters.py   # IR 转换器
│   │   ├── library/             # Token/算子库
│   │   │   ├── operators.py    # 内置算子 (add, mul, sin, cos, ...)
│   │   │   ├── diff_ops.py     # 微分算子 (diff, diff2, ...)
│   │   │   └── registry.py     # 算子注册表
│   │   ├── executor/            # 执行引擎
│   │   │   ├── numpy_exec.py   # NumPy 执行器
│   │   │   ├── torch_exec.py   # PyTorch 可微执行器
│   │   │   └── safety.py       # 数值稳定性保护
│   │   ├── evaluator/           # 评估器
│   │   │   ├── metrics.py      # MSE, NMSE, R2, AIC
│   │   │   └── complexity.py   # 复杂度评估
│   │   ├── constraints/         # 约束系统
│   │   │   ├── structural.py   # 长度、深度约束
│   │   │   ├── physical.py     # 微分约束
│   │   │   └── joint_prior.py  # 联合先验
│   │   └── linear_solve/        # 线性求解器
│   │       ├── stridge.py      # STRidge 稀疏回归
│   │       └── lasso.py        # Lasso
│   │
│   ├── data/                    # 数据层
│   │   ├── schema.py           # PDEDataset 规范
│   │   ├── loaders.py          # 数据加载器
│   │   └── derivatives/         # 导数提供者
│   │       ├── finite_diff.py  # 有限差分
│   │       ├── autograd.py     # 自动微分
│   │       └── metanet.py      # MetaNet 神经网络
│   │
│   ├── plugins/                 # 插件系统
│   │   ├── base.py             # AlgorithmPlugin 接口
│   │   ├── discover/           # DISCOVER 插件
│   │   ├── sga/                # SGA 插件
│   │   ├── dlga/               # DLGA 插件
│   │   └── pysr/               # PySR 插件
│   │
│   ├── experiment/              # 实验管理
│   ├── visualization/           # 可视化
│   └── config/                  # 配置管理 (YAML)
│
├── tests/
├── examples/
└── pyproject.toml
```

---

## 四、核心设计详解

### 4.1 两层 IR 设计

**设计决策**：先实现两层 IR（GenIR + AnalysisIR），ExecIR (DAG+CSE) 作为后续优化。

| IR 层 | 表示形式 | 用途 | 对应参考 |
|-------|---------|------|---------|
| **GenIR** | 前缀 Token 序列 | RL/序列生成、GA 编码、哈希去重 | DISCOVER `Program.traversal` |
| **AnalysisIR** | AST 树 | 结构分析、约束检查、可视化、执行 | SGA `Tree` |

**转换关系**：
```
GenIR ──parse──> AnalysisIR
                     │
              split_terms() → 多个 term (用于 STRidge)
```

**GenIR 示例**：
```python
# 表示 u * u_x + u_xx
tokens = ["add", "mul", "u", "ux", "uxx"]
gen_ir = GenIR(tokens)
```

**AnalysisIR 示例**：
```
        add
       /   \
     mul   uxx
    /   \
   u    ux
```

### 4.2 Token 与算子库设计

```python
@dataclass(frozen=True)
class Token:
    name: str                    # "add", "mul", "diff", "u", "x"
    arity: int                   # 参数数量 (0=终端, 1=一元, 2=二元)
    complexity: float            # 复杂度权重
    token_type: TokenType        # TERMINAL, UNARY, BINARY, DIFF
    function: Optional[Callable] # 实际计算函数

class TokenType(Enum):
    TERMINAL = 0      # 变量、常数
    UNARY = 1         # sin, cos, exp, n2, n3
    BINARY = 2        # add, sub, mul, div
    DIFF = 3          # diff, diff2 (特殊处理)
```

**内置算子**：
- 基础：`add`, `sub`, `mul`, `div` (保护版本防除零)
- 数学：`sin`, `cos`, `exp`, `log`, `sqrt`
- 幂次：`n2` (x²), `n3` (x³)
- 微分：`diff`, `diff2`, `diff3`, `lap` (拉普拉斯)

### 4.3 数据层设计

```python
@dataclass
class PDEDataset:
    name: str
    task_type: TaskType          # PDE, ODE, REGRESSION

    # 坐标轴
    axes: Dict[str, AxisInfo]    # {"x": ..., "t": ...}
    axis_order: List[str]        # ["x", "t"]

    # 场变量
    fields: Dict[str, FieldData] # {"u": ...}

    # LHS 定义
    lhs_field: str               # "u"
    lhs_derivative: str          # "t" → 表示 u_t = RHS

    # 元数据
    noise_level: float
    ground_truth: Optional[str]  # 真实方程（用于验证）
```

**导数提供者接口**：
```python
class DerivativeProvider(ABC):
    def get_derivative(field: str, axis: str, order: int) -> ndarray
        """获取预计算的导数，如 u_x, u_xx"""

    def diff(expression: Tensor, axis: str, order: int) -> Tensor
        """开放形式导数：对任意表达式求导"""
```

**两种模式**：
1. **有限差分** (FiniteDiffProvider)：简单高效，适合高质量数据
2. **自动微分** (AutogradProvider)：精确，需要预训练 NN，适合噪声数据

### 4.4 执行器设计

```python
class Executor(ABC):
    def execute(ir: AnalysisIR, context: ExecutionContext) -> ExecutionResult

class ExecutionContext:
    state_vars: Dict[str, ndarray]      # {"u": array, "v": array}
    coord_vars: Dict[str, ndarray]      # {"x": array, "t": array}
    derivative_provider: DerivativeProvider

class ExecutionResult:
    value: ndarray
    is_valid: bool
    error_type: Optional[str]
```

**StackExecutor**：基于栈的 NumPy 执行器
**TorchExecutor**：PyTorch 可微执行器（支持 autograd）

### 4.5 评估器设计

```python
class EvaluationResult:
    # 准确度指标
    mse: float
    nmse: float                  # 归一化 MSE
    r2: float

    # 复杂度指标
    complexity: float            # Token 加权复杂度
    term_count: int              # 项数
    depth: int                   # 树深度

    # 综合分数
    reward: float                # DISCOVER 风格
    aic: float                   # SGA 风格 (AIC = 2k + 2ln(MSE))

    # 系数
    coefficients: List[float]    # STRidge 求得的系数
```

**评估流程**：
1. 将表达式按 +/- 拆分为多个 term
2. 对每个 term 执行得到数值
3. 组成特征矩阵 Θ = [term1, term2, ...]
4. 用 STRidge/Lasso 求解 y = Θξ
5. 计算残差和各项指标

### 4.6 约束系统设计

```python
class Constraint(ABC):
    def is_violated(ir: AnalysisIR) -> bool

class GenerationPrior(Constraint):
    """生成时的软约束，调整采样概率"""
    def compute_prior(actions, parent, sibling, dangling) -> ndarray
```

**核心约束**：
1. **LengthConstraint**：限制表达式长度
2. **DiffConstraint**：
   - diff 的右子节点必须是坐标变量
   - RHS 不能对 LHS 轴求导（避免 u_t = u_t 退化）
3. **RelationalConstraint**：禁止某些父子/兄弟关系
4. **TrigConstraint**：限制三角函数嵌套

### 4.7 线性求解器

**STRidge (Sequential Threshold Ridge)**：
```
输入: 特征矩阵 Θ [n_samples, n_terms], 目标 y [n_samples]
输出: 稀疏系数 ξ

算法:
1. 初始 Ridge 回归: ξ = (Θ'Θ + αI)^{-1} Θ'y
2. 迭代阈值化:
   - 将 |ξ_i| < threshold 的系数置零
   - 在非零系数上重新回归
3. 返回稀疏解
```

这是 SGA 和 DISCOVER 的核心组件，**必须作为一等公民实现**。

### 4.8 算法插件接口

```python
class AlgorithmPlugin(ABC):
    name: str
    version: str
    algorithm_type: AlgorithmType  # RL, GA, HYBRID

    def configure(config: Dict) -> None
        """设置算法参数"""

    def setup(dataset, library, constraints, evaluator) -> None
        """初始化搜索环境"""

    def search(max_iterations, callback) -> SearchResult
        """执行搜索"""

class SearchResult:
    best_ir: AnalysisIR
    best_expression: str         # 人类可读形式
    best_score: float
    evaluation: EvaluationResult
    history: List[SearchState]
    runtime_seconds: float
```

---

## 五、算法插件集成方案

### 5.1 SGA 插件（首个实现）

**适配要点**：
- 包装 `ref_libs/sga/` 现有代码
- Context 转换：kd2 PDEDataset → SGA Context
- IR 转换：SGA Tree → kd2 AnalysisIR
- 评估：使用 AIC = 2k + 2ln(MSE)

**为什么先实现 SGA**：
- 代码相对简单，GA 逻辑清晰
- 适合验证核心架构（IR、执行器、评估器）
- 不依赖深度学习框架的复杂部分

### 5.2 DISCOVER 插件

**适配要点**：
- 包装 `ref_libs/DISCOVER/` 现有代码
- 需要 TorchExecutor 和 AutogradProvider
- Controller (LSTM) 包装
- Priority Queue Training
- 评估：reward = (1 - penalty*complexity) / (1 + sqrt(NMSE))

### 5.3 DLGA 插件

**适配要点**：
- 包装 `ref_libs/dlga/` 现有代码
- 基因编码：模块化基因 → 乘积的和 (SOP)
- 需要强制 NN 训练
- 表达能力受限，但实现简单

**DLGA 集成方案**：
- 基因编码本质是选择 Θ 矩阵的哪些列相乘
- 可以统一到 "特征选择 + 线性回归" 框架
- 与 SGA/DISCOVER 的 STRidge 流程兼容

### 5.4 PySR 插件

**适配要点**：
- 外部库包装
- 用于普通符号回归（非 PDE）
- 接口相对简单

---

## 六、设计决策总结

| 决策点 | 选择 | 理由 |
|--------|------|------|
| **IR 层数** | 两层优先 | GenIR + AnalysisIR 先行，ExecIR 作为后续优化 |
| **首个插件** | SGA | 代码简单，适合验证核心架构 |
| **导数模式** | 两者并行 | 有限差分 + 自动微分，灵活适配不同场景 |
| **技术栈** | Python 3.11+ | 最新语法，PyTorch 2.x；未来可 C/C++ 优化 |

---

## 七、分阶段实现计划

### Phase 1: 核心基础 (2-3 周)
- [ ] IR 系统 (GenIR, AnalysisIR, 转换器)
- [ ] Token 和 Library 基础
- [ ] StackExecutor (NumPy)
- [ ] PDEDataset 数据规范
- [ ] FiniteDiffProvider + AutogradProvider
- [ ] 基本评估器 (MSE, NMSE, R2, AIC)
- [ ] LeastSquaresSolver + STRidgeSolver

**里程碑**: 手动构建表达式、执行并评估

### Phase 2: 约束与线性求解 (1-2 周)
- [ ] 约束系统 (Length, Diff, Relational)
- [ ] JointPrior
- [ ] LassoSolver
- [ ] 高阶微分算子

**里程碑**: 评估包含微分的复杂表达式

### Phase 3: SGA 插件 (2 周)
- [ ] AlgorithmPlugin 基类
- [ ] 插件注册机制
- [ ] 配置系统 (YAML)
- [ ] SGA 适配器
- [ ] ExperimentManager 基础

**里程碑**: 使用 kd2 运行 SGA 发现 Burgers 方程

### Phase 4: DISCOVER 插件 (2-3 周)
- [ ] TorchExecutor
- [ ] DISCOVER 适配器
- [ ] MetaNet 支持

**里程碑**: 使用 kd2 运行 DISCOVER

### Phase 5: 完善与扩展 (2-3 周)
- [ ] ExecIR (DAG + CSE 优化)
- [ ] 可视化模块
- [ ] DLGA 插件
- [ ] PySR 插件
- [ ] 文档与示例

### Phase 6: Agent 接口 (未来)
- [ ] 自动实验调度
- [ ] 超参数搜索
- [ ] 结果分析

---

## 八、验证方案

### 8.1 Phase 1 验证代码
```python
# 手动构建并评估表达式
from kd2.core.ir import GenIR, AnalysisIR
from kd2.core.executor import StackExecutor
from kd2.core.evaluator import Evaluator

# 构建 u * u_x + u_xx
gen_ir = GenIR.from_string("add,mul,u,ux,uxx", library)
ast = converter.convert(gen_ir)
result = executor.execute(ast, context)
eval_result = evaluator.evaluate(ast, result.value, y_true)
```

### 8.2 Phase 3 验证代码
```python
# 使用 SGA 发现 Burgers 方程
from kd2.plugins import SGAPlugin
from kd2.data import load_dataset

dataset = load_dataset("burgers")
plugin = SGAPlugin()
plugin.configure({"population_size": 30, "generations": 100})
plugin.setup(dataset, library, constraints, evaluator)
result = plugin.search()
print(f"发现的方程: {result.best_expression}")
# 预期: u_t = -u * u_x + nu * u_xx
```

### 8.3 端到端测试方程
- **Burgers**: `u_t = -u * u_x + 0.1 * u_xx`
- **KdV**: `u_t = -6 * u * u_x - u_xxx`
- **Chafee-Infante**: `u_t = u_xx + u - u^3`

---

## 九、关键参考文件

| 功能 | 参考文件 |
|------|---------|
| IR 与执行 | `ref_libs/DISCOVER/dso/dso/program.py` |
| 约束系统 | `ref_libs/DISCOVER/dso/dso/prior.py` |
| 树结构与评估 | `ref_libs/sga/sgapde/pde.py`, `tree.py` |
| 自动微分 | `ref_libs/DISCOVER/dso/dso/task/pde/utils_nn.py` |
| 计算图解析 | `ref_libs/AutoKE/parse/compute.py` |
| STRidge | `ref_libs/DISCOVER/dso/dso/stridge.py` |

---

## 十、待讨论问题

1. **n 维支持**：当前设计支持任意维度，但需要确认是否有特殊的高维需求？
必须支持n维, 物理方程几乎都是多维的（2D Navier-Stokes, 3D 麦克斯韦方程）。如果你的代码把 x, t 写死了，这项目就废了. 代码里不要出现 x 或 y 这样的变量名，统称 coords[:, i], 或者参考discover等矩阵+配置文件存, 或者, 有更好的办法么?

2. **分布式计算**：是否需要支持多 GPU / 分布式评估？
不要过早优化, 目前问题规模不大, 暂时够用, 未来扩展不要太麻烦就行.

3. **实验追踪**：是否需要集成 MLflow/WandB 等实验追踪工具？
必须要有, 但是选哪个还没定.

4. **数据格式**：除了 numpy/mat，是否需要支持其他数据格式？
目前就支持这俩就行了, 我们先做MVP, 不要把时间浪费在parser上, 而且未来也方便扩展的.

5. **可视化需求**：除了树形和场数据，还需要哪些可视化？
Pareto Front, 残差图, 场图, 45度线图, 方程 latex 渲染成公式等等, 留好接口, 未来需要加新类型必须很方便.


---

# 补充意见:

- 从头到尾，数据永远是 torch.Tensor, 这样可以保留梯度, 虽然目前的算法不是End-to-End Differentiable的, 但是如果以后有pinn等, 可能需要.



## Gemini 意见:

1. 建议： 必须在 Evaluator 中增加一个**“图扩展模式” (Graph Extension Mode)**。

不仅仅是求值：Evaluator 不应只返回 ndarray。对于 AutoKE 风格的算法，Evaluator 应该返回一个 PyTorch 计算图节点。

联合优化：架构中需要预留位置，允许 Loss 不仅更新方程的常数，还能反向传播更新 Layer B (Surrogate NN) 的权重。这是 kd2 超越 SGA 的关键点


2. 关于 Batching (批处理) 的性能隐患
4.4 执行器设计 看起来是一次执行一个 IR。

问题：GA 或 RL 一代通常有 500-1000 个个体。如果在 Python 里写 for tree in population: execute(tree)，速度会非常慢（Python 循环开销）。

建议：

Phase 1 就要考虑 Batching 接口。

即使底层先用 List comprehension 实现，接口层面也应该是 executor.execute_batch(list_of_irs)。

未来利用 torch.vmap 或编译技术优化时，不用改上层代码

3. 关于 IR 设计的细节
GenIR 示例采用了前缀序列 ["add", "mul", "u", "ux", "uxx"]。

点赞：这是波兰表达式，解析极其简单，且无歧义，非常适合作为 Canonical Representation。

提醒：在 AnalysisIR (AST) 中，建议增加 .to_canonical_string() 方法，并且强制对可交换算子（add, mul）的子节点排序。这是做 Caching (缓存) 的前提。如果不排序，a+b 和 b+a 会被当成两个不同的方程，浪费计算资源

## 工具

1. SymPy: 需要一个现成的库来把字符串 "x + y" 变成一棵树，并告诉你它是合法的。虽然它慢，但我们只在 管理层 用它，不在 评估层 的内循环里用它，所以性能不是问题。

2. Hydra: 科研神器。你以后会有几十个参数（学习率、种群大小、变异率、惩罚系数...）。用 argparse 写会疯掉，Hydra 可以让你优雅地管理 conf/config.yaml。

3. WandB: 科研标配。你在跑 GA 的时候，肯定想实时看到“现在的最优方程是什么”、“Loss 下降了吗”。WandB 能直接在网页上画出漂亮的图，汇报给导师看很方便。

4. DiskCache: 务实之选。Redis 需要你在 Mac 上装服务，还得维护。DiskCache 直接在项目目录下生成一个缓存文件，用法和 Redis 一样，但无需运维，非常适合单机科研



## codex 意见

你们的分层与“Core(数据/IR/执行/评估/约束/回归)+Plugin(算法)”方向是对的（arc_drafts.md (line 49)），并且“两层 IR（GenIR 前缀序列 + AnalysisIR AST）”非常贴合 DISCOVER 的 preorder 生成范式（arc_drafts.md (line 132)；task.py (line 113)）。
真正会“卡死项目”的不是目录结构，而是 3 个工程硬点：①可微/可批量的执行与求导（open-form diff）②PDE 的“项库+稀疏回归(STRidge)”一等公民化 ③数值防护+缓存体系（否则 GA/RL 的万级评估会崩）。
你们草稿最强的点（建议保留）

STRidge 作为核心组件（arc_drafts.md (line 294)）是正确的：DISCOVER/SGA 都把“拆项→Θ→回归→reward/AIC”当主路径（stridge.py (line 182)；pde.py (line 111)）。
约束系统分离出来（arc_drafts.md (line 275)）也是必须的：DISCOVER 明确有 priors+regulations 双层（prior.py (line 93)）。
必须尽早补齐/改写的关键设计（否则未来很难集成算法）

把“显式 PDE = Σ ξᵢ termᵢ”做成一等公民：不仅是 split_terms()+STRidge 的 evaluator 内部技巧（arc_drafts.md (line 246)），而应抽象成可复用的“FeatureLibraryModel/LinearPDE”能力；SGA 本质是“terms 的 GA”，DISCOVER 本质是“terms 的 RL”，共享同一回归内环会大幅降低未来集成成本。
Canonical IR 需要更明确：prefix 作为唯一缓存键：DISCOVER 直接用 tokens bytes 做 Program cache key（program.py (line 111)），并依赖 dangling 完整化序列（task.py (line 133)）。建议你们把 GenIR 定义为 canonical（可哈希/可日志/可复现），AnalysisIR 只是 view（arc_drafts.md (line 132)），并补上：
finish_tokens()/截断规则（对齐 DISCOVER 的 dangling 逻辑）
stable stringify（用于结果存档），但不要把 string 当“唯一真相”
DerivativeProvider 要分两层能力（否则实现会变成“要么慢死、要么不支持 open-form”）：
DerivativeStore.get(field, axis, order)：预计算导数（SGA/大部分 GA 高吞吐依赖这个）
diff(expr, axis, order)：对任意子表达式求导（open-form 关键；DISCOVER/AutoKE 都用 autograd.grad 语义：utils_nn.py (line 209)、compute.py (line 11)）
落地建议：先限制 diff(u, axis, order) 跑通 SGA/Burgers；再扩展到 diff(any_expr, …)（这一步最难、也最耗性能）
约束要“两段式”：采样期 mask + 评估期 fail-fast：仅靠生成阶段的 prior 不够（SGA 评估里也会直接丢弃 RHS 对 lhs_axis 求导的 term，防退化：pde.py (line 158)；DISCOVER 有 Regulations 二次检查：stridge.py (line 122)）。你们草稿里建议把 Constraint 拆成：
mask(partial_gen_state)->allowed_tokens（对齐 parent/sibling/dangling：task.py (line 113)）
post_check(full_tree)->valid/reason（数值/物理护栏）
Plugin 接口建议从 search() 升级为 propose()/update()：DISCOVER/SGA 都天然是“提案→评估→更新”的闭环；让 ExperimentManager 统一调度评估（并行/缓存/记录）会比每个 plugin 自己调 evaluator 更可控、更容易做性能与复现。
不要指望“直接包装 ref_libs/DISCOVER”：它核心训练是 TF1；你们目标是 PyTorch（见 discover.md (line 1) 的风险提示）。建议把 DISCOVER 当“算法思想与接口契约”，用 PyTorch 复刻 policy+trainer，平台提供 evaluator。
按风险排序的里程碑（可作为验收标准）

M1（先保底可用）：GenIR<->AnalysisIR + DataSpec(结构化 grid: coords_1d/axis_order/fields) + FiniteDiffProvider + StackExecutor(numpy) + STRidge；用 Burgers/KdV 的合成数据跑通“手动表达式→回归→指标”（对齐 arc_drafts.md (line 449)）
M2（决定性能生死）：表达式 hash 缓存 + 子表达式 memo + 数值 guard（safe-div/NaN/Inf）+ 评估期 regulations；目标是“万级候选评估不退化”
M3（首个插件）：SGA plugin 只做“terms 的 GA”，完全复用平台 evaluator；验收：Burgers 端到端（arc_drafts.md (line 463)）
M4（最难块）：TorchExecutor + AutogradProvider/MetaNet + open-form diff(any_expr, axis, order)；再上 DISCOVER（PyTorch 版）与 risk-seeking 训练（对照 train.py (line 360)）
