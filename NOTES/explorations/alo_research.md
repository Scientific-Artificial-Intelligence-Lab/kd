# LLM-based 符号回归/方程发现方法深度分析

## 研究的工作总览

| 方法 | 发表 | 核心思想 | GitHub |
|------|------|---------|--------|
| **LLM-SR** | ICLR 2025 Oral | 方程作为程序骨架 + 演化搜索 | deep-symbolic-mathematics/LLM-SR |
| **LLM4ED** | Physics of Fluids 2024 | LLM 作为黑盒优化器 + 演化操作 | menggedu/EDL |
| **DrSR** | arXiv 2025 | 双重推理：数据感知 + 归纳思想 | (未公开) |
| **LaSR** | NeurIPS 2024 | 学习概念库 + PySR 混合搜索 | trishullab/LibraryAugmentedSymbolicRegression.jl |
| **ICSR** | ACL 2024 SRW | In-Context 迭代优化 | merlerm/In-Context-Symbolic-Regression |
| **IdeaSearchFitter** | arXiv 2025 | LLM 作为语义变异/交叉算子 | ideasearch.cn |

---

## 核心发现：共同的架构模式

**所有 LLM-based 方法都遵循相同的四阶段模式：**

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM-based 方程发现通用流程                 │
├─────────────────────────────────────────────────────────────┤
│  1. LLM 生成方程骨架 (skeleton/ansatz)                       │
│     - 输入: 问题描述、数据特征、历史反馈                       │
│     - 输出: 带有常数占位符的表达式 (如 "c1 * x + c2 * sin(x)")│
│                                                             │
│  2. 常数优化 (外部优化器)                                    │
│     - BFGS/Nelder-Mead (scipy.optimize)                     │
│     - STRidge/LASSO (线性系数)                              │
│     - Adam (PyTorch, 可微分场景)                            │
│                                                             │
│  3. 评估打分                                                │
│     - 数值精度: MSE, NMSE, R²                               │
│     - 复杂度: 节点数、深度、参数数量                         │
│     - 有效性: 语法正确、数值稳定 (无 NaN/Inf)                │
│                                                             │
│  4. 反馈/搜索循环                                           │
│     - 演化: mutation, crossover (LaSR, IdeaSearchFitter)    │
│     - 自我改进: 历史表现反馈 (LLM4ED, DrSR)                  │
│     - 概念抽象: 提取可复用模式 (LaSR)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 各方法详细分析

### 1. LLM-SR (ICLR 2025 Oral) - 最有影响力

**核心思想**: 方程表示为 Python 程序骨架，LLM 生成骨架 + 演化搜索

**架构**:
```
问题描述 (spec) → LLM 生成程序骨架 → scipy.optimize.minimize(BFGS) 优化常数
                        ↓
         评估 (MSE) → 演化搜索 (FunSearch 风格) → 迭代
```

**关键设计**:
- 方程是 NumPy/PyTorch **程序**，不是字符串
- 常数用占位符 `c0, c1, ...` 表示
- 演化采用 FunSearch 的 "programs database" 设计
- 支持 local LLM (vLLM) 和 OpenAI API

**代码结构** (从 GitHub 分析):
```
LLM-SR/
├── llm_engine/       # LLM 服务器
│   └── engine.py     # vLLM/HuggingFace 推理
├── llmsr/
│   ├── sampler.py    # LLM 采样
│   ├── evaluator.py  # 评估器
│   └── config.py     # 配置
├── specs/            # prompt 模板 (问题描述)
└── data/             # 数据集
```

**对 kd2 的启示**:
- **程序表示比字符串更灵活** - 但需要 sandbox 执行
- **常数优化是独立模块** - kd2 的 LinearSolver 正好对应
- **LLM 引擎可以抽象** - local/API 统一接口

---

### 2. LLM4ED - 专注 PDE 发现

**核心思想**: 两阶段优化 - 自我改进 + 演化操作

**架构**:
```
┌─────────────────────────────────────────┐
│          LLM4ED 双策略优化              │
├─────────────────────────────────────────┤
│  策略1: 自我改进 (Self-Improvement)     │
│  - LLM 分析历史方程的性能               │
│  - 基于关系发现，局部修改方程           │
│                                         │
│  策略2: 演化操作 (Evolutionary)         │
│  - LLM 执行用户定义的变异/交叉          │
│  - 全局搜索，生成多样化组合             │
└─────────────────────────────────────────┘
```

**PDE 特有设计**:
- 方程按 `+/-` 拆分为 terms
- 每个 term 的系数用 **稀疏回归** (STRidge) 求解
- 这和 kd2 plan 中的 "split_terms → STRidge" 完全一致！

**代码结构**:
```
EDL/
├── scripts_pde/      # PDE 发现脚本
├── script_ode/       # ODE 发现脚本
├── prompt.py         # prompt 模板
├── prompt_utils.py   # prompt 工具
├── optimzier_utils.py # 常数优化
└── evaluation/       # 评估模块
```

**对 kd2 的启示**:
- **PDE 的 term-based 评估** 是 kd2 已经计划的
- **双策略优化** 可以作为 LLM 插件的设计模式
- prompt 设计对 PDE 有特殊考量（微分算子描述）

---

### 3. DrSR - 双重推理 (2025 最新)

**核心思想**: 数据感知 + 经验归纳的闭环

**架构**:
```
┌────────────────────────────────────────────────────────┐
│                    DrSR 双重推理                        │
├────────────────────────────────────────────────────────┤
│  π_data (数据感知)                                      │
│  - 分析数据的结构关系：单调性、非线性、相关性            │
│  - 生成 structured description                         │
│                                                        │
│  π_idea (归纳思想)                                     │
│  - 从历史生成中提取 valid ideas / invalid ideas        │
│  - 有效策略 → 复用；无效策略 → 避免                    │
│                                                        │
│  π_main (方程生成)                                     │
│  - 基于 data insight + ideas 生成方程                  │
└────────────────────────────────────────────────────────┘
```

**创新点**:
- **Invalid idea extraction** - 避免重复错误（语法错误、数值溢出）
- **Data-aware insight** - LLM 先分析数据，再生成方程
- 大幅提升 **有效解比例** (valid solution rate)

**对 kd2 的启示**:
- **数据分析可以作为预处理步骤**
- **错误追踪** 对 LLM 方法很重要
- kd2 的评估模块应该返回详细的失败原因

---

### 4. LaSR - 概念库 + PySR 混合

**核心思想**: 用 LLM 学习可复用的概念库，指导 PySR 搜索

**架构**:
```
┌─────────────────────────────────────────────────────────┐
│                    LaSR 三阶段循环                       │
├─────────────────────────────────────────────────────────┤
│  Phase 1: Hypothesis Evolution                          │
│  - 基于 PySR 的多种群遗传搜索                           │
│  - 以概率 p (1%) 替换传统操作为 LLM 操作                │
│  - LLM 操作基于 concept library                        │
│                                                         │
│  Phase 2: Concept Abstraction                           │
│  - 从高性能方程中提取概念 (如 "power law")              │
│  - 存入 concept library                                │
│                                                         │
│  Phase 3: Concept Evolution                             │
│  - LLM 推理概念的蕴含关系                               │
│  - 扩展 concept library                                │
└─────────────────────────────────────────────────────────┘
```

**关键创新**:
- **不替代 PySR，而是增强它** - LLM 只有 1% 概率介入
- **概念库** 是可累积的知识
- **语义搜索 vs 语法搜索** - 克服 PySR 的 exploration bottleneck

**对 kd2 的启示**:
- **混合方法可能是最佳策略**
- **概念库** 可以跨实验复用
- kd2 应该支持 "增强现有算法" 的模式

---

### 5. ICSR - In-Context 符号回归

**核心思想**: 最简单的 LLM 方法 - 纯 in-context learning

**架构**:
```
Prompt: "这些数据点: [(x1,y1), ...], 之前尝试过的方程: [f1(score), f2(score), ...], 请生成更好的方程"
  ↓
LLM 生成新方程 → scipy 优化常数 → 评估 → 加入历史 → 迭代
```

**关键设计**:
- **无需训练/微调** - 纯 zero-shot
- **历史反馈** 在 prompt 中
- **外部优化器** 处理常数

**对 kd2 的启示**:
- **最简单的 LLM 集成方式**
- 可以作为 kd2 的 "LLM baseline"

---

### 6. IdeaSearchFitter - 语义变异/交叉

**核心思想**: LLM 作为语义层面的演化算子

**架构**:
```
传统 GP: 语法变异 (改变树节点)
IdeaSearchFitter: 语义变异 (LLM 理解含义后改进)

例如:
- 传统: f(x) = x² → f(x) = x³ (随机改一个算子)
- 语义: f(x) = x² → "这是二次函数，也许应该加入线性项" → f(x) = ax² + bx
```

**关键创新**:
- **Explain-then-formalize** - LLM 先解释为什么要改，再改
- **Multi-island evolutionary** - 多种群并行
- **Pareto 前沿** - accuracy + complexity + interpretability

**对 kd2 的启示**:
- **变异/交叉可以用 LLM 实现**
- kd2 的 GA 插件可以支持 "LLM-guided" 模式

---

## 对 kd2 架构的核心启示

### 1. propose/update 接口可能太简化

你们当前的设计:
```python
class AlgorithmPlugin:
    def propose(self, n: int) -> List[GenIR]
    def update(self, expressions, rewards)
```

**问题**: LLM 方法的搜索循环更复杂:
- LLM-SR: 需要 prompt + feedback 历史
- DrSR: 需要 data insight + idea library
- LaSR: 需要 concept library + PySR 状态

**建议**: 采用 **组件层 + 可选接口层** 设计

```python
# 核心组件 - 所有方法都可以直接使用
class kd2.Components:
    executor: Executor
    evaluator: Evaluator
    constant_optimizer: ConstantOptimizer
    library: Library
    cache: Cache
    llm_interface: LLMInterface  # 新增！

# 可选接口 - 适合 GA/GP 类
class AlgorithmPlugin:
    def propose(self, n) -> List[GenIR]
    def update(self, expressions, rewards)

# 直接组合组件 - 适合 LLM 类
class LLMBasedRunner:
    def __init__(self, components: kd2.Components):
        self.components = components
    
    def run(self, dataset, custom_loop):
        # 算法自己控制循环
        pass
```

### 2. 需要新增 LLM 接口组件

```python
# 新增 LLM 相关组件
kd2/
├── llm/
│   ├── interface.py     # 统一 LLM 接口
│   │   ├── LocalLLM     # vLLM / HuggingFace
│   │   └── APIClient    # OpenAI / Anthropic
│   ├── sampler.py       # 采样策略 (temperature, top_p, etc.)
│   ├── prompt.py        # prompt 模板管理
│   └── parser.py        # 字符串 → GenIR 解析
```

### 3. 常数优化需要更灵活

LLM 方法生成的是 **骨架**，常数是占位符:
```python
# LLM 输出: "c0 * x + c1 * sin(c2 * x)"
# 需要: 解析 → 识别常数 → 优化

class ConstantOptimizer:
    def optimize(self, skeleton: GenIR, dataset) -> Tuple[GenIR, List[float]]:
        """
        输入: 带占位符的表达式
        输出: 优化后的表达式 + 常数值
        """
        # 两种模式:
        # 1. LINEAR: split terms → STRidge (你们已有)
        # 2. NONLINEAR: BFGS/Adam (需要添加)
```

### 4. 评估模块需要返回更多信息

DrSR 的经验: **记录失败原因** 对 LLM 很重要

```python
class EvaluationResult:
    # 数值评估
    mse: float
    nmse: float
    r2: float
    
    # 复杂度
    complexity: int
    depth: int
    
    # 有效性
    is_valid: bool
    error_type: Optional[str]  # "syntax_error", "nan", "inf", "timeout"
    error_message: Optional[str]
    
    # 用于 LLM 反馈
    def to_feedback_str(self) -> str:
        """生成可以放入 LLM prompt 的反馈字符串"""
```

### 5. 方程表示需要支持常数占位符

```python
# 当前 GenIR 设计:
GenIR = List[Token]  # ["add", "mul", "u", "u_x", "u_xx"]

# 需要支持:
# 1. 数值常数: ["add", "mul", "3.14", "u", "u_x"]
# 2. 占位符: ["add", "mul", "c0", "u", "c1", "u_x"]

class Token:
    type: TokenType  # OPERATOR, VARIABLE, CONSTANT, PLACEHOLDER
    value: str
    
class TokenType(Enum):
    OPERATOR = "op"
    VARIABLE = "var" 
    CONSTANT = "const"      # 已知数值
    PLACEHOLDER = "param"   # 待优化常数
```

---

## 建议的 kd2 架构修订

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│   CLI Tool  │  Web UI  │  Agent API  │  Notebooks           │
├─────────────────────────────────────────────────────────────┤
│                     Experiment Layer                         │
│   ExperimentManager  │  ResultStore  │  Visualization        │
├─────────────────────────────────────────────────────────────┤
│                    Algorithm Layer                           │
│                                                              │
│  ┌─────────────────────┐  ┌─────────────────────────────┐   │
│  │   Plugin Interface   │  │    Direct Component Usage    │   │
│  │   (propose/update)   │  │    (for LLM methods, etc.)   │   │
│  │                      │  │                              │   │
│  │   - SGA             │  │   - LLM-SR Runner           │   │
│  │   - DISCOVER        │  │   - DrSR Runner             │   │
│  │   - DLGA            │  │   - LaSR Runner (wrap PySR) │   │
│  └─────────────────────┘  └─────────────────────────────┘   │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                       Core Layer                             │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  IR System                                          │    │
│  │  - GenIR (支持常数占位符)                           │    │
│  │  - AnalysisIR                                       │    │
│  │  - Converters (string ↔ IR ↔ SymPy)                │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────────────┐  │
│  │  Executor   │ │  Evaluator  │ │  ConstantOptimizer   │  │
│  │  - NumPy    │ │  - MSE/NMSE │ │  - STRidge (linear)  │  │
│  │  - PyTorch  │ │  - R²/AIC   │ │  - BFGS (nonlinear)  │  │
│  └─────────────┘ │  - Validity │ │  - Adam (diff.)      │  │
│                  └─────────────┘ └──────────────────────┘  │
│                                                              │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────────────┐  │
│  │  Library    │ │    Cache    │ │   LLM Interface ★    │  │
│  │  - Operators│ │  - DiskCache│ │  - Local (vLLM)      │  │
│  │  - Tokens   │ │  - Memory   │ │  - API (OpenAI)      │  │
│  └─────────────┘ └─────────────┘ │  - Prompt Templates  │  │
│                                  │  - Parser            │  │
│                                  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Foundation Layer                          │
│   Data Layer │ DerivativeProvider │ PDEDataset              │
│   PyTorch    │ NumPy              │ SymPy                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 实现优先级建议

### Phase 1-3 保持不变
按原计划完成核心功能 + SGA 插件

### Phase 4 调整: 加入 LLM 支持
```
原计划: DISCOVER 插件 (PyTorch 重写)
新增:
  - LLM Interface 组件
  - Prompt Template 系统
  - String → GenIR 解析器
  - 非线性常数优化器 (BFGS)
  - LLM-SR 风格的示例 runner
```

### Phase 5+: LLM 方法生态
```
- DrSR 风格的 data-aware insight
- LaSR 风格的 concept library
- Agent 自动实验 (使用 LLM 选择算法/参数)
```

---

## 结论

**LLM 方法代表了方程发现的新范式**，但它们都需要相同的核心组件：
- 表达式表示与执行
- 常数优化
- 评估打分
- (可选) LLM 接口

**kd2 的定位应该是**:
> 提供 PDE 发现算法（无论是传统搜索还是 LLM-based）都需要的可复用组件，让算法开发者专注于搜索策略本身。

这样 kd2 就可以同时支持:
1. 传统搜索方法 (SGA, DISCOVER) - 通过 propose/update 接口
2. LLM-based 方法 - 直接使用组件层
3. 混合方法 (LaSR 风格) - 组合使用

**核心价值**: 无论用什么搜索策略，评估、优化、缓存、可视化都是通用的。