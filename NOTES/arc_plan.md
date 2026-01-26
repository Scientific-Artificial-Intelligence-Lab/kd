# kd2 符号回归平台 - 实现计划

> **状态**：待教授讨论确认
> **关联文档**：[arc_final.md](arc_final.md) - 完整架构设计

---

## 一、项目概述

kd2 是一个符号回归领域（特别是 PDE 发现）的通用实验平台，具有以下核心目标：
- **数据层**：统一处理 PDE/ODE 数据，支持多种导数计算方式
- **通用 IR**：两层中间表示（GenIR + AnalysisIR），解耦算法与执行
- **插件化算法**：DISCOVER、SGA、DLGA、PySR 等作为可插拔插件
- **灵活评估**：支持多种评估指标，可定制
- **可视化**：结果和实验过程可视化
- **Web 兼容**：核心零 UI 依赖，预留 Web 接口
- **未来扩展**：Agent 自动实验接口

---

## 二、系统架构

### 2.1 五层结构

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
│   │   ├── library/             # Token/算子库
│   │   ├── executor/            # 执行引擎
│   │   ├── evaluator/           # 评估器
│   │   ├── constraints/         # 约束系统 (两阶段)
│   │   ├── linear_solve/        # 线性求解器 (STRidge/Lasso)
│   │   └── cache/               # 缓存系统 (DiskCache)
│   │
│   ├── data/                    # 数据层
│   │   ├── schema.py           # PDEDataset 规范
│   │   ├── loaders.py          # 数据加载器
│   │   └── derivatives/         # 导数提供者 (两层设计)
│   │
│   ├── plugins/                 # 插件系统
│   │   ├── base.py             # AlgorithmPlugin (propose/update)
│   │   ├── sga/                # SGA 插件
│   │   ├── discover/           # DISCOVER 插件 (PyTorch 重写)
│   │   ├── dlga/               # DLGA 插件
│   │   └── pysr/               # PySR 插件
│   │
│   ├── experiment/              # 实验管理
│   ├── visualization/           # 可视化 (双模式输出)
│   ├── dto/                     # 数据传输对象 (Pydantic)
│   └── config/                  # 配置管理 (Hydra)
│
├── src/kd2_cli/                 # CLI 入口（可选包）
├── src/kd2_api/                 # API 入口（未来 Web 用）
├── tests/
├── examples/
└── pyproject.toml
```

---

## 三、设计决策总结

| 决策点 | 选择 | 说明 |
|--------|------|------|
| **数据类型** | `torch.Tensor` 全程 | 保留计算图，未来可端到端优化 |
| **IR 层数** | 两层优先 | GenIR + AnalysisIR 先行，ExecIR (DAG+CSE) 作为 Phase 5 优化 |
| **首个插件** | SGA | 代码相对简单，GA 逻辑清晰，适合验证核心架构 |
| **导数模式** | 两者并行 | 同时实现有限差分和自动微分，灵活适配不同场景 |
| **插件接口** | `propose()/update()` | 平台统一控制评估，便于缓存和并行 |
| **缓存方案** | DiskCache | 无需运维，单机科研友好 |
| **哈希策略** | 可交换算子子节点排序后哈希 | `a+b` 与 `b+a` 共享缓存 |
| **实验追踪** | WandB + Hydra | 科研标配，配置管理优雅 |
| **DISCOVER 集成** | PyTorch 重写 | TF1 不兼容，保留算法逻辑 |
| **技术栈** | Python 3.11+ | 使用最新语法特性，PyTorch 2.x |
| **Web 兼容** | 核心零 UI 依赖 | Pydantic DTO、可视化双模式、Callback 机制预留 |
| **数据拓扑** | Grid 优先 | Phase 1 定义 topology 字段，MVP 只实现 Grid，Scattered 留 Phase 5 |
| **Checkpointing** | Phase 3 纳入 | 插件 `get_state/set_state` + ExperimentManager 保存/恢复 |
| **结果档案** | 简化版 ResultArchive | 收集结果 + 后处理提取 Pareto 前沿 (Phase 3) |

---

## 四、分阶段实现计划

### Phase 1: 核心基础

**目标**：手动构建表达式、执行并评估

- [ ] IR 系统 (GenIR, AnalysisIR, 转换器)
- [ ] Token 和 Library（默认算子集：add, sub, mul, div, sin, cos, exp, n2, n3, diff, diff2）
- [ ] StackExecutor (NumPy)
- [ ] PDEDataset 数据规范（含 `topology` 字段，MVP 只实现 Grid）
- [ ] FiniteDiffProvider（检查并拒绝 Scattered 数据）
- [ ] 基本评估器 (MSE, NMSE, R2)
- [ ] LeastSquaresSolver
- [ ] Pydantic DTO（Web 兼容预留）

**里程碑**：手动构建表达式、执行并评估

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

- [ ] 约束系统 (采样期 + 评估期两阶段)
- [ ] JointPrior
- [ ] STRidgeSolver
- [ ] LassoSolver
- [ ] AutogradProvider（基础版）
- [ ] 常数优化（内嵌 BFGS）

**里程碑**：评估包含微分的复杂表达式

### Phase 3: SGA 插件

**目标**：使用 kd2 运行 SGA 发现 Burgers 方程

- [ ] AlgorithmPlugin 基类 (propose/update + get_state/set_state)
- [ ] 插件注册机制
- [ ] 配置系统 (Hydra)
- [ ] SGA 适配器
- [ ] ExperimentManager（含 Checkpointing 支持）
- [ ] ResultArchive（收集结果 + 后处理提取 Pareto）
- [ ] 缓存系统 (DiskCache)
- [ ] WandB 集成
- [ ] ProgressCallback（Web 兼容预留）

**里程碑**：使用 kd2 运行 SGA 发现 Burgers 方程

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
- [ ] DISCOVER Controller (LSTM, PyTorch 重写)
- [ ] Risk-seeking policy gradient
- [ ] Priority Queue Training
- [ ] MetaNet 支持

**里程碑**：使用 kd2 运行 DISCOVER

### Phase 5: 完善与扩展

- [ ] ExecIR (DAG + CSE 优化)
- [ ] 可视化模块 (Pareto, 残差图, LaTeX) - 双模式输出
- [ ] DLGA 插件
- [ ] PySR 插件
- [ ] 辅助网络 (AutoKE N2 网络)
- [ ] 文档与示例

### Phase 6: Agent 接口（未来）

- [ ] 自动实验调度
- [ ] 超参数搜索
- [ ] 结果分析
- [ ] LLM 搜索插件
- [ ] Web API (FastAPI/Gradio)

---

## 五、验证方程

| 方程 | 表达式 | 难度 |
|------|--------|------|
| Burgers | `u_t = -u * u_x + 0.1 * u_xx` | 入门 |
| KdV | `u_t = -6 * u * u_x - u_xxx` | 中等 |
| Chafee-Infante | `u_t = u_xx + u - u^3` | 中等 |
| 2D Navier-Stokes | `u_t = -u*u_x - v*u_y + nu*lap(u) - p_x` | 高级 |

---

## 六、关键参考文件

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

## 七、风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| open-form diff 性能 | 先用缓存 + 批量化；Phase 5 用 CSE 优化 |
| DISCOVER TF1 依赖 | PyTorch 重写，保留算法逻辑 |
| 万级候选评估瓶颈 | DiskCache + canonical hash 去重 |
| 数值不稳定 | safety.py 统一护栏 (safe_div, NaN/Inf 检测) |
| 配置管理复杂 | Hydra 优雅管理 |

---

## 八、设计原则

1. **String as Interface, Tensor as Data**
2. **GenIR 是唯一真相，AnalysisIR 是视图**
3. **全程 torch.Tensor，保留计算图**
4. **约束两阶段：采样期 mask + 评估期 fail-fast**
5. **插件只做 propose/update，平台统一评估**
6. **n 维支持，禁止硬编码坐标名**
7. **接口优先，实现渐进（批量化、CSE 留给后续）**
8. **核心零 UI 依赖，结果可序列化**
