# kd2 架构设计文档

> **状态**：正式版 v1.0
> **定位**：符号回归领域（特别是 PDE 发现）的通用实验平台
> **设计原则**：String as Interface, Tensor as Data

---

## 文档结构

| 文档 | 内容 |
|------|------|
| **arc_final.md**（本文档） | 主文档：决策表、架构图、目录结构 |
| [arc_core.md](arc_core.md) | 核心层：IR、执行器、评估器、约束、缓存 |
| [arc_data.md](arc_data.md) | 数据层：PDEDataset、导数提供者、ScaleHandler |
| [arc_plugin.md](arc_plugin.md) | 插件层：接口、SGA、DISCOVER、实验管理 |
| [arc_web.md](arc_web.md) | Web兼容：DTO、可视化、Callback |
| [arc_plan.md](arc_plan.md) | 实现计划：分阶段任务列表 |
| [arc_codex.md](arc_codex.md) | Codex Review：详细建议记录 |

---

## 一、设计目标

1. **统一平台**：整合 DISCOVER、SGA、DLGA、PySR 等算法为可插拔插件
2. **n 维支持**：代码中不硬编码 x、y、t，统一使用坐标配置
3. **可微执行**：全程使用 `torch.Tensor`，保留计算图用于未来端到端优化
4. **高效评估**：支持万级候选的批量评估，带缓存和数值护栏
5. **可扩展性**：未来可接入 Agent 自动实验、LLM 搜索等

---

## 二、设计决策总结

### 2.1 架构决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| **数据类型** | `torch.Tensor` 全程 | 保留计算图，未来可端到端优化 |
| **IR 层数** | 两层 (GenIR + AnalysisIR) | ExecIR (DAG+CSE) 作为 Phase 5 优化 |
| **插件接口** | `propose()/update()` | 平台统一控制评估，便于缓存和并行 |
| **缓存方案** | DiskCache + 分桶目录 | 无需运维，单机科研友好，隔离不同实验 |
| **哈希策略** | sha256(canonical_string) | 稳定哈希，可交换算子子节点排序 |
| **实验追踪** | WandB + Hydra | 科研标配，配置管理优雅 |
| **技术栈** | Python 3.11+, PyTorch 2.x | 未来可 C/C++ 优化瓶颈 |

### 2.2 数据与导数

| 决策点 | 选择 | 理由 |
|--------|------|------|
| **数据拓扑** | Grid 优先，Scattered 延后 | MVP 只实现 Grid |
| **导数模式** | 有限差分 + 自动微分并行 | 灵活适配不同场景 |
| **导数表示** | 混合方案 + 约束消别名 | `u_x` terminal + `diff` operator |
| **导数命名** | `u_xx` 显示 + `deriv:u:x:2` 内部 | MVP 单轴导数 |
| **归一化** | ScaleHandler (Phase 2) | Phase 1 无归一化 |
| **dataset_fp** | 元信息 + 采样 hash | 用于缓存隔离 |

### 2.3 评估与常数

| 决策点 | 选择 | 理由 |
|--------|------|------|
| **评估模式** | LINEAR 默认，DIRECT 可选 | split_terms → STRidge 为主流程 |
| **常数策略** | ξ 优先，θ 延后 | Phase 1-3 仅线性系数 |
| **评估指标** | 可插拔 Metric | 内置 AIC/BIC，支持自定义 |
| **计算图保留** | 默认关闭 + 可选开启 | 性能优先，PINN 等可开启 |
| **额外基函数** | 可配置列表，默认空 | 常数项用 `["1"]` |

### 2.4 实验管理

| 决策点 | 选择 | 理由 |
|--------|------|------|
| **首个插件** | SGA | 代码简单，适合验证架构 |
| **Checkpointing** | Phase 3 纳入 | 插件 get_state/set_state |
| **结果档案** | 简化版 ResultArchive | 收集结果 + 后处理 Pareto |
| **Web 兼容** | 核心零 UI 依赖 | Pydantic DTO、可视化双模式 |
| **可视化架构** | Facade + Registry | 统一入口，插件可扩展，kd1 验证成熟 |

---

## 三、系统架构

### 3.1 五层架构

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

### 3.2 目录结构

```
kd2/
├── src/kd2/
│   ├── core/                    # 核心模块
│   │   ├── ir/                  # 两层 IR 系统
│   │   ├── library/             # Token/算子库
│   │   ├── executor/            # 执行引擎
│   │   ├── evaluator/           # 评估器
│   │   ├── constraints/         # 约束系统
│   │   ├── linear_solve/        # 线性求解器
│   │   └── cache/               # 缓存系统
│   │
│   ├── data/                    # 数据层
│   │   ├── schema.py           # PDEDataset 规范
│   │   ├── loaders.py          # 数据加载器
│   │   └── derivatives/         # 导数提供者
│   │
│   ├── plugins/                 # 插件系统
│   │   ├── base.py             # AlgorithmPlugin
│   │   ├── sga/                # SGA 插件
│   │   ├── discover/           # DISCOVER 插件
│   │   └── ...
│   │
│   ├── experiment/              # 实验管理
│   ├── visualization/           # 可视化 (Facade + Registry)
│   ├── dto/                     # 数据传输对象
│   └── config/                  # 配置管理
│
├── src/kd2_cli/                 # CLI 入口（可选包）
├── src/kd2_api/                 # API 入口（未来 Web 用）
├── tests/
├── examples/
└── pyproject.toml
```

---

## 四、分阶段计划概览

| Phase | 目标 | 里程碑 |
|-------|------|--------|
| **Phase 1** | 核心基础 | 手动构建表达式、执行并评估 |
| **Phase 2** | 约束与线性求解 | 评估包含微分的复杂表达式 |
| **Phase 3** | SGA 插件 | 使用 kd2 运行 SGA 发现 Burgers 方程 |
| **Phase 4** | DISCOVER 插件 | 使用 kd2 运行 DISCOVER |
| **Phase 5** | 完善与扩展 | ExecIR、可视化、DLGA、PySR |
| **Phase 6** | Agent 接口 | 自动实验调度、LLM 搜索 |

详见 [arc_plan.md](arc_plan.md)。

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

## 七、设计原则

1. **String as Interface, Tensor as Data**
2. **GenIR 是唯一真相，AnalysisIR 是视图**
3. **全程 torch.Tensor，保留计算图**
4. **约束两阶段：采样期 mask + 评估期 fail-fast**
5. **插件只做 propose/update，平台统一评估**
6. **n 维支持，禁止硬编码坐标名**
7. **接口优先，实现渐进**
8. **核心零 UI 依赖，结果可序列化**

---

## 八、风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| open-form diff 性能 | 先用缓存 + 批量化；Phase 5 用 CSE 优化 |
| DISCOVER TF1 依赖 | PyTorch 重写，保留算法逻辑 |
| 万级候选评估瓶颈 | DiskCache + canonical hash 去重 |
| 数值不稳定 | safety.py 统一护栏 |
| 配置管理复杂 | Hydra 管理 |
