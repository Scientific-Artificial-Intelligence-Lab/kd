# kd2 架构设计（Codex 版 / Working Doc）

> 状态：草案（会不断迭代）。  
> 目标：把 `NOTES/arc_info.md` 的教授思路落成“可插件化的 PDE Discovery 实验平台”架构，并能接入 `ref_libs/` 的算法实现。  
> 约定：不把任何单一算法的历史包袱带进 core；core 只沉淀“跨算法可复用的稳定边界”。  

## 1. 背景（来自 `NOTES/arc_info.md`）

- kd2 是 kd 的新版本，定位为符号回归平台，**优先 PDE 发现**，同时是可扩展的实验平台。
- 平台应包含：
  - DataLayer：数据加载/清洗/导数准备（FD 或 NN+Autograd）。
  - 通用 IR：表达式/计算顺序的统一表示（具体形式未定）。
  - 算法插件：SGA / DISCOVER / DLGA / PySR 等。
  - 评估模块：平台级，允许算法开发者定制/复用。
  - 结果与可视化：一等公民。
  - 未来：接入 agent，做自动化实验平台。
- 教授核心洞察：**符号树/字符串本质都是“计算顺序”**；用 NN 的自动微分 + 链式法则（以及 PyTorch 计算图的 autograd）可以实现任意计算顺序并计算残差与导数。

## 2. 从 `ref_libs/` 反推的“必须做对”的核心能力

> 本节是“平台必须提供什么”，否则插件会强耦合/无法落地。

### 2.1 导数能力必须是“开放形式”（open-form）

- 不仅要支持 `u_x, u_xx` 这类预计算导数，还要支持 `diff(u*u_x, x)` 这类对**复合子表达式**求导。
- `ref_libs/AutoKE` 与 `ref_libs/DISCOVER` 的 torch 执行路径都在用 `torch.autograd.grad` 语义；`ref_libs/sga` 也明确把 open-form 作为关键能力。

### 2.2 线性稀疏回归（STRidge/AIC）应成为平台可复用组件

- SGA-PDE/DISCOVER 都在用“拆项 + 线性回归求系数”做加速/稳定（不仅是一个插件私货）。
- 如果平台只支持“单棵树直接 residual”，很多 PDE discovery 方法会慢一个量级。

### 2.3 批量候选的高效执行（含缓存与数值护栏）是成败关键

- GA/RL 会评估海量候选；每次都重新 parse/构图/求导会成为瓶颈。
- 必须统一处理：safe-div / NaN/Inf 传播策略 / domain scaling / 子表达式缓存（memo/CSE）。

## 3. 核心模块边界（建议先固定 6–8 个稳定接口）

建议把 core 固定为以下稳定边界，其余（算法、具体 UI、benchmark 适配）都做插件或上层应用：

1. **DataSpec / DataLayer**：数据协议与加载（结构化 grid 优先，nD & multi-field）。
2. **DerivativeProvider**：FD / NN+autograd / hybrid；提供 `eval(expr)` 与 `diff(expr, axis, order)` 所需能力。
3. **SymbolLibrary**：token/算子/变量注册（arity/type/complexity/cost）。
4. **ExpressionIR + Codec**：表达式表示与编解码（prefix tokens ↔ tree ↔ 执行视图）。
5. **Executor**：把 ExpressionIR 编译/执行为张量计算（可选 DAG/CSE 加速）。
6. **Evaluator**：accuracy/simplicity/…；可组合；可内嵌 inner-solver（STRidge 等）。
7. **ConstraintSystem**：生成阶段 mask + 评估阶段 post-check（例如 RHS 禁止沿 lhs_axis 求导）。
8. **OptimizerPlugin**：算法插件只依赖 propose/update，不关心底层实现细节。

最小伪 API（语言无关）：

```text
Dataset = DataLayer.load(spec, raw)
Derivs  = DerivativeProvider.build(dataset, requests, method=fd|nn_autograd|hybrid)

expr    = Codec.decode(prefix_tokens | canonical_string) -> ExpressionIR
exec    = Executor.compile(expr, derivs, backend=torch|numpy) -> Executable
report  = Evaluator.evaluate(exec, dataset) -> EvalReport

state   = Optimizer.init(search_space, seed, budget)
batch   = Optimizer.propose(state, k) -> [Candidate(prefix_tokens|expr)]
state   = Optimizer.update(state, feedback=[{candidate_id, report}])
```

## 4. IR 设计：三层视图（强烈建议）

从 DISCOVER/SGA 的“生成侧需求”与“执行侧加速需求”同时满足的角度，建议采用三层 IR：

1. **Canonical / Interchange IR：prefix token sequence（或其 canonical string）**  
   - 直接对齐 DISCOVER 的生成空间（dangling/parent/sibling 状态都是树语义）。
   - 易哈希、易日志、易复现、易去重。
2. **Analysis IR：AST Tree**  
   - 复杂度（深度/长度/项数）、结构约束、可视化、规则检查最自然。
3. **Execution IR（可选）：DAG（hash-cons/CSE 后的共享子图）**  
   - 只用于加速执行与缓存；**不作为生成 IR**（否则 parent/sibling/dangling 语义不再良定义）。

结论（来自 `ref_libs/discover*.md` 的一致建议）：**不要把 DAG 当成生成 IR**；可以把 DAG 当执行加速层。

## 5. 数据与导数：DataSpec / DerivativeProvider 的关键约束

从 `ref_libs/sga/sgapde/context.py` 的 nD 升级路径可抽象出稳定数据协议：

- `coords_1d: Dict[axis_name, np.ndarray]`（每个轴一维坐标）
- `axis_order: List[axis_name]`（定义 field 张量维度顺序）
- `fields_data: Dict[field_name, np.ndarray]`（每个场一个 nD 张量）
- `target_field: str`（默认 "u"）
- `lhs_axis: str`（默认 "t"，表示发现 `d(target_field)/d(lhs_axis) = RHS(...)`）

DerivativeProvider 至少要支持两条策略：

- **FiniteDiffProvider**：快；对高质量数据友好。
- **NN+AutogradProvider（MetaNet / PINN 风格）**：抗噪；支持 open-form；需处理 normalization 与链式法则缩放（`ref_libs/sga/sgapde/context.py:_calculate_derivatives_autograd` 有可复用的尺度处理思路）。

## 6. Evaluator：统一 reward/metrics + inner-solver

建议把 Evaluator 设计为可组合的“多头报告”：

- accuracy：MSE/NMSE/R2、residual 分布等
- simplicity：长度/深度/项数/算子加权复杂度
- constraints：非法原因（NaN/Inf、除零、RHS 含 lhs_axis 导数、越界等）
- 可选：physical consistency、math solvability（先预留接口）

并把 **inner-solver** 作为可插拔组件（但由平台提供实现）：

- 输入：terms 的数值列（Θ）与 LHS（如 `u_t`）
- 输出：系数、选择的 term、AIC/score、诊断信息
- 典型：STRidge / Lasso / Ridge + AIC

这会让 SGA、SINDy、DISCOVER 的“拆项+拟合”路径复用同一套基础设施。

## 7. 插件接口：OptimizerPlugin（算法只做搜索/更新）

插件只做两件事：

- `propose(state, k)`：给出候选（prefix tokens 或 ExpressionIR）。
- `update(state, feedback)`：基于 report/reward 更新内部状态（RL/GA/…）。

平台负责：

- 合法性/数值护栏
- 统一求值与回归
- 统一结果协议（用于比较、复现、可视化、agent 调用）

## 8. 各参考算法的“集成要点”（当前理解）

### 8.1 DISCOVER（`ref_libs/DISCOVER`）

- 生成侧是 **prefix traversal + dangling/parent/sibling** 的树语义（`dso/dso/program.py` 等）；因此生成 IR 必须是 tree/prefix。
- 控制器是 TF1（`dso/dso/controller.py`）；若 kd2 以 PyTorch 为核心，建议**重写 policy 网络与训练 loop**，但保持 risk-seeking top-ε + baseline + entropy 的算法逻辑不变。
- 执行侧已有 torch 执行语义（`dso/dso/execute.py:python_execute_torch`），并以 NaN/Inf 做 early reject。
- 建议集成形态：把 DISCOVER 做成“策略插件”，平台提供 Evaluator（含可选 STRidge inner-solver）。

### 8.2 SGA-PDE（`ref_libs/sga`）

- 方法假设是 `u_t = Θ · ξ` 的“对系数线性”的显式 PDE；SGA 搜索的是 **term forest**（若干项相加）。
- `evaluate_mse` 里明确：
  - RHS 禁止对 `lhs_axis` 求导（fail-fast）
  - 除零策略与 bad term 丢弃
  - Θ 构建 + STRidge + AIC 是核心评分
- 集成关键：平台必须提供批量候选的 `eval/diff` 与 inner-solver；否则性能会崩。

### 8.3 AutoKE（`ref_libs/AutoKE`）

- 核心价值：把方程字符串解析成 AST，并在遍历 AST 时用 `torch.autograd.grad` 实现 diff（`net.py:autocal_residual` + parse 模块）。
- 对 kd2 的启发：**string 作为稳定 interchange 很好**，但执行时应转成结构化 IR 并可缓存。

### 8.4 DLGA（`ref_libs/dlga/dlga.py`）

- 采用“NN 拟合数据 + GA 搜索基因模块”的混合框架；当前代码依赖更大工程（相对导入 `..base` 等）。
- 暂定：作为后续插件接入；先把 core 做稳。

## 9. 待你确认的关键问题（会直接决定 IR/接口/里程碑）

1. **任务边界**：kd2 v1 只做显式形式 `u_{lhs_axis} = RHS` 吗？是否需要支持隐式 `F(...)=0`、多方程/多场耦合？
2. **数据形态**：v1 是否可以 grid-first（`coords_1d + axis_order + fields_data`）？是否必须同时支持散点/不规则采样？
3. **导数策略默认值**：默认 FD 还是 NN+autograd？噪声数据是第一优先级吗？
4. **IR canonical**：是否接受“canonical=prefix tokens/string；analysis=tree；execution=optional DAG”的三层方案作为 v1？
5. **表达式语义**：`diff(expr, axis, order)` 是否作为一等算子进入 token 库（像 DISCOVER/AutoKE/SGA 一样）？
6. **系数拟合**：是否把“feature library + sparse regression(AIC)”作为 PDE discovery 的一等公民（强烈建议）？
7. **插件粒度**：结构搜索（propose）与系数拟合（inner-solver）是否强制解耦？还是允许插件自带拟合器？
8. **最先要集成的算法顺序**：SGA vs DISCOVER vs PySR（DLGA 可后置）？你希望 v1 的最小可用 demo 是哪个？
9. **产物协议**：你最想一键得到哪些 artifact（最佳方程、Pareto 前沿、训练曲线、残差图、可复现实验包）？
10. **对外形态**：v1 以 Python API 为主还是需要 CLI/Notebook 入口同步提供？

## 10. 下一步（建议）

- 你先回答第 1/2/3/4/6/8 这 6 个“会改变骨架”的问题；我据此把 IR 与核心接口收敛成可执行的工程规范。
- 然后我们用 TDD 方式做一个最小 core：`DataSpec + DerivativeProvider + ExpressionIR + Evaluator(inner-solver)`，再选一个算法（建议先 SGA 或 DISCOVER 的简化版）接入做端到端验证。
