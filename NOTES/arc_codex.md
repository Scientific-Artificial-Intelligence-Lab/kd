# kd2 架构 Review（Codex 记录）

> 目的：把对架构的 review 结论/风险/已确认决策记录在此，作为后续实现与讨论的单一入口。  
> 依据：`NOTES/arc_info.md`、`NOTES/arc_plan.md`、`NOTES/arc_final.md`、`ref_libs/*`（DISCOVER/SGA/AutoKE 与讨论汇总）。  
> Review 方式：分块慢速审查；每轮只审一部分并提出需要确认的问题。

## 已确认决策（你已在对话中确认）

### Task Boundaries（Phase 1–3）
- MVP 严格限定为**显式单方程**：`u_{lhs_axis} = RHS(...)`。
- 暂不支持隐式形式 `F(...)=0`（需要不同 loss/约束体系，且更易出现平凡解与不适定）。

### 表达式/导数语义（混合）
- `u_x / u_xx / ...` 等作为**独立终结符（terminals）长期保留**（高吞吐、利于 GA/回归路线）。
- `diff(expr, axis, order)` 作为 **open-form diff**（对任意子表达式求导）保留，用于 DISCOVER 等需要的表达能力。
- `diff` 约束范围（已决定）：
  - 只禁止对**原始场**做 `diff`：例如 `diff(u, x)`、`diff(v, y)`（避免与 `u_x` 等 terminals 重复、也避免平凡解）。
  - 允许对**导数 terminals**做 `diff`：例如 `diff(u_x, x) → u_xx`（支持高阶导的“递推生成”）。
  - 允许对**复合表达式**做 `diff`：例如 `diff(sin(u), x)`、`diff(u*u_x, x)`（open-form diff）。
- 建议：当 `diff(deriv_terminal, axis)` 等价于某个已存在的导数 terminal 时，在 canonicalization 阶段直接 rewrite（避免重复表达、提升缓存命中）。
- MVP（Phase 1–4）仍只支持单轴高阶导：允许 `diff(u_xx, x) → u_xxx`；禁止 `diff(u_x, y)` 这类混合导数（Phase 5 再扩展）。

### 评估模式 / 常数 / 缓存隔离
| 决策点 | 选择 | 备注 |
|---|---|---|
| 评估模式 | `LINEAR` 默认，`DIRECT` 可选 | `split_terms → STRidge` 为主流程，允许整棵 RHS 直接评估 |
| 常数策略 | `ξ` 优先，`θ` 延后 | Phase 1–3 仅线性系数；Phase 4 引入非线性常数 BFGS |
| 缓存隔离 | 复合键 + 分桶目录 | `dataset_fp/deriv_config` 分目录，避免跨实验污染 |

## 本轮 review 结论（IR / 插件边界 / 缓存风险）

### 1) IR 总体方向是对的，但要补齐“可复现/可缓存”的硬约束
- 你们的“两层 IR”（`GenIR` canonical + `AnalysisIR` view）与 `propose()/update()` 插件边界，与 DISCOVER 的生成假设（prefix + dangling/parent/sibling）高度对齐；也能覆盖 SGA 的 tree 操作需求。
- 建议坚持“生成侧=Tree/prefix；执行侧可选 DAG/CSE”——**不要把 DAG 当生成 IR**（否则 DISCOVER 的结构感知状态不再良定义）。

### 2) 必须修正：canonical hash 不能用 Python `hash()`
- `NOTES/arc_final.md` 里示例 `canonical_hash()` 使用了 `hash((...))` 风格；Python 的 `hash()` **跨进程/跨运行不稳定**，会直接破坏 DiskCache 命中与复现。
- 建议：以“可版本化的 canonical_string”为唯一输入，做稳定哈希（例如 `sha256(canonical_string)`），并显式纳入：
  - `library` 版本/算子表（token->arity/type/commutativity/complexity）
  - canonicalization 规则版本（commutative 排序、常数规范化、term split 规范等）

### 3) 缓存键必须包含上下文（否则会错误复用）
仅用表达式结构做 key 不够：同一表达式在不同 dataset / 导数策略 / 归一化尺度 / solver 超参下评估结果不同。
- 建议缓存“分桶目录 + 复合键”：
  - 分桶：`cache_root/<dataset_fp>/<deriv_fp>/...`
  - 键：`expr_fp + eval_mode + solver_fp + safety_fp (+ seed/version)`  
    （至少保证跨实验不会污染/错用）

### 4) DiskCache 不应直接存大 Tensor（尤其是 residual 向量）
`EvaluationResult` 里若含 `residual: torch.Tensor`，缓存会变得巨大、慢、不可控；而且插件反馈也不需要这么重的对象。
- 建议拆分：
  - `EvalSummary`：标量/小对象（nmse/aic/complexity/coeffs/valid_reason），用于缓存与 `Feedback`
  - `EvalArtifacts`：残差向量、term 明细、调试数据（可选落盘/按需保存）

### 5) `finish_tokens()` 的语义需要重定义（当前示例不自洽）
`dangling` 代表“还需要多少个叶子/子树”才能闭合；用“terminal 种类数”判断可完成性是错误维度。
- 建议你们明确 `finish_tokens` 的用途：
  - 如果是“返回允许的终结符集合”，应直接返回 `all_terminals`（再交给 constraint/mask 过滤）。
  - 如果是“生成一个补全序列”，应返回一个**具体补全方案**（长度受 `max_len`/约束影响），并在插件侧做 fail-fast。

### 6) `diff` 的 axis 参数必须强类型化
需要避免 `x` 既能作为“坐标张量 terminal”又能作为“axis-id”被 `diff` 消费导致的歧义。
- 建议：把 axis 设计为**axis-id token**（只用于 `diff` 参数，不参与数值计算），或者为 `diff` 定义专用语法节点（而非普通 unary/binary）。

### 7) 常数策略的“可辨识性”要提前写进规则
Phase 1–3 只做线性系数 `ξ` 是对的（速度与稳定性），但需要明确与表达式内部常数 token `C` 的关系，否则会出现“缩放不辨识”（term 内常数与外部 `ξ` 同时拟合）。
- 建议：Phase 1–3 禁止/弱化内部可学习常数；仅保留 `ξ` 线性回归。Phase 4 再引入 `θ`（非线性常数）并把其优化与缓存键版本化。

## 下一轮审查要点（数据层/导数层）
我接下来会细审 `PDEDataset` / `DerivativeProvider` / `ScaleHandler`，重点关注：
- n 维与 `axis_order` 的一致性、dataset fingerprint 设计（可复现且成本可控）
- 预计算导数 terminals 与 open-form `diff(expr)` 的一致性（尺度/噪声/缓存）
- `FiniteDiffProvider` 与 `AutogradProvider` 的能力边界（特别是 SGA/DISCOVER 的吞吐瓶颈）

## 已确认：dataset_fp / 归一化与 ScaleHandler / 导数命名

### 1) `dataset_fp`（混合方案：元信息 + 采样 hash）
权衡：
| 方案 | 计算成本 | 精确度 | 适用场景 |
|---|---:|---:|---|
| 全量 hash | O(数据大小) | 100% | 小数据集 |
| 元信息 + 采样 hash | O(1) | 99%+ | 大数据集 |

建议实现（你给出的草案）：
```python
def compute_dataset_fingerprint(dataset: PDEDataset) -> str:
    # 1) 元信息（必须纳入）
    meta = f"{dataset.name}:{dataset.topology.value}"
    meta += f":{dataset.lhs_field}:{dataset.lhs_axis}"

    # 2) 形状信息
    shapes = "_".join(f"{k}{v.values.shape}" for k, v in sorted(dataset.fields.items()))

    # 3) 数据 hash（大数据集采样）
    content_hash = hashlib.sha256()
    for field in sorted(dataset.fields.values(), key=lambda f: f.name):
        data = field.values.numpy()
        if data.nbytes > 10_000_000:  # > 10MB: 采样
            content_hash.update(data.ravel()[::1000].tobytes())
        else:
            content_hash.update(data.tobytes())

    return f"{meta}_{shapes}_{content_hash.hexdigest()[:8]}"
```

强制纳入项（已决定）：
| 字段 | 纳入 | 理由 |
|---|---|---|
| `lhs_field` | ✅ | 影响目标函数 |
| `lhs_axis` | ✅ | 决定 `u_{lhs_axis} = RHS` 的形式 |
| `topology` | ✅ | 影响导数计算方式 |
| `axis_order` | ❌（你建议） | 你认为 shapes 已隐含 |

Codex 备注（建议补强，避免碰撞/错用缓存）：
- shapes **不一定**能唯一决定 `axis_order`（例如多个轴长度相同的情况），建议仍显式纳入 `axis_order` 字符串（成本极低）。
- 对 PDE 来说，坐标 `axes[*].values` 也会改变导数与 residual，建议把 axis 值的采样 hash 一并纳入 fingerprint（或单独 `axes_fp`）。

### 2) 归一化与 `ScaleHandler`（Phase 1 NONE；Phase 2 引入）
核心：尺度不一致会导致混合 term 错误（`u_x` terminal 与 `diff(expr,x)` 必须同尺度）。

建议（你给出的草案）：
```python
class ScaleMode(Enum):
    NONE = "none"           # Phase 1 默认
    MINMAX = "minmax"       # [0, 1]
    ZSCORE = "zscore"       # (x - μ) / σ

@dataclass
class ScaleInfo:
    mode: ScaleMode
    shift: float      # μ or min（导数不使用，但用于反归一化）
    scale: float      # σ or (max - min)

class ScaleHandler:
    def __init__(self, dataset: PDEDataset, mode: ScaleMode = ScaleMode.NONE):
        self.mode = mode
        self.field_scales: Dict[str, ScaleInfo] = {}
        self.coord_scales: Dict[str, ScaleInfo] = {}

    def get_derivative_scale(self, field: str, axis: str, order: int) -> float:
        f_scale = self.field_scales[field].scale
        c_scale = self.coord_scales[axis].scale
        return f_scale / (c_scale ** order)
```

关键约定（已决定）：
- 预计算 `u_x` 时，乘以 `get_derivative_scale(field, axis, order)`；
- open-form `diff(expr, axis)` 的输出也自动应用相同尺度因子；
- Evaluator 在进入 `_evaluate_linear()` 前，保证 `Θ` 与 target 在同一尺度体系里。

### 3) 导数 terminals 命名（两层：canonical + display；MVP 单轴）
权衡：
| 格式 | 可读性 | 可解析性 | n 维扩展 |
|---|---:|---:|---:|
| `u_xx` | ✅ | ⚠️ | ⚠️（混合导数麻烦） |
| `d(u,x,2)` | ⚠️ | ✅ | ✅ |

建议（你给出的方案）：
- 内部 canonical（用于 hash/canonical/interchange）：`deriv:{field}:{axes}:{orders}`
  - 例：`deriv:u:x:2`、`deriv:u:xy:11`（混合导数）
- 外部 display（用于输出/LaTeX）：`u_xx` / `u_xy`

MVP 简化（已决定）：
- Phase 1–4：只支持单轴导数 `u_x/u_xx/u_xxx`；
- Phase 5：再扩展混合导数（并定义 canonical 排序策略）。

## 数据层 / 导数层（初步 review 笔记）

### 1) 强烈建议明确一个“评估用的标准 layout”
当前文档里 `fields/coords/derivatives` 的 shape 约定未锁定（nD grid vs flatten），但线性回归/批量评估最终都需要 `(N,)` 或 `(N,1)` 的点级向量。
- 建议：平台内部统一一个 point-layout（`N = Π n_axis`），所有 `Executor/Evaluator/Solver` 都以 flat 向量工作；
- 同时保留 grid-view（用于 FD 与可视化），通过 `reshape(axis_order)` 往返。

### 2) `FiniteDiffProvider.diff()` 不支持 open-form diff 会卡住 SGA/通用性
你们表达式语言已经允许 `diff(u*u_x, x)` 这种复合 diff；如果 Phase 1–3 仍大量使用 FD 导数（常见且更快），则：
- 要么：实现 `FiniteDiffProvider.diff(value_tensor, axis, order)` = 对“表达式数值结果”再做 FD（这在 grid 上是可行且可微的线性算子）；
- 要么：强约束“当 provider=FD 时禁用 diff token”，否则会产生运行时错误与大量无效候选。

### 3) `ScaleHandler` 目前存在“公式/实现不一致”的风险
（已用新决策替换）Phase 1 默认不归一化；Phase 2 引入显式 `ScaleHandler`，并用 `field_scale / coord_scale^order` 统一：
- 预计算导数 terminals（查表路径）
- open-form `diff(expr, axis)`（autograd 路径）
避免混用导致 `Θ` 列尺度不一致而回归出错。

### 4) `AutogradProvider` 需要补齐两件“可实现性细节”
- `PDEDataset.get_coords_with_grad()` 的规范（grid/scattered 都要返回同一语义：flat 点坐标 dict，且 `requires_grad=True`）。
- `surrogate` 的 I/O 契约：输入 coords（flat），输出 fields（flat）；多场情况下输出 shape/命名的约定要写死，否则插件会难以复用。

### 5) 预计算导数 terminals 的命名规范需要兼容“非单字符 axis 名”
已决定采用“两层命名”（canonical + display），可直接规避非单字符 axis 的歧义：
- 内部 canonical：`deriv:{field}:{axes}:{orders}`（稳定、可解析、可扩展）
- 外部 display：`u_xx`（人类友好；MVP 单轴足够）

### 6) FD 边界点/周期性会影响回归与指标，需要一个“valid mask”
`AxisInfo.is_periodic` 已出现，但 FD 在边界如何处理（裁剪/周期/一阶/填充）会直接改变样本集合与 NMSE/AIC。
- 建议：DerivativeProvider 输出除数值外，再输出可选 `valid_mask`（哪些点导数可信/可用），Evaluator 在构造 `Θ` 与 `y` 时统一应用。

## Evaluator / 缓存 / 性能（review 笔记）

### 1) “万级候选评估瓶颈”的主因不只是 DiskCache
`NOTES/arc_final.md` 把缓解写成 “DiskCache + canonical hash 去重”，但真实瓶颈通常来自：
- Python 级 AST 递归执行（解释器开销）；
- open-form `diff` 的 autograd 构图/反向（尤其是高阶导与 batch 大）；
- 线性回归（STRidge/Lasso）的反复拟合与数值病态（Theta 条件数）；
- 大 Tensor 的频繁分配/拷贝（Theta 拼接、residual 存储）。

建议把性能策略拆成 3 个层级，并按 Phase 渐进：
- Phase 1–3（MVP）：**默认不保留计算图**（不做 θ 优化），尽量用预计算导数 terminals + `LINEAR` 回归；
- Phase 4（DISCOVER/θ）：只在需要时开启“可微模式”（对 θ/Policy loss），否则仍走数值模式；
- Phase 5：再上 `ExecIR(DAG+CSE)` / vmap / compile。

### 2) `EvaluationResult` 建议拆成 Summary/Artifacts（避免大对象进入 cache/feedback）
你们当前 `EvaluationResult` 结构里包含 `residual: torch.Tensor`（潜在巨大），不适合：
- 写入 DiskCache（慢、占空间、跨进程兼容性差）
- 作为 `Feedback` 回传给插件（带宽/序列化灾难）

建议：保持你们现有 `EvaluationResult` 对外语义，但实现上拆为两层：
- `EvalSummary`（可缓存/可序列化）：`nmse/aic/r2/complexity/coeffs/selected_terms/invalid_reason/...`
- `EvalArtifacts`（按需保存）：`residual_vector/Theta/term_values/debug_traces/...`

### 3) 缓存层级：表达式级去重不够，`LINEAR` 需要 term/subtree 级缓存
`LINEAR` 模式本质是反复构造 `Θ = [term_1, term_2, ...]`；大量候选会共享很多 term（尤其是 GA/RL 迭代收敛后）。
- 表达式级缓存（whole RHS）只能命中“完全相同的表达式”，收益有限；
- 更高价值的是 **term-level**（甚至 subtree-level）缓存：`term_fp -> term_value_vector`。

建议的两层实现（实现难度低→高）：
1) `TermValueCache`（内存 LRU，run 内）：缓存 term 的数值列（可选存 float32 CPU 以省 GPU 显存）。
2) `ExpressionSummaryCache`（DiskCache，可跨 run）：只存 `EvalSummary`（不存 residual/Θ）。

### 4) `LINEAR` 评估链路需要显式的“mask/standardize/diagnostics”
为了让 STRidge/Lasso 稳定、可 debug，建议在 `_evaluate_linear()` 固化以下步骤：
- `valid_mask`：来自 DerivativeProvider（边界/缺失点），以及数值护栏（NaN/Inf）产生的 mask；
- `Theta` 标准化：对每列做标准化（记录 scale，最后还原系数）；否则 STRidge 阈值在不同量纲下不可比；
- `额外基函数`：已决定“可配置列表、默认空”；但建议把常数 `1`、`x/t` 等作为可选基函数（它们属于 ξ 的线性列，不等同于 Phase 4 的 θ）；
- `diagnostics`：记录条件数、被丢弃列比例、有效样本数等，方便定位“拟合差是搜索差还是回归差”。

### 5) open-form `diff` 的成本控制（关键风险）
你们的语义决策允许 `diff(composite, axis)`，这对 DISCOVER 必要，但对 Phase 1–3（SGA）可能是纯负担。

建议把能力/约束与 provider 强绑定，避免运行时大量无效候选：
- 当 `DerivativeProvider = FiniteDiffProvider`：
  - 至少支持你们已决定的“递推导数”用法：`diff(u_x, x) → u_xx`（优先 rewrite 成导数 terminal/查表）
  - 对 `diff(composite, axis)`：要么禁用（采样期 mask），要么实现 `FiniteDiffProvider.diff(value_tensor, axis, order)`（对表达式数值结果再做 FD，grid 上可行）
- 当 `DerivativeProvider = AutogradProvider`：
  - open-form `diff` 可用；在 Phase 1–3 仍建议提高其复杂度权重/降低采样概率，避免吞吐被 autograd 构图拖垮

（可选优化）如果你们愿意为性能投入：对一部分常见算子实现“符号求导 + 查表求值”（链式法则），把 `diff(sin(u),x)` rewrite 成 `mul(cos(u), u_x)`，可大幅减少 autograd 构图。

### 6) AIC/Reward 的数值定义需要统一（否则不同算法不可比）
当前文档的 AIC 写法是 `2k + 2ln(MSE)`（简化版）；不同实现很容易出现常数项/`n` 因子差异，导致跨实验/跨数据集不可比。
- 建议在 `metrics.py` 里把 AIC 定义成单一权威实现，并记录 `n_samples`、`k`、`RSS/MSE` 的口径；
- Reward（DISCOVER 风格）同理：明确输入使用 `NMSE` 还是 `MSE`，是否做 clamp/log，避免奖励分布极端导致 RL 不稳定。

## 已确认：Evaluator/缓存/指标的 5 个关键决策

| 问题 | 决策 |
|---|---|
| `diff` 约束范围 | 只禁止原始场 `u/v`；允许 `diff(u_x, x) → u_xx` |
| 计算图保留 | 默认关闭；可选开启（`requires_grad`/配置控制） |
| 额外基函数 | 可配置列表；默认空 |
| Term 缓存 | 内存 LRU；实验级作用域 |
| 评估指标 | 可插拔 `Metric` 系统；支持自定义 |

> 备注：AIC/Reward 的“默认口径”可以作为 Metric 插件的默认实现来固化；若不固化，也需要在实验记录里把 metric 配置完整存档，确保可复现与可比性。

## Experiment Layer（review）

### 1) 建议把 WandB/Hydra 变成“可选外设”，避免侵入核心循环
`ExperimentManager` 构造函数里直接依赖 `WandBLogger`（见 `NOTES/arc_final.md` 伪代码）会把依赖传播到所有用户/插件。
- 建议：定义 `Logger` 协议（`log_metrics/log_artifact/close`），WandB 只是一个实现；CLI/Notebook/Web 都可替换。

### 2) Checkpoint 建议采用“版本化 + 原子写入 + 目录结构”
你们的 `Checkpoint` dataclass 思路对，但落地时建议：
- `checkpoint/metadata.json`：dataset_fp/deriv_config/library_fp/metric_config/eval_config/git_commit/env/rng_state 等（可读可 diff）
- `checkpoint/plugin_state.pt`：`torch.save`（或 pickle，但建议统一 torch.save）
- `checkpoint/archive.jsonl`：可选，把“关键结果流”单独落盘，避免 result_archive 过大时 checkpoint 变慢
- 原子写：先写临时目录/文件，再 rename（防止中途崩溃损坏）

### 3) `Feedback` / `ResultArchive` 建议只依赖“轻量可序列化摘要”
你们已决定 term cache 仅内存 LRU、计算图默认不保留；因此建议从接口层就避免把大 Tensor 传播出去：
- `Feedback.result` 建议换为 `EvalSummary + metrics: dict[str, float] + invalid_reason`
- `ResultArchive` 默认只存 `EvalSummary` 与可复现的表达式信息（canonical + display + coeffs + metric values）
- residual/Theta/term_values 这类大对象归入“可选 artifacts”，按需保存（debug/可视化时再生成或按配置落盘）

### 4) 可复现性：rng_state 需要覆盖 torch CPU/CUDA + numpy + random
Checkpoint 的 `rng_state` 建议显式包含：
- `random.getstate()`、`np.random.get_state()`
- `torch.random.get_rng_state()`、`torch.cuda.get_rng_state_all()`（如用 CUDA）
- 以及 `torch.backends.cudnn.deterministic/benchmark` 与 `torch.use_deterministic_algorithms` 的开关状态（否则“同 seed 不同结果”很常见）

## Visualization Layer（review）

### 1) `NOTES/viz_design_notes.md` 的 Facade + Adapter + Registry 非常适合 kd2
优点是：算法插件与可视化扩展解耦，且 “capability discovery” 对交互式/Agent 非常关键。

建议把 viz 的核心契约固定成 4 个对象：
- `VizRequest(target, intent, options, output_mode, strict=False)`
- `VizResult(figure=None, data=None, warnings=[], ok=True)`
- `VizAdapter.capabilities(target) -> list[str]` + `render(request) -> VizResult`
- `VizRegistry.register(target_type, adapter)`（支持插件覆盖 core）

### 2) “零异常”建议改成“默认不抛 + 可选 strict”
批量实验/Web 服务里不抛异常很稳，但调试阶段需要 fail-fast。
- 建议：Facade 提供 `strict` 开关；strict 时将 warnings 升级为异常（或至少 `ok=False` 并记录 error）。

### 3) 双模式输出建议“data-first”
为了 Web/Agent/日志复现：
- `data` 应该是 JSON-serializable（Pydantic DTO/纯 dict），并作为主要产物；
- `figure` 只是本地渲染加成（matplotlib/plotly 任意），不进入 ResultArchive 的关键路径。

## 已确认 / TBD（实验/可视化）

| 问题 | 决策 |
|---|---|
| Checkpoint 对外格式 | ✅ 目录结构（`metadata + state + artifacts`），便于演进与兼容 |
| 可视化默认后端 | TBD：后续研究（本阶段坚持 data-first；后端可选 matplotlib/seaborn/plotly/wandb 等） |
| ResultArchive Pareto objectives | TBD：后续讨论（需写入 archive 元数据以保证可复现） |

## Metric System（review + 建议规范）

你们已决定“评估指标可插拔 Metric 系统，支持自定义”。为了让缓存/复现/可视化更稳，建议把评估拆成两层：
- **FitResult（可缓存、与指标无关）**：执行 +（可选）线性回归得到 `coeffs/selected_terms/residual_stats/complexity/valid_mask/invalid_reason/...`
- **MetricResult（可配置、可扩展）**：基于 FitResult 与配置计算出 `metrics: dict[str, float]`，并记录 metric 版本/配置到实验元数据

### 1) 最小接口（建议）
```python
class Metric(Protocol):
    name: str
    direction: Literal["min", "max"]  # 用于 Pareto/排序

    def compute(self, fit: FitResult, ctx: "MetricContext") -> float:
        ...

@dataclass(frozen=True)
class MetricContext:
    dataset_fp: str
    eval_mode: Literal["linear", "direct"]
    lhs_field: str
    lhs_axis: str
    config: dict[str, Any]  # metric 自己的超参
```

建议内置一组“默认 metric”（实现为插件）：
- `mse/nmse/r2`：基于 residual（支持 valid_mask）
- `complexity`：来自 IR（长度/深度/加权复杂度）
- `aic`：基于 `n_samples/k/rss` 的权威口径（写清楚）
- `reward_*`：DISCOVER 风格 reward（明确使用 nmse 或 mse、以及 clamp/log 规则）

### 2) 指标与缓存的关系
因为 metric 是可插拔的，**不建议**把 metric 结果作为 DiskCache 的唯一真相；否则切换 metric 配置会导致缓存不可比/误用。
- DiskCache：存 `FitSummary`（与 metric 无关）
- 实验日志/ResultArchive：存 `MetricResult` + `metric_config_fp`（保证可复现与可比性）

### 3) invalid / 数值护栏约定
建议在 FitResult 层就统一输出：
- `ok: bool`
- `invalid_reason: str | None`（来自约束/数值护栏）
- `valid_mask`（可选）
Metric 只处理“ok 的 FitResult”；对 invalid 的输入统一返回 `+inf/-inf/NaN` 由上层策略决定（推荐：直接跳过、不进入 archive）。

### 4) 与 Pareto 的衔接（TBD 友好）
Pareto objectives 未定，因此建议把“哪些 metric 是 objectives”做成纯配置，并写入 archive 元数据：
- `objectives = [{"name": "nmse", "direction": "min"}, {"name": "complexity", "direction": "min"}]`（示例）

## Viz Data Contracts（data-first）

目标：让 CLI/Notebook/Web/Agent 都能消费同一份 viz-data；图形只是可选渲染。

### 1) 基础 DTO（建议固定）
```python
@dataclass
class EquationDTO:
    expr: str                 # display string
    expr_canonical: str       # canonical string（GenIR 或等价）
    latex: str | None
    coeffs: list[float] | None
    metrics: dict[str, float]

@dataclass
class ParetoPointDTO:
    id: str                   # canonical hash
    objectives: dict[str, float]
    equation: EquationDTO

@dataclass
class ParetoPlotDTO:
    x: str                    # objective name
    y: str
    points: list[ParetoPointDTO]
    warnings: list[str] = field(default_factory=list)
```

建议补一个统一包裹类型（便于 Facade/Registry 统一返回）：
```python
@dataclass
class VizResultDTO:
    ok: bool
    data: dict[str, Any] | None      # 某个 DTO 的 dict 形式
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
```

### 2) 大数组/场数据的传输（建议预留 ArrayRef）
残差/场可视化会携带大数组，建议定义“内联 or 引用”的两种模式（便于 Web）：
- `InlineArray`：小数组直接存 list（debug/小数据）
- `ArrayRef`：只存 `{artifact_id, shape, dtype}`，真实数据走 artifact（`.npy/.pt/.parquet`）或 web blob

### 3) VizRequest intents（建议先固定最小集合）
结合 `NOTES/viz_design_notes.md`，建议 core 先支持：
- `pareto`：输入 `ResultArchive`，输出 `ParetoPlotDTO`
- `equation`：输入 `SearchResult/ArchivedResult`，输出 `EquationDTO`
- `residual`：输入 `FitResult/EvalArtifacts`，输出（ArrayRef + 元数据）

其余（tree_plot/field_plot）Phase 5 再补。
