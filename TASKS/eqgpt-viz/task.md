# Task: eqgpt-viz

Status: **Phase 1 completed** (Phase 2/3 backlog)
Priority: Medium
Created: 2026-03-23

## Goal

为 EqGPT 添加可视化支持，让 Gradio app 的 Visualization tab 对 EqGPT 可用。

## Non-goals

- 不重构 EqGPT 的训练/推理管道
- 不做论文级的对比 benchmark 图（Fig 3 热力图）
- 不修改 `docs/`

## Background

当前 EqGPT 训练完后 Visualization tab 完全空白（`app.py` 硬编码 `caps = []`）。
`kd/viz/` 没有 EqGPT adapter。

论文（arxiv 2505.05869）中的核心图：
- Fig 4/5/6：Reward 柱状图（Top-10 排名）+ Reward 进化曲线（每 epoch）
- Fig 6d：LHS vs RHS 散点（类似 parity）
- Fig 6e：观测 vs 预测曲线

EqGPT 内部有零散的 matplotlib 代码（`plot_reward`、`posterial_solution`），但全部不可用（硬编码、死代码、只 show 不 save）。

### 框架适配性分析

viz adapter 框架（`registry.py`）**不阻塞**——注册新 adapter 只需实现 Protocol（`capabilities` + `render()`）。

**可复用**：
- `_contracts.py` 的 `RewardEvolutionData` 数据类
- `ParityPlotData`（Phase 3 LHS vs RHS 时用）

**不可复用**：
- `equation_renderer` — EqGPT 方程不是 LaTeX 格式（见下方"方程格式"）
- SGA/Discover 的 `_calculate_pde_fields()` 等依赖 PDEDataset + 有限差分的逻辑
- 所有依赖 `u, ut, x, t` 网格数据的绘图函数
- `plot_reward()` — 画的是单个方程在各 case 上的奖励（12-23 bars），不是 top-10 排名

**最大瓶颈**：`fit_pretrained()` 当前只返回最终 top-10（equations + rewards），没有过程数据。

### 方程格式

EqGPT 使用自定义 token 词汇表（`dict_datas_0725.json`，57 个 token）。
方程输出是 token 拼接字符串，如 `ut+u*ux+uxx`，**不是 LaTeX**。

常见 token 示例：
- 变量/导数：`u`, `ut`, `ux`, `uxx`, `uxxx`, `uxxxx`, `uxt`, `u^2`, `u^3`, `ux^2`
- 复合项：`(uux)x`, `(u^4)xx`, `Laplace(u)`, `sin(u)`, `sqrt(u)`
- 运算符：`+`, `*`, `/`
- 坐标：`x`, `y`, `t`, `x^2`, `y^2`

Phase 1 需要一个简单的 token→LaTeX 映射函数（`_eqgpt_to_latex()`），覆盖：
- `ut` → `u_t`，`uxx` → `u_{xx}` 等简单替换
- `u^2` → `u^2`（已是 LaTeX）
- `(uux)x` → `(u u_x)_x` 等复合项
- 运算符保持不变

## Current state

- `kd/viz/_adapters/`：有 sga/dlga/discover/pysr adapter，无 eqgpt
- `kd/model/kd_eqgpt.py:fit_pretrained()` 返回：
  ```python
  {"equations": List[str], "rewards": List[float],
   "best_equation": str, "best_reward": float}
  ```
- `app.py:302`：硬编码 `caps = [] if "EqGPT" in model_name`
- `kd/model/eqgpt/continue_train_GPT_all.py:149`：`plot_reward()` 存在但硬编码+被禁用（注意：它画的是 per-case 奖励，不是 top-10 排名）
- `kd/model/eqgpt/dict_datas_0725.json`：词汇表（57 token），方程格式参考

## Proposed change (small steps)

### Phase 1：Reward 柱状图 + 方程渲染（零模型改动）

**数据来源**：`fit_pretrained()` 已有的返回值

1. 新建 `kd/viz/_adapters/eqgpt.py`
   - `EqGPTVizAdapter` 实现 `capabilities = {"equation", "reward_ranking"}`
   - `_eqgpt_to_latex(eq_str)`：token→LaTeX 转换（见"方程格式"节）
   - `_equation()`：转换 best_equation → LaTeX，渲染为 PNG
   - `_reward_ranking()`：Top-10 方程 reward 柱状图（从零设计，不参考 `plot_reward`）
     - x 轴：方程排名（1-10）
     - y 轴：reward 值（动态范围，不硬编码 ylim）
     - 每个 bar 标注对应方程文本
2. 在 `kd/viz/adapters.py` 的 `register_default_adapters()` 中注册（参考 PySR 的延迟注册模式）
3. `app.py:302`：删除 EqGPT 的 `caps = []` 硬编码

**前置条件**：`fit_pretrained()` 的返回值需要存到 model 属性上（当前返回 dict 但不存 self），adapter 才能从 `model` 对象读取。在 `kd_eqgpt.py` 中加：
```python
self.result_ = result  # store for viz adapter access
```

### Phase 2：Reward 进化曲线（小改 fit_pretrained）

**数据来源**：需要训练循环保存每 epoch 的 top-10 reward

1. `kd_eqgpt.py` 训练循环中，在 `_update_top10()` 之后累积：
   ```python
   reward_history: List[List[float]] = []
   for epoch in range(self.optimize_epochs):
       ...
       best_award, best_sentence = self._update_top10(...)
       reward_history.append(list(best_award[:10]))  # best_award 是降序 List[float]
       ...
   ```
   存入返回 dict：`"reward_history": reward_history`
2. adapter 新增 capability `"reward_evolution"`
   - 折线图：x=epoch, y=reward, 多条线表示 top-1/top-5/top-10
   - 复用 `RewardEvolutionData` 数据类

### Phase 3：LHS vs RHS + 观测 vs 预测（较大改动，可选）

**数据来源**：需要暴露 surrogate 中间计算

1. `calculate_reward` 返回扩展：除了标量 reward，还返回 LHS/RHS 数组
2. `posterial_solution` 重构：接受参数化的坐标范围，返回 (observed, predicted) 数组
3. adapter 新增：
   - `"parity"`：LHS vs RHS 散点（复用 `ParityPlotData`）
   - `"posterior_prediction"`：观测 vs 预测曲线

## Acceptance criteria

### Phase 1
- `_eqgpt_to_latex()` 正确转换所有 57 个词汇表 token
- Gradio app 训练 EqGPT 后，Visualization tab 显示 `equation` 和 `reward_ranking` 两个选项
- `reward_ranking` 生成正确的 Top-10 柱状图（动态 y 轴范围）
- `equation` 渲染最佳方程（EqGPT token → LaTeX → PNG）
- 现有测试不受影响

### Phase 2
- `fit_pretrained()` 返回 `reward_history` 字段
- Visualization tab 新增 `reward_evolution` 选项
- 折线图正确展示每 epoch 的 reward 趋势

### Phase 3（可选）
- Parity plot 展示 LHS vs RHS
- 观测 vs 预测曲线可用

## Validation commands

```bash
# 单元测试
pytest tests/test_app_helpers.py -q -p no:faulthandler
pytest tests/ -q -p no:faulthandler -k "not slow"

# 手动验证（需要 EqGPT 数据文件 + gradio）
python app.py
# → 选任意数据集 + KD_EqGPT → 训练 → Visualization tab → 选 reward_ranking → Generate
```

## Rollback plan

```bash
git checkout -- kd/viz/_adapters/eqgpt.py kd/viz/adapters.py
git checkout -- kd/model/kd_eqgpt.py app.py
```

## 参考

- 论文 PDF：`ref_lib/EqGPT/EqGPT_arxiv_2505.05869.pdf`（Fig 2/4/5/6）
- 现有可视化代码（不可用但有参考价值）：`kd/model/eqgpt/continue_train_GPT_all.py:149`
- viz 框架：`kd/viz/core.py`、`kd/viz/registry.py`
- adapter 注册入口：`kd/viz/adapters.py:register_default_adapters()`
- 现有 adapter 示例：`kd/viz/_adapters/sga.py`（handler dispatch 模式）、`kd/viz/_adapters/pysr.py`（最小化实现）
- EqGPT 词汇表：`kd/model/eqgpt/dict_datas_0725.json`（57 token）
