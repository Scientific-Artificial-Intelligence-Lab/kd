# Examples 全量运行分析结果

> 日期：2026-02-09
> 范围：examples/ 下全部 11 个示例脚本（不含 _bootstrap.py）

---

## 一、运行结果总览

| 脚本 | 模型 | 数据集 | Exit | 发现的方程 | Reward | Viz 生成 | 问题 |
|------|------|--------|------|-----------|--------|---------|------|
| kd_dscv_example.py | KD_DSCV | burgers | OK | (legacy, 11 epochs) | — | legacy dscv_viz | plt.show 无文件 |
| kd_dscv_viz_api_example.py | KD_DSCV | burgers | OK | (unified, 11 epochs) | — | 7/7 unified | artifacts/dscv_viz |
| kd_dscvspr_example.py | KD_DSCV_SPR | burgers | OK | `-0.2041*d/dx(u^2+u)` | 0.7373 | 1/7 legacy | Legacy viz 无 savefig |
| kd_dscvspr_viz_api_example.py | KD_DSCV_SPR | burgers | OK | `-0.3446*u_x` | 0.7609 | 6/6 unified | 完美 |
| kd_dscvspr_nd_example.py | KD_DSCV_SPR | 合成 3D | OK | 失败(预期) | N/A | 0/0 正确跳过 | PINN 注册缺失 |
| kd_dscv_nd_example.py | KD_DSCV | 合成 3D | OK | (11 epochs) | — | 7/9 (2 注释) | #3 #4 N-D 崩溃 |
| kd_sga_example.py | KD_SGA | burgers | OK | — | — | 5+legacy | 正常 |
| kd_sga_custom_example.py | KD_SGA | 合成 sin*cos | OK | — | — | 5 unified | equation_latex try/except |
| kd_sga_nd_example.py | KD_SGA | 合成 3D | OK | — | — | 3/5 (+1空) | #1 注释, #2 空输出 |
| kd_dlga_example.py | KD_DLGA | kdv | OK | — | — | 10/11 unified | 正常 |
| kd_pysr_example.py | KD_PySR | 合成回归 | OK | — | — | 3/3 unified | 需 pysr 包 |

**所有 11 个脚本均正常退出（exit code 0），无未预期的崩溃。**

---

## 二、各脚本详细分析

### 2.1 kd_dscv_example.py — DSCV 基础示例

- **数据集**: `load_pde('burgers')` — Burgers 方程 `u_t + u*u_x = nu*u_xx`
- **模型**: `KD_DSCV(binary_operators=["add", "mul", "diff"], unary_operators=['n2'], n_samples_per_batch=500)`
- **训练**: 11 epochs
- **Viz (Legacy dscv_viz, 7 调用)**:
  1. `render_latex_to_image(discover_program_to_latex(...))` — 方程图片
  2. `plot_expression_tree(model)` — 表达式树
  3. `plot_density(model, epoches=[2, 5, 10])` — 奖励密度
  4. `plot_evolution(model)` — 进化曲线
  5. `plot_pde_residual_analysis(model, program)` — PDE 残差
  6. `plot_field_comparison(model, program)` — 场对比
  7. `plot_actual_vs_predicted(model, program)` — 散点对比
- **输出目录**: 默认（legacy dscv_viz 内部处理）
- **try/except**: 无

### 2.2 kd_dscv_viz_api_example.py — DSCV 统一 Viz API 示例

- **数据集**: `load_pde('burgers')`
- **模型**: 同上
- **训练**: 11 epochs, verbose=False
- **Viz (统一 facade, 7 调用)**:
  1. `render(VizRequest('search_evolution', ...))` → `reward_evolution.png`
  2. `render(VizRequest('density', ..., epoches=[0,1,2]))` → `reward_density.png`
  3. `render(VizRequest('tree', ...))` → `expression_tree.png`
  4. `render(VizRequest('equation', ...))` → `equation.png`
  5. `render(VizRequest('residual', ...))` → `residual_analysis.png`
  6. `render(VizRequest('field_comparison', ...))` → `field_comparison.png`
  7. `render(VizRequest('parity', ...))` → `parity_plot.png`
- **输出目录**: `artifacts/dscv_viz/dscv/`
- **全部 7 个 intent 成功保存 PNG**

### 2.3 kd_dscvspr_example.py — DSCV_SPR 基础示例

- **数据集**: `load_pde('burgers')`
- **模型**: `KD_DSCV_SPR(n_samples_per_batch=300, binary_operators=["add_t","mul_t","div_t","diff_t","diff2_t"], unary_operators=['n2_t'])`
- **训练**: 200 epochs, sample_ratio=0.1, colloc_num=20000
- **发现方程**: `-0.2041 * diff_t(add_t(n2_t(u1),u1),x1)` = `u_t = -0.2041 * d/dx(u^2 + u)`
  - Reward: 0.7373
  - 结构上部分发现了 `u*u_x` 项，但系数不够精确且缺少扩散项
  - 稳定性测试 3 候选中 #1 以 99/100 票胜出
- **Viz (Legacy dscv_viz, 7 调用)**: 非 GUI 环境下 legacy viz 使用 plt.show()，仅生成 1 个文件 `log/discover_Burgers2_0data.png`
- **try/except**: 无

### 2.4 kd_dscvspr_viz_api_example.py — DSCV_SPR 统一 Viz API 示例

- **数据集**: `load_pde('burgers')`
- **模型**: `KD_DSCV_SPR(n_samples_per_batch=100, ..., colloc_num=512)`
- **训练**: 100 epochs
- **发现方程**: `-0.3446 * diff_t(u1,x1)` = `u_t = -0.3446 * u_x`
  - Reward: 0.7609
  - 仅发现对流项近似（100 epochs 不够发现完整结构）
- **Adapter Capabilities**: density, equation, field_comparison, parity, residual, search_evolution, spr_field_comparison, spr_residual, tree
- **Viz (统一 facade, 6/6 成功)** → `artifacts/dscvspr_viz/dscv/`:
  | 文件 | 大小 | Intent |
  |------|------|--------|
  | reward_evolution.png | 451KB | search_evolution |
  | reward_density.png | 152KB | density |
  | expression_tree.png | 7.8KB | tree |
  | equation.png | 16KB | equation |
  | spr_residual_analysis.png | 302KB | spr_residual |
  | spr_field_comparison.png | 131KB | spr_field_comparison |

### 2.5 kd_dscvspr_nd_example.py — DSCV_SPR N-D 示例

- **数据集**: 合成 3D `u(x,y,t) = sin(x)*cos(y)*exp(-t)`, shape (20,20,30)
- **模型**: `KD_DSCV_SPR`, 同标准 SPR 算子
- **训练**: 预期失败 — `AssertionError: Dataset 2d_decay is not existed`
  - 根因: PINN_model 要求数据集名称在内置注册表中
- **Viz**: 全部被 `if step_output is not None:` 保护跳过，0 个 PNG 输出
- **try/except**: 有，包裹 import_dataset + train

### 2.6 kd_dscv_nd_example.py — DSCV N-D 示例

- **数据集**: 合成 3D `u(x,y,t) = sin(x)*cos(y)*exp(-t)`, shape (20,20,30)
- **Ground truth**: `u_t = -u`
- **模型**: `KD_DSCV(binary_operators=["add", "mul", "Diff"], unary_operators=['n2'])`
  - 注意大写 `"Diff"`
- **训练**: 11 epochs
- **Viz (混合 legacy + unified)**:
  - OK: render_latex_to_image, plot_expression_tree, plot_density, plot_evolution, plot_actual_vs_predicted, render_equation(unified), plot_parity(unified) — 共 7 个
  - 已注释掉: `plot_pde_residual_analysis` (TypeError: NoneType), `dscv_plot_field_comparison` (shape 不匹配)
- **输出目录**: 默认 + `artifacts/dscv_nd_viz/dscv/`

### 2.7 kd_sga_example.py — SGA 标准示例

- **数据集**: `load_pde("burgers")`
- **模型**: `KD_SGA(sga_run=100, num=20, depth=4, width=5, seed=0)`
- **训练**: `model.fit_dataset(dataset)`
- **Viz (统一 facade + legacy)**:
  1. `render_equation(model)` — OK
  2. `plot_field_comparison(model, ...)` — OK
  3. `plot_time_slices(model, ..., slice_times=[0.0, 0.5, 1.0])` — OK
  4. `plot_parity(model, ...)` — OK
  5. `plot_residuals(model, ...)` — OK
  6. `model.plot_results()` — Legacy, 写入 `artifacts/legacy_sga/`
- **输出目录**: `artifacts/sga_viz/sga/` + `artifacts/legacy_sga/burgers/`

### 2.8 kd_sga_custom_example.py — SGA 自定义数据集示例

- **数据集**: 合成 `u(x,t) = sin(pi*x)*cos(pi*t)`, 32x33 网格, PDEDataset(pde_data=None)
- **模型**: `KD_SGA(sga_run=100, num=20, depth=4, width=5, seed=0, use_autograd=False, use_metadata=False)`
- **Viz (统一 facade, 5 调用)**:
  1. `render_equation(model)` — OK
  2. `render(VizRequest('field_comparison', ...))` — OK
  3. `render(VizRequest('time_slices', ..., slice_times=[0.0, 0.5, 1.0]))` — OK
  4. `render(VizRequest('parity', ...))` — OK
  5. `render(VizRequest('residual', ...))` — OK
- **输出目录**: `artifacts/sga_custom/sga/`
- **try/except**: `model.equation_latex()` 包裹处理预期失败

### 2.9 kd_sga_nd_example.py — SGA N-D 示例

- **数据集**: 合成 3D `u(x,y,t) = sin(x)*cos(y)*exp(-t)`, shape (20,20,30)
- **Ground truth**: `u_t = -u`
- **模型**: `KD_SGA(sga_run=50, num=15, depth=3, width=4, seed=42)`
- **Viz (统一 facade)**:
  - OK: render_equation, plot_field_comparison, plot_parity — 3 个正常 PNG
  - 已注释掉: `plot_time_slices` — IndexError (axis 1 size 20 vs index 29)
  - 调用但空输出: `plot_residuals` — warning: 'ProblemContext' 缺少 right_side_full_origin
- **输出目录**: `artifacts/sga_nd_viz/sga/`

### 2.10 kd_dlga_example.py — DLGA 完整示例

- **数据集**: `load_pde("kdv")` — KdV 方程 `u_t + 6*u*u_x + u_xxx = 0`
- **模型**: `KD_DLGA(operators=["u","u_x","u_xx","u_xxx"], epi=0.1, input_dim=2, max_iter=9000)`
- **训练**: `fit_dataset(kdv_data, sample=1000, sample_method="random", Xy_from="sample")`
- **Viz (统一 facade helpers, 11 调用)**:
  1. render_equation → equation.png
  2. plot_training_curve → training_loss.png
  3. plot_validation_curve → validation_loss.png
  4. plot_search_evolution → evolution_analysis.png
  5. plot_optimization → optimization_analysis.png
  6. plot_field_comparison → field_comparison.png
  7. plot_residuals → residual_analysis.png
  8. plot_time_slices → time_slices_comparison.png
  9. plot_derivative_relationships → derivative_relationships.png
  10. plot_parity → pde_parity_plot.png
- **输出目录**: `artifacts/dlga_example/dlga/` — 10 个 PNG 全部生成

### 2.11 kd_pysr_example.py — PySR 最小示例

- **数据集**: 合成回归 `y = x0^2 + 0.5*x1 + noise`, 256 样本
- **模型**: `KD_PySR(niterations=40, binary_operators=["+","-","*","/"], unary_operators=["square"])`
- **训练**: `model.fit(X, y)`（直接 fit，非 PDE）
- **Viz (统一 facade, 3 调用)**:
  1. render_equation → equation.png
  2. render(VizRequest('parity', ...)) → parity.png
  3. render(VizRequest('residual', ...)) → residual.png
- **输出目录**: `artifacts/pysr_minimal/pysr/` — 3 个 PNG 全部生成
- **依赖**: 需要 `pysr` 包

---

## 三、已知 N-D 问题汇总

来源: `notes/viz-nd-issues.md` (archived)

| # | Demo | viz 调用 | 严重度 | 当前处理 |
|---|------|----------|--------|---------|
| 1 | SGA N-D | `plot_time_slices` | High | 源码中注释掉 |
| 2 | SGA N-D | `plot_residuals` | Medium | 调用但空输出/warning |
| 3 | DSCV N-D | `plot_pde_residual_analysis` | High | 源码中注释掉 |
| 4 | DSCV N-D | `dscv_plot_field_comparison` | High | 源码中注释掉 |
| 5 | SPR N-D | 全部 | High | try/except 捕获，跳过 |

正常工作的 N-D viz:
- SGA: render_equation, field_comparison, parity (3/5)
- DSCV: render_latex, expression_tree, density, evolution, actual_vs_predicted, render_equation, parity (7/9)

---

## 四、生成的 Artifacts 清单

```
artifacts/
├── dscv_viz/dscv/           # kd_dscv_viz_api_example.py
│   ├── equation.png
│   ├── expression_tree.png
│   ├── field_comparison.png
│   ├── parity_plot.png
│   ├── residual_analysis.png
│   ├── reward_density.png
│   └── reward_evolution.png
├── dscvspr_viz/dscv/        # kd_dscvspr_viz_api_example.py
│   ├── equation.png
│   ├── expression_tree.png
│   ├── reward_density.png
│   ├── reward_evolution.png
│   ├── spr_field_comparison.png
│   └── spr_residual_analysis.png
├── dscv_nd_viz/dscv/        # kd_dscv_nd_example.py
│   ├── equation.png
│   └── parity_plot.png
├── sga_viz/sga/             # kd_sga_example.py
│   ├── equation.png
│   ├── equation_structure.png
│   ├── field_comparison.png
│   ├── parity_plot.png
│   ├── residual_analysis.png
│   └── time_slices_comparison.png
├── sga_custom/sga/          # kd_sga_custom_example.py
│   ├── equation.png
│   ├── equation_structure.png
│   ├── field_comparison.png
│   ├── parity_plot.png
│   └── time_slices_comparison.png
├── sga_nd_viz/sga/          # kd_sga_nd_example.py
│   ├── equation.png
│   ├── equation_structure.png
│   ├── field_comparison.png
│   └── parity_plot.png
├── dlga_example/dlga/       # kd_dlga_example.py
│   ├── derivative_relationships.png
│   ├── equation.png
│   ├── evolution_analysis.png
│   ├── field_comparison.png
│   ├── optimization_analysis.png
│   ├── pde_parity_plot.png
│   ├── residual_analysis.png
│   ├── time_slices_comparison.png
│   ├── training_loss.png
│   └── validation_loss.png
├── pysr_minimal/pysr/       # kd_pysr_example.py
│   ├── equation.png
│   ├── parity.png
│   └── residual.png
└── legacy_sga/burgers/      # kd_sga_example.py (legacy)
    ├── metadata_field.png
    ├── metadata_residual_heatmap.png
    ├── original_field.png
    ├── original_residual_heatmap.png
    ├── residual_slice.png
    ├── rhs_slice.png
    ├── u_slice.png
    ├── ut_slice.png
    ├── ux_slice.png
    └── uxx_slice.png
```

---

## 五、关键发现

1. **所有 11 个脚本均正常退出**，无未预期崩溃
2. **统一 viz facade 可靠性远高于 legacy dscv_viz** — unified facade 在非 GUI 环境下均正确 savefig，legacy 使用 plt.show() 不保存文件
3. **N-D 支持现状**: 核心训练正常，5 个 viz 调用存在已知问题（3 个注释掉避免崩溃，1 个空输出，1 个训练阶段阻塞）
4. **SPR N-D 完全阻塞**: PINN 数据集注册表不支持自定义名称，导致训练和所有下游 viz 均不可用
5. **Legacy SPR viz 在 headless 环境下无输出**: `kd_dscvspr_example.py` 的 7 个 legacy viz 调用仅产生 1 个 data.png，其余使用 plt.show() 丢失
