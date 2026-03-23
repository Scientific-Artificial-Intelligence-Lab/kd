# General Review: param_fields Feature (Phase 1-4)

**Reviewer**: reviewer (general)
**Date**: 2026-02-13
**Scope**: Full param_fields feature across PDEDataset, SGA, DSCV Regular/Sparse/PINN, LaTeX rendering

---

## Summary
- 结论：**APPROVE** (with minor notes)
- High: 0, Medium: 2, Low: 3

整体设计清晰合理，四个 Phase 层次分明：
1. **Phase 1** (PDEDataset + SGA): param_fields 作为 Dict[str, ndarray] 存储，形状校验到位；SGA 端通过 `_build_vars` 加入 VARS 但排除在 `den` 之外（不求导），语义正确。
2. **Phase 2** (DSCV Token + Library + execute): Token 新增 `param_var` 属性，Library 维护 `param_tokens` 索引，`execute.py` 两个引擎（numpy/torch）都正确处理 param_var 分支且不设置 `dim_flag`（不用于求导），设计精确。
3. **Phase 3** (LaTeX): SGA 侧在 `_render_leaf` 增加 Greek 字母映射；DSCV 侧在 `discover_eq2latex.py` 添加 p1/p2/p3 的 sympy 符号和显示映射。
4. **Phase 4** (Sparse/PINN): DSCVSparseAdapter 在 legacy 和 N-D 路径都正确提取 param_fields（N-D 还做了 transpose 对齐）；PDEPINNTask 接受 n_param_var 并在 `_refresh_param_refs` 中处理 numpy/torch 适配和常量校验。Library.np_names 排除 param tokens，DiffConstraint_right 用 set 差集逻辑正确覆盖 param tokens。

---

## Medium Priority

### 1. **PDEPINNTask._refresh_param_refs 对 spatially-varying 的检测使用 np.allclose** @ `kd/model/discover/task/pde/pde_pinn.py:131`
- `np.allclose(raw, raw.flat[0])` 使用默认 atol=1e-8, rtol=1e-5。对于非常小的 param 值（如 1e-9 量级），这个容差可能将 slightly-varying 的参数误判为 constant，导致信息丢失而非报错。
- 建议：考虑使用更严格的容差（如 `atol=0, rtol=1e-12`），或者改用 `np.ptp(raw) < eps` 的写法，使"常量"的判定更明确。
- 当前行为不会导致崩溃（只是会把微小变化的参数当常量处理），所以不是 High。

### 2. **Token.__init__ 参数顺序不一致** @ `kd/model/discover/library.py:38`
- `Token.__init__(self, function, name, arity, complexity, input_var=None, state_var=None, param_var=None)` 前两个位置参数是 `function, name`，但 `create_tokens` 中创建 param/input/state tokens 时使用 `Token(name=..., function=None, ...)`，而测试中有 `Token(np.multiply, "mul", arity=2, complexity=1)` 这种位置参数调用。
- 这种不一致不会导致 bug（keyword args 正确），但容易在后续扩展时产生混淆。考虑在 Token docstring 中明确说明 Terminal tokens 的标准创建方式。

---

## Low Priority

### 3. **LaTeX 符号覆盖有限** @ `kd/viz/discover_eq2latex.py:13`
- `DEEPRL_SYMBOLS_FOR_SYMPY` 和 `_SYMBOL_DISPLAY` 只预定义了 p1/p2/p3。如果用户自定义 param_names（如 "nu"）在 DSCV 模式下，sympy 解析会退化为纯文本符号。
- 不影响正确性（sympy 会创建 generic Symbol），但 LaTeX 输出可能不够美观。
- SGA 侧的 `_render_leaf` 已有完整 Greek 映射，DSCV 侧缺少等价机制。

### 4. **execute.py 中 param_var 分支的 print 和 pdb** @ `kd/model/discover/execute.py:42-44`
- `python_execute` 的 except 分支包含 `print(e)` 和 `import pdb;pdb.set_trace()`。虽然这是历史代码不在此次修改范围内，但 param_var 作为新的分支入口，如果上游传入错误的 param token（比如 `_param_data_ref` 为 None），会直接掉进这个 catch-all，调试困难。
- 建议：在 param_var 分支加一个 `assert token._param_data_ref is not None` 的早期检查。

### 5. **测试中 RED 注释未清理** @ `tests/test_param_fields.py` (multiple locations)
- 多处注释如 "RED: will fail until ..." 来自 TDD 阶段，现在 Phase 4 已完成，这些注释已过时。
- 不影响功能，但建议清理以避免混淆。

---

## 设计评价

### 架构层面
- **param_fields 作为 Dict[str, ndarray]** 的设计选择很好：与 fields_data 保持一致的风格，shape 校验复用了 usol 的 shape，简洁直观。
- **在 Token 层面用 param_var 区分**比在外部维护映射表更优雅，且与 input_var/state_var 保持对称。
- **_param_data_ref 模式**（在 Token 上挂载数据引用）是对现有 execute 引擎最小侵入的做法，避免了修改 execute 函数签名。

### 跨层一致性
- PDEDataset -> Adapter -> Task -> Library -> execute 的数据流路径完整，每层都有对应的测试覆盖。
- SGA 和 DSCV 两条独立管线都实现了 param_fields 支持，且语义一致（出现在方程中但不求导）。

### 与历史教训的对照
- **common-mistakes #5 (测试从 buggy 实现逆向推断)**：本次测试是先写的（TDD），且有明确的语义预期（param 在 VARS 中但不在 den 中，param 不在 np_names 中，param 是 DiffConstraint_right 的 target），不存在这个问题。
- **common-mistakes #7 (try/except 吞错误)**：PDEPINNTask._refresh_param_refs 没有使用宽泛 except，而是显式 raise NotImplementedError。execute.py 的历史 catch-all 仍在但不是本次引入的。

### 测试覆盖度
测试文件包含 7 个测试类、23 个测试方法，覆盖了：
- PDEDataset 存储/默认值/shape 校验/多参数/N-D 模式
- SGA adapter/config/context 全链路
- DSCV Token/Library/create_tokens
- execute.py param_var 分支（单独 + 表达式中）
- DSCV Regular/Sparse adapter（legacy + N-D + 无参数）
- LaTeX 渲染（SGA Greek + DSCV symbol display + sympy symbols）
- PINN Task 创建/load_data/refresh_param_refs（numpy/torch/varying）
- Library.np_names 排除/DiffConstraint_right 覆盖

覆盖度充分，边界情况处理良好。
