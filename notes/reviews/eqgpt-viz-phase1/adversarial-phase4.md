# Adversarial Edge Case Review: param_fields

## Summary
- 结论: REQUEST_CHANGES
- High: 3, Medium: 4, Low: 2

---

## High Priority

### 1. **[High] NaN in param_fields passes `np.allclose` check silently** @ `kd/model/discover/task/pde/pde_pinn.py:131`
- **场景**: 用户测量数据中包含 NaN（传感器故障、缺失值），传入 `param_fields={"nu": nu_with_nan}`。在 PINN 模式中，`_refresh_param_refs` 使用 `np.allclose(raw, raw.flat[0])` 检查是否为常数场。
- **后果**: `np.allclose(nan, nan)` 返回 `False`（因为 NaN != NaN），但 `float(raw.flat[0])` 也会产生 `float('nan')`，所以 **如果 raw 全是 NaN**，`np.allclose(np.nan, np.nan)` 返回 False -> 抛出 NotImplementedError，尚可接受。**但如果只有部分 NaN（如 `[0.1, nan, 0.1]`），allclose 也返回 False -> 抛出 NotImplementedError**，错误消息说"spatially varying"但实际问题是数据包含 NaN。用户会被误导。
- **更危险的路径**: Regular mode (PDETask) 的 `load_data` 完全不检查 param_data 的 NaN，直接绑定引用。NaN 会静默传播到 execute 中的 `token._param_data_ref`，导致 reward 计算中 NaN 扩散。
- **历史参考**: 匹配 common-mistakes.md #2 "未检查 NaN/Inf"
- **建议修复**:
  1. 在 `PDEDataset.__init__` 中验证 param_fields 值有限: `if not np.all(np.isfinite(arr)):`
  2. 在 `_refresh_param_refs` 中，先做 NaN 检查再做 allclose

### 2. **[High] `python_execute_torch` 中 param token 的 `_param_data_ref` 不参与 autograd 计算图** @ `kd/model/discover/execute.py:71`
- **场景**: PINN 模式 (MODE2) 中，`_refresh_param_refs` 为 torch 参数创建 `torch.full(...)` 张量。但在 `python_execute_torch` 中 (line 71)，`intermediate_result = token._param_data_ref` 直接返回这个张量。这个张量 **没有** `requires_grad=True`，不是计算图的一部分。
- **后果**: 如果下游 diff 操作（`torch_diff`）对包含 param token 的表达式求导，param 的梯度将为零或 autograd 会报错。例如表达式 `diff(mul(nu, u1), x1)` 中 `nu * u1` 的求导需要 nu 在计算图上，否则梯度断链。
- **但**: 仔细看 `python_execute_torch` 的 diff 分支 (L82-89)，diff_t token 不使用 `dim_flag`（不像 numpy 版本），而是直接对 `terminals` 进行 autograd。如果 `nu` 是一个不需要梯度的常数张量，`mul_t(nu, u1)` 的结果仍然需要梯度（因为 u1 需要梯度），所以 `diff_t` 对 `mul_t(nu, u1)` 求导时，`nu` 作为常数被正确处理。**这实际上是正确的行为**：`d(nu * u)/dx = nu * du/dx` 当 nu 为常数。
- **实际风险降低为 Medium**: 但如果用户意外使 param 具有 `requires_grad=True`，diff 会对 param 求导，产生非零梯度，导致错误的物理方程。`_refresh_param_refs` 应该显式 `requires_grad_(False)` 确保安全。
- **建议修复**: 在 `_refresh_param_refs` 中 torch 分支加 `.requires_grad_(False)` 或验证创建的张量不追踪梯度。

### 3. **[High] `switch_tokens` 在有 param token 时会 KeyError** @ `kd/model/discover/program.py:406`
- **场景**: PINN 模式中，程序表达式包含 param token (如 `nu`)。当 `switch_tokens(token_type='torch')` 被调用时（见 `pinn.py:296`），代码在 L399 取 `np_tokens_name = Program.library.np_names`，然后在 L406 尝试 `torch_tokens[t.name+'_t']`。
- **后果**: 修复后 `np_names` 已排除 param token（测试 `TestNpNamesExcludesParam` 验证了这一点），所以 param token 不在 `np_tokens_name` 中，会走 `else` 分支直接 `append(t)` —— 这是正确的。
- **但**: 如果未来有人回退了 np_names 的修复，param token 会出现在 np_names 中，然后 `torch_tokens["nu_t"]` 会 KeyError。这是一个脆弱点。
- **风险评估**: 测试已覆盖此路径，实际风险取决于测试是否在 CI 中运行。**如果 CI 没有运行 test_param_fields.py，风险为 High**。
- **建议**: 确保 test_param_fields.py 在 CI 中运行。可选：在 switch_tokens 中加一个 `.get()` 防御。

---

## Medium Priority

### 4. **[Medium] `_refresh_param_refs` 中 `self.u` 结构在不同 `generate` 路径下不一致** @ `kd/model/discover/task/pde/pde_pinn.py:126`
- **场景**: `_refresh_param_refs` 假设 `self.u[0]` 存在且是单个数组/张量。但查看不同的 `generate_meta_data` 分支:
  - `AD_generate_1D` (L280): `self.u = [model.net_u(xt)]` -- list of 1 tensor, OK
  - `AD_generate_mD` (L295): `self.u = [model.net_u(xt)]` -- list of 1 tensor, OK
  - `FD_generate` (L312): `self.u = [tensor2np(u).reshape(m,n).T]` -- list of 1 array, OK
  - `FD_generate_2d` (L373): `self.u = self.u_new` 其中 `self.u_new = [Wn, Un, Vn]` -- **list of 3 arrays!**
  - `weak_form_cal` (L258): `self.u = [u1, u2]` -- **list of 2 tensors!**
- **后果**: 在 `FD_generate_2d` 和 `weak_form_cal` 路径下，`self.u[0]` 只是第一个场变量 (W 或 u1)，不是完整的 u。`_refresh_param_refs` 用 `u_ref.shape` 创建 param tensor，shape 会匹配第一个场变量而非全部。
- **实际影响**: `FD_generate_2d` 主要用于多变量 (n_state_var > 1) 的流体问题，通常不会同时使用 param_fields。但没有防御代码阻止这种组合。
- **建议**: 在 `_refresh_param_refs` 中添加注释说明假设，或在 `n_state_var > 1` 时抛出 NotImplementedError for param_fields。

### 5. **[Medium] param_fields key 与 token 名称冲突无任何检查 (DSCV 侧)** @ `kd/model/discover/functions.py:253-257`
- **场景**: 用户传入 `param_fields={"x1": some_array}` 或 `param_fields={"add": some_array}`。DSCV 的 `create_tokens` 会创建一个 `Token(name="x1", param_var=0)` 与已有的 `Token(name="x1", input_var=0)` 同名。
- **后果**: `Library` 中会有两个名为 "x1" 的 token。`lib.names.index("x1")` 只返回第一个出现位置，`lib["x1"]` 也只返回第一个。这不会崩溃，但会导致：
  - Prior 约束可能应用到错误的 token
  - LaTeX 渲染可能选错 token
  - `actionize()` 基于名称查找，会映射到错误 token
- **SGA 侧**: `context.py:988` 中 `_build_vars` 直接将 param_fields 的 key 作为变量名添加到 VARS。如果 key 是 "x" 或 "u"，会与已有 VARS 条目冲突，导致 AIC 回归中出现共线性列。但 SGA 侧有 `_validate_registry_names` 检查 field/coord 冲突，**只是不检查 param_fields 名称冲突**。
- **建议**: 在 `PDEDataset` 或两个 adapter 中验证 param_fields key 不与已有 token 名称冲突。至少检查: `{"x1", "x2", "x3", "u1", "u2", "add", "sub", "mul", "div", "diff", "diff2", "const", ...}`。

### 6. **[Medium] `np.allclose(raw, raw.flat[0])` 对大数组的性能问题** @ `kd/model/discover/task/pde/pde_pinn.py:131`
- **场景**: `_refresh_param_refs` 在每次 `generate_meta_data` 被调用时执行（每个 epoch/iteration）。`raw` 是原始参数场数据，大小 = full grid。对于 256x201 = 51456 点的数据，allclose 开销可忽略。但对于 3D PINN (如 50x50x50x100 = 12.5M 点)，每次迭代都做全量比较。
- **后果**: 在大规模 3D 数据上可能造成不必要的性能损耗。不过 PINN 的 collocation 点通常远少于 full grid，且 `_param_fields_raw` 保持的是原始数据（不会膨胀）。
- **建议**: 可以在 `load_data` 时做一次常数检查并缓存结果，而不是每次 refresh 都重新检查。

### 7. **[Medium] SGA `_build_vars` 中 param_fields 的值直接来自 config，可能未经 permute** @ `kd/model/sga/sgapde/context.py:988-990`
- **场景**: 在 N-D 模式下，`u` 和 coord_grids 已经根据 axis_order 构建了正确的 broadcast shape。但 `_build_vars` 直接使用 `param_fields.items()` 的值，这些值来自用户通过 SGADataAdapter 传入的 `config.param_fields`。
- **后果**: SGADataAdapter (L47) 做了 `np.asarray(v)`，但没有做 permute/broadcast。如果用户的 param_fields 数组 shape 是 (nt, nx, ny) 但 SGA 内部期望 (nx, ny, nt)（或经过 broadast），shape 可能不匹配。
- **但**: SGA adapter 用的是 N-D registry payload 路径，不做 permute —— 它直接传递 axis_order 给 SolverConfig，然后 context 按 axis_order 构建 fields。如果 param_fields 的 shape 与 fields_data 一致（在 PDEDataset 中已验证），且不经额外 permute，则 param_fields 的 shape 也与内部 u 一致。**不过** context 中的 fields 是通过 `_init_variable_registry` -> `fields = {name: np.asarray(value)}` 直接赋值的，不做 transpose。所以只要 param_fields shape == usol.shape，就是正确的。
- **建议**: 加一个 shape 一致性断言在 `_build_vars` 中: `assert value.shape == self.u.shape`。

---

## Low Priority

### 8. **[Low] `param_names` 是 mutable list 参数，但多次注入安全** @ `kd/model/discover/functions.py:254`
- **场景**: `create_tokens(param_names=["nu"])` 中 `param_names` 是可变 list。如果调用者保持引用并之后修改，会影响 token 的 name 吗？
- **分析**: `create_tokens` 在 L254 只做 `name = param_names[i] if param_names ...`，取出的是字符串（不可变），赋给新 Token 对象。即使后续修改 list，已创建的 Token 不受影响。
- **结论**: 不是实际 bug，只是代码风格建议。

### 9. **[Low] DSCV LaTeX 渲染只支持 p1/p2/p3 硬编码** @ `kd/viz/discover_eq2latex.py:13,20-23`
- **场景**: `DEEPRL_SYMBOLS_FOR_SYMPY` 和 `_SYMBOL_DISPLAY` 只包含 `p1`, `p2`, `p3`。如果用户使用自定义名称 (如 "nu") 或超过 3 个参数 (p4+)，LaTeX 渲染会 fallback 到原始名称。
- **后果**: 不会崩溃，只是渲染不够美观 —— "nu" 在 DSCV LaTeX 中不会被渲染为 `\nu`（但在 SGA 侧的 `_render_leaf` 已处理了 Greek 名称映射）。
- **建议**: 将 DSCV 的 symbol display 也做成动态注入或扩展 Greek 映射。非紧急。

---

## 隐藏假设

| 假设 | 合理？ |
|------|--------|
| param_fields 的值是有限数 (no NaN/Inf) | **否** - 用户数据可能包含缺失值，但代码仅在 adapter 层检查 u 的 NaN，不检查 param_fields |
| param_fields 在 PINN 模式下是空间常数 | **是** - PINN 的 collocation 是随机采样的，非网格对齐，空间变化的参数无法直接映射。显式检查并抛出 NotImplementedError |
| param token 名称不会与已有 token 名称冲突 | **否** - 无任何验证。用户传 `param_names=["x1"]` 会导致重名 token |
| `self.u` 在 `_refresh_param_refs` 调用时总是 list with >= 1 element | **大部分是** - 但 `weak_form_cal` 和 `FD_generate_2d` 使 u 有多个元素 |
| `_param_fields_raw` 在 `_refresh_param_refs` 被调用前已设置 | **是** - `load_data` 在 `generate_meta_data` 之前被调用 |
| 不超过 3 个参数变量 | **实际上是** - LaTeX 只支持 p1-p3，搜索空间爆炸也使 >3 不实际 |

## 过度工程

1. **`_refresh_param_refs` 中的 torch import try/except** (pde_pinn.py:137-146): 每次调用都 try import torch。考虑到 PINN 模式必然依赖 torch，这个 try/except 永远不会走 except 分支。可以直接 import torch，或在类级别做一次性检查。但这是非常小的问题，不影响正确性。

## 建议测试

```python
# 1. NaN in param_fields
def test_param_fields_nan_raises():
    """param_fields containing NaN should be rejected at PDEDataset level."""
    nu_with_nan = np.full((32, 20), 0.1)
    nu_with_nan[10, 5] = np.nan
    with pytest.raises(ValueError, match="NaN|Inf|finite"):
        PDEDataset(equation_name="test", x=x, t=t, usol=u,
                   param_fields={"nu": nu_with_nan})

# 2. Name collision
def test_param_name_collision_raises():
    """param_fields key matching existing token name should raise."""
    # This should fail at some validation point
    tokens = create_tokens(n_input_var=1, function_set=["add"],
                           protected=False, n_state_var=1, task_type="pde",
                           n_param_var=1, param_names=["x1"])
    lib = Library(tokens)
    # lib.names should have two "x1" entries -> ambiguity

# 3. allclose with NaN raw data in _refresh_param_refs
def test_refresh_param_refs_nan_raw():
    """_refresh_param_refs should handle NaN in raw param data gracefully."""
    task = _make_task_with_param()
    task._param_fields_raw = [np.array([np.nan, np.nan, np.nan])]
    task.u = [np.random.randn(50)]
    with pytest.raises((ValueError, NotImplementedError)):
        task._refresh_param_refs()

# 4. Multi-state var + param_fields in PINN
def test_pinn_multi_state_param_not_supported():
    """FD_generate_2d with param_fields should warn or raise."""
    # Currently no guard exists

# 5. Param token in switch_tokens path
def test_switch_tokens_with_param_no_error():
    """switch_tokens should not crash when library contains param tokens."""
    # Requires a mock library with torch_tokens and param tokens
```
