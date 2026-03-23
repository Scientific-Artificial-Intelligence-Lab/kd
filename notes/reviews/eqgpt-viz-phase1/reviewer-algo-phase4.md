# Algorithm Correctness Review: Phase 4 param_fields Sparse Mode

## Summary
- 结论：**APPROVE**（附带 2 个 Medium 建议）
- 算法风险等级: **Low**
- High: 0, Medium: 2, Low: 3

---

## High Priority

无。

---

## Medium Priority

### 1. `_refresh_param_refs` 中的 `np.allclose` 常量检测可能误判

**位置**: `kd/model/discover/task/pde/pde_pinn.py:131`

**风险类型**: Precision / 隐含假设

**问题描述**:

```python
if not np.allclose(raw, raw.flat[0]):
    raise NotImplementedError(
        f"Spatially-varying param '{token.name}' not supported "
        "in PINN mode. Use constant parameters or Regular mode."
    )
```

`np.allclose` 使用默认 `rtol=1e-5, atol=1e-8`。如果 param_field 实际上是用户用 `np.full_like(u, value)` 构建的（文档推荐做法），那么值完全相同，`allclose` 一定通过。

但如果用户的 param_field 来自数据文件（例如实验测量的 viscosity），即使"几乎常量"（如 `0.1 +/- 1e-6` 的测量噪声），`allclose` 也会通过，此时 `const_val = float(raw.flat[0])` 会丢掉微小的空间变化。

**影响**: 用户可能不知道微小的空间变化被静默忽略了。当前实现选择 raise NotImplementedError 对于 truly varying 参数，这是正确的。但边界情况——"接近常量但有微量噪声"——的行为需要明确文档。

**建议**:
- 如果认为这属于合理行为（PINN 不支持空间变化参数），当前实现已足够
- 考虑对"通过 allclose 但非完全相同"的情况添加 `logging.info` 级别提示

### 2. Sparse Legacy `param_data` 传递 raw 数组，但 PINN `_refresh_param_refs` 期望与 u 形状对齐

**位置**: `kd/model/discover/adapter.py:411-421` (legacy) 和 `pde_pinn.py:114-120` (load_data)

**风险类型**: Shape consistency / 数据流

**问题描述**:

在 `DSCVSparseAdapter._prepare_legacy` 中，param_data 是 raw 数组（与 `usol` 同形 `(nx, nt)`），未经采样。这与 `X_u_train`（采样后）形状不一致。

但这不构成 bug，因为:
1. `PDEPINNTask.load_data()` 仅将 raw 数组绑定到 token 的 `_param_data_ref`
2. `_refresh_param_refs()` 在 `generate_meta_data()` 末尾调用，此时 `self.u` 已经被 PINN 重生成（形状是 collocation points），`_refresh_param_refs` 使用 `torch.full(u_ref.shape, const_val)` 创建新的匹配形状 tensor

所以数据流实际上是：raw → load_data（临时绑定）→ generate_meta_data → _refresh_param_refs（用常量值重建匹配 u 的 tensor）。

**关键验证**: `_refresh_param_refs` 正确检查了参数是否为常量，然后用 `torch.full/np.full_like` 重建——这保证了 shape 对齐。

**建议**: 在 `load_data` 的注释中明确说明 raw param_data 只是"临时存储，等待 _refresh_param_refs 重建"，避免后续维护者误解。

---

## Low Priority

### 3. Library `np_names` 排除 param tokens — 正确

**位置**: `kd/model/discover/library.py:137`

```python
self.np_names = [t.name for t in tokens if t.name not in ['const'] and ('u' not in t.name or t.name=='mul') and t.input_var is None and t.param_var is None]
```

**验证**: `t.param_var is None` 条件正确排除了 param tokens。

这是必要的，因为 `switch_tokens` (program.py:393-408) 遍历 `np_names` 并查找 `torch_tokens[name+'_t']`。Param tokens 没有 torch 对应版本（如 `nu_t`），如果不排除会导致 `KeyError`。

**结论**: 正确。

### 4. `DiffConstraint_right` 使用 set-based exclusion — 正确

**位置**: `kd/model/discover/dso/prior.py:664-688`

```python
input_set = set(library.input_tokens.tolist())
non_input_terminals = np.array(
    [t for t in library.terminal_tokens if t not in input_set],
    dtype=np.int32)
targets = np.concatenate([
    library.unary_tokens,
    library.binary_tokens,
    non_input_terminals,
])
```

**验证**: 旧代码使用 `terminal_tokens[-2:]` 硬编码偏移，当 param token 插入后偏移错位。新实现使用 set exclusion（`terminal_tokens - input_tokens`）。

- `terminal_tokens` = 所有 arity=0 的 token（包括 x1, u1, p1, const）
- `input_tokens` = 只有 input_var is not None 的 token（x1, x2, ...）
- `non_input_terminals` = u1, p1, const — 正确排除了 input 变量

**关键语义**: diff(f, x_i) 的右孩子必须是空间坐标（x1, x2, ...），不应是 u1、param token 或 const。新实现正确保证了这一点，且不依赖 token 顺序。

**结论**: 正确，且比旧实现更健壮。

### 5. N-D Sparse path param permutation

**位置**: `kd/model/discover/adapter.py:543-555`

```python
# Sparse perm: spatial_axes + [lhs_axis] (time-last)
param_fields = getattr(self.dataset, "param_fields", None) or {}
if param_fields:
    for name, arr in param_fields.items():
        p_raw = np.asarray(arr, dtype=float)
        p = np.transpose(p_raw, perm)
        param_data_list.append(p)
```

**验证**: Sparse N-D path 的 `perm` 将 `axis_order → [spatial_axes..., lhs_axis]`（time-last）。这与 `u = np.transpose(u_raw, perm)` 使用相同的 perm。

对比 Regular N-D path (`adapter.py:252-265`): perm 将 `axis_order → [lhs_axis, *spatial_axes]`（time-first）。两者各自一致。

**结论**: 正确。param_data 和 u 使用相同的 transpose，shape 保证对齐。

---

## 等价性分析

| 测试场景 | 结果 | 备注 |
|----------|------|------|
| Regular Legacy 1D: param_data shape = usol shape | PASS | 无 perm，直接传递 |
| Regular N-D: param_data shape = u (time-first) | PASS | 同 perm |
| Sparse Legacy 1D: param_data = raw | PASS | _refresh_param_refs 重建 |
| Sparse N-D: param_data shape = u (time-last) | PASS | 同 perm |
| PDEPINNTask: param tokens 创建 | PASS | create_tokens(n_param_var, param_names) |
| PDEPINNTask: load_data 绑定 _param_data_ref | PASS | 按 param_var 索引 |
| PDEPINNTask: _refresh_param_refs (numpy) | PASS | np.full_like |
| PDEPINNTask: _refresh_param_refs (torch) | PASS | torch.full |
| PDEPINNTask: _refresh_param_refs (varying) | PASS | NotImplementedError |
| Library.np_names 排除 param | PASS | t.param_var is None |
| DiffConstraint_right 包含 param | PASS | set-based exclusion |
| execute.py param_var branch | PASS | token._param_data_ref |
| SGA _build_vars 包含 param | PASS | 不在 den 中 |
| KD_DSCV_SPR._inject_nd_config param 注入 | PASS | n_param_var + param_names |

---

## 数学公式验证

本次修改不涉及数学公式变更。param_fields 的核心语义是"作为 terminal token 出现在表达式中，但不参与微分"。这通过以下机制保证：

1. **DSCV**: param token 的 `param_var is not None`，`input_var is None`，`state_var is None`。在 `execute.py` 中，param_var 分支直接返回 `_param_data_ref`，不设置 `dim_flag`（因此后续 diff 操作不会将 param 视为坐标方向）。
2. **SGA**: param_fields 加入 `VARS`（可用作叶子变量），但不加入 `den`（微分分母列表），因此 SGA 不会计算 d(f)/d(param)。
3. **DiffConstraint_right**: param token 被禁止作为 diff 的右孩子，从搜索空间层面阻止 `diff(f, param)` 表达式生成。

三层防护一致，无逻辑漏洞。

---

## 建议的数值测试

现有测试（`tests/test_param_fields.py`）已覆盖:
- `test_refresh_numpy` / `test_refresh_torch` — shape/dtype/device 对齐
- `test_refresh_varying_raises` — 空间变化参数拒绝
- `test_np_names_no_param_tokens` — np_names 排除
- `test_diff_constraint_excludes_param` — prior constraint 覆盖
- `test_param_in_expression` — execute 引擎正确性

额外建议（优先级低）：
- `test_refresh_near_constant`: 测试 param_field = `np.full_like(u, 0.1) + 1e-9 * np.random.randn(...)` 时行为是否符合预期（当前会通过 allclose，静默取 flat[0]）
- `test_sparse_nd_param_perm_matches_u`: 验证 param_data 和 u 的 shape 完全一致

---

## 历史模式检查

| 历史模式 | 本次是否命中 | 备注 |
|----------|-------------|------|
| #1 未保护的除法 | 否 | 无新除法 |
| #2 未检查 NaN/Inf | 否 | 使用 np.full/torch.full 创建常量，不引入 NaN |
| #5 测试从 buggy 实现逆向推断 | 否 | 测试基于明确 spec（param 不参与微分、shape 对齐等） |
| #6 半支持比不支持更危险 | **注意** | PINN 模式只支持常量 param，varying 会 raise。这是"显式拒绝"而非"半支持"，符合最佳实践 |
| #7 try/except 吞错误 | 否 | _refresh_param_refs 中无宽泛 except |
