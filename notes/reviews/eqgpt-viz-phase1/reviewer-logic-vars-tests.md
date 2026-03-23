# Logic Completeness Review: VARS Leakage Fix (Full Review)

> Reviewer: reviewer-logic | Scope: context.py + equation.py + test_vars_hybrid.py
> Plan: `jolly-seeking-quasar.md` | Ref: `ref_lib/SGA-PDE/codes/setup.py:338-355`

## Summary
- 结论：**APPROVE** (with 1 Medium, 3 Low)
- High: 0, Medium: 1, Low: 3

修复正确实现了 Strategy C（源头删除 legacy alias + lhs_axis 过滤），与 ref_lib 行为一致。
三个变更点（legacy alias 删除、lhs_axis 跳过、guardrail 添加）逻辑完备。
测试基于 ref_lib spec 编写，覆盖 2D/3D/dedup/guardrail。以下是逐项审查。

---

## Focus Area 1: lhs_axis filter in `_build_vars` — N-D 正确性

**结论：正确。**

`context.py:978-980`:
```python
for axis in self.axis_order:
    if axis == self.lhs_axis:
        continue
```

- 动态比较 `axis == self.lhs_axis`，不依赖硬编码 `"t"`。
- 与同文件 `den` 构建逻辑（line 475-477）结构一致，避免了 plan 中描述的"den 正确跳过但 VARS 遗漏"的根因。
- N-D 场景（4+ axes）：循环遍历 `self.axis_order` 所有轴，只跳过一个 lhs_axis，其余都加入 VARS。正确。
- `coord_grids.get(axis)` 的 `None` 检查（line 981-982）：如果某个 axis 没有对应的 coord_grid，会被跳过。这是防御性合理的。

**Branch 完备性**：`_build_vars` 有两个 call site：
1. `_init_operators` line 472: `include_grad=False` — 初始化阶段，grad_fields 尚未计算。
2. `_update_operators_with_derivatives` line 964: `include_grad=True` — 梯度计算后更新。

两处都通过同一函数，lhs_axis 过滤逻辑统一。

---

## Focus Area 2: Legacy `ux` alias 删除安全性

**结论：安全。**

Plan 中删除了 `_compute_grad_fields` 末尾的 legacy alias 注册（原 lines 573-578）。

消费者分析（`"ux"` 在 `kd/model/sga/` 中的所有引用）：

| 位置 | 用途 | 影响 |
|------|------|------|
| `context.py:450` | `ALL` 表初始化 `['ux', 0, None]` | 不受影响。`_update_operators_with_derivatives` line 970 通过 `self.grad_fields.get("ux", self.ux)` fallback 到 `self.ux`。数据正确。 |
| `context.py:969-970` | `ALL` 表更新 `grad_fields.get("ux", self.ux)` | fallback 生效，`self.ux` 在 line 946 已赋值。正确。 |
| `context.py:293` | 冲突检测 `field_names & {"ux"}` | 不受影响。这是校验用户输入的 field name 不能叫 `"ux"`。 |
| `context.py:1066` | `_get_debug_info` | 使用 `self.ux`（属性），不依赖 `grad_fields["ux"]`。正确。 |
| `equation.py:240` | LaTeX mapping `'ux': 'u_{x}'` | 保留。ALL 表中 `"ux"` entry 仍存在，tree rendering 可能产生 `"ux"` leaf。需要这个 mapping。 |

**关键验证**：VARS 中不再有 `"ux"` entry（因为 `grad_fields` 不再注册 `"ux"`），只有 canonical `"u_x"`。但 ALL 表中 `"ux"` entry 仍在（hardcoded in `_init_operators`），其 data 通过 fallback 正确指向 `self.ux`。两个表的职责不同：VARS 是 GP 搜索的 terminal set，ALL 是完整 operator table（包含 OPS + VARS + legacy）。分离合理。

---

## Focus Area 3: Guardrail `_assert_no_lhs_in_vars` 边界情况

**结论：基本正确，有一个 Low 级别注意点。**

`context.py:993-1000`:
```python
def _assert_no_lhs_in_vars(self) -> None:
    var_names = {entry[0] for entry in self.VARS}
    if self.lhs_axis in var_names:
        raise ValueError(...)
```

检查项：
- [x] 检查的是 VARS 中的 name（`entry[0]`），正确匹配坐标轴名。
- [x] `self.lhs_axis` 动态引用，不硬编码。
- [x] 在 `_update_operators_with_derivatives` line 965 中被调用，位于 `_build_vars` 之后。

**注意点**：
- `_init_operators` line 472 的第一次 `_build_vars(include_grad=False)` 调用**没有**调用 guardrail。这意味着如果 `_build_vars` 在 `include_grad=False` 路径有 bug，guardrail 不会 catch。但实际上 `include_grad=False` 不包含 grad_fields，只包含 fields + coords + zeros，而 coords 的 lhs_axis 过滤逻辑与 `include_grad=True` 路径完全相同（同一个 for loop）。所以不是真正的风险。
- 假设 `self.VARS` 不为空且格式正确。合理，因为 `_build_vars` 总是至少添加 `['0', 0, zeros]`。

---

## Focus Area 4: 测试充分性

**结论：良好，有 1 Medium 建议。**

### 已覆盖的场景

| 测试 | 场景 | 覆盖 |
|------|------|------|
| `test_vars_include_fields_coords_and_spatial_gradients` | 3D multi-field，lhs_axis 排除 + 梯度包含 | 全面 |
| `test_vars_spatial_gradients_have_expected_values` | 梯度数值正确性 | 全面 |
| `test_vars_exclude_lhs_first_derivatives_for_all_fields` | lhs-axis 导数不在 VARS | 全面 |
| `test_legacy_vars_keep_ux_alias` | 2D 去重：只有 `u_x`，没有 `ux` | 全面 |
| `test_operator_table_all_ux_points_to_precomputed_gradient` | ALL 表 `ux` fallback 正确 | 全面 |
| `test_build_vars_excludes_gradients_when_disabled` | `include_grad=False` 路径 | 全面 |
| `TestVarsExcludesLhsAxisCoordinate` 2D/3D | 不同维度的 lhs_axis 排除 | 2D+3D |
| `TestVarsNoDuplicateGradData` | 去重（data pointers + names） | 全面 |
| `TestAssertNoLhsInVarsGuardrail` | guardrail 存在/raises/passes | 全面 |

### 缺失的边界

**Medium: 无 `lhs_axis != "t"` 测试。** 所有 11 个测试都使用 `lhs_axis="t"`。虽然实现正确使用动态比较，但 `tests/unit/test_pdedataset_nd.py:209` 已有 `lhs_axis="x"` 的 dataset 测试，证明这是真实可达的配置。建议在 `TestVarsExcludesLhsAxisCoordinate` 中增加一个 case：

```python
def test_2d_lhs_x(self):
    """When lhs_axis='x', 'x' must not be in VARS but 't' should be."""
    x = np.linspace(0.0, 1.0, 5)
    t = np.linspace(0.0, 2.0, 6)
    X = x[:, None]
    T = t[None, :]
    u = X + 0.2 * T
    context = _build_context(
        fields={"u": u},
        coords_1d={"x": x, "t": t},
        axis_order=["x", "t"],
        lhs_axis="x",
    )
    names = _vars_names(context)
    assert "x" not in names
    assert "t" in names
```

---

## Focus Area 5: equation.py LaTeX mapping

**结论：正确，有 1 Low 注意点。**

`equation.py:235-247` `_render_leaf` mapping:
```python
mapping = {
    'u': 'u', 'x': 'x', 't': 't',
    'ux': 'u_{x}',
    'u_x': 'u_{x}',   # <-- 新增
    'uxx': 'u_{xx}', 'uxxx': 'u_{xxx}',
    'ut': 'u_{t}', '0': '0',
}
```

- `'u_x'` 条目已添加，确保 canonical 名称正确渲染为 `u_{x}`。正确。
- `'ux'` 条目保留，因为 ALL 表中仍有 `"ux"` hardcoded entry，tree rendering 可能生成 `"ux"` leaf。正确。

**Low: N-D 梯度名称（`u_y`, `v_x`, `v_y` 等）不在 mapping 中。** 这些名称由 `build_derivative_key` 生成（格式 `{field}_{axis}`），不在 `_render_leaf` 的 hardcoded mapping 中。但 `mapping.get(name, name)` 的 fallback 行为是返回原名，所以 `u_y` 会被渲染为 `u_y`（raw text），而非 `u_{y}`（LaTeX subscript）。

影响分析：
- 2D 场景（Burgers 等）：VARS 中只有 `u_x`，已在 mapping 中。无问题。
- 3D+ 场景：如果 SGA 发现的方程包含 `u_y` 或 `v_x` terminal，LaTeX 输出会显示 raw `u_y` 而非 `u_{y}`。这是 cosmetic 问题，不影响数值正确性。
- 当前项目 SGA 的实际使用以 2D Burgers/KdV/Chafee-Infante 为主。3D+ 场景暂时较少。

建议：可以在 `_render_leaf` 中加一个 regex fallback（如 `re.match(r'^(\w+)_(\w+)$', name)` → `f'{field}_{{{axis}}}'`），但这不是本次修复的范围，可作为 follow-up。

---

## High Priority

无。

---

## Medium Priority

1. **缺少 `lhs_axis != "t"` 的测试** @ `test_vars_hybrid.py` (全局)
   - 场景：所有 11 个测试都使用 `lhs_axis="t"`。`lhs_axis` 是可配置参数，`"x"` 是已知的真实配置。
   - 后果：如果实现退化为硬编码 `"t"` 检查，此测试套件不会捕获。
   - 历史参考：common-mistakes.md #4 — 测试未覆盖边界情况。
   - 建议：见 Focus Area 4 中的代码示例。

---

## Low Priority

1. **`_assert_no_lhs_in_vars` 未在 `_init_operators` 首次 `_build_vars` 调用后执行** @ `context.py:472`
   - 场景：`_init_operators` line 472 调用 `_build_vars(include_grad=False)` 但不调用 guardrail。
   - 后果：理论上如果 `_build_vars` 在 `include_grad=False` 路径有 bug，首次构建的 VARS 可能包含 lhs_axis。但实际上这个 VARS 很快会被 `_update_operators_with_derivatives` 覆盖（line 964），且那里有 guardrail。风险极低。
   - 建议：保持原样。如果要彻底防御，可以在 line 472 后也加一行 `self._assert_no_lhs_in_vars()`，但不是必要的。

2. **N-D 梯度名称缺少 LaTeX mapping** @ `equation.py:236-246`
   - 场景：`u_y`, `v_x`, `v_y` 等 N-D canonical 梯度名称不在 `_render_leaf` mapping 中。
   - 后果：3D+ 场景的 LaTeX 输出会显示 raw `u_y` 而非 `u_{y}`。Cosmetic 问题。
   - 建议：Follow-up task，不阻塞本次修复。

3. **Guardrail 注入测试耦合 VARS 内部结构** @ `test_vars_hybrid.py:382-383`
   - 场景：`np.vstack` 注入假设 VARS 是 `(N,3)` ndarray。
   - 后果：如果 VARS 结构变化，测试会 break。但 guardrail 测试本质需要模拟异常状态，这是合理的 tradeoff。

---

## Branch Coverage Analysis

| Function | Branches | Covered by Tests | Missing |
|----------|----------|-----------------|---------|
| `_build_vars(include_grad=True)` | fields / coords-skip-lhs / coords-include / grads / zeros | All | - |
| `_build_vars(include_grad=False)` | fields / coords-skip-lhs / coords-include / zeros | All | - |
| `_compute_grad_fields` lhs_axis skip | skip / include | Indirectly (via VARS content assertions) | - |
| `_assert_no_lhs_in_vars` | clean / dirty | Both | - |
| `_update_operators_with_derivatives` ALL `ux` fallback | `grad_fields` has `ux` / fallback to `self.ux` | Fallback path tested | `grad_fields` has `ux` path (now unreachable, correct) |
| `_render_leaf` `u_x` mapping | hit | `test_legacy_vars_keep_ux_alias` 验证 VARS 使用 canonical name | - |
| lhs_axis = "t" | tested | 11/11 tests | - |
| lhs_axis != "t" | **untested** | 0/11 tests | **Medium #1** |

## Hidden Assumptions

| 假设 | 合理？ | 说明 |
|------|--------|------|
| `self.ux` 在 `_update_operators_with_derivatives` 之前已赋值 | 是 | `_compute_primary_derivatives` (line 937-946) 在 `_update_operators_with_derivatives` 之前调用 |
| ALL 表中 `"ux"` entry 永远存在 | 是 | hardcoded in `_init_operators` line 450 |
| `grad_fields` 永远不会包含 `"ux"` key（after fix） | 是 | legacy alias 注册已删除，且 `build_derivative_key("u","x",order=1)` 返回 `"u_x"` 而非 `"ux"` |
| VARS entry 的 index 2 是 data array | 是 | 与 ref_lib `setup.py:354` 结构一致 `[name, child_num, data]` |
| `axis_order` 中只有一个轴等于 `lhs_axis` | 是 | `_init_variable_registry` 校验 axis_order 不重复 |

## Over-Engineering

未发现。修复遵循 Strategy C（最简方案），净改动为负行数。guardrail 与已有 `_assert_no_lhs_in_den` 风格一致，不是额外复杂性。
