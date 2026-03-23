# Logic Completeness Review: Phase 4 param_fields

## Summary
- 结论：**REQUEST_CHANGES**
- High: 2, Medium: 4, Low: 2

---

## High Priority

### 1. **PDEMultiPINNTask 继承不传递 n_param_var** @ `kd/model/discover/task/pde/pde_pinn.py:448`
- **场景**：`PDEMultiPINNTask` 继承 `PDEPINNTask`，但没有 override `__init__`。由于 `PDEPINNTask.__init__` 不调用 `super().__init__()`（只调用了 `super(PDETask).__init__()` 即跳到 `HierarchicalTask`），`PDEMultiPINNTask` 确实"自动"继承了 `n_param_var` 和 `param_names` 参数。
- **但问题在**：`PDEMultiPINNTask` overrides `AD_generate` (line 454)，而该 override 只是 `super().AD_generate(x, model)` —— 但父类 `PDEPINNTask` 中不存在 `AD_generate` 方法（只有 `AD_generate_1D` 和 `AD_generate_mD`）。这说明 `PDEMultiPINNTask.AD_generate` 是一个**死代码路径**，调用会抛出 `AttributeError`。
- **后果**：如果有人通过 `PDEMultiPINNTask` 使用 param_fields，`generate_meta_data` 中的 `self._refresh_param_refs()` 调用仍然有效（它在 `PDEPINNTask.generate_meta_data` 中），但 `AD_generate` override 表明此子类可能有未完成的设计意图。使用者可能陷入混乱。
- **建议**：确认 `PDEMultiPINNTask.AD_generate` 是否应该是 `AD_generate_1D` 或 `AD_generate_mD`，修复或删除这段死代码。

### 2. **NoInputsConstraint 中 non_state_tokens 包含 param tokens，可能导致 param-only 表达式** @ `kd/model/discover/library.py:192` + `kd/model/discover/dso/prior.py:550`
- **场景**：`Library.non_state_tokens` 定义为 `arity == 0 and state_var is None`，这意味着 param tokens（`param_var is not None`）会被包含进去。`NoInputsConstraint.__call__` 使用 `non_state_tokens` 来约束"表达式必须至少包含一个 state variable"时，将 param tokens 也列入"被约束掉"的名单。
- **问题**：当 `dangling == 1`（表达式即将结束）且还没有任何 state token 时，`NoInputsConstraint` 会对所有 `non_state_tokens` 加约束（即禁止它们出现）。这是正确的意图（强制选择 state token）。但 param tokens 也在此约束中，意味着 param tokens 在"必须放 state var"的位置被正确禁止了。
- **实际问题出在反面**：如果表达式已经有了 state token，param token 可以出现在任何 terminal 位置——此时 controller 可能生成一个 `mul(nu, nu)` 这样的纯 param 表达式片段，而没有 input_var。这不一定是 bug（param*u 是有意义的），但需要确认这是否符合设计意图。
- **更关键的问题**：`NoInputsConstraint` 检查的是 `state_tokens`（u1, u2 等），但不检查 `input_tokens`（x1, x2 等）。param tokens 既不是 state 也不是 input，因此一个表达式可能只包含 `nu * u1`（有 state，没 input）——这在语义上可能是合理的，但也可能是无意义的搜索浪费（对于 PDE discovery，每个 RHS term 通常至少包含一个 diff(u, x) 结构）。
- **后果**：搜索空间膨胀，可能生成无意义的候选方程。不会崩溃，但影响效率。
- **历史参考**：匹配 common-mistakes #5 "半支持比不支持更危险" —— param tokens 在约束系统中被"部分考虑"，可能需要更完整的 prior 约束设计。
- **建议**：评估是否需要一个 `NoParamOnlyConstraint`，确保 param tokens 总是与 state/diff 组合使用。至少在文档中说明当前 prior 系统对 param tokens 的约束策略。

---

## Medium Priority

### 3. **switch_tokens 对 param token 的安全性取决于 np_names 的正确过滤** @ `kd/model/discover/program.py:393-422`
- **分析**：`switch_tokens` 遍历 traversal 中的 token，对每个 `name in np_tokens_name` 的 token 查找 `torch_tokens[t.name+'_t']`。如果 param token 的 name 出现在 `np_names` 中，就会尝试查找 `nu_t` 这个 torch token，导致 `KeyError`。
- **当前状态**：`Library.np_names`（line 137）已经添加了 `t.param_var is None` 过滤条件，所以 param tokens 会被正确排除。`switch_tokens` 对无法匹配的 token（包括 param tokens）执行 `else: new_traversal.append(t)`，即直接保留 numpy param token。
- **潜在问题**：在 torch execution 模式下，param token 的 `_param_data_ref` 需要是 torch.Tensor（由 `_refresh_param_refs` 负责）。如果 `_refresh_param_refs` 正确执行，param token 返回 torch tensor，与其他 torch tokens 的输出一致。但如果 `_refresh_param_refs` 未被调用或失败（例如 spatially-varying param），param token 仍然持有 numpy array，与 torch tokens 混合运算可能导致类型错误。
- **后果**：在 PINN 模式下 spatially-varying param 已被 `NotImplementedError` 拦截，所以实际不会触发。但如果未来放宽此限制，需要注意。
- **建议**：当前实现是安全的。在 `_refresh_param_refs` 的 docstring 中注明此依赖关系。

### 4. **LaplaceConstraint targets 包含 param tokens** @ `kd/model/discover/dso/prior.py:749-751`
- **场景**：`LaplaceConstraint` 的 targets 使用 `token.state_var is None` 来收集"非 state 变量"作为 laplace 的禁止子节点。param tokens 有 `state_var is None`，因此会被包含在 targets 中。
- **行为**：这意味着 `lap(nu)` 或 `lap_t(nu)` 会被 prior 约束禁止。从物理语义上看，这是正确的——对参数取 Laplacian 没有意义。
- **但**：约束使用的是 `token.state_var is None` 而非"所有非 u token"这一语义。如果将来添加更多 token 类型，此过滤条件可能需要更明确的语义。
- **后果**：当前行为正确。Low risk。

### 5. **DSCV Regular mode 的 PDETask.load_data 不会在 dataset 复用时 reset param** @ `kd/model/discover/task/pde/pde.py:116-147`
- **场景**：用户先用有 param 的 dataset 创建 KD_DSCV 实例并 fit，然后用无 param 的 dataset 再次 fit。
- **调用链**：`KD_DSCV_SPR._inject_nd_config()` 会先 reset `config_task["n_param_var"] = 0` 和 pop `param_names`。然后 `set_task` 调用 `make_task(**config_task)` 创建新的 `PDEPINNTask`（n_param_var=0），再调用 `task.load_data(data)`。由于创建了全新的 task 和 library，旧的 param tokens 不会泄漏。
- **但是**：对于 `KD_DSCV`（Regular mode）的路径，`set_task` 同样创建新的 `PDETask`，而 `PDETask.__init__` 不接受 `n_param_var` 参数！
- **调用链分析**：`make_task(task_type="pde", ...)` → `PDETask(**config_task)`。如果 `config_task` 包含 `n_param_var`，`PDETask.__init__` 不接受此参数 → `TypeError: __init__() got an unexpected keyword argument 'n_param_var'`。
- **实际情况**：检查 `KD_DSCV.set_task()`（非 SPR 版本）是否会在 `config_task` 中注入 `n_param_var`。查看 `_inject_nd_config` 只在 `KD_DSCV_SPR` 中存在。对于 `KD_DSCV`（Regular），如果 adapter 的 data 中包含 `n_param_var`，`PDETask.load_data` 正确读取它。`config_task` 默认是 `task_type="pde"` 且不含 `n_param_var`，所以不会传给 `__init__`。
- **后果**：Regular mode 中 `PDETask.__init__` 不接受 `n_param_var`，但 `load_data` 从 data dict 中读取。如果用户手动在 `config_task` 中添加 `n_param_var`，会崩溃。但在正常使用流程中不会触发。
- **建议**：确保 Regular mode 的 `config_task` 不会意外包含 `n_param_var`，或让 `PDETask.__init__` 也接受此参数（即使忽略）。

### 6. **param_fields 的 dict 迭代顺序隐含假设** @ multiple files
- **场景**：`param_fields` 是 `Dict[str, np.ndarray]`。在多处代码中（adapter.py L121-128, L255-265, L414-421, L546-555; pde.py L123-128），param_data 列表和 param_names 列表的顺序依赖于 `dict.items()` 的迭代顺序。
- **隐含假设**：Python 3.7+ 保证 dict 按插入顺序迭代，所以这是安全的。但 param_var 索引（0, 1, 2...）与 param_data 列表的位置对应关系，完全依赖于 dict 迭代顺序的一致性。
- **后果**：在 Python 3.7+ 上是安全的，但语义上脆弱。如果用户在不同位置以不同顺序构造 dict，可能导致名称和数据不匹配（但这需要两个不同 dict 实例，在正常流程中不会发生，因为始终是从 `dataset.param_fields` 读取）。
- **建议**：Low risk，但可以考虑在 adapter 中使用 `sorted()` 确保确定性顺序，或在 PDEDataset 中使用 OrderedDict。

---

## Low Priority

### 7. **DSCV discover_eq2latex.py 只支持 p1, p2, p3** @ `kd/viz/discover_eq2latex.py:12-14, 21-23`
- **场景**：`DEEPRL_SYMBOLS_FOR_SYMPY` 和 `_SYMBOL_DISPLAY` 只定义了 `p1`, `p2`, `p3`。如果用户使用 4+ 个 param tokens，或使用自定义名称（如 "nu"），SymPy 解析可能失败或显示不友好。
- **后果**：LaTeX 渲染时，超过 3 个参数或非 `pN` 命名的参数不会在 symbol dict 中找到，但 `parse_expr` 会将它们作为普通 SymPy symbols 处理，不会崩溃。显示效果可能不理想（如 "nu" 显示为 `\nu` 是期望的，但未配置）。
- **建议**：考虑动态生成 symbol dict（基于 Library.param_tokens）或在 `_SYMBOL_DISPLAY` 中添加更多默认映射。这是文档中提到的 FIXME（line 9）。

### 8. **execute.py 中 param_var 分支不设置 dim_flag** @ `kd/model/discover/execute.py:28-29`
- **分析**：当 traversal 中的 token 是 param_var 时（line 28-29），`dim_flag` 不会被更新。这是**正确的**行为——param tokens 不表示空间坐标，不应影响 diff 操作的维度标记。
- **但**：如果 traversal 以 `diff(param, x1)` 开头（param 作为 diff 左子节点），`dim_flag` 仍由 `x1` 的 `input_var` 设置（在后续遍历中）。由于 traversal 是前序遍历，diff 会先入栈，然后处理 param（不设 dim_flag），然后处理 x1（设置 dim_flag），最终 diff 使用正确的 dim_flag。
- **后果**：当前实现正确，无问题。
- **注意**：`DiffConstraint_right` 已正确禁止 param 作为 diff 的右子节点（即 diff 的"对谁求导"不能是 param）。`DiffConstraint_left` 也禁止了 input_tokens 作为 diff 左子节点。但 param tokens 没有被 `DiffConstraint_left` 特别处理——param tokens 不在 `input_tokens` 也不在 `const_token` 中，所以它们**可以**作为 diff 左子节点，即 `diff(param, x1)` 是允许的。对常数 param 来说，这个表达式值为零，不影响方程发现结果但浪费搜索空间。
- **建议**：可以在 `DiffConstraint_left` 中将 param tokens 也加入 targets（禁止作为 diff 左子节点），减少无效搜索空间。Low priority。

---

## Branch Coverage Analysis

| Function/Area | Branches | Covered | Missing |
|---|---|---|---|
| `PDEDataset.__init__` param_fields | 3 (None / valid / mismatch) | 3 | - |
| `DSCVRegularAdapter._prepare_legacy` param | 2 (has / no param) | 2 | - |
| `DSCVRegularAdapter._prepare_nd` param | 2 (has / no param) | 2 | - |
| `DSCVSparseAdapter._prepare_legacy` param | 2 (has / no param) | 2 | - |
| `DSCVSparseAdapter._prepare_nd` param | 2 (has / no param) | 2 | - |
| `PDETask.load_data` param binding | 2 (has / no param_data) | 2 | - |
| `PDEPINNTask.load_data` param binding | 2 (has / no param_data) | 2 | - |
| `PDEPINNTask._refresh_param_refs` | 4 (no raw / const np / const torch / varying) | 4 | - |
| `ProblemContext._build_vars` param | 2 (has / no param) | 2 | - |
| `Library.__init__` param_tokens | 2 (has / no param) | 2 | - |
| `switch_tokens` param interaction | 2 (param in/not in np_names) | 1 | param in np_names (correctly prevented by filter) |
| `NoInputsConstraint` param interaction | 1 implicit | 1 | No explicit param handling |

## Hidden Assumptions

| 假设 | 合理？ |
|---|---|
| param_fields dict 迭代顺序一致（Python 3.7+） | 是，但语义脆弱 |
| param_fields shape == target field shape（在所有路径中） | 是，PDEDataset 验证 |
| PINN mode param 必须是常数（不支持 spatially-varying） | 是，有 NotImplementedError 守卫 |
| param token name 不含 'u'（避免 np_names 过滤器误命中） | 是，但未显式验证 |
| param token name 不与 diff/Diff token name 冲突 | 是，但未显式验证 |
| create_tokens 中 param_names 长度 >= n_param_var | 否 —— 使用 `i < len(param_names)` 防御，fallback 到 `pN` |

## Over-Engineering

- 未发现过度工程问题。param_fields 的实现是 minimal 的，每个层只做必要的传递和转换。
- `_refresh_param_refs` 的设计合理——PINN 模式下 u 的 shape 在每次 generate_meta_data 后变化，需要同步更新 param 数据。

---

## 调用链完整性分析

### Regular mode (KD_DSCV)
```
PDEDataset(param_fields={"nu": ...})
  -> DSCVRegularAdapter._prepare_legacy()
     -> data["param_data"], data["n_param_var"], data["param_names"]
  -> set_task(config_task, data)
     -> PDETask(**config_task)  # n_param_var 不在 config_task 中
     -> task.load_data(data)    # 从 data dict 中读取 param_data
        -> create_tokens(n_param_var=..., param_names=...)
        -> Library(tokens) 包含 param tokens
        -> token._param_data_ref 绑定
```
**完整性**：OK。

### Sparse mode (KD_DSCV_SPR)
```
PDEDataset(param_fields={"nu": ...})
  -> DSCVSparseAdapter._prepare_legacy()/_prepare_nd()
     -> data["param_data"], data["n_param_var"], data["param_names"]
  -> _inject_nd_config()
     -> config_task["n_param_var"] = n_param_var
     -> config_task["param_names"] = [...]
  -> set_task(config_task, data)
     -> PDEPINNTask(n_param_var=..., param_names=...)  # init 中建 library + param tokens
     -> task.load_data(data)    # 绑定 _param_data_ref
  -> pretrain() -> Program.reset_task()
     -> generate_meta_data() -> _refresh_param_refs()  # 同步 param data 到新 u shape
```
**完整性**：OK。

### SGA mode (KD_SGA)
```
PDEDataset(param_fields={"nu": ...})
  -> SGADataAdapter.to_solver_kwargs()
     -> kwargs["param_fields"] = {...}
  -> SolverConfig(param_fields={"nu": ...})
  -> ProblemContext(config)
     -> _build_vars() 中添加 param_fields entries
     -> VARS 包含 "nu"
     -> den 不包含 "nu"（正确）
```
**完整性**：OK。
