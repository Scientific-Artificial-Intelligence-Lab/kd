# Code Quality Review: Phase 4 param_fields Implementation

## Summary
- 结论：**APPROVE** (with minor suggestions)
- High: 0, Medium: 3, Low: 5

Phase 4 的 param_fields 实现整体代码质量良好。新增代码遵循项目现有的 dataclass 模式，type hints 完整度高，命名清晰。以下是逐条检查结果。

---

## High Priority

无。

---

## Medium Priority

### 1. **`context.py` 文件超长 (1180 行)** @ `/Users/hao/PhD/project/kd/kd/model/sga/sgapde/context.py`
- `ProblemContext` 已经超过项目规范的 500 行上限（当前 1180 行）。param_fields 增改本身很小（`_build_vars` 加了约 5 行），但需注意该文件的技术债务。
- 建议：当前 PR 不需要拆分，但后续应考虑将 autograd 相关逻辑抽离为独立模块。
- 注意：这不是 param_fields 引入的问题，而是已有技术债务。

### 2. **`execute.py` 中存在遗留 `print()` 和 `pdb`** @ `/Users/hao/PhD/project/kd/kd/model/discover/execute.py:42-44, 100`
- param_var 分支本身（L28-29）没有 print，但同文件的相邻代码有 `print(e)` 和 `import pdb;pdb.set_trace()` 调试语句。这些不是本次变更引入的，但影响所在函数 `python_execute` 的整体质量。
- 历史参考：`common-mistakes.md` #7 - try/except 吞错误隐藏 bug。`execute.py:41` 的 `except Exception as e:` 加 print+pdb 正是这个模式。
- 建议：不在本 PR 范围，但建议 follow-up 将这些调试代码移除。

### 3. **`pde_pinn.py` load_data 缺少对已有属性的安全处理** @ `/Users/hao/PhD/project/kd/kd/model/discover/task/pde/pde_pinn.py:114-120`
- `load_data` 直接存储 `self._param_fields_raw = data.get('param_data', None)`，但不会验证 param token 的 `param_var` 索引是否越界，只用 `< len(self._param_fields_raw)` 做上界检查。如果 `_param_fields_raw` 为空列表 `[]` 而不是 `None`，truthy 检查 `if self._param_fields_raw:` 不会执行绑定。这合理但微妙。
- 建议：保持现状即可，但建议在 docstring 中说明空列表和 None 的区别。

---

## Low Priority

### 1. **Token.__init__ 缺少 type hints** @ `/Users/hao/PhD/project/kd/kd/model/discover/library.py:38`
- `Token.__init__(self, function, name, arity, complexity, input_var=None, state_var=None, param_var=None)` 所有参数均无 type annotations。新增的 `param_var` 参数也沿用了这一风格。
- 这是已有代码的遗留问题，param_fields 变更保持了一致性，未引入退化。
- 建议：后续统一添加 type hints 到 `Token` 和 `Library` 类。

### 2. **`create_tokens` 文档字符串缺少 Return type** @ `/Users/hao/PhD/project/kd/kd/model/discover/functions.py:206`
- `create_tokens` 函数有 docstring 且正确记录了新增的 `n_param_var` 和 `param_names` 参数，但返回类型未标注（`-> list` 或 `-> List[Token]`）。
- 建议：添加返回类型注解。

### 3. **测试文件 `test_param_fields.py` 中 RED 注释可清理** @ `/Users/hao/PhD/project/kd/tests/test_param_fields.py:411,433,484,505,568,585,605,629`
- 多处 `# RED: will fail until ...` 注释在实现完成后已失去意义。这些是 TDD 过程的遗留标记。
- 建议：清理过时的 `# RED:` 注释，保留那些仍有解释价值的。

### 4. **`_render_leaf` mapping 表可考虑常量化** @ `/Users/hao/PhD/project/kd/kd/model/sga/sgapde/equation.py:241-264`
- Greek letter mapping 是模块级常量的好候选人，目前每次调用 `_render_leaf` 都会创建字典。
- 影响极小（调用频率低），仅为风格建议。

### 5. **`adapter.py` 中 param_fields 提取逻辑重复 3 次** @ `/Users/hao/PhD/project/kd/kd/model/discover/adapter.py:119-129, 253-265, 412-421`
- `DSCVRegularAdapter._prepare_legacy`、`DSCVRegularAdapter._prepare_nd`、`DSCVSparseAdapter._prepare_legacy` 三处都有相同的 param_fields 提取逻辑（读 `getattr(self.dataset, "param_fields", None) or {}`，遍历写入 `param_data_list`/`param_names`）。
- `DSCVSparseAdapter._prepare_nd` 有额外的 transpose 逻辑所以略有不同。
- 建议：可以提取为一个共用的 `_extract_param_fields(dataset, perm=None)` helper 方法，减少重复。但考虑到每处上下文略有差异（raw vs transposed），当前的重复尚可接受。

---

## Checklist Results

### 1. Type Hints
- [x] 新增 public 函数（`_build_vars`, `_render_leaf` 扩展, `create_tokens` param 参数）有合理的注解
- [x] `PDEDataset.__init__` 的 `param_fields` 参数有 `Optional[Dict[str, np.ndarray]]` 注解
- [ ] `Token.__init__` 无 type hints（遗留问题，非本次引入）
- [x] `_refresh_param_refs` 无返回类型但为 private 方法（可接受）

### 2. Code Structure
- [x] 新增函数长度合理（`_refresh_param_refs` ~22 行，`_build_vars` param 段 ~5 行）
- [x] `adapter.py` 634 行，在限制内
- [ ] `context.py` 1180 行，超过 500 行上限（遗留问题）

### 3. Naming
- [x] snake_case: `param_fields`, `param_data`, `param_names`, `n_param_var`, `_refresh_param_refs`
- [x] PascalCase: `PDEPINNTask`, `DSCVRegularAdapter`
- [x] UPPER_SNAKE_CASE: 无新常量（合理）
- [x] 命名清晰表达意图

### 4. Error Handling
- [x] `_refresh_param_refs` 使用具体异常 `NotImplementedError`
- [x] `PDEDataset` 中 param shape mismatch 使用 `ValueError`
- [x] 无 bare except
- [x] 无 mutable default arguments（`param_fields=None` 正确处理）

### 5. Style Consistency
- [x] 新代码无 `print()` 语句
- [x] 无 magic numbers
- [x] 与项目现有风格一致（dataclass 模式、`getattr` fallback 模式）
- [x] 导入分组正确

### 6. Tests Alignment
- [x] 测试覆盖全部新功能路径（PDEDataset, SGA adapter/config/context, DSCV token/library/execute/adapter, LaTeX, Sparse, PINN, DiffConstraint）
- [x] 测试包含正向和反向用例（有/无 param_fields）
- [x] 边界情况有覆盖（shape mismatch, spatially-varying 参数, backward compat）

---

## Notes

1. **代码风格一致性**：新增代码与项目现有代码保持了良好的一致性。特别是 `getattr(self.dataset, "param_fields", None) or {}` 模式与已有的 `fields_data`/`coords_1d` 处理方式统一。

2. **历史模式检查**：
   - common-mistakes #3 (缺少 Type Hints)：新代码的 public API 已有注解，`Token` 类的缺失是遗留问题。
   - common-mistakes #5 (测试固化 bug)：测试基于 spec 编写，断言明确（如 param 不应在 `np_names`、param 不应在 `den`），无反向推断问题。
   - common-mistakes #7 (try/except 吞错误)：新代码无此问题。`_refresh_param_refs` 中 `try: import torch` 的 pattern 合理（处理可选依赖）。

3. **测试质量**：`tests/test_param_fields.py` (763 行) 组织良好，按 Phase 分组，fixtures 共享，覆盖了 full stack（Dataset -> Adapter -> Token/Library -> Execute -> Prior -> LaTeX）。
