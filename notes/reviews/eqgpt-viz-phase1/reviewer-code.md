# Code Quality Review: fix-latex

**Reviewer**: reviewer-code
**Scope**: `kd/viz/discover_eq2latex.py:41-58` (`_print_Derivative` method) + `tests/test_discover_notation.py`

---

## Summary
- 结论：**APPROVE**
- High: 0, Medium: 2, Low: 3

---

## High Priority

无。

---

## Medium Priority

### 1. `_print_Derivative` 缺少 type hints @ `kd/viz/discover_eq2latex.py:41`

- **问题**：方法签名 `def _print_Derivative(self, expr)` 没有类型注解。
- **历史参考**：匹配 common-mistakes.md #3（缺少 Type Hints）
- **说明**：这是 SymPy `LatexPrinter` 的 override 方法，SymPy 基类本身也没有 type hints，所以保持签名一致有合理性。但按项目规范，至少应标注 `expr: sympy.Expr` 和返回类型 `-> str`。`_print_Symbol` (line 60) 也同样缺少。
- **建议修复**：
  ```python
  def _print_Derivative(self, expr: sympy.Expr) -> str:
  def _print_Symbol(self, expr: sympy.Expr) -> str:
  ```
- **严重程度**：Medium（不影响功能，但违反项目规范）

### 2. `_discover_term_node_to_latex` 缺少 type hints @ `kd/viz/discover_eq2latex.py:66`

- **问题**：签名 `def _discover_term_node_to_latex(term_node_obj, local_sympy_symbols=None, *, notation="subscript")` 完全没有类型注解。
- **历史参考**：匹配 common-mistakes.md #3
- **说明**：这是本次修改涉及的函数（新增 `notation` 参数），按规范应标注。虽然 `term_node_obj` 没有明确的 protocol/ABC 类型（使用鸭子类型），可用 `Any` 或 protocol。
- **建议修复**：
  ```python
  def _discover_term_node_to_latex(
      term_node_obj: Any,
      local_sympy_symbols: Optional[dict] = None,
      *,
      notation: str = "subscript",
  ) -> str:
  ```
- **注**：此函数是 private（下划线前缀），严格来说不是 public API，但按项目惯例仍建议加注解。

---

## Low Priority

### 1. 残留无用注释 @ `kd/viz/discover_eq2latex.py:19`

- **问题**：第 19 行有一个看起来像是编辑遗留的注释：`# ... (imports, DEEPRL_SYMBOLS_FOR_SYMPY, DEBUG_RENDERER_MODE) ...`
- 这行注释没有提供有用信息，像是某次代码折叠或生成时的占位符。
- **建议**：删除。

### 2. `hasattr` 鸭子类型检查可考虑更清晰的注释 @ `kd/viz/discover_eq2latex.py:45`

- **问题**：`hasattr(func, 'func') and hasattr(func.func, 'name')` 是合理的鸭子类型检查（区分 Symbol vs AppliedUndef vs compound expression），但代码中的注释只说明了 "Applied function" 场景。
- **评估**：考虑过用 `isinstance(func, sympy.function.AppliedUndef)` 替代，但 `hasattr` 在此处更安全——SymPy 的类层次复杂，`AppliedUndef` 可能不覆盖所有 applied function 子类。且 else 分支已正确 fallback 到 `self._print(func)`，所以即使 `hasattr` 判断有遗漏也不会崩溃。
- **结论**：`hasattr` 用法**可以接受**，当前实现在此处是 defensive 且正确的。注释已足够说明意图。

### 3. 测试文件 docstring 中 "All tests are expected to FAIL" 已过时 @ `tests/test_discover_notation.py:9`

- **问题**：文件开头 docstring 说 "All tests are expected to FAIL until the implementation is written."，但实现已完成，应更新或移除此说明。
- **建议**：改为描述当前测试的用途，例如 "Tests for DSCV LaTeX dual-mode notation rendering."

---

## 代码质量检查清单

### Type Hints
- [x] `_print_Derivative`：缺少，但不影响功能（Medium）
- [x] `_print_Symbol`：缺少（同上，已存在的代码）
- [x] `_discover_term_node_to_latex`：缺少（Medium）
- [x] `discover_program_to_latex`：已有部分注解（`lhs_axis: Optional[str]`, `notation: str`），但 `program_object` 和其他参数仍缺

### Code Structure
- [x] 方法长度：`_print_Derivative` 18 行（41-58），符合 < 50 行
- [x] 文件长度：207 行，符合 < 500 行
- [x] 内聚性好：三层渲染逻辑（Printer → term → program）自顶向下
- [x] 无重复代码

### Naming
- [x] snake_case：所有函数和变量均符合
- [x] PascalCase：`_SubscriptLatexPrinter` 符合
- [x] UPPER_SNAKE_CASE：常量 `DEEPRL_SYMBOLS_FOR_SYMPY`, `_SYMBOL_DISPLAY`, `_VALID_NOTATIONS` 等符合
- [x] 命名清晰：`func_name`, `subscript_parts`, `var_name` 意图清楚

### Error Handling
- [x] 无 bare except（line 94 用 `except Exception as e`）
- [x] 错误消息有上下文（line 98-101）
- [x] 无 mutable default arguments

### Style Consistency
- [x] 无 `print()`，使用 `logger`
- [x] 无 magic numbers
- [x] 导入分组：stdlib → third-party → local 正确（line 1-10）
- [x] 与项目现有风格一致

### Tests Alignment
- [x] 测试覆盖了三层：Printer 直接测试、term 转换测试、program 完整测试
- [x] 覆盖了 Symbol 和非 Symbol 两种 func 类型
- [x] 有 regression 测试确保简单场景不退化
- [x] 测试命名清晰，docstring 说明预期行为
- [x] helper 函数（`_make_mock_node`, `_make_mock_program`）组织合理

---

## 总体评价

修改范围小且精准（18 行核心逻辑），三路分支（Symbol / Applied Function / Compound Expression）覆盖了所有可能的 `func` 类型。Compound expression 的 fallback 路径 `self._print(func)` 是正确的防御性设计。测试覆盖充分，包含 crash case、regression case 和 edge case。

主要改进点是 type hints（Medium），但不阻塞合并。
