# Reviewer-Algo: `_SubscriptLatexPrinter._print_Derivative()` 三分支修复

## Summary
- 结论：**APPROVE**
- 算法风险等级: Low
- High: 0, Medium: 1, Low: 2

---

## 审查范围

文件：`kd/viz/discover_eq2latex.py:41-58`

修改内容：将 `_print_Derivative` 的 else 分支从直接访问 `func.func.name`（对 `Pow`/`Mul`/`Add` 会 AttributeError）改为三分支逻辑：

```python
if isinstance(func, sympy.Symbol):
    func_name = _SYMBOL_DISPLAY.get(func.name, func.name)
elif hasattr(func, 'func') and hasattr(func.func, 'name'):
    func_name = _SYMBOL_DISPLAY.get(func.func.name, func.func.name)
else:
    func_name = "{" + self._print(func) + "}"
```

---

## High Priority

无。

---

## Medium Priority

### 1. **`variable_count` 中的 `var` 可能不是 Symbol** @ `discover_eq2latex.py:54`

- 风险类型: Robustness / Potential AttributeError
- 描述：第 54 行 `var.name` 假设 `var` 始终是 `sympy.Symbol`。在标准的 `Derivative` 使用中（对 `x1`, `x2` 求导），`variable_count` 返回的确实是 `(Symbol, int)` 元组。但 SymPy 允许对更复杂的表达式求导（例如对一个函数求导），理论上 `var` 可能不是 `Symbol`。
- 实际影响分析：从 `Node.to_sympy_string()` 的实现看（`stridge.py:127-134`），`diff`/`diff2`/`diff3`/`diff4`/`Diff`/`Diff2` 的第二个参数始终来自叶子节点（`x1`, `x2`, `x3`），这些在 `DEEPRL_SYMBOLS_FOR_SYMPY` 中都被定义为 `sympy.Symbol`。因此在本项目的实际使用中，`var` 始终是 `Symbol`，**风险极低**。
- 建议：当前不需要修改。如果未来扩展导数变量类型，应加防御：
  ```python
  var_name = _SYMBOL_DISPLAY.get(var.name, var.name) if isinstance(var, sympy.Symbol) else self._print(var)
  ```

---

## Low Priority

### 1. **`{...}` 包裹的 LaTeX 正确性** @ `discover_eq2latex.py:50`

- 风险类型: LaTeX 渲染
- 描述：`"{" + self._print(func) + "}"` 生成如 `{u^{2}}_{x}` 的 LaTeX。在 LaTeX 中，`{u^{2}}_{x}` 是合法的——外层花括号将 `u^{2}` 作为一个整体添加下标。这等价于 `{u^{2}}_{x}`，渲染结果正确：先显示 u 的平方，再添加下标 x。
- 与 `u^{2}_{x}` 对比：后者在 LaTeX 中会产生"double superscript/subscript"的歧义警告（虽然多数引擎能渲染）。使用 `{...}` 包裹是更规范的做法。
- 结论：**正确且规范**。

### 2. **分支条件区分精确性** @ `discover_eq2latex.py:43-50`

- 风险类型: 分支覆盖
- 描述：三个分支的覆盖分析：
  1. `isinstance(func, sympy.Symbol)` — 覆盖简单符号（如 `u1`、`x1`）
  2. `hasattr(func, 'func') and hasattr(func.func, 'name')` — 覆盖应用函数（如 `u1(x1, x2)`），因为 `Function('u1')(x1, x2).func` 是 `Function('u1')`，其 `.name` 属性为 `'u1'`
  3. `else` — 覆盖所有其他表达式（`Pow`, `Mul`, `Add`, `Integral`, 嵌套 `Derivative` 等）

  关键问题：`Pow`、`Mul`、`Add` 类确实有 `.func` 属性（它是 Python 类本身，如 `<class 'sympy.core.power.Pow'>`），但 Python 的内置类型的 `.name` 通过 `__name__` 访问而非 `.name`。所以 `hasattr(Pow, 'name')` 为 `False`，这些类型正确地落入 else 分支。

  SymPy 的 `UndefinedFunction`（通过 `Function('u1')` 创建的）具有 `.name` 属性，而内置数学类如 `Pow`/`Mul`/`Add`/`sin`/`cos` 等没有 `.name`（它们使用 `__name__`）。因此第二个条件 `hasattr(func.func, 'name')` **准确区分了用户定义的应用函数和内置运算**。

  注意：`sympy.sin(u1)` 的 `.func` 是 `sin` 类，`sin` 类**没有** `.name` 属性（使用 `__name__`），所以 `Derivative(sin(u1), x1)` 也会正确落入 else 分支，通过 `self._print(func)` 渲染。这是正确的行为。

- 结论：**分支条件设计正确**。

---

## 等价性分析

本改动不涉及数值计算，仅影响 LaTeX 字符串生成。无需数值等价性测试。

### 行为等价性

| 测试场景 | 修改前 | 修改后 | 正确性 |
|----------|--------|--------|--------|
| `Derivative(u1(x1,x2), x1)` | `u_{x}` | `u_{x}` | 不变 |
| `Derivative(Symbol('u1'), x1)` | `u_{x}` | `u_{x}` | 不变 |
| `Derivative(Pow(u1, 2), x1)` | **AttributeError → Error fallback** | `{u^{2}}_{x}` | **修复** |
| `Derivative(Mul(u1,u1), x1)` | **AttributeError → Error fallback** | `{u^{2}}_{x}`（SymPy 简化后） | **修复** |
| `Derivative(Add(u1, u1**2), x1)` | **AttributeError → Error fallback** | `{u + u^{2}}_{x}` | **修复** |

---

## 建议的数值测试

本改动不涉及数值计算，无需额外数值测试。

现有测试已充分覆盖：
- `tests/test_discover_notation.py::TestNonSymbolDerivative` — 7 个测试覆盖 Pow/Mul/Add 及嵌套场景
- `tests/test_discover_notation.py::TestSubscriptPrinterNonSymbolFunc` — 4 个直接 SymPy 表达式测试
- 回归测试确保 simple `u_{x}`, `u_{xx}` 不受影响

---

## 算法分析总结

1. **`self._print(func)` 对所有 SymPy 表达式类型的正确性**：`self._print()` 是 SymPy `LatexPrinter` 的核心递归方法，能正确处理所有 SymPy 表达式类型（Pow, Mul, Add, Function, Integral, Sum 等）。自定义的 `_print_Symbol` 方法会在递归中被调用，确保 `u1` → `u`、`x1` → `x` 的映射在嵌套表达式中也生效。

2. **三分支逻辑完备性**：三个分支穷尽了 `Derivative.args[0]` 的所有可能类型——`Symbol`、applied `Function`、以及其他一切表达式。`else` 分支作为兜底，通过 `self._print()` 处理任何未预见的类型，安全且正确。

3. **无数值计算风险**：整个改动仅涉及字符串生成，不存在 NaN/Inf/除零/精度等数值风险。
