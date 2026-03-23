# Logic Completeness Review: `_SubscriptLatexPrinter._print_Derivative()`

**Reviewer**: reviewer-logic
**Scope**: `kd/viz/discover_eq2latex.py:41-58` + `tests/test_discover_notation.py`

---

## Summary
- 结论：**APPROVE**
- High: 0, Medium: 1, Low: 2

修改的三分支条件逻辑能正确覆盖 SymPy 中 `Derivative.args[0]` 的所有实际可能类型。`else` 分支通过 `self._print(func)` 兜底所有复合表达式。没有发现会导致崩溃或错误结果的逻辑遗漏。

---

## High Priority

无。

---

## Medium Priority

### 1. **`var.name` 在 `variable_count` 循环中缺少 `_SYMBOL_DISPLAY` fallback 的显式说明** @ `discover_eq2latex.py:54`

- **场景**：如果 derivative 的变量是一个不在 `_SYMBOL_DISPLAY` 中的 Symbol（例如 `t`、`r`、或自定义变量），`_SYMBOL_DISPLAY.get(var.name, var.name)` 会 fallback 到原始名称，这是正确的。但 `var_name * count` 会产生重复字符串（如 `"tt"`），这对用户来说可能不是最直观的表示。
- **后果**：不会崩溃，输出语义上正确，但对于非标准变量名可能产生 `u_{tt}` 而非 `u_{t^2}` 之类的歧义（当然对 PDE 社区来说 `u_{tt}` 是常规写法，所以实际上这是合理的）。
- **结论**：这不是 bug，`str * int` 对于 subscript notation 是正确的 PDE 惯例。但如果未来需要支持非单字符变量名（如 `theta`），`var_name * count` 会产生 `"thetatheta"` 而非 `"theta^2"`。当前项目变量名都是单字符映射，所以暂无问题。
- **建议**：可以作为 TODO 标记，不阻塞本次修改。

---

## Low Priority

### 1. **测试未覆盖变量不在 `_SYMBOL_DISPLAY` 中的 Derivative 场景** @ `tests/test_discover_notation.py`

- **场景**：例如 `Derivative(u1, t)` 其中 `t` 不在 `_SYMBOL_DISPLAY` 中。
- **后果**：当前 fallback `var.name` 是正确的，不会崩溃。但缺少测试意味着 future regression 不会被检测到。
- **建议**：添加一个如下测试：
  ```python
  def test_derivative_unknown_var(self):
      """Derivative wrt unknown var falls back to raw name."""
      u1 = Function("u1")(Symbol("t"))
      expr = Derivative(u1, Symbol("t"))
      result = self._render(expr)
      assert result == "u_{t}"
  ```

### 2. **缺少 `Number` 作为 Derivative func 的测试** @ `tests/test_discover_notation.py`

- **场景**：`Derivative(Integer(5), x1)` — 数学上无意义（结果为 0），但 SymPy 允许构造。
- **后果**：SymPy 会自动求值为 `0`（整数常数的导数为零），所以 `_print_Derivative` 不会被调用。不会触发任何 bug。
- **结论**：不需要测试。仅作为"已分析但不需要处理"的记录。

---

## Branch Coverage Analysis

### `_print_Derivative` (lines 41-58)

| Branch | 条件 | 覆盖 | 分析 |
|--------|------|------|------|
| 1 | `isinstance(func, sympy.Symbol)` | 测试覆盖 (`test_derivative_of_pow` 等直接传 Symbol 进 Derivative) | OK |
| 2 | `hasattr(func, 'func') and hasattr(func.func, 'name')` | 测试覆盖 (`test_first_order_derivative_u_x` 等用 `Function("u1")(x1, x2)`) | OK |
| 3 | `else` (compound expr) | 测试覆盖 (`test_derivative_of_pow`, `test_derivative_of_mul`, `test_derivative_of_add`) | OK |

**所有可能的 `expr.args[0]` 类型分析**：

| `func` 类型 | 走哪个分支 | 在项目中是否实际产生 | 测试覆盖 |
|-------------|-----------|-------------------|---------|
| `Symbol` (如 `u1`) | Branch 1 | 是（`diff(u1, x1)` → `Derivative(u1, x1)`） | 是 |
| `AppliedUndef` (如 `u1(x1, x2)`) | Branch 2 | 是（leibniz mode 用 `Function('u1')(x1, x2, x3)`） | 是 |
| `Pow` (如 `u1**2`) | Branch 3 | 是（`diff(n2(u1), x1)` → `Derivative(Pow(u1, 2), x1)`） | 是 |
| `Mul` (如 `u1*u1`) | Branch 3 | 是（`diff(mul(u1, u1), x1)`） | 是 |
| `Add` (如 `u1 + u1**2`) | Branch 3 | 是（`diff(add(n2(u1), u1), x1)`） | 是 |
| `Derivative` (嵌套导数) | Branch 3 | 可能（`diff(diff(u1, x1), x1)`，但 SymPy 常自动合并为二阶） | 是（`test_nested_diff_diff_u1_x1_x1`） |
| `exp/sin/cos/...` (SymPy 函数) | Branch 2 (有 `.func.name`) | 可能（`Derivative(sin(u1), x1)`），但项目 Node 系统不生成此结构 | 否（合理忽略） |
| `Number/Integer/Rational` | 不会到达 | SymPy 自动求值为 0 | N/A |
| `Piecewise/MatrixExpr/Integral` | Branch 3 | 项目中不产生 | N/A |

**条件优先级分析**：

- Branch 1 (`isinstance Symbol`) 和 Branch 2 (`hasattr func + name`) 互斥：`Symbol` 没有 `.func` attribute（它有 `.name` 直接属性但没有 `.func`），所以不会错误匹配 Branch 2。
- `AppliedUndef`（如 `u1(x1, x2)`）不是 `Symbol` 的实例，所以不会匹配 Branch 1。
- 条件顺序正确，不存在一个类型同时满足多个条件的情况。

### `variable_count` 循环 (lines 52-55)

| 检查项 | 结果 |
|--------|------|
| `expr.variable_count` 是否总存在 | 是：这是 SymPy `Derivative` 的内置属性，返回 `((var, count), ...)` |
| `var` 是否总是 `Symbol` | 是：SymPy `Derivative` 的 differentiation variable 必须是 `Symbol` |
| `var.name` 是否总存在 | 是：所有 `Symbol` 都有 `.name` |
| `count` 是否总是正整数 | 是：SymPy 保证 count >= 1 |
| `var_name * count` 是否安全 | 是：`str * int` 对正整数安全 |

---

## Hidden Assumptions

| 假设 | 合理？ |
|------|--------|
| `expr.args[0]` 是 Derivative 中被求导的表达式 | 是（SymPy API 保证） |
| `expr.variable_count` 返回 `((Symbol, int), ...)` | 是（SymPy API 保证） |
| `_SYMBOL_DISPLAY` 覆盖所有实际使用的变量名 | 是（项目变量固定为 u1, x1, x2, x3, p1, p2, p3, c） |
| `self._print(func)` 对任何 SymPy 表达式返回有效 LaTeX | 是（SymPy LatexPrinter 的基础保证，不会崩溃） |
| 变量名在 `_SYMBOL_DISPLAY` 中是单字符映射 | 是（当前 "u", "x", "y", "z"）。`str * count` 依赖此假设 |

---

## Over-Engineering

无发现。修改是最小化的：
- 只增加了一个 `else` 分支和一个 `AppliedUndef` 条件
- 没有引入新的抽象或辅助函数
- 修改范围严格限于 `_print_Derivative` 方法

---

## 历史教训匹配

检查了 `notes/patterns/common-mistakes.md` 中的 10 个已知模式：

| 模式 | 是否相关 | 分析 |
|------|---------|------|
| #7: try/except 吞错误 | 部分相关 | `_discover_term_node_to_latex` 第 94 行有宽泛的 `except Exception`，但这是 **已有代码**，不在本次修改范围内。该 except 会 log error 并返回 `\text{Error...}`，测试中有断言 `"Error" not in result`，所以不是静默吞掉。可接受。 |
| #4: 测试未覆盖边界 | 轻微相关 | 测试覆盖了主要的 Pow/Mul/Add 场景，缺少未知变量名的测试（见 Low #1）。 |
| 其他 8 个模式 | 不相关 | 本次修改不涉及数值计算、维度扩展、ref_lib 移植等。 |
