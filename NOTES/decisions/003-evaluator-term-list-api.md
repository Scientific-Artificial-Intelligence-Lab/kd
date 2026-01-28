# Decision 003: Evaluator 以 term 列表为核心 API

> Date: 2026-01-28
> Status: accepted
> Task: Phase 1 (核心基础)

## Context

Phase 1 需要实现 Evaluator（评估器）。核心设计问题：

1. **LINEAR vs DIRECT**：Phase 1 是否包含 LINEAR 评估模式（split_terms → Θ 矩阵 → 最小二乘）？
2. **split_terms 归属**：在 AST 顶层 add/sub 处拆分为 term 列表的操作，应放在 Evaluator 内部还是 IR 层？
3. **Evaluator 的主入口**：接收"完整表达式"还是"term 列表"？

调研了 DISCOVER 和 SGA 的评估流程后做出以下决策。

## 调研发现

### DISCOVER 的评估流程

```
LSTM 生成完整 token 序列
  → STRidge.__init__() 调用 split_forest()
    → rebuild_tree() 建树
    → split_sum(root) 在 add/sub 处递归拆分
    → self.terms = [Node₁, Node₂, ...]
  → evaluate_terms() 逐个执行 term → results 列表
  → coef_calculate() → np.linalg.lstsq(Θ, ut)
```

DISCOVER 搜索**完整表达式**，split_terms 在 STRidge 内部完成。STRidge 类同时负责树拆分、term 执行、系数求解——职责过重。

### SGA 的评估流程

```
GA 生成 PDE.elements = [Tree₁, Tree₂, ...]  ← 天然就是 term 列表
  → evaluate_mse() 遍历 elements，逐层执行每个 tree
  → 拼接 context.default_terms（固定候选集）
  → Train() → STRidge 求系数
```

SGA 搜索空间本身就是 **term 集合**，不存在"完整表达式"概念，无需 split_terms。

### 对比

| | DISCOVER | SGA | PySR |
|---|---|---|---|
| 搜索单位 | 完整表达式 | term 集合 | 带系数的完整表达式 |
| split_terms | 需要（STRidge 内部做） | 不需要 | 不需要 |
| 评估模式 | LINEAR | LINEAR | DIRECT |

## Decision

### 1. Phase 1 包含 LINEAR 模式

**决定**：Phase 1 同时实现 LINEAR 和 DIRECT 两种评估模式。

**理由**：
- LINEAR 是 SGA/DISCOVER 的核心评估方式，是 PDE 发现的基本功能而非高级功能
- 所需额外组件不多：split_terms ~20 行，Θ 构建 ~10 行，lstsq ~5 行核心
- 如果推迟到 Phase 2，Phase 2 负担过重（约束系统 + STRidge + Lasso + AutogradProvider）
- 有 LINEAR 后，Phase 1 结束时能看到系数被自动恢复（如 Burgers 的 [-1.0, 0.1]），验证更有说服力

Phase 1 只实现 LeastSquares（`torch.linalg.lstsq`），STRidge/Lasso 留给 Phase 2。

### 2. Evaluator 核心 API 以 term 列表为输入

**决定**：Evaluator 的主入口接收 `List[ASTNode]`（term 列表），而非完整表达式。

```python
# 核心 API — 主入口
evaluate_terms(
    terms: List[ASTNode],
    context: ExecutionContext,
    target: Tensor,
) -> EvaluationResult

# 便捷 API — 语法糖
evaluate_expression(
    expr: ASTNode,
    context: ExecutionContext,
    target: Tensor,
    mode: EvaluationMode = LINEAR,
) -> EvaluationResult
    # LINEAR: split_terms(expr) → evaluate_terms(...)
    # DIRECT: execute(expr) → 直接算 MSE
```

| 方案 | 优点 | 缺点 |
|------|------|------|
| 表达式为主入口（DISCOVER 做法） | DISCOVER 插件直接传入 | SGA 需要假装把 terms join 成表达式再拆回来 |
| **term 列表为主入口（选择）** | SGA 直传；DISCOVER 调 split_terms 后传入；职责清晰 | DISCOVER 需要多调一步 split_terms |

**理由**：
- "term 列表"是所有 LINEAR 评估算法的公约数——SGA 天然产出 term 列表，DISCOVER 经 split 后也是
- DIRECT 模式是特例：等价于一个 term，系数固定为 1.0
- 避免 SGA 插件被迫拼接假表达式

### 3. split_terms 归属 IR 层，不归属 Evaluator

**决定**：`split_terms()` 是 AnalysisIR 上的结构操作工具函数，不是 Evaluator 的内部方法。

```python
# IR 层工具函数
def split_terms(ast: ASTNode) -> List[ASTNode]:
    """在 AST 顶层沿 add/sub 递归拆分为 term 列表"""
```

| 方案 | 优点 | 缺点 |
|------|------|------|
| 放在 Evaluator 内部（DISCOVER STRidge 做法） | 调用方便 | 把 IR 结构操作和数值评估耦合 |
| **放在 IR 层（选择）** | 职责清晰；IR 层操作和评估解耦；可被其他模块复用 | 调用方需要自己先 split |

**理由**：split_terms 只关心树结构（在哪里有 add/sub），不关心数值。放在 IR 层可被约束系统、可视化等其他模块复用。

## Consequences

- Phase 1 需要实现：`evaluate_terms()` + `evaluate_expression()` + `split_terms()` + `LeastSquaresSolver`
- Phase 1 不需要 STRidge/Lasso（Phase 2）
- Phase 1 验证代码可以手动构建 term 列表直接传入，不依赖 split_terms
- 插件接口 `propose()` 可以返回完整表达式或 term 列表，平台两种都能处理
- DISCOVER 的 STRidge 角色被解构为：split_terms（IR 层）+ evaluate_terms（Evaluator）+ SparseSolver（Solver 层）——职责分离
- `arc_plan.md` 和 `arc_core.md` 中的 Evaluator 接口需更新
