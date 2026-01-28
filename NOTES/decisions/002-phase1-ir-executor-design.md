# Decision 002: Phase 1 IR 与执行器设计

> Date: 2026-01-28
> Status: accepted
> Task: Phase 1 (核心基础)

## Context

Phase 1 需要实现 IR 系统和执行器。架构文档（`arc_core.md`, `arc_plan.md`）已定义了两层 IR（GenIR + AnalysisIR）和执行器接口，但以下三个实现细节需要明确：

1. AnalysisIR（AST 树）应该是 mutable 还是 immutable？
2. `diff` 算子应该用 arity=2（`diff(expr, axis)`）还是一族一元算子（`diff_x(expr)`）？
3. 执行器应该用栈式（在 GenIR token 序列上）还是树遍历（在 AnalysisIR AST 上）？

调研了三个参考实现（DISCOVER, SGA, AutoKE）的做法后做出以下决策。

## Decisions

### 1. AnalysisIR 为 Immutable

**决定**: `ASTNode` 使用 `frozen=True`，`children` 用 `tuple` 而非 `list`。

```python
@dataclass(frozen=True)
class ASTNode:
    token: Token
    children: tuple["ASTNode", ...]  # immutable
```

| 方案 | 优点 | 缺点 |
|------|------|------|
| Mutable（SGA 做法） | GA 变异可原地修改 | 需要 deep copy 防执行污染；缓存不安全；第三方插件可能误改 |
| **Immutable（选择）** | 安全、可缓存、线程安全、对开源 infra 更健壮 | GA 变异需创建新树（但 AST 通常 <20 节点，开销可忽略） |

**调研依据**:
- SGA: mutable tree，执行时把中间结果存在 `node.cache` 里，每次执行前需 `deepcopy`，忘了重置会出 bug
- DISCOVER: token 序列是真相，树是临时构建的只读视图，用完丢弃
- AutoKE: AST 只读，纯解析+执行

**关键原则**: kd2 是平台，AnalysisIR 是平台层数据结构。平台只做只读操作（执行、评估、约束检查）。GA 变异是算法插件内部的事，插件可在内部维护 mutable 结构，提交给平台时转为 immutable AnalysisIR。

### 2. diff 为一族一元算子 + 自动注册

**决定**: 不使用 `diff(expr, axis)` (arity=2)，改用 `diff_x(expr)`, `diff2_x(expr)` 等 (arity=1)。根据数据集坐标轴自动注册。

```python
# 根据数据集自动生成
# axes=["x", "t"], max_order=3 → diff_x, diff2_x, diff3_x, diff_t, diff2_t, diff3_t
```

| 方案 | 优点 | 缺点 |
|------|------|------|
| arity=2（DISCOVER/SGA 标准做法） | 新坐标轴自动可用 | 执行器/约束系统需大量特殊分支 `if 'diff' in name` |
| **一族一元算子（选择）** | 执行器完全统一，无特殊分支；约束简化；代码更干净 | 新坐标轴需注册新算子（通过自动注册解决） |

**调研依据**:
- DISCOVER 自身在 subgrid 模式下已改用一元算子（`ddx`, `ddy`, `laplacian`，arity=1）
- DISCOVER 标准模式的 arity=2 导致执行器中大量 `if 'diff' in token.name` 特殊处理
- SGA 的 `pde.py` 中 `d`/`d^2` 有 ~20 行专门的特殊逻辑
- 对搜索算法透明：LSTM/GA 都是从 token 库中选 token，不关心 token 名字

**自动注册机制**: Library 构建时根据 `PDEDataset.axes` 和配置的 `max_diff_order` 自动生成 diff token，无需硬编码。

### 3. 递归树遍历执行器 (TreeExecutor) + 全程 torch.Tensor

**决定**: Phase 1 实现 `TreeExecutor`，递归遍历 AnalysisIR AST，每个节点调用对应的 `torch` 函数。不使用 NumPy。

```python
class TreeExecutor:
    def execute(self, node: ASTNode, context: ExecutionContext) -> Tensor:
        if node.token.arity == 0:
            return self._resolve_terminal(node.token, context)
        child_values = [self.execute(c, context) for c in node.children]
        return node.token.function(*child_values)
```

| 方案 | 优点 | 缺点 |
|------|------|------|
| 栈式 on GenIR（DISCOVER 做法） | 不需建树；O(n) 遍历 | 需处理 diff 特殊逻辑；不如树遍历直观 |
| 自底向上 on 按层树（SGA 做法） | 无递归深度限制 | SGA 特有数据结构，不通用；mutable cache 易出 bug |
| **递归树遍历 on AST（选择）** | 最直观（~10 行核心代码）；所有算子统一路径；torch 原生 | 递归深度受限（Python 默认 1000，表达式深度通常 <15，无问题） |

**调研依据**:
- AutoKE 的 `compute()` 就是递归树遍历 + torch，代码最简洁
- DISCOVER 的栈式执行器为处理 diff 引入了 `dim_flag` 全局状态和字符串匹配特殊分支
- 已有 AnalysisIR（AST），在其上递归遍历是最自然的选择

**关于 `arc_plan.md` 的 "StackExecutor (NumPy)"**: 这是早期草案遗留，与"全程 torch.Tensor"的架构决策矛盾。本决策将其更正为 TreeExecutor + torch。

**未来优化**: 栈式执行器可作为 Phase 5 性能优化（跳过 AST 构建，直接从 GenIR 执行），但 Phase 1 不需要。

## Consequences

- `arc_plan.md` 中 Phase 1 的 "StackExecutor (NumPy)" 应更正为 "TreeExecutor (torch)"
- `arc_core.md` 中 AnalysisIR 的 `children: List` 应改为 `children: Tuple`
- `arc_core.md` 中 diff/diff2 的 Token 定义需从 arity=2 改为一族 arity=1 Token + 自动注册说明
- 约束系统简化：不再需要 `DiffArgumentConstraint`（禁止 `diff(terminal, axis)` 的别名约束），因为 `diff_x` 天然只接受一个表达式参数
- 执行器代码量大幅减少：所有算子（包括 diff）走统一执行路径，无特殊分支
- 三个决策互相配合：immutable AST → 递归遍历安全；一元 diff → 执行器统一；torch 全程 → diff 内部可直接调用 autograd
