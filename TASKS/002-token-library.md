# Task: 002 - Token & Library

> Status: `ready`
> Parent: SPEC.md (Phase 1)
> Assignee: claude

## Goal

实现 Token 数据结构和 Library 算子管理系统，作为整个 IR 系统的基石。

## Non-goals

- [ ] 不实现 open-form diff 算子（Phase 2）
- [ ] 不实现常数 token 优化（Phase 4）
- [ ] 不实现自定义算子注册 API（按需后续添加）

## Context

### Relevant files

```
src/kd2/core/ir/token.py       # 新建
src/kd2/core/library.py        # 新建
src/kd2/core/__init__.py       # 更新导出
tests/unit/test_token.py       # 新建
tests/unit/test_library.py     # 新建
```

### Current behavior

N/A - 新模块

### Dependencies

- Requires: 无（仅依赖已有的 safety.py）
- Blocks: 003 (GenIR), 004 (AnalysisIR), 008 (TreeExecutor)

## Design

### Approach

1. `Token` 作为 frozen dataclass，存储算子元信息（名称、元数、计算函数、类型）
2. `Library` 作为算子注册表，支持按名称查找、按元数过滤、自动注册 diff 系列 token
3. 默认算子集使用 safety.py 中的保护函数

### Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Token 可变性 | frozen=True | 不可变，可作为字典键 |
| diff 表示 | 一元 token 家族 | Decision 002: `diff_x`, `diff2_x` 作为一元 token |
| 函数签名 | `Callable[[Tensor, ...], Tensor]` | 统一 torch.Tensor 接口 |
| 默认算子 | safe_* 函数 | 数值安全第一 |

### API 设计

```python
class TokenType(Enum):
    FUNCTION = "function"    # add, mul, sin, ...
    VARIABLE = "variable"    # u, v, ...
    CONSTANT = "constant"    # 数值常数占位符
    DIFF = "diff"            # diff_x, diff2_x, ... (一元)

@dataclass(frozen=True)
class Token:
    name: str
    arity: int                                    # 0=terminal, 1=unary, 2=binary
    token_type: TokenType
    function: Optional[Callable[..., Tensor]]     # None for terminals
    is_commutative: bool = False                  # add, mul 为 True

class Library:
    def register(self, token: Token) -> None
    def get(self, name: str) -> Token
    def get_by_arity(self, arity: int) -> List[Token]
    def get_terminals(self) -> List[Token]
    def get_functions(self) -> List[Token]
    def register_diff_tokens(self, axes: List[str], fields: List[str], max_order: int) -> None
    def create_default() -> "Library"             # 类方法，创建默认库
```

## Implementation steps

1. [ ] **Step 1**: 实现 TokenType 和 Token
   - Files: `src/kd2/core/ir/token.py`
   - Test: `pytest tests/unit/test_token.py -k "test_token"`
   - Agent: Dev

2. [ ] **Step 2**: 实现 Library 基础功能
   - Files: `src/kd2/core/library.py`
   - Test: `pytest tests/unit/test_library.py -k "test_library_basic"`
   - Agent: Dev

3. [ ] **Step 3**: 实现默认算子集
   - Files: `src/kd2/core/library.py`
   - Test: `pytest tests/unit/test_library.py -k "test_default_operators"`
   - Agent: Dev

4. [ ] **Step 4**: 实现 diff token 自动注册
   - Files: `src/kd2/core/library.py`
   - Test: `pytest tests/unit/test_library.py -k "test_diff_registration"`
   - Agent: Dev

## Acceptance criteria

### Functional

- [ ] Token 创建正确，frozen 不可修改
- [ ] Library 注册、查找、过滤功能正常
- [ ] 默认算子集包含：add, sub, mul, div, sin, cos, exp, log, n2, n3
- [ ] diff 自动注册生成正确的一元 token（如 `diff_x`, `diff2_x`, `diff_t`）
- [ ] 可交换标记（is_commutative）对 add, mul 为 True

### Tests

- [ ] Unit tests pass: `pytest tests/unit/test_token.py tests/unit/test_library.py`
- [ ] Coverage >= 95%: `pytest --cov=src/kd2/core/ir/token --cov=src/kd2/core/library`

### Quality

- [ ] Type hints complete: `mypy src/kd2/core/ir/token.py src/kd2/core/library.py`
- [ ] No lint errors: `ruff check src/kd2/core/`
- [ ] 算子函数使用 safe_div, safe_exp, safe_log

## Validation

```bash
# Minimal validation
pytest tests/unit/test_token.py tests/unit/test_library.py -v

# Full validation
pytest tests/unit/ -v --cov=src/kd2/core
mypy src/kd2/core
ruff check src/kd2/core
```

## Constraints

- [ ] 无破坏性变更
- [ ] Token 必须 hashable（用于缓存键）

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| diff token 命名冲突 | Low | Med | 使用 `diff_` 前缀，`diff{order}_{axis}` 格式 |

## Rollback plan

```bash
git revert <commit-hash>
```

## Notes

_实现时注意：_
- diff token 的 function 字段为 None（由 Executor 特殊处理）
- 常数 token 暂用占位符，Phase 4 再完善

---

## Completion checklist

Before marking as `done`:

- [ ] All implementation steps completed
- [ ] All acceptance criteria met
- [ ] Tests written and passing
- [ ] Code reviewed (`/code-review`)
- [ ] No TODOs left in code
- [ ] `/wrap-up` completed
