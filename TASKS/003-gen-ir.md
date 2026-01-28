# Task: 003 - GenIR

> Status: `ready`
> Parent: SPEC.md (Phase 1)
> Assignee: claude

## Goal

实现前缀 Token 序列表示（GenIR），作为表达式的 canonical 表示和缓存键。

## Non-goals

- [ ] 不实现表达式变异/交叉操作（插件层职责）
- [ ] 不实现 ExecIR（Phase 5 优化）
- [ ] 不实现字符串美化输出（如中缀表示）

## Context

### Relevant files

```
src/kd2/core/ir/gen_ir.py      # 新建
tests/unit/test_gen_ir.py      # 新建
```

### Current behavior

N/A - 新模块

### Dependencies

- Requires: 002 (Token, Library)
- Blocks: 005 (IR Converters)

## Design

### Approach

GenIR 是前缀 token 序列的 frozen 包装，核心职责：
1. 从字符串解析（`from_string`）
2. 检查完整性（`is_complete`）
3. 作为 hashable 缓存键

### Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| 存储格式 | `tuple[str, ...]` | 不可变，hashable |
| 完整性检查 | 基于 arity 的 dangling 计数 | O(n) 遍历 |
| canonical 化 | 由 Converter 负责 | GenIR 只存储原始序列 |

### API 设计

```python
@dataclass(frozen=True)
class GenIR:
    tokens: tuple[str, ...]

    @classmethod
    def from_string(cls, s: str, library: Library) -> "GenIR":
        """解析逗号分隔的 token 序列"""

    def to_string(self) -> str:
        """返回逗号分隔的字符串"""

    def is_complete(self, library: Library) -> bool:
        """检查是否是完整表达式（无 dangling）"""

    def dangling(self, library: Library) -> int:
        """返回 dangling 槽位数（0 表示完整）"""

    def __hash__(self) -> int:
        """基于 tokens tuple 的 hash"""

    def __len__(self) -> int:
        """token 数量"""
```

### 完整性检查算法

```
dangling = 1  # 初始需要一个表达式
for token in tokens:
    dangling -= 1        # 消耗一个槽位
    dangling += arity    # 产生 arity 个新槽位
    if dangling < 0:     # 早期终止
        raise ValueError
return dangling == 0
```

## Implementation steps

1. [ ] **Step 1**: 实现 GenIR 基础结构
   - Files: `src/kd2/core/ir/gen_ir.py`
   - Test: `pytest tests/unit/test_gen_ir.py -k "test_gen_ir_basic"`
   - Agent: Dev

2. [ ] **Step 2**: 实现 from_string 解析
   - Files: `src/kd2/core/ir/gen_ir.py`
   - Test: `pytest tests/unit/test_gen_ir.py -k "test_from_string"`
   - Agent: Dev

3. [ ] **Step 3**: 实现完整性检查
   - Files: `src/kd2/core/ir/gen_ir.py`
   - Test: `pytest tests/unit/test_gen_ir.py -k "test_completeness"`
   - Agent: Dev

4. [ ] **Step 4**: 实现 hash 和相等性
   - Files: `src/kd2/core/ir/gen_ir.py`
   - Test: `pytest tests/unit/test_gen_ir.py -k "test_hash"`
   - Agent: Dev

## Acceptance criteria

### Functional

- [ ] `from_string("add,u,v", library)` 正确解析
- [ ] `is_complete()` 正确识别完整/不完整表达式
- [ ] `dangling()` 返回正确的槽位数
- [ ] 相同 tokens 的 GenIR hash 相同
- [ ] 不同 tokens 的 GenIR hash 不同（高概率）

### Tests

- [ ] Unit tests pass: `pytest tests/unit/test_gen_ir.py`
- [ ] Coverage >= 95%

### Quality

- [ ] Type hints complete: `mypy src/kd2/core/ir/gen_ir.py`
- [ ] No lint errors: `ruff check src/kd2/core/ir/`

## Validation

```bash
# Minimal validation
pytest tests/unit/test_gen_ir.py -v

# Full validation
pytest tests/unit/ -v --cov=src/kd2/core/ir
mypy src/kd2/core/ir
ruff check src/kd2/core/ir
```

## Constraints

- [ ] GenIR 必须 frozen 和 hashable
- [ ] 解析时验证 token 存在于 Library

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 未知 token | Med | Low | 解析时抛出明确 ValueError |

## Rollback plan

```bash
git revert <commit-hash>
```

## Notes

_测试用例示例：_
- 完整：`"add,u,v"`, `"mul,add,u,v,w"`, `"sin,u"`
- 不完整：`"add,u"`, `"mul,u"`, `"add"`
- 无效：`"unknown_token,u,v"`

---

## Completion checklist

Before marking as `done`:

- [ ] All implementation steps completed
- [ ] All acceptance criteria met
- [ ] Tests written and passing
- [ ] Code reviewed (`/code-review`)
- [ ] No TODOs left in code
- [ ] `/wrap-up` completed
