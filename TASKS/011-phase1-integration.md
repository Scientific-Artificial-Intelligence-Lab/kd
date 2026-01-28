# Task: 011 - Phase 1 端到端集成验证

> Status: `ready`
> Parent: SPEC.md (Phase 1)
> Assignee: claude

## Goal

验证 Phase 1 里程碑——手动构建表达式、执行并评估，确保所有模块正确集成。

## Non-goals

- [ ] 不进行性能优化
- [ ] 不实现新功能
- [ ] 不编写用户文档

## Context

### Relevant files

```
tests/integration/test_phase1_e2e.py     # 新建
tests/integration/__init__.py            # 新建
examples/phase1_demo.py                  # 新建（可选）
```

### Current behavior

完成 Task 002-010 后，所有基础模块已就位。

### Dependencies

- Requires: 002-010 全部任务
- Blocks: Phase 2 开始

## Design

### Approach

1. 复现 arc_plan.md 中的验证代码
2. 验证 Burgers 方程系数恢复
3. 确保各模块正确集成

### 验证场景

**场景 1：字符串 → 执行**
```python
gen_ir = GenIR.from_string("add,mul,u,u_x,u_xx", library)
ast = to_analysis_ir(gen_ir, library)
result = executor.execute(ast)
# 验证：result.is_valid == True, result.value.shape 正确
```

**场景 2：DIRECT 评估**
```python
# 手动构建 RHS 表达式
# 评估 MSE(RHS, LHS)
eval_result = evaluator.evaluate_expression(ast, mode=DIRECT)
# 验证：eval_result.mse 合理
```

**场景 3：LINEAR 评估（核心）**
```python
# Burgers: u_t = -u*u_x + 0.1*u_xx
# terms = [mul(u, u_x), u_xx]
terms = [
    ASTNode("mul", (ASTNode("u"), ASTNode("u_x"))),
    ASTNode("u_xx")
]
eval_result = evaluator.evaluate_terms(terms, mode=LINEAR)
# 验证：coefficients ≈ [-1.0, 0.1]（相对误差 < 1%）
```

**场景 4：完整管线**
```python
# 字符串 → GenIR → AST → split_terms → evaluate_terms
gen_ir = GenIR.from_string("add,mul,u,u_x,u_xx", library)
ast = to_analysis_ir(gen_ir, library)
terms = split_terms(ast, library)
eval_result = evaluator.evaluate_terms(terms, mode=LINEAR)
# 验证：完整流程无错误
```

## Implementation steps

1. [ ] **Step 1**: 创建集成测试框架
   - Files: `tests/integration/__init__.py`, `tests/integration/test_phase1_e2e.py`
   - Test: 框架可运行
   - Agent: Tester

2. [ ] **Step 2**: 实现场景 1（字符串 → 执行）
   - Files: `tests/integration/test_phase1_e2e.py`
   - Test: `pytest tests/integration/test_phase1_e2e.py -k "test_string_to_execution"`
   - Agent: Tester

3. [ ] **Step 3**: 实现场景 2（DIRECT 评估）
   - Files: `tests/integration/test_phase1_e2e.py`
   - Test: `pytest tests/integration/test_phase1_e2e.py -k "test_direct_evaluation"`
   - Agent: Tester

4. [ ] **Step 4**: 实现场景 3（LINEAR 评估 + 系数恢复）
   - Files: `tests/integration/test_phase1_e2e.py`
   - Test: `pytest tests/integration/test_phase1_e2e.py -k "test_linear_coefficients"`
   - Agent: Tester

5. [ ] **Step 5**: 实现场景 4（完整管线）
   - Files: `tests/integration/test_phase1_e2e.py`
   - Test: `pytest tests/integration/test_phase1_e2e.py -k "test_full_pipeline"`
   - Agent: Tester

6. [ ] **Step 6**: 编写 smoke test
   - Files: `tests/smoke/test_phase1_smoke.py`
   - Test: `pytest -m smoke`
   - Agent: Tester

7. [ ] **Step 7**: （可选）创建 demo 脚本
   - Files: `examples/phase1_demo.py`
   - Agent: Dev

## Acceptance criteria

### Functional

- [ ] 验证代码运行无错误
- [ ] Burgers 系数恢复：`[-1.0, 0.1]`（相对误差 < 5%）
- [ ] R² > 0.99（在无噪声数据上）
- [ ] 完整管线各步骤输出类型正确

### Tests

- [ ] Integration tests pass: `pytest tests/integration/test_phase1_e2e.py -v`
- [ ] Smoke tests pass: `pytest -m smoke`

### Quality

- [ ] 测试代码有清晰注释
- [ ] 测试覆盖所有验证场景

## Validation

```bash
# Phase 1 完整验证
pytest tests/ -v --ignore=tests/benchmark

# Smoke tests（快速检查）
pytest -m smoke -v

# 仅集成测试
pytest tests/integration/ -v
```

## Constraints

- [ ] 测试数据使用合成 Burgers 数据
- [ ] 不依赖外部文件

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 系数恢复精度不足 | Med | Med | 检查数据质量，放宽容差 |
| 模块接口不匹配 | Low | High | 集成测试早期发现 |

## Rollback plan

```bash
# 如果集成测试失败，回滚到上一个稳定状态
git bisect start
git bisect bad HEAD
git bisect good <last-known-good-commit>
```

## Notes

_Phase 1 里程碑检验清单：_
1. IR 系统：GenIR ↔ AnalysisIR 双向转换
2. Library：默认算子 + diff token 自动注册
3. 数据：PDEDataset + FiniteDiffProvider
4. 执行：TreeExecutor 递归遍历
5. 评估：DIRECT + LINEAR 模式
6. 求解：LeastSquaresSolver 系数优化

_验证代码（来自 arc_plan.md）：_
```python
from kd2.core.ir import GenIR, to_analysis_ir
from kd2.core.executor import TreeExecutor
from kd2.core.evaluator import Evaluator

gen_ir = GenIR.from_string("add,mul,u,u_x,u_xx", library)
ast = to_analysis_ir(gen_ir, library)
result = executor.execute(ast)
eval_result = evaluator.evaluate_expression(ast)
```

---

## Completion checklist

Before marking as `done`:

- [ ] All implementation steps completed
- [ ] All acceptance criteria met
- [ ] Tests written and passing
- [ ] 验证代码可直接复制运行
- [ ] Phase 1 里程碑确认完成
- [ ] SPEC.md / arc_plan.md Phase 1 状态更新为 done
