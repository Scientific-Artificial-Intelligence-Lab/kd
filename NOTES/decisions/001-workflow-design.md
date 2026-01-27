# Decision 001: 多 Agent 工作流设计

> Date: 2026-01-27
> Status: accepted
> Task: TASKS/001-workflow-setup.md

## Context

kd2 项目需要建立多 Agent 协作的 TDD 开发工作流。Task 000 已创建基础 .claude/ 配置（8 agents, 7 commands），现需根据实际需求调整。

## Decisions

### 1. NOTES/ 目录结构

**决定**: 现有 arc_*.md 移入 `NOTES/architecture/`，新增三个子目录。

```
NOTES/
├── index.md
├── architecture/    # arc_*.md 迁移至此
├── decisions/       # 技术决策记录
├── concepts/        # 概念解释
└── explorations/    # 探索性研究
```

**理由**: 现有文件是一次性架构设计产物，新子目录是持续性开发记录，性质不同。

### 2. Hooks 策略

**决定**: 适度方案 — `src/` 文件变更时跑 `ruff check`，不 auto-fix，不跑 pytest。

**理由**: Phase 0 无代码，pytest 无意义。Ruff 提供即时 lint 反馈，不阻塞操作。

### 3. Tester 权限

**决定**: 务实 TDD — Tester 可读 `src/`，但写测试时优先基于接口。

**理由**: 科学计算项目需要理解数值行为；等价性测试需对照 ref_libs/ 和 src/。

### 4. Teacher 权限

**决定**: 允许写示例代码（放 NOTES/ 或 examples/），不写生产代码。

**理由**: 科学计算概念（prefix notation IR、CSE 等）用代码示例解释更清晰。

### 5. Agent 角色（7 个）

**决定**: 从 8 个精简合并为 6 个，后追加 MentorDev 共 7 个。

| 角色 | 常驻/按需 | 合并来源 | 职责 |
|------|----------|---------|------|
| Architect | 常驻 | architect + planner | 设计决策、架构规划、任务拆解 |
| Tester | 常驻 | tdd-guide + integration-runner | TDD、单元/集成/等价性测试 |
| Dev | 常驻 | 新增 | 写功能代码、实现新功能 |
| Porter | 常驻 | porter | 算法移植（ref_libs/ → src/） |
| Teacher | 按需 | mentor | 教学解释、示例代码 |
| Researcher | 按需 | researcher | 研究参考实现、算法分析 |
| MentorDev | 按需 | 新增 | 手把手教学开发，一次写一个函数，逐行讲解含语法 |

**降级**: code-reviewer → slash command `/code-review`。

**理由**: 减少 session 管理负担；Dev 和 Porter 分开因为算法移植（TF1.x → PyTorch）足够专业化。MentorDev 面向用户学习需求，与 Dev（高效）和 Teacher（不写生产代码）互补。

**弹性原则**: 所有 agent 的默认流程和参数可由用户随时调整。Agent 遇到不确定情况时应主动询问用户。

### 6. 终端管理

**决定**: 两者都提供 — 启动文档 + 可选 tmux 脚本。

**理由**: 灵活适应不同工作习惯。

### 7. Slash Commands（10 个）

**决定**: 保留 5 + 合并 2 + 新增 3。

```
# 任务生命周期
start-task     # 开始任务（含 Teacher 概念讲解）[替代 plan]
wrap-up        # 结束任务（含 Teacher 代码讲解）[新增]
record         # 记录决策到 decisions/ [新增]

# 开发流程
tdd            # TDD 流程 [合并 tdd-step]
code-review    # 代码审查 [保留]
integration    # 集成测试 [保留]
test-coverage  # 覆盖率分析 [保留]

# 知识管理
explain        # 概念讲解 [新增]
learn          # 从参考代码提取模式 [保留]
port           # 算法移植工作流 [保留]
```

### 8. Teacher 自动触发

**决定**: Teacher 讲解融入任务生命周期命令。

- `/start-task`: 开始时自动包含概念讲解
- `/wrap-up`: 结束时自动包含代码讲解

**理由**: "理论先行，实践复盘" — 学习效果最佳。

## Consequences

- NOTES/ 中现有文档的路径变化，需更新引用（CLAUDE.md, AGENTS.md, SPEC.md）
- .claude/agents/ 从 8 个变为 6 个，需删除旧文件
- .claude/commands/ 从 7 个变为 10 个，需删除 plan.md 和旧 tdd.md
- 需创建 tmux 脚本和启动文档
