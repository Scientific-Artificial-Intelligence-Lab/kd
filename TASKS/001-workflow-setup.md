# Task: 001 - 重构多 Agent 工作流与项目文档结构

> Status: `ready`
> Parent: SPEC.md Phase 1
> Assignee: claude (Architect session)

## Goal

建立支持 4-5 个 Agent 协作的 TDD 开发工作流，包含持久化记忆系统和自动化命令。

## Non-goals

- [ ] 不修改任何 src/ 代码
- [ ] 不改变 SPEC.md 的技术架构
- [ ] 不实现复杂的自动化编排（保持人工协调）

## Context

### Relevant files
```
CLAUDE.md
AGENTS.md
NOTES/
.claude/
TASKS/
```

### Current behavior

- 单 session 开发
- 无标准化的决策记录
- 无 slash commands
- 无 hooks 自动化

### Dependencies

- Requires: Task 000 (项目基础配置) ✅
- Blocks: 所有后续开发任务

## Design

### Approach

1. 重构 NOTES/ 目录，增加 decisions/, concepts/, explorations/
2. 配置 .claude/ 目录：commands, hooks, agent prompts
3. 更新 CLAUDE.md 和 AGENTS.md 反映新工作流
4. 创建 tmux 启动脚本方便开多 session

### Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Agent 数量 | 4 常驻 + 1 按需 | Architect/Tester/Dev/Teacher 常驻，Researcher 按需 |
| 协调方式 | 人工 + 文件系统 | 避免过度自动化，保持可控 |
| 记忆持久化 | NOTES/ 子目录 | 与现有文档体系一致 |
| Subagent 使用 | 仅简单任务 | 核心算法需要完整 session 能力 |

## Implementation steps

1. [ ] **Step 1**: 创建 NOTES/ 子目录结构和模板
   - Files: `NOTES/decisions/`, `NOTES/concepts/`, `NOTES/explorations/`
   - Test: 目录存在，index.md 可读

2. [ ] **Step 2**: 配置 .claude/commands/ slash commands
   - Files: `.claude/commands/start-task.md`, `tdd-step.md`, `record.md`, `explain.md`
   - Test: `/start-task` 命令可用

3. [ ] **Step 3**: 配置 .claude/settings.json hooks
   - Files: `.claude/settings.json`
   - Test: 文件修改后自动跑 pytest

4. [ ] **Step 4**: 创建 Agent 启动 prompts
   - Files: `.claude/agents/architect.md`, `tester.md`, `dev.md`, `teacher.md`, `researcher.md`
   - Test: prompts 清晰完整

5. [ ] **Step 5**: 创建 tmux 启动脚本
   - Files: `scripts/start-dev.sh`
   - Test: `./scripts/start-dev.sh` 能开 4 个 session

6. [ ] **Step 6**: 更新 CLAUDE.md 和 AGENTS.md
   - Files: `CLAUDE.md`, `AGENTS.md`
   - Test: 新 session 能根据文档理解工作流

7. [ ] **Step 7**: 更新 Task 模板，增加 Agent 协作说明
   - Files: `TASKS/TEMPLATE.md`
   - Test: 模板包含 agent 分工指引

## Acceptance criteria

### Functional

- [ ] 4 个 Agent session 能独立工作
- [ ] Agent 间通过文件系统通信
- [ ] 决策能被记录并被新 session 读取
- [ ] Slash commands 正常工作

### Structure
```
NOTES/
├── index.md
├── decisions/
│   ├── index.md
│   └── TEMPLATE.md
├── concepts/
│   ├── index.md
│   └── TEMPLATE.md
└── explorations/
    ├── index.md
    └── TEMPLATE.md

.claude/
├── settings.json
├── commands/
│   ├── start-task.md
│   ├── tdd-step.md
│   ├── record.md
│   └── explain.md
└── agents/
    ├── architect.md
    ├── tester.md
    ├── dev.md
    ├── teacher.md
    └── researcher.md

scripts/
└── start-dev.sh
```

### Quality

- [ ] 所有 markdown 格式正确
- [ ] Shell 脚本可执行
- [ ] 无冗余或重复内容

## Validation
```bash
# 结构验证
ls NOTES/decisions NOTES/concepts NOTES/explorations
ls .claude/commands .claude/agents
cat .claude/settings.json | jq .

# 功能验证
chmod +x scripts/start-dev.sh
./scripts/start-dev.sh  # 应开启 tmux 多窗口

# 在新 claude session 中测试
# /start-task TASKS/001-workflow-setup.md
```

## Constraints

- [ ] 不引入额外依赖
- [ ] 保持与现有 NOTES/ 内容兼容
- [ ] Agent prompts 控制在 50 行以内

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 工作流过于复杂 | Med | High | 先最小化实现，迭代优化 |
| Hooks 干扰正常开发 | Low | Med | Hooks 只做通知，不阻塞 |
| Agent 间信息不同步 | Med | Med | 每个 task 开始时强制读 decisions/ |

## Rollback plan
```bash
# 保守回滚
git checkout HEAD~1 -- .claude/ NOTES/ CLAUDE.md AGENTS.md

# 或删除新增内容，保留原有
rm -rf NOTES/decisions NOTES/concepts NOTES/explorations
rm -rf .claude/commands .claude/agents
```

## Notes

### 讨论要点（与 Architect 确认）

1. **NOTES/ 现有内容迁移**:
   - 现有 3.6K 行内容是否需要重组？
   - 还是只新增子目录，现有内容不动？

2. **Hooks 激进程度**:
   - 每次改文件都跑 pytest? 还是只在特定目录？
   - 是否需要自动 ruff fix?

3. **Agent Prompt 细节**:
   - Tester 是否允许看 src/ 实现？（纯 TDD 应该不看）
   - Teacher 是否允许写代码示例？

4. **tmux vs 手动开 terminal**:
   - 用户偏好 tmux 还是手动管理？

---

## Completion checklist

Before marking as `done`:

- [ ] 所有目录和文件创建完成
- [ ] Slash commands 在新 session 中测试通过
- [ ] Hooks 功能验证
- [ ] 4 个 agent 各开一个 session 测试
- [ ] CLAUDE.md 更新，新 session 能理解整个工作流
- [ ] 与用户确认工作流合理性