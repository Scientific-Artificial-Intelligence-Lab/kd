# Agent collaboration guide

This file exists to help Codex CLI (and humans) work effectively in this repo.

## Non-negotiables

- Do not modify anything under `data/` and `ref_libs` unless the user explicitly asks.
- Prefer small, reviewable changes. Summarize what changed and why.
- Avoid “drive-by refactors” and unrelated cleanups.
- Use TDD style, write many test_* under ./tests/*

## Communication style (default)

- Use Chinese for discussion and status updates unless the user requests English.
- Be pragmatic: optimize for maintainability and predictable behavior.

## Environment


## Where things live

- Code: `./src/`
- Examples for users: `./examples/`
- Reference projects (partial), view only when I ask: `./ref_libs/`
- QA / verify / benchmark scripts (not shipped): TBD
- Tests: `tests/` 
- Internal long-lived notes: `NOTES/`
- Work-in-progress plans / task packets / Future work: `TASKS/task*.md`

## Task workflow (recommended)

- Use `TASKS/TEMPLATE.md` as the template for a task packet (goal, constraints, acceptance, commands).
- Keep conclusions and “final decisions” in `NOTES/`.
- MOVE the finished work form `TASKS/` to `ARCHIVE/`


Before executing any changes:

Read TODO.md or the current task packet (if available).
First, provide 3–7 planned steps, and list the files expected to be modified.
Clarify: In-scope / Out-of-scope for this change.
Ensure small iterations; do not add hundreds of lines of code at once. Only proceed to the next step after passing tests and review.

实现时：
- 强制 TDD：若行为/接口有变化，先补测试或同步更新测试
- 如需跨文件重构核心算法：说明原因、风险, 补充足够测试
- 代码必须有注释

完成时（必须输出）：
- **变更摘要**：改了什么、为什么改, 改之后和之前的区别
- **验证结果**：运行了哪些命令，结果如何（成功/失败与失败原因）
