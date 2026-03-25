# Project Guidelines

## Collaboration Rules

- 先读代码再改代码，不凭猜测修改
- 每轮对话开始前，先读 RESEARCH.md 和 DESIGN.md 获取上下文
- 修改前说明意图和影响范围，等用户确认后再执行
- 不主动重构未被要求改动的代码
- commit 由用户显式要求时才执行

## Doc Sync (自动)

每轮代码改动完成后，自动同步以下文件，无需用户显式要求：

- `RESEARCH.md` — TODO 打勾、推进阶段、补充新发现的子任务
- `DESIGN.md` — 更新 Known Issues（划掉已修复 / 补充新发现）、记录架构变更到 Changelog
- `CLAUDE.md` — 如果阶段切换，更新 Current Focus

不改代码的纯讨论轮次不触发同步。

## File Deletion Policy

- **禁止使用 `rm` 删除任何文件** — `settings.local.json` 已通过 deny 规则强制执行
- 需要删除文件时，移入回收站：`mv <file> trash/`
- 同名文件加日期后缀避免覆盖：`mv foo.py trash/foo.py.20260321`
- `trash/` 已加入 `.gitignore`，不会被提交到 git
- `trash/` 由用户手动清理，Claude 不主动清空

## Bash Usage

- `cat`/`head`/`tail`/`grep`/`find` 等常用命令已在 `settings.local.json` 的 allow 列表中，可直接使用
- 内置工具 `Read`/`Grep`/`Glob` 同样可用，按场景选择即可

## Quick Reference

```bash
# Environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate kd-env

# Run tests
python -m pytest tests/ -x -q                    # all tests
python -m pytest tests/ -m smoke -q              # smoke only
python -m pytest tests/ -m "not slow" -q         # skip slow

# Install (editable)
pip install -e .
```

## Current Focus

Phase 4: 玻尔平台部署。代码已全部就绪并同步到两台开发机。CPU 侧方程渲染 (`_build_viz_proxy` + `_render_equation_fig`) 替代了 equation.png 文件搜索。ndarray 序列化 + SystemExit 保护 + 下载路径持久化 (`/data/outputs`) 均已完成。下一步：保存 CPU/GPU 镜像并发布 + 三模型全链路线上验证。

## Key Files

- `RESEARCH.md` — 项目目标、原则、非目标、当前阶段
- `DESIGN.md` — 系统架构、模块、数据流、约束
- `app.py` — Gradio Web UI 入口（玻尔平台部署用）
- `kd/job.py` — GPU 任务管理（提交/查询/下载）
- `kd/runner.py` — GPU 离线任务执行 + viz 数据序列化