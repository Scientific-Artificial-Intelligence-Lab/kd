# Learnings

Accumulated experience across Claude Code sessions.

## Environment & Tool Invocation

### 1. 使用 conda 环境路径

- **环境路径**：`/Users/hao/miniconda3/envs/kd-env/`
- **Python**：`/Users/hao/miniconda3/envs/kd-env/bin/python`
- **pytest**：`/Users/hao/miniconda3/envs/kd-env/bin/python -m pytest`
- **不使用 uv**：项目用 conda 管理环境，不要用 `uv run`

### 2. `git stash` may revert pyproject.toml

- **Problem**: `git stash` restores pyproject.toml to its committed state, which may trigger a system-reminder notification about changed context
- **Note**: After `stash pop`, verify that the file is correctly restored
