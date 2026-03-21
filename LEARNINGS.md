# Learnings

Accumulated experience across Claude Code sessions.

## Environment & Tool Invocation

### 1. 使用 conda 环境路径

- **conda init**：`source /root/miniconda3/etc/profile.d/conda.sh`
- **环境名**：`kd-env`（Python 3.9）
- **激活**：`conda activate kd-env`
- **pytest**：`python -m pytest`
- **不使用 uv**：项目用 conda 管理环境，不要用 `uv run`
- **pip mirror**：清华镜像可能 SSL 故障，备用 `-i https://pypi.org/simple/`

### 2. `git stash` may revert pyproject.toml

- **Problem**: `git stash` restores pyproject.toml to its committed state, which may trigger a system-reminder notification about changed context
- **Note**: After `stash pop`, verify that the file is correctly restored
