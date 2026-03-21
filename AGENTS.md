# KD - Knowledge Discovery Framework

> PDE 方程发现与可视化框架

## Non-negotiables

- **不要修改 `docs/`** 除非用户明确要求
- **小步可 review 的变更**：总结改了什么、为什么
- **避免 "顺手重构"**：不做无关的清理
- **不要 git add 生成物**：`artifacts/`, `outputs/`, `model_save/`, `result_save/`

## 环境管理

- **工作流**：conda 管理环境 + `pip install -e .` 安装项目
- **环境定义**：`environment.yml`（锁定精确版本）
- **依赖声明**：`pyproject.toml`（兼容范围，不加版本上界，由 environment.yml 保证版本）
- **conda 环境路径**：`/Users/hao/miniconda3/envs/kd-env/`

```bash
# 运行命令
/Users/hao/miniconda3/envs/kd-env/bin/python -c "import kd"
/Users/hao/miniconda3/envs/kd-env/bin/python -m pytest tests/ -q
```

## Quick Start

```bash
# 安装（可编辑模式）
pip install -e .

# 运行测试
pytest tests/ -v

# 快速验证（提交前）
pytest -q

# Smoke tests
pytest -q tests/test_examples_smoke.py

# Viz tests (需要可选依赖)
pytest -q tests/test_viz_adapters.py
```

## 开发流程（TDD）

**严格遵循 TDD：先写测试，再写实现。**

1. **读取任务文档** — 如果指向 `TASKS/xxx/TASK.md`，先读取理解需求
2. **RED — 写失败的测试** — 根据 acceptance criteria 写测试，运行确认全部 FAIL
3. **GREEN — 最小实现** — 写刚好让测试通过的代码，不多不少
4. **验证** — 运行全量测试确认无回归：`pytest tests/ -q`
5. **总结** — 说明改了什么、为什么

## 项目结构

```
kd/
├── kd/                    # 主代码库
│   ├── model/             # 模型实现
│   │   ├── discover/      # Discover 发现算法
│   │   ├── sga/           # SGA 遗传算法
│   │   └── kd_*.py        # KD 封装层
│   ├── viz/               # 可视化模块
│   ├── utils/             # 工具函数
│   └── dataset/           # 数据集加载
├── examples/              # 用户示例
├── scripts/               # 开发者工具
├── tests/                 # 测试
│   └── manual_scripts/    # 手动可复现脚本
├── ref_lib/               # 参考实现（用于对比和研究）
├── notes/                 # 长期知识库
├── TASKS/                 # 任务包（活跃任务）
└── docs/                  # 公开文档（勿改）
```

## 任务管理

- 活跃任务放在 `TASKS/<task-name>/`
- 任务模板：`TASKS/TASK_SPEC.md`
- 结论和最终决定放在 `notes/`，不要埋在任务包里

## 开发规范

### 代码风格
- 所有 public 函数必须有 type hints
- 函数 < 50 行，文件 < 500 行
- 使用常量代替 magic numbers
- 不要用 `print()`，使用 logging

### 数值安全
- 除法必须有零保护：`safe_div(a, b)` 或 `np.divide(..., where=...)`
- 指数运算限制范围：`np.clip(x, -700, 700)`
- 关键路径检查：`np.isfinite(result)`

### 测试要求
- Line coverage >= 80%
- 必须测试数值边界情况 (NaN, Inf, 0)
- 移植代码需要 equivalence tests

## 关键入口

- **数据加载**: `kd.dataset.load_pde()`
- **模型**:
  - `kd/model/kd_dlga.py` - KD_DLGA
  - `kd/model/kd_sga.py` - KD_SGA
  - `kd/model/kd_discover.py` - KD_Discover, KD_Discover_SPR
  - `kd/model/kd_pysr.py` - KD_PySR (可选依赖)
- **可视化**: `kd/viz/`

详见 `REPO_MAP.md`。

## 参考实现

`ref_lib/` 包含参考实现，用于：
- 理解算法原理
- 验证新实现的正确性
- equivalence testing

## 常见问题

查看 `notes/patterns/common-mistakes.md` 了解历史上反复出现的问题模式。

## 语言

项目使用**中英混合**：技术术语用英文，说明用中文。
讨论和状态更新默认使用中文。
