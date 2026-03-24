# System Design

## Architecture Overview

```
用户数据 (numpy/csv/mat)
       ↓
   PDEDataset (统一容器, 2D/N-D)
       ↓
   Adapter (SGA/DLGA/DSCV 各自适配)
       ↓
   Discovery Engine (GA/RL/PINN)
       ↓
   符号方程 (LaTeX / SymPy)
       ↓
   VizRequest → render() → VizResult (图片/文件)
```

## Module Map

| Module | File | Role |
|--------|------|------|
| BaseEstimator | `kd/base.py` | 参数管理基类 (get_params/set_params) |
| PDEDataset | `kd/dataset/_base.py` | 统一数据容器 (2D legacy + N-D structured) |
| Registry | `kd/dataset/_registry.py` | 数据集注册表 + load_pde 分发 |
| KD_SGA | `kd/model/kd_sga.py` | 符号遗传算法 wrapper |
| KD_DLGA | `kd/model/kd_dlga.py` | 深度学习遗传算法 wrapper |
| KD_DSCV | `kd/model/kd_dscv.py` | RL + 稀疏回归 (有限差分模式) |
| KD_DSCV_SPR | `kd/model/kd_dscv.py` | DSCV + PINN (稀疏/噪声模式) |
| KD_PySR | `kd/model/kd_pysr.py` | PySR wrapper (Julia, lazy-import) |
| SGA Engine | `kd/model/sga/` | 上游 sgapde solver |
| DISCOVER Engine | `kd/model/discover/` | 上游 RL 控制器 + searcher + PINN |
| Viz Core | `kd/viz/core.py` | VizRequest/VizResult/render() |
| Viz API | `kd/viz/api.py` | 高层函数: render_equation, plot_* |
| Viz Adapters | `kd/viz/_adapters/` | 模型 → 可视化数据转换 |
| Utils | `kd/utils/` | FiniteDiff, Attrdict, logging |
| GPU Runner | `kd/runner.py` | 离线任务执行脚本 (DLGA/DSCV_SPR on GPU) |
| Job Helpers | `kd/job.py` | GPU 任务封装 (提交/查询/下载)；支持 bohrium/mock 后端 |

## Data Flow

### 训练路径
1. `load_pde("burgers")` → Registry 查表 → 加载 .mat/.npy → PDEDataset
2. `model.fit_dataset(dataset)` → Adapter 转换 → Engine 训练 → 符号方程
3. `model.equation_latex()` → LaTeX 字符串

### DSCV_SPR 路径 (稀疏数据)
1. PDEDataset → `import_dataset(sample_ratio=0.1)` → 稀疏采样
2. PINN 预训练 → 光滑场 + 自动微分求导
3. RL 控制器搜索 → 符号方程

### 可视化路径
1. `configure(save_dir=...)` → 全局配置
2. `render(VizRequest(kind, model))` → Registry 查 adapter → 绘图 → VizResult

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| PySR lazy-import | Julia + torch 同时加载导致 SIGABRT，必须延迟 |
| Adapter 模式 | 模型内部格式各异，需要桥接层而非改上游代码 |
| 2D/N-D 双模式 PDEDataset | 兼容历史代码 (x,t,usol) 同时支持任意维度 |
| N-D 算子命名约定 | 1D: diff/diff2 (小写), 2D+: Diff/Diff2/lap (大写) |
| Gradio 而非 FastAPI | 玻尔平台推荐 Gradio；KD viz 层已产出 matplotlib fig，直接对接 gr.Plot |
| PySR 不上线 | Julia 运行时体积大且与 torch 冲突，首版不含 PySR |

## Constraints

- Python >= 3.9, PyTorch >= 1.10
- PySR 可选 (需 Julia 运行时)
- DSCV_SPR 的 PINN 预训练可能几分钟到几十分钟
- GPU 模型 (DLGA, DSCV_SPR) 在 Bohrium 上通过离线任务 API 提交到 GPU 节点执行
- 清华 pip 镜像 SSL 不稳定，需备用 pypi.org
- Bohrium CPU 节点无外网，依赖通过本机 `pip download` + `scp` + `--no-index` 离线安装
- Bohrium 容器内有 HTTP 代理 (`ga.dp.tech:8118`)，curl localhost 需加 `--noproxy`
- Bohrium SDK 1.0.5 无 `user.list_project()` 方法，需用 REST `openapi/v1/project/list` 替代
- 多用户共享容器，`/tmp` 文件存 job_id 不可靠，需配合 BrowserState 恢复

## Changelog

| Date | Change | Rationale |
|------|--------|-----------|
| 2026-03-21 | 新增 Changelog 区 | 记录架构级变更，避免决策遗失 |
| 2026-03-21 | 新增 `app.py` Gradio Web UI | 玻尔平台部署入口，零侵入现有代码 |
| 2026-03-21 | 修复 `execute.py` 残留 pdb 断点 | 无效 token 时进入调试器导致 Web 服务卡死 |
| 2026-03-21 | 修复 `stridge.py` None result 崩溃 | execute 返回 None 后 `.copy()` 报 AttributeError |
| 2026-03-22 | 新增 GPU 离线任务分流架构 | CPU Web 节点 + GPU 离线任务，降低部署成本 |
| 2026-03-22 | CPU Web 节点部署到 bohrium | 离线安装 torch(CPU)+gradio+bohrium-sdk 等 98 包，Gradio :50001 启动验证通过 |
| 2026-03-23 | GPU 节点环境验证 (T4) | torch 2.5.1+cu121, DLGA/DSCV_SPR runner.py 跑通 |
| 2026-03-23 | 新增 `launching.py` | 对接 Bohrium Launching 框架 (dp.launching)，标准离线任务入口 |
| 2026-03-23 | 重构 `job.py` submit | 从上传 config JSON 改为扁平 inputs，匹配 Launching Options schema |
| 2026-03-23 | GPU 离线应用注册 + 全链路验证 | app_key=kd-gpu, OAuth Cookie 认证，DLGA 提交→执行→下载→展示 OK |
| 2026-03-23 | 合并 bohrium-deploy 分支 | EqGPT 模型 + Regression 任务 + 模型重命名 (Discover→DSCV) |
| 2026-03-23 | Web App 发布 + OAuth | 端口 50000，HTTPS 修复，gr.Timer 自动轮询，job_id 文件持久化 |
| 2026-03-23 | 修复 `collections.Mapping` | Python 3.10 需用 `collections.abc.Mapping` |
| 2026-03-23 | 修复 GPU 轮询前端报错 | Bohrium 代理间歇返回 HTML→前端 SyntaxError；timer 30s + show_error=False + 顶层 try/except |
| 2026-03-23 | 修复 MODEL_KEYS 与 registry 不匹配 | 合并时 Discover→DSCV 重命名漏改 registry key，导致 DSCV/SPR 在所有 PDE 数据集上不可见 |
| 2026-03-23 | 对齐 Bohrium SDK 官方用法 | output_dir `./outputs`→`./output`；submit 加 `sourceAppKey`；download 优先用 `web.sub_model.download` |
| 2026-03-23 | 修复 EqGPT 权重路径测试 | 测试错误检查 `kd/model/eqgpt/gpt_model/`，实际权重在 `_DATA_DIR`（ref_lib/） |
| 2026-03-24 | `job.py` 新增 mock 后端 | `KD_JOB_BACKEND=mock` 使用 `/tmp/kd_mock_jobs` 本地 JSON 模拟 GPU 任务全流程，无需部署 Bohrium |
| 2026-03-24 | 动态 project_id 解析 | SDK 1.0.5 无 `list_project()`，改用 REST `openapi/v1/project/list` 自动获取用户项目 ID，支持多用户隔离 |
| 2026-03-24 | 页面刷新任务恢复 | `app.load` 时检查 file/BrowserState 恢复未完成 GPU 任务轮询 |
| 2026-03-24 | ForceHTTPS 条件激活 | 仅在 Host 含 `bohrium` 时注入 HTTPS headers，避免本地开发 SSL 问题 |
| 2026-03-24 | 持久化存储迁移 `/tmp/` → `/personal/.kd/` | FC 容器重建时 `/tmp` 清空导致 job_id 丢失；`/personal/` 是 NAS 挂载，跨容器持久化 |
| 2026-03-24 | recover_row 始终可见 | GPU 任务相关路径全部显示 Resume 输入框，SSE 断裂后用户有手动恢复入口 |
| 2026-03-24 | 页面恢复支持已完成任务 | `_on_load_recover` 恢复链：NAS 文件 → BrowserState → API；已完成任务也触发下载展示 |
| 2026-03-24 | Job ID 醒目显示 | `job_status` 格式改为 `>>> Job ID: xxx <<<`，确保用户在 SSE 错误前能记住 |
| 2026-03-24 | 下载结果目录迁移 | `JOB_OUTPUT_DIR` 从 `/home/outputs` 改为 `/personal/outputs`，结果跨容器持久化 |
| 2026-03-24 | 修复 `_train_outputs` 引用顺序 | 变量定义移至 `app.load()` 之前，修复 `UnboundLocalError` |

## Known Issues

1. `kd/__init__.py` `__all__` 声明 `KD_DSCV_Pinn` 但实际类名是 `KD_DSCV_SPR`
2. `DLGA`, `KD_DLGA` 在 `__all__` 但未在顶层 import — 需用 `from kd.model import ...`
3. `load_kdv_equation` 在 `__all__` 但未在 `kd/__init__.py` 中 import
4. `fit_dataset` 返回类型不一致：SGA/DLGA 返回 self，DSCV 返回 result dict
5. `KD_DLGA` 继承 DLGA (非 BaseEstimator) — `get_params()`/`set_params()` 可能异常
6. ~~`execute.py` 残留 `pdb.set_trace()` 调试断点~~ → 已修复 (2026-03-21)
7. ~~`stridge.py` 未处理 execute 返回 None 的情况~~ → 已修复 (2026-03-21)
