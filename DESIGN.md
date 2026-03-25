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

### GPU 离线任务路径 (Bohrium)
```
CPU Web Node (app.py)                    GPU Node (launching.py → runner.py)
─────────────────────                    ──────────────────────────────────
1. submit_gpu_job()  ──[Bohrium API]──→  2. entry_function() → model.fit_dataset()
                                         3. save_output(): _serialize_viz_data()
                                            → result.json (含 model/equation/viz_*)
4. download_result() ←─[SDK download]──
5. _build_viz_proxy(result_data)
   → 动态 ProxyClass + result_ 属性
6. _render_equation_fig(proxy, eq)
   → viz adapter LaTeX 渲染
7. _get_model_capabilities(proxy)
   → Viz tab 下拉菜单
```
GPU 只传 JSON 数据（`result.json`），CPU 用代理模型 + 同一套 viz adapter 渲染所有图像。

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
| 2026-03-24 | ~~下载结果目录迁移~~ | ~~`/home/outputs` → `/personal/outputs`~~ (后被撤回，`/personal/` 跨发布丢失) |
| 2026-03-24 | 修复 `_train_outputs` 引用顺序 | 变量定义移至 `app.load()` 之前，修复 `UnboundLocalError` |
| 2026-03-24 | JS 前端错误抑制 | MutationObserver 自动移除 Bohrium 代理导致的 SyntaxError 弹窗，无法修复代理本身 |
| 2026-03-24 | GPU 面板重构 (`gpu_panel`) | job_status/check_btn/recover 合并为 `gr.Column`，本地模型时整体隐藏消除空白横线 |
| 2026-03-24 | 修复 `result.json` 双下载损坏 | timer tick 与手动 Check 并发下载导致 JSON 追加；下载前清理 + `raw_decode` 容错 |
| 2026-03-24 | Viz 错误提示友好化 | 不支持的 plot type 显示 "not yet implemented" 而非堆栈跟踪 |
| 2026-03-24 | check_btn 完成后不隐藏 | job 完成/失败后 Check GPU Job 按钮保持可见，避免用户困惑 |
| 2026-03-24 | Job History 功能 | `/personal/.kd/{uid}.history` 记录 GPU 任务历史；折叠面板展示最近 10 条，含方程结果 |
| 2026-03-24 | 合并合作者 `3e650b6` | sympy 表达式渲染、Regression viz capabilities、BrowserState 兼容、viz adapter in-memory figure |
| 2026-03-24 | 修复 `launching.py` save_output 缺少 model 参数 | GPU 离线任务 result.json 不含 viz 数据，导致 Web 端 proxy model 为 None，Viz tab 不可用 |
| 2026-03-24 | Viz 下拉菜单 `allow_custom_value` | Gradio FC 环境下 `gr.update(choices=...)` 不同步服务端状态，导致选择报错；加 `allow_custom_value=True` 绕过验证 |
| 2026-03-24 | 持久化存储迁移 → `/data/` | `/personal/` 跨发布清空、`/share/` 生产容器为空、`/home/` 不完全持久；发现 `/data/` 是 App 专属 NAS（跨发布持久）|
| 2026-03-24 | History 存储路径最终确定 | `/data/kd_users/{userId}/job_list.txt`（App NAS）|
| 2026-03-24 | 对齐官方 SDK 文档 | download API 优先 `web.sub_model.download` + fallback；`output_dir`/`remote_target` 统一为 `"output"`；output 目录去掉 `kd_` 前缀 |
| 2026-03-24 | 修复 SDK 下载 `s/` 子目录问题 | Bohrium SDK 下载文件放在 `{save_dir}/s/` 下；`result.json` 和 `equation.png` 查找路径扩展到 `s/` |
| 2026-03-24 | History 三层恢复机制 | 1) `job_list.txt` 主数据源 2) API 实时查询未完成任务 3) 扫描 `/home/outputs/` 目录兜底恢复 |
| 2026-03-24 | 恢复的 job 自动补 history | 从 BrowserState/API 恢复的 job_id 如果 `job_list.txt` 无记录，自动补写条目 |
| 2026-03-24 | BrowserState 固定 `secret` | FC 冷启动时 `secret` 随机重生成导致旧 localStorage 解密失败；固定为 `"kd_browser_v1"` 使 BrowserState 跨重启持久 |
| 2026-03-24 | 下载后递归列出 `s/` 内容 | `os.walk(s_dir)` 打印 SDK 实际文件布局，一轮部署即可确认 result.json 路径 |
| 2026-03-24 | `find_active_job` 用 GPU app_key | `_bohrium_find_active` 原用 `clientName` cookie（Web App key），`job.list()` 按 header 过滤查不到 GPU 任务；改用 `KD_GPU_APP_KEY` |
| 2026-03-24 | 修复 `launching.py` save_output | 缺少 `model=model` 参数，GPU 端 EqGPT 的 viz 数据（equations/rewards/reward_history）不写入 result.json |
| 2026-03-24 | GPU 镜像全量文件同步 | `kd_eqgpt.py` 旧版缺少 `self.result_ = result`（根因）；`kd/viz/_adapters/eqgpt.py` 完全缺失导致 `save_viz` 跳过 EqGPT 图片生成；md5 全量对比修复 13+2 个文件差异 |
| 2026-03-24 | GPU Viz 数据序列化 (`_serialize_viz_data`) | 新增 `runner.py` 辅助函数：提取 Discover `searcher.r_train/r_history`、DLGA `train_loss_history/evolution_history/Chrom/coef`、EqGPT `result_` 写入 result.json，使 CPU 端可重建代理模型 |
| 2026-03-24 | CPU 代理模型增强 (`_build_viz_proxy`) | 从 result.json 的 `viz_*` 前缀字段还原模型属性：`proxy.searcher.r_train`（Discover）、`proxy.train_loss_history`（DLGA）等；Viz tab 下拉菜单不再为空 |
| 2026-03-24 | 修复代理类名 registry 匹配 | `dscv_spr` 的 class_map 从 `KD_DSCV_SPR` 改为 `KD_Discover`，因为 registry 注册的是父类 `KD_Discover`，名称回退机制需类名完全匹配 |
| 2026-03-24 | equation.png 4 路径搜索 | 扩展 `save_dir/` `s/` `s/output/` `s/outputs/` 四个候选路径，兼容 SDK 下载的嵌套目录结构 |
| 2026-03-24 | GPU 镜像清理 5.7G | 删除 `/tmp/*.whl`(2.7G) + `/root/.cache/pip`(3G) + `.git`(30M) + `tests/docs/log/trash` 等，镜像从 13G 降至 7.3G |
| 2026-03-24 | CPU 侧方程渲染 | 废弃 equation.png 文件搜索；CPU 用 `_build_viz_proxy()` 构建代理模型 + `_render_equation_fig()` 调用 viz adapter 的 LaTeX 渲染，GPU 只传 JSON 数据 |
| 2026-03-24 | ndarray 序列化修复 | `_to_list()` 增加 `isinstance(val, dict)` 递归处理；`json.dump` 增加 `default=lambda o: o.tolist()` 兜底；修复 EqGPT parity_data 133500×2 的 TypeError |
| 2026-03-24 | SystemExit 保护 | Bohrium SDK `APIResponseManager.__exit__` 在 401 时调用 `exit(1)` 杀死 ASGI 进程；在 job.py (3处) 和 app.py (4处) 添加 `except SystemExit` 捕获 |
| 2026-03-24 | 友好模型显示名 | `MODEL_DISPLAY` 映射 KD_EqGPT→EqGPT 等，UI 文案更清晰 |
| 2026-03-24 | 下载结果持久化 `/data/outputs` | `JOB_OUTPUT_DIR` 从 `/home/outputs`(容器 overlay) 改为 `/data/outputs`(App NAS)，结果跨发布持久 |

## Post-Deploy Issues (2026-03-24 线上验证)

上版本 (`9904f8e`) 部署后通过线上日志发现以下问题并逐一修复：

### 1. SyntaxError 弹窗干扰用户
- **现象**：页面显示 4 个 "Could not parse server response: SyntaxError: Unexpected token '<'" 错误
- **根因**：Bohrium FC 代理偶发返回 HTML 而非 JSON，Gradio SSE 解析失败
- **影响**：冷启动时加载事件并发高更容易触发；SSE 断裂后 timer 死掉
- **修复**：`gr.Blocks(js=...)` 注入 MutationObserver，自动移除含 SyntaxError 的 DOM 节点
- **局限**：只隐藏提示，不修复代理本身；如果 submit 首次响应就被代理拦截，Job ID 不会到达前端

### 2. 刷新页面后任务和结果丢失
- **现象**：用户刷新页面后 Job ID、状态、结果全部消失
- **根因**：`/tmp/` 在 FC 容器重建时清空；BrowserState 可能未写入；`find_active_job` 只找运行中任务
- **修复**：
  - ~~持久化迁移到 `/personal/.kd/`~~ → 最终迁移到 `/data/kd_users/`（App 专属 NAS，跨发布持久）
  - `_on_load_recover` 恢复链：NAS 文件 → BrowserState → API，已完成任务也触发下载
- **存储路径排查**：`/personal/` 跨发布清空 | `/share/` 生产容器为空 | `/home/` 保存镜像时打包但不完全持久 | `/data/` ✅ App NAS 跨发布持久
- **最终方案**：History → `/data/kd_users/`，下载结果 → `/data/outputs/`（均为 App NAS 持久存储）
- **验证**：日志确认 FC 容器重建后 BrowserState 第二级恢复成功

### 3. Resume 入口不可见
- **现象**：用户从未看到 "Enter Job ID to resume" 输入框
- **根因**：`_on_load_recover` 和所有 GPU 路径返回 `recover_row=visible=False`
- **修复**：GPU 面板 (`gpu_panel`) 包含 check_btn + recover 输入框，GPU 路径下整体显示

### 4. 非 GPU 模型页面显示空白横线
- **现象**：选择 KD_SGA/Regression 等本地模型时，结果区出现多条空白横线
- **根因**：隐藏的 GPU 组件各自 `visible=False` 但仍占据空间
- **修复**：GPU 组件合并为 `gpu_panel`（`gr.Column`），本地模型时整体隐藏

### 5. result.json 解析失败 "Extra data"
- **现象**：`Download failed: Extra data: line 13 column 2 (char 251)`，但实际有正确结果
- **根因**：timer tick 与手动 Check GPU Job 并发触发 `client.app.job.download()`，SDK 追加写入导致文件包含两份 JSON
- **修复**：下载前 `os.remove()` 清理旧文件 + `json.JSONDecoder().raw_decode()` 容错解析

### 6. Check GPU Job 按钮消失
- **现象**：多次点击 Check GPU Job 后按钮消失
- **根因**：job 完成后返回 `check_btn=visible=False`，后续点击 job_id 已清空，按钮被隐藏
- **修复**：gpu_panel 内组件不再单独控制 visible，由 gpu_panel 统一管理

## Known Issues

1. `kd/__init__.py` `__all__` 声明 `KD_DSCV_Pinn` 但实际类名是 `KD_DSCV_SPR`
2. `DLGA`, `KD_DLGA` 在 `__all__` 但未在顶层 import — 需用 `from kd.model import ...`
3. `load_kdv_equation` 在 `__all__` 但未在 `kd/__init__.py` 中 import
4. `fit_dataset` 返回类型不一致：SGA/DLGA 返回 self，DSCV 返回 result dict
5. `KD_DLGA` 继承 DLGA (非 BaseEstimator) — `get_params()`/`set_params()` 可能异常
6. ~~`execute.py` 残留 `pdb.set_trace()` 调试断点~~ → 已修复 (2026-03-21)
7. ~~`stridge.py` 未处理 execute 返回 None 的情况~~ → 已修复 (2026-03-21)
8. ~~Viz 下拉菜单 choices=[] 服务端验证失败~~ → 已修复 (`allow_custom_value=True`)
9. ~~持久化存储路径 `/personal/` 跨发布丢失~~ → 已修复（迁移到 `/data/` App NAS）
10. ~~SDK 下载结果在 `s/` 子目录，`result.json` 找不到~~ → 已修复（多路径搜索）
11. `web.sub_model.download` 在生产 SDK 中不可用，始终 fallback 到 `app.job.download`（功能正常，待确认 SDK 版本差异）
12. GPU 任务 DSCV_SPR 的 `spr_residual` / `spr_field_comparison` 两个 viz 能力不可用 — 需要完整数据集和模型内部对象，无法通过 JSON 序列化还原（设计限制，非 bug）
13. Bohrium SDK `APIResponseManager.__exit__` 在 API 错误时调用 `exit(1)` — 当前通过 `except SystemExit` 保护，但属于 SDK 架构缺陷，升级 SDK 版本时需复验
