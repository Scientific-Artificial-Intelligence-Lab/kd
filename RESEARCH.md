# Research

## Project Goal

KD (Knowledge Discovery) 是一个从数值数据中自动发现偏微分方程 (PDE) 符号表达式的工具包。
提供 4 种互补的发现引擎、统一的数据集接口和出版级可视化，支持 1D/2D/3D 空间数据。

## Core Idea

- **稀疏回归路线 (SGA/DLGA)**：构建候选算子库，用遗传算法搜索最优组合
- **强化学习路线 (DSCV/DSCV_SPR)**：用 RL 控制器生成候选符号表达式，评估奖励
- **PINN 辅助 (DSCV_SPR)**：对稀疏/噪声数据先用 PINN 学光滑场，再在上面求导和搜索
- **通用符号回归 (PySR)**：对非 PDE 问题的 fallback，基于 Julia 的进化搜索

## Principles

- 统一接口：所有模型遵循 scikit-learn 风格 (fit_dataset / equation_latex)
- Adapter 模式：PDEDataset → 模型内部格式由适配器桥接，松耦合
- 注册表驱动：数据集通过 PDE_REGISTRY 管理，一行加载
- 可视化门面：VizRequest/render() 分发到模型适配器，模型不感知绘图细节

## Non-Goals

- 不做 PDE 正向求解器（只做逆问题：从数据发现方程）
- 不做通用机器学习框架
- 不做大规模分布式训练

## Current Phase

项目 v1.0.0 已完成，核心功能稳定。当前目标：**集成到玻尔 (Bohrium) 线上平台**。
采用 Gradio 自定义 Web App 方案（非 Launching 框架），已完成 `app.py` 基础版。

## TODO

- [ ] Phase 1: 修复 __all__ 导出、统一 fit_dataset 返回类型
- [ ] Phase 2: 设计序列化接口 (to_dict/from_dict)、统一 equation_latex
- [x] Phase 3: ~~REST API wrapper (FastAPI)~~ → 改用 Gradio Web App (`app.py`)，已完成基础版
  - [x] 数据集选择 + 模型选择 + 参数配置 + 训练 + 可视化
  - [x] Cancel training 功能
  - [x] Viz tab 根据模型自动过滤可用 plot type
  - [x] 修复 execute.py 残留 pdb 断点 + stridge.py None result 崩溃
  - [ ] 用户数据上传 (.mat/.npy)
  - [ ] 训练进度实时显示
- [ ] Phase 4: 镜像制作（Bohrium 内置工具或 Dockerfile）、上架发布
  - [x] GPU 离线任务分流架构 (`kd/runner.py` + `kd/job.py` + `app.py` 路由)
  - [x] CPU Web 节点部署 — 代码同步 + 离线安装 98 个 wheel 包到 bohrium (2026-03-22)
  - [x] Gradio Web 服务启动验证 — `:50001` HTTP 200 OK (2026-03-22)
  - [x] GPU 节点环境搭建 — torch 2.5.1+cu121 + dp-launching-sdk，DLGA/DSCV_SPR 验证通过 (2026-03-23)
  - [x] `launching.py` — Bohrium Launching 框架入口，测试通过 (2026-03-23)
  - [x] `job.py` 重构 — 扁平 inputs 匹配 Launching Options schema (2026-03-23)
  - [x] GPU 离线应用注册 — app_key=`kd-gpu`, sub_model_name=`Options` (2026-03-23)
  - [x] Web 端 GPU 提交全链路验证 — OAuth + Cookie 认证 + 自动轮询 (2026-03-23)
  - [x] 合并 bohrium-deploy 分支 — EqGPT 模型 + Regression 任务 + 模型重命名 (2026-03-23)
  - [x] 自定义 Web App 注册 + OAuth 开启 (2026-03-23)
  - [x] 修复 GPU 轮询 SyntaxError — timer 30s + show_error=False + 顶层安全网 (2026-03-23)
  - [x] 修复 MODEL_KEYS 与 registry key 不匹配 — DSCV/SPR 在所有数据集不可见 (2026-03-23)
  - [x] 对齐 Bohrium SDK 官方用法 — output_dir/sourceAppKey/download API (2026-03-23)
  - [x] `job.py` mock 后端 — `KD_JOB_BACKEND=mock` 本地模拟 GPU 任务 (2026-03-24)
  - [x] 动态 project_id — REST API 自动获取用户项目，解决多用户隔离 (2026-03-24)
  - [x] 页面刷新恢复 — app.load 自动恢复未完成的 GPU 任务轮询 (2026-03-24)
  - [x] ForceHTTPS 仅在 Bohrium 网关生效，本地开发不受影响 (2026-03-24)
  - [ ] 最终镜像发布 + 验证稳定性
  - [ ] 日志监控
