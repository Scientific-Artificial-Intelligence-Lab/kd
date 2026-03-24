# Tasks Index

> 全局路线图和执行顺序见 [ROADMAP.md](ROADMAP.md)

活跃任务包。

## 当前任务

| 任务 | 状态 | 描述 |
|------|------|------|
| [fix-dscv-latex/](fix-dscv-latex/) | **completed** | DSCV LaTeX 嵌套导数渲染修复。chain rule 展开 + 162 个验证测试 |
| [fix-sga-searchspace/](fix-sga-searchspace/) | **completed** | SGA Burgers 方程过于复杂。根因：数据+AIC。修复：减 sga_run + 重新生成 Gallery (e6a67c3) |
| [dscv-nd/](dscv-nd/) | **completed** | DSCV N-D 升级：全部 3 Phase 完成 |
| [viz-nd/](viz-nd/) | **completed** | viz 模块 N-D 兼容性（Phase 1 + Phase 2 + review fixes） |
| [fix-diff2-3/](fix-diff2-3/) | **completed** | 修复 `Diff2_3` name==3 索引 bug（3 bugs: m→p, 3D→4D index, missing return） |
| [nd-pdedataset/](nd-pdedataset/) | **completed（有遗留 issues）** | PDEDataset N-D 扩展。遗留：2D-only API 无 guard、get_data() N-D 信息丢失 |
| [preexist-cleanup/](preexist-cleanup/) | **deferred** | 清理预存阻塞：pdb/raise/jit |
| [sga-nd-remaining/](sga-nd-remaining/) | **deferred** | SGA N-D 残留：metadata/delete_edges/autograd |
| [mvp-stop-bleeding/](mvp-stop-bleeding/) | suspended | Run 隔离 × 可复现 × 可观测 |
| [viz-paper-charts/](viz-paper-charts/) | **backlog** | 论文级可视化：coefficient bar chart, error map, Pareto front 等 |
| [gradio-quick-wins/](gradio-quick-wins/) | **completed** | Gradio app 快速改进：默认算子修正、数据预览、算子校验、EqGPT 提示 |
| [eqgpt-viz/](eqgpt-viz/) | **Phase 1 done** | EqGPT 可视化：Phase 1 equation+reward_ranking 完成，Phase 2/3 backlog |
| [platform-ux/](platform-ux/) | **backlog** | 平台体验：__init__.py 修复, plot_all, 统一 API, 低代码示例 |
| [md-audit/](md-audit/) | **completed** | Markdown 文档全面审计：83 findings，报告见 notes/reviews/md-full-audit/ |
| [dscv-reflib-benchmark/](dscv-reflib-benchmark/) | **completed** | kd DSCV vs ref_lib DISCOVER 对齐验证（5 dataset benchmark） |
| [nd-example-data/](nd-example-data/) | **backlog** | N-D example 替换更合理的合成方程（当前 u_t=-u 过于简单） |
| [verify-examples/](verify-examples/) | **completed** | Examples 自动化验证框架（15/15 PASS），发现 3 个 viz bug 待修 |

## 任务模板

新建任务时，复制 [TASK_SPEC.md](TASK_SPEC.md) 模板。

## 工作流

1. 在 `TASKS/<task-name>/` 创建任务包
2. 使用 `/tdd` -> `/dev` -> `/wrap-up` 流程
3. 完成后结论放入 `notes/`，任务包可保留或归档
