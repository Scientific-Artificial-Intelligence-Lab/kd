"""
Minimal SGA example using the unified PDEDataset入口。

设计目标：
- 与 dlga/dscv 等示例保持同样的运行方式（直接 python 执行即可）。
- 仅演示从统一入口加载数据并用 KD_SGA.fit_dataset 跑通一次。
- 额外的可视化暂不默认执行，如需尝试请按注释自行取消注释。
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

# --- 依赖导入 / Dependency Imports ---
from kd.dataset import load_pde
from kd.model.kd_sga import KD_SGA
# from kd.viz import VizRequest, render, render_equation  # 可选：新 viz 适配器


# --- 配置 / Configuration ---
# 可选数据集：'chafee-infante' / 'burgers' / 'kdv'。目前仍以内置 benchmark 为主。
pde_name = "burgers"

model = KD_SGA(
    sga_run=20,     # 遗传搜索的迭代次数 / Number of evolutionary iterations
    num=20,         # 种群规模 / Population size
    depth=4,        # 表达式树最大深度 / Max depth of expression trees
    width=5,        # 表达式树最大宽度 / Max width of expression trees
    p_var=0.5,      # 变量替换概率 / Variable mutation probability
    p_mute=0.3,     # 节点变异概率 / Node mutation probability
    p_cro=0.5,      # 交叉概率 / Crossover probability
    seed=0,         # 随机种子 / Random seed
    use_autograd=False,  # 建议保持 False，custom 数据尚未支持对应元数据模型 / Keep False unless autograd models exist
    use_metadata=False,  # 建议保持 False，默认不加载预训练 metadata / Keep False to skip metadata
)


# --- 数据加载 / Data Loading ---
print(f"Loading dataset '{pde_name}' via kd.dataset.load_pde ...")
pde_dataset = load_pde(pde_name)


# --- 模型训练 / Model Training ---
print("\nRunning KD_SGA.fit_dataset ...")
model.fit_dataset(pde_dataset)


# --- 结果输出 / Results ---
print("\nBest PDE (raw string):")
print(model.best_pde_)

try:
    print("\nEquation in LaTeX:")
    print(model.equation_latex())
except Exception as exc:  # pragma: no cover - 仅用于示例运行时的友好提示
    print(f"暂时无法生成 LaTeX 表达式（{exc}），后续实现到位后可取消注释重试。")


# --- 可选可视化 / Optional Visualization ---
# 新的 kd.viz 适配器在 custom 数据上仍在完善，如需测试内置 benchmark，可取消以下注释：
# render_equation(model)
# render(VizRequest(kind="field_comparison", target=model))
# render(VizRequest(kind="time_slices", target=model, options={"slice_times": [0.0, 0.5, 1.0]}))
# render(VizRequest(kind="parity", target=model))
# render(VizRequest(kind="residual", target=model))

# legacy 可视化仅适用于内置三种 benchmark。暂不默认调用，避免 custom 数据场景报错。
# model.plot_results()
