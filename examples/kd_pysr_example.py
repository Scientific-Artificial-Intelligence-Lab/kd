"""Minimal KD_PySR example on synthetic tabular data.

This script shows how to:

- Generate a simple synthetic regression dataset
  (``y = x0**2 + 0.5 * x1 + noise``);
- Fit ``KD_PySR`` as a generic symbolic regressor;
- Inspect the discovered equation and visualise basic diagnostics via
  the unified ``kd.viz`` façade (equation / parity / residual).

Note: running this example requires the ``pysr`` package and its
dependencies to be installed in the current environment.
"""

from pathlib import Path

import numpy as np

from _bootstrap import ensure_project_root_on_syspath

PROJECT_ROOT = ensure_project_root_on_syspath()

from kd.model.kd_pysr import KD_PySR
from kd.viz import VizRequest, configure, render, render_equation


# 1. Basic configuration / 基本配置
N_SAMPLES = 256
SEED = 0
SAVE_DIR = PROJECT_ROOT / "artifacts" / "pysr_minimal"

# Configure viz façade so that all figures are written under SAVE_DIR.
# 配置可视化前端，所有图像会输出到 SAVE_DIR 目录下。
configure(save_dir=SAVE_DIR)


# 2. Synthetic data generation / 构造合成回归数据
rng = np.random.default_rng(SEED)
X = rng.uniform(-2.0, 2.0, size=(N_SAMPLES, 2))
# Target relation: y = x0^2 + 0.5 * x1 + small Gaussian noise.
# 目标关系：y = x0^2 + 0.5 * x1 + 少量高斯噪声。
y = X[:, 0] ** 2 + 0.5 * X[:, 1] + 0.01 * rng.normal(size=N_SAMPLES)


# 3. Create and fit KD_PySR / 创建并拟合 KD_PySR 模型
model = KD_PySR(
    niterations=40,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square"],
    extra_model_kwargs={
        # Keep the configuration simple; options may vary across pysr versions.
        # 这里刻意保持配置简单，不依赖版本特有的高级参数。
        "progress": False,
    },
)

print("[KD_PySR Minimal] Fitting model on synthetic data...")
model.fit(X, y)


# 4. Inspect equation and training error / 查看发现的方程与误差
y_pred = model.predict(X)
mse = float(np.mean((y_pred - y) ** 2))

print("\n[Result] best_equation_:")
print(f"  {model.best_equation_}")
print("[Result] equation_latex:")
print(f"  {model.equation_latex()}")
print(f"[Result] train MSE: {mse:.6e}")


# 5. Visualisation via kd.viz / 使用 kd.viz 进行可视化

eq_result = render_equation(model)
print("\n[Viz] Equation figure paths:")
for path in eq_result.paths:
    print(f"  - {path}")

intents = [
    ("parity", {}),
    ("residual", {}),
]

for intent, options in intents:
    result = render(VizRequest(kind=intent, target=model, options=options))
    print(f"\n[Viz] {intent.replace('_', ' ').title()} paths:")
    for path in result.paths:
        print(f"  - {path}")
    if result.metadata:
        print(f"[Viz] {intent.title()} metadata: {result.metadata}")
