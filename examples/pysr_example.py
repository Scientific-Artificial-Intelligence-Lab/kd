"""KD_PySR example: symbolic regression on synthetic tabular data.

Demonstrates KD's wrapper around PySR for generic symbolic regression
(not PDE-specific). Requires the `pysr` package to be installed.

Usage:
    python examples/pysr_example.py
"""

# IMPORTANT: PySR uses Julia via juliacall, which crashes (SIGABRT) if
# torch or tensorflow signal handlers are installed first.  We must
# initialize juliacall BEFORE importing kd (which pulls in torch/tf
# via kd.model.__init__ → kd_dscv).
# See: pytorch/pytorch#78829
import os
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
import pysr  # noqa: E402,F401 — init Julia runtime before torch/tf load

import numpy as np

from kd.model.kd_pysr import KD_PySR
from kd.viz import VizRequest, configure, render, render_equation


# ============================================================
# 1. Synthetic Data / 合成数据
# ============================================================

N_SAMPLES = 256
rng = np.random.default_rng(0)
X = rng.uniform(-2.0, 2.0, size=(N_SAMPLES, 2))

# Target: y = x0^2 + 0.5*x1 + noise / 目标关系
y = X[:, 0] ** 2 + 0.5 * X[:, 1] + 0.01 * rng.normal(size=N_SAMPLES)


# ============================================================
# 2. Model / 模型
# ============================================================

model = KD_PySR(
    niterations=40,                            # Search iterations / 搜索迭代次数
    binary_operators=["+", "-", "*", "/"],      # Binary ops / 二元算子
    unary_operators=["square"],                 # Unary ops / 一元算子
    extra_model_kwargs={"progress": False},     # Extra PySR kwargs / 额外 PySR 参数
)

model.fit(X, y)

y_pred = model.predict(X)
mse = float(np.mean((y_pred - y) ** 2))
print(f"Equation: {model.best_equation_}")
print(f"LaTeX:    {model.equation_latex()}")
print(f"MSE:      {mse:.6e}")


# ============================================================
# 3. Visualization / 可视化
# ============================================================

configure(save_dir="artifacts/pysr_viz")

render_equation(model)
render(VizRequest("parity", model))
render(VizRequest("residual", model))
