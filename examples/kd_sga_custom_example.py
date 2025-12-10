"""Custom KD_SGA example on a synthetic PDE field.

This script constructs a simple toy field
``u(x, t) = sin(pi * x) * cos(pi * t)`` on a regular grid, wraps it into
``PDEDataset``, and runs ``KD_SGA`` in a custom mode without any
built-in problem template.

Key points:
- Use ``PDEDataset`` with ``pde_data=None`` and explicit ``x``, ``t``,
  ``usol`` for fully custom data;
- ``problem_name`` is a free-form label (not one of the built-in three
  benchmarks);
- Metadata/autograd are disabled to keep the example lightweight.
"""

from pathlib import Path
import numpy as np

from _bootstrap import ensure_project_root_on_syspath

PROJECT_ROOT = ensure_project_root_on_syspath()

from kd.dataset import PDEDataset
from kd.model.kd_sga import KD_SGA
from kd.viz import VizRequest, configure, render, render_equation

# 1. Construct custom data / 构造自定义数据
x = np.linspace(0.0, 1.0, 32)
t = np.linspace(0.0, 1.0, 33)
xx, tt = np.meshgrid(x, t, indexing="ij")
u = np.sin(np.pi * xx) * np.cos(np.pi * tt)

# PDEDataset requires the positional argument pde_data; for fully custom
# data we pass None and rely on x/t/usol instead.
dataset = PDEDataset(
    equation_name="custom_sga_test",
    pde_data=None,
    domain=None,
    epi=0.0,
    x=x,
    t=t,
    usol=u,
)

# 2. Basic configuration / 基本配置
SAVE_DIR = PROJECT_ROOT / "artifacts" / "sga_custom"

# Configure viz façade so all figures will be stored under SAVE_DIR.
configure(save_dir=SAVE_DIR)

# 3. Initialise and train KD_SGA / 初始化并训练 KD_SGA
model = KD_SGA(
    sga_run=100,      # fewer iterations for a quick smoke test
    num=20,
    depth=4,
    width=5,
    seed=0,
    use_autograd=False,
    use_metadata=False,
)

print("[KD_SGA Custom] Running KD_SGA on custom dataset (no ground truth templates)...")
model.fit_dataset(dataset, problem_name="custom_sga_test")

print("\n[Result] Best PDE (raw string):")
print(model.best_pde_)

try:
    print("\n[Result] Equation in LaTeX:")
    print(model.equation_latex())
except Exception as exc:
    print(f"[KD_SGA Custom] Failed to generate LaTeX expression ({exc}); skipping.")

# 4. Visualisation via kd.viz / 使用 kd.viz 进行可视化

eq_result = render_equation(model)
print("\n[Viz] Equation figure paths:")
for path in eq_result.paths:
    print(f"  - {path}")

intents = [
    ("field_comparison", {}),
    ("time_slices", {"slice_times": [0.0, 0.5, 1.0]}),
    ("parity", {}),
    ("residual", {}),  # residuals may be approximate since there is no analytic PDE template
]

for intent, options in intents:
    result = render(VizRequest(kind=intent, target=model, options=options))
    print(f"\n[Viz] {intent.replace('_', ' ').title()} paths:")
    for path in result.paths:
        print(f"  - {path}")
    if result.metadata:
        print(f"[Viz] {intent.title()} metadata: {result.metadata}")

# Note: we do not call legacy model.plot_results() here, since it is
# primarily designed for the three built‑in benchmark problems.
