"""Example 03 - Visualize an SGA fit with the HTML report.

After ``model.fit(...)``, kd can render a multi-figure HTML report
showing convergence curves, term importance, residuals, the discovered
equation as a tree, and more.

Uses the built-in Chafee-Infante benchmark (real PDE data bundled
with kd). Output goes to ``examples/out/chafee_infante/``.

Run: python examples/03_visualize.py
      open examples/out/chafee_infante/report.html
"""

from pathlib import Path

import kd

# 1. Load a real benchmark dataset bundled with kd.
dataset = kd.load_chafee_infante()
print(f"Loaded: {dataset.name}, ground truth: {dataset.ground_truth}")
kd.preview(dataset)

# 2. Fit. Chafee-Infante has a richer term library than Burgers, so we
# bump the SGA budget above the defaults.
model = kd.Model(
    algorithm="sga",
    generations=60,
    population=20,
    seed=0,
    verbose=False,
)
model.fit(dataset)

print()
print(f"Discovered: {model.result_.equation}")
print(f"Best AIC: {model.best_score_:.4f}")

# 3. Render all supported figures, including SGA's search diagnostics.
out_dir = Path(__file__).parent / "out" / "chafee_infante"
report = model.report(out_dir)

print(f"Report: {report.report}")
print(f"Figures: {len(report.figures)} files")
