"""Example 05 - Save and reload a fitted result.

Fit takes time; you don't want to re-run it every time. ``ExperimentResult``
serializes to JSON: fit once, save, load anywhere (including a different
machine) without re-running SGA.

Uses the bundled PDE_divide benchmark for variety.

NOTE: this persists a FINISHED result. To survive a crash MID-run (and
resume the search itself), see ``10_checkpoint_resume.py``.

Run: python examples/05_save_load.py
"""

from pathlib import Path

import kd

# 1. Load a real benchmark dataset and fit.
dataset = kd.load_pde_divide()
print(f"Loaded: {dataset.name}")
print(f"Ground truth: {dataset.ground_truth}")

model = kd.Model(
    algorithm="sga",
    generations=30,
    population=15,
    seed=0,
    verbose=False,
).fit(dataset)
print(f"Discovered: {model.best_expr_}")
print(f"Best AIC: {model.best_score_:.4f}")

# 2. Save the full result (best expression, AIC trace, residuals, recorder data).
out_path = Path(__file__).parent / "out" / "pde_divide.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
model.result_.save(out_path)
print(f"Saved: {out_path}")

# 3. Round-trip: load it back and verify.
restored = kd.ExperimentResult.load(out_path)
print(f"Loaded back: best={restored.best_expression!r} AIC={restored.best_score:.4f}")

assert restored.best_expression == model.best_expr_, "Round-trip failed!"
print("Round-trip OK.")
