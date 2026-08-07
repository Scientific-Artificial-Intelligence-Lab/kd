"""Example 12 - Standalone scalar symbolic regression with PySR.

This is the pure scalar-SR bypass: fit ``y = f(X)`` on tabular data without a
PDE dataset, derivatives, Theta matrix, or the ``kd.Model`` facade.

The example uses the bundled TLC-CC dataset: real-world column chromatography
measurements (Xu et al., Nat Commun 16, 832, 2025), where the start retention
volume V_S is predicted from the TLC retardation factor R_F and the eluent
composition r.

Run: python examples/12_symbolic_regression.py
"""

import numpy as np

from kd import PySRConfig, load_tlc_cc
from kd.search.pysr import PySRSymbolicRegressor

# Small budget: enough to demonstrate the end-to-end API quickly. Increase
# niterations/populations/maxsize for a serious search.
CONFIG = PySRConfig(
    niterations=5,
    population_size=20,
    populations=2,
    maxsize=10,
    seed=0,
)

dataset = load_tlc_cc(target="start")

print(f"Dataset: {dataset.name} ({dataset.description})")
print(f"Source: {dataset.source}")
print(f"Samples: {dataset.X.shape[0]}")
print(f"Features: {list(dataset.var_names)} -> {dataset.target_name}")

model = PySRSymbolicRegressor(config=CONFIG)
model.fit(dataset.X, dataset.y, var_names=list(dataset.var_names))
y_hat = model.predict(dataset.X)

print()
print(f"Best expr: {model.best_expr_}")
print(f"Best NMSE: {model.best_score_:.6g}")
print(f"Predictions: shape={y_hat.shape}, finite={np.isfinite(y_hat).all()}")
