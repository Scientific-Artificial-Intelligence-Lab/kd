
from pathlib import Path

import numpy as np

from kd.search.pysr.config import PySRConfig
from kd.search.pysr.sr import PySRSymbolicRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "tests" / "fixtures" / "pysr" / "tlc_cc_Rf_t1.npy"
VAR_NAMES = ["R_F", "r"]



CONFIG = PySRConfig(
    niterations=5,
    population_size=20,
    populations=2,
    maxsize=10,
    seed=0,
)

data = np.load(DATA_PATH)
X = data[:, [0, 1]]
y = data[:, 2]

print(f"Dataset: {DATA_PATH}")
print(f"Samples: {X.shape[0]}")
print(f"Features: {VAR_NAMES}")

model = PySRSymbolicRegressor(config=CONFIG)
model.fit(X, y, var_names=VAR_NAMES)
y_hat = model.predict(X)

print()
print(f"Best expr: {model.best_expr_}")
print(f"Best NMSE: {model.best_score_:.6g}")
print(f"Predictions: shape={y_hat.shape}, finite={np.isfinite(y_hat).all()}")
