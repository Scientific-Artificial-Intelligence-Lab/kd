
import numpy as np

from kd.data.regression import load_tlc_cc
from kd.search.pysr.config import PySRConfig
from kd.search.pysr.sr import PySRSymbolicRegressor



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
