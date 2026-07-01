
import numpy as np

from kd.search.sindy.sr import SINDyRegressor



rng = np.random.default_rng(0)
X = rng.normal(size=(200, 2))
y = 2.0 * np.sin(X[:, 0]) - 3.0 * (X[:, 0] * X[:, 1])


LIBRARY = ["x1", "sin(x1)", "mul(x1, x2)", "x2", "cos(x2)", "mul(x2, x2)"]

print(f"Samples: {X.shape[0]}")
print(f"Library: {LIBRARY}")

model = SINDyRegressor()
model.fit(X, y, LIBRARY)
y_hat = model.predict(X)

print()
print(f"Selected: {model.selected_terms_}")
print(f"Coefficients: {np.asarray(model.coefficients_).round(4).tolist()}")
print(f"Expression: {model.expression_}")
print(f"NMSE: {model.nmse_:.6g}")
print(f"Predictions: shape={y_hat.shape}, finite={np.isfinite(y_hat).all()}")



print()
try:
    model.fit(X, y, ["x1", "tanh(x1)"])
except ValueError as exc:
    print(f"Fail-loud: {exc}")
