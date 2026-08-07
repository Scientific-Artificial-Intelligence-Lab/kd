"""Example 13 - Semi-custom SINDy-style symbolic regression (user-supplied basis).

This is the basis-SR bypass: the USER supplies a library of candidate kd-IR terms
and the regressor does ONE sparse linear solve, returning ``y ~= sum_i c_i *
term_i(X)`` (selected support + coefficients). It is physically isolated from the
PDE platform (no PDEDataset/derivatives/Theta-LHS/Model facade) -- it reuses only
the kd numerical kernel (Evaluator + STRidge).

Unlike PySR (Example 12), it does NOT search free-form structure and cannot fit
constants *inside* nonlinear functions (e.g. the -1.85 in ``exp(-1.85*x)``). Its
value is the opposite: the user brings domain knowledge as the basis, and the
solver picks the sparse combination + coefficients.

Run: python examples/13_sindy_basis_sr.py
"""

import numpy as np

from kd.search.sindy import SINDyRegressor

# Synthetic ground truth: y = 2.0*sin(x1) - 3.0*(x1*x2), exact (no noise).
# The true support is exactly {sin(x1): 2.0, mul(x1, x2): -3.0}.
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 2)) # noqa: N816
y = 2.0 * np.sin(X[:, 0]) - 3.0 * (X[:, 0] * X[:, 1])

# A user library = the two true terms PLUS four decoys that must be dropped.
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

# Fail-loud contract (agent-facing): a bad term raises, naming the offender,
# rather than being silently dropped.
print()
try:
    model.fit(X, y, ["x1", "tanh(x1)"]) # tanh is not a kd default operator
except ValueError as exc:
    print(f"Fail-loud: {exc}")
