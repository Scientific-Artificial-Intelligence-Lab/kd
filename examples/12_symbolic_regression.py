"""Example 12 - Scalar symbolic regression on tabular data with kd.Model.

Tabular mode: fit ``y = f(X)`` on a plain feature table -- no PDE fields, no
derivatives, no Theta matrix. ``Model.fit`` accepts a ``TabularDataset``
directly and derives the term catalog from the feature columns.

The example uses the bundled TLC-CC dataset: real-world column chromatography
measurements (Xu et al., Nat Commun 16, 832, 2025), where the start retention
volume V_S is predicted from the TLC retardation factor R_F and the eluent
composition r.

Two engines run on the same 74-row table:

- PySR: genetic-programming symbolic regression (small demo budget).
- DISCOVER: RL-guided expression search with a Pareto front over accuracy
  and expression size.

Run: python examples/12_symbolic_regression.py
"""

from kd import Model, load_tlc_cc
from kd.core.expr import to_sympy

dataset = load_tlc_cc(target="start")

print(f"Dataset: {dataset.name} ({dataset.description})")
print(f"Source: {dataset.source}")
print(f"Samples: {dataset.X.shape[0]}")
print(f"Features: {list(dataset.var_names)} -> {dataset.target_name}")

# --- PySR -------------------------------------------------------------------
# Small budget: enough to demonstrate the end-to-end API quickly. Increase
# generations/populations/maxsize for a serious search.
pysr_model = Model(
    "pysr",
    generations=5,
    population_size=20,
    populations=2,
    maxsize=10,
    verbose=False,
).fit(dataset)

print()
print("[pysr]")
print(f"Best expr: {pysr_model.result_.equation}")
print(f"Best NMSE: {pysr_model.best_score_:.6g}")

# --- DISCOVER ---------------------------------------------------------------
# RL-guided expression search with per-candidate constant fitting. Budget:
# 200 generations x batch 1000: about 17 minutes with OMP_NUM_THREADS=1,
# about 30 minutes with PyTorch's default thread count on a 24-core CPU.
#
# Tabular scoring is scale-free: each Pareto entry's expression is the raw
# candidate and the fitted outer coefficient is published in ``entry.scale``.
# At seed 0 this budget puts the published law on the front: the
# complexity-5 entry ``div(r, add(0.0737, R_F))`` with scale 6.634
# normalizes to r/(0.151*R_F + 0.0111), next to the paper's
# r/(0.147*R_F + 0.0114). At 100 generations x batch 500 the seed-0 front
# does not contain the form r/(a*R_F + b).
discover_model = Model(
    "discover",
    generations=200,
    seed=0,
    batch_size=1000,
    reward_alpha=0.005,
    max_length=15,
    verbose=False,
).fit(dataset)

print()
print("[discover]")
print(f"Best expr: {discover_model.result_.equation}")
print(f"Best reward: {discover_model.best_score_:.6g}")
print("Pareto front (loss = NMSE of the scale-fitted candidate):")
for entry in discover_model.result_.pareto_front():
    scale = "-" if entry.scale is None else f"{entry.scale:.4g}"
    print(
        f" complexity={entry.complexity:2d} loss={entry.loss:.4g} "
        f"scale={scale} {to_sympy(entry.expression)}"
    )
