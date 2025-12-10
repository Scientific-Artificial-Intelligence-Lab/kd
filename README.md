# Knowledge Discovery (KD) – Symbolic Equation Discovery Toolkit

Knowledge Discovery (KD) is a modular toolkit for symbolic modelling and
equation discovery, with a strong focus on partial differential equations
(PDEs). It combines reinforcement-learning search (DSCV), genetic algorithms
(DLGA), sparse regression (SGA), an optional generic SR backend powered by
PySR, and high-quality visual diagnostics in a single workflow.

## Models at a glance

- `KD_SGA` – sparse-regression–based PDE discovery on hand-crafted libraries,
  targeting local PDEs of the form `u_t = N(u, u_x, ...)`.
- `KD_DLGA` – deep-learning–aided genetic algorithm for PDE discovery, with
  rich optimisation and search diagnostics.
- `KD_DSCV` – DISCOVER-based local PDE discovery in regular (STRidge) mode.
- `KD_DSCV_SPR` – DISCOVER sparse + PINN mode for PDE discovery with
  physics-informed neural networks.
- `KD_PySR` (optional) – thin wrapper around
  [PySR](https://github.com/MilesCranmer/PySR) for generic symbolic regression
  tasks `fit(X, y)`.

## Quick start

```bash
# 1) create the recommended environment
conda create -n kd-env python=3.9
conda activate kd-env

# 2) install dependencies for working from the repo
pip install -r requirements.txt

# (optional) install KD as a package in the same environment
pip install .

# (optional) enable PySR-based symbolic regression
pip install 'kd[pysr]'
```

Run a minimal end-to-end example:

```bash
# SGA – Burgers benchmark with unified viz
python examples/kd_sga_example.py

# DLGA – KdV benchmark with unified viz
python examples/kd_dlga_example.py

# DSCV – Burgers benchmark (regular mode)
python examples/kd_dscv_example.py

# DSCV_SPR – Burgers benchmark with PINN (sparse mode)
python examples/kd_dscvspr_example.py

# PySR (optional) – generic SR on synthetic (X, y)
python examples/kd_pysr_example.py
```

For a full catalogue of example scripts and their intended usage, see
`examples/README.md`.

Figures are written under `artifacts/` by default. Graphviz (for equation trees)
can be installed via `conda install python-graphviz` or your system package
manager.

## Usage

A minimal SGA workflow (imports omitted for brevity):

```python
dataset = load_pde("burgers")
model = KD_SGA()
model.fit_dataset(dataset)
print(model.best_equation_latex_)
```

To generate basic diagnostics with the unified visualization façade:

```python
configure(save_dir="artifacts/run-1")
plot_training_curve(model)
plot_residuals(model, actual=y_true, predicted=y_hat, coordinates=X)
plot_field_comparison(
    model,
    x_coords=x,
    t_coords=t,
    true_field=u_true,
    predicted_field=u_pred,
)
```

Each helper returns a `VizResult` containing saved paths, warnings, and the
normalized contract data (`ResidualPlotData`, `OptimizationHistoryData`,
`FieldComparisonData`). See `notes/viz/viz_helpers.md` for intent details and
adapter status.

## Repository layout

```
kd/
├── dataset/    # built-in PDE datasets and loaders
├── model/      # DSCV, DLGA, SGA, PySR wrappers and discover/ core
├── viz/        # unified façade, adapters, legacy helpers
├── tests/      # automated tests (pytest)
└── examples/   # runnable tutorials (see examples/README.md)
```


## Acknowledgements

KD draws inspiration from open-source projects including DISCOVER, Deep
Symbolic Optimization, SymPy, DeepXDE, and PySR. We are especially grateful to
the PySR project for providing a powerful and flexible symbolic regression
engine that KD_PySR builds upon.

https://github.com/menggedu/DISCOVER  
https://github.com/dso-org/deep-symbolic-optimization  
https://github.com/sympy/sympy  
https://github.com/lululxvi/deepxde  
https://github.com/MilesCranmer/PySR  
