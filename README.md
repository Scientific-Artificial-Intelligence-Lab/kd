# KD – PDE Symbol Discovery Toolkit

KD is a modular toolkit for discovering partial differential equations (PDEs)
and symbolic models. It combines reinforcement-learning search (DSCV), genetic
algorithms (DLGA), sparse regression (SGA), and high-quality visual diagnostics
in a single workflow.

## Quick start

```bash
# 1) create the recommended environment
conda create -n kd-env python=3.9
conda activate kd-env

# 2) install dependencies
pip install -r requirements.txt

# 3) run a complete DLGA example with the new viz API
python examples/kd_dlga_viz_api_example.py
```

Figures are written to `artifacts/dlga_viz/` by default. Graphviz (for equation
trees) can be installed via `conda install python-graphviz` or your system
package manager.

## Visualization façade (`kd.viz`)

```python
from kd.viz import (
    configure,
    plot_training_curve, plot_validation_curve,
    plot_search_evolution, plot_optimization,
    plot_residuals, plot_field_comparison,
    render_equation,
)

configure(save_dir="artifacts/run-1")
plot_training_curve(model)
plot_residuals(model, actual=y_true, predicted=y_hat, coordinates=X)
plot_field_comparison(model, x_coords=x, t_coords=t, true_field=u_true, predicted_field=u_pred)
```

Each helper returns a `VizResult` containing saved paths, warnings, and the
normalized contract data (`ResidualPlotData`, `OptimizationHistoryData`,
`FieldComparisonData`). See `docs/viz_helpers.md` for intent details and adapter
status.

## Repository layout

```
kd/
├── dataset/    # built-in PDE datasets and loaders
├── model/      # DSCV, DLGA, SGA wrappers and discover/ core
├── viz/        # unified façade, adapters, legacy helpers
├── tests/      # automated tests (pytest)
└── examples/   # runnable tutorials (see kd_dlga_viz_api_example.py)
```


## Acknowledgements

KD draws inspiration from OPEN-source projects including DISCOVER, Deep
Symbolic Optimization, SymPy, DeepXDE, and PySR.

https://github.com/menggedu/DISCOVER
https://github.com/dso-org/deep-symbolic-optimization
https://github.com/sympy/sympy
https://github.com/lululxvi/deepxde
https://github.com/MilesCranmer/PySR
