# KD Examples Overview

This document summarises the example scripts under `examples/` and how
they are intended to be used. For concrete hyperparameters and output
paths, please refer to the comments inside each script.

---

## 1. SGA – Symbolic Genetic Algorithm for PDE discovery

- `kd_sga_example.py`  
  - Goal: Introductory SGA example with visualisation.  
  - Data: `load_pde("burgers")`.  
  - Usage: `python examples/kd_sga_example.py`  
  - Notes: Uses `KD_SGA.fit_dataset(PDEDataset)` together with the
    unified `kd.viz` façade to produce equation figures, field
    comparison, residual and parity diagnostics. Calls the legacy
    `plot_results` at the end as an optional supplement. This is the
    primary SGA + viz entry point.

- `kd_sga_custom_example.py`  
  - Goal: Demonstrate SGA on a custom `u(x, t)` field (advanced usage).  
  - Data: Constructs `u(x,t) = sin(pi*x) * cos(pi*t)` in-script and
    wraps it into a `PDEDataset`.  
  - Usage: `python examples/kd_sga_custom_example.py`  
  - Notes: Shows how to use `KD_SGA.fit_dataset` and `kd.viz` on
    non‑benchmark data without a built‑in PDE template. Suitable for
    users who are already comfortable with the basic API.

---

## 2. DLGA – Deep Learning Genetic Algorithm for PDE discovery

- `kd_dlga_example.py`  
  - Goal: Introductory DLGA example with visualisation.  
  - Data: `load_pde("kdv")`.  
  - Usage: `python examples/kd_dlga_example.py`  
  - Notes: Trains DLGA via `KD_DLGA.fit_dataset(PDEDataset, ...)` and
    uses the unified `kd.viz` façade to generate training/validation
    curves, search evolution, optimisation diagnostics, field
    comparison, residual plots, time-slice comparisons, derivative-term
    relationships and parity plots. This is the main DLGA + viz example.

---

## 3. DSCV – DISCOVER-based PDE discovery (Regular / SPR modes)

### 3.1 Regular PDE (`KD_DSCV`)

- `kd_dscv_example.py`  
  - Goal: Classic KD_DSCV example in regular (local PDE) mode.  
  - Data: typically `load_pde("burgers")`.  
  - Usage: `python examples/kd_dscv_example.py`  
  - Notes: Keeps the flavour of the original DISCOVER examples. Uses
    `KD_DSCV.import_dataset + train` together with specialised functions
    from `dscv_viz` (expression tree, reward density, residual,
    field comparison, etc.) to run a complete PDE search and
    visualisation. Suitable for reproducing paper-style plots.

- `kd_dscv_viz_api_example.py`  
  - Goal: Demonstrate KD_DSCV working with the unified `kd.viz` façade
    (advanced / developer example).  
  - Data: typically `load_pde("burgers")`.  
  - Usage: `python examples/kd_dscv_viz_api_example.py`  
  - Notes: Does not call `dscv_viz` directly; instead it uses
    `VizRequest` + `render` to trigger adapter intents such as
    `search_evolution`, `density`, `tree`, `equation`, `residual`,
    `field_comparison`, and `parity`. This script is a reference for the
    unified visualisation interface.

### 3.2 Sparse + PINN (`KD_DSCV_SPR`)

- `kd_dscvspr_example.py`  
  - Goal: Full KD_DSCV_SPR example (PDE + PINN in sparse mode).  
  - Data: typically `load_pde("burgers")`.  
  - Usage: `python examples/kd_dscvspr_example.py`  
  - Notes: Shows how to configure sparse sampling, collocation points,
    and PINN training via `KD_DSCV_SPR.import_dataset`, and uses
    SPR-specific plots from `dscv_viz` (residual analysis, field
    comparison, actual vs predicted) to understand the iterative
    PDE+PINN process.

- `kd_dscvspr_viz_api_example.py`  
  - Goal: Demonstrate KD_DSCV_SPR working with the unified `kd.viz`
    façade (advanced / developer example).  
  - Data: `load_pde("burgers")` or other SPR-enabled datasets.  
  - Usage: `python examples/kd_dscvspr_viz_api_example.py`  
  - Notes: Uses SPR-specific intents such as `spr_residual` and
    `spr_field_comparison` to visualise PINN diagnostics through the
    façade, providing a unified entry point for SPR-related views.

---

## 4. PySR – Generic symbolic regression (optional dependency)

- `kd_pysr_example.py`  
  - Goal: Demonstrate KD’s wrapper around PySR (`KD_PySR`).  
  - Data: simple synthetic regression dataset `(X, y)`.  
  - Usage: `python examples/kd_pysr_minimal.py`
    (requires `pysr` and its dependencies in the environment).  
  - Notes: Shows how to use PySR as a generic symbolic regressor inside
    KD: build synthetic `(X, y)`, call `KD_PySR.fit`, and use `kd.viz`
    equation/parity/residual plots for basic diagnostics.

---

## 5. N-D Spatial Examples (2D/3D)

- `kd_sga_nd_example.py`
  - Goal: SGA PDE discovery on 2D/3D spatial data.
  - Usage: `python examples/kd_sga_nd_example.py`

- `kd_dscv_nd_example.py`
  - Goal: DSCV Regular mode on 2D/3D spatial data.
  - Usage: `python examples/kd_dscv_nd_example.py`

- `kd_dscvspr_nd_example.py`
  - Goal: DSCV Sparse/PINN mode on 2D/3D spatial data.
  - Usage: `python examples/kd_dscvspr_nd_example.py`

---

## 6. Usage suggestions

- Beginners / quick start:  
  Start with the `*_example.py` scripts (e.g.
  `kd_sga_example.py`, `kd_dlga_example.py`, `kd_dscv_example.py`) to
  get familiar with the `load_pde + fit_dataset` style entry points (or
  their equivalents) and the unified visual outputs.

- Model / algorithm exploration:  
  Look at advanced examples such as `kd_sga_custom_example.py`,
  `kd_dlga_viz_api_example.py`, `kd_dscv_example.py`,
  `kd_dscvspr_example.py` to see more detailed configuration options and
  additional diagnostics.

- Visualisation and diagnostics:  
  Scripts with `viz_api_example` in their names focus on the
  `kd.viz` façade and show how to drive SGA/DLGA/DSCV/SPR/PySR
  visualisations through a unified adapter interface.

These examples are not installed as part of the `kd` package (e.g. via
PyPI). To run them, clone the repository locally and execute the
scripts from the project root.
