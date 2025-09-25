# KD Visualization Helpers

This note captures the current capabilities of the unified `kd.viz` façade. It
is meant for teammates who want a quick reference outside of the README.

## Core entry points

```python
from kd.viz import configure, plot_training_curve, plot_validation_curve,
    plot_search_evolution, plot_optimization, plot_residuals,
    plot_field_comparison, render_equation
```

Usage pattern:

```python
from kd.viz import configure, plot_training_curve

configure(save_dir="artifacts/viz")  # optional but recommended
result = plot_training_curve(model)
print(result.paths)  # saved figure paths, empty if nothing written
```

Every helper returns a `VizResult` with:

- `paths`: list of saved figure files
- `warnings`: non-empty if the intent is unsupported or required data missing
- `metadata`: intent-specific extras (`ResidualPlotData`, summaries, etc.)

Helpers auto-apply the global Matplotlib style configured via `configure()`.

## Intent overview

| Intent | Helper | Notes |
| --- | --- | --- |
| `training_curve` | `plot_training_curve(model)` | Requires `model.train_loss_history` |
| `validation_curve` | `plot_validation_curve(model)` | Requires validation history |
| `search_evolution` | `plot_search_evolution(model)` | Handled directly by adapter |
| `optimization` | `plot_optimization(model)` | Uses `OptimizationHistoryData` |
| `equation` | `render_equation(model, font_size=16, ...)` | Produces LaTeX PNG |
| `residual` | `plot_residuals(model, actual, predicted, coordinates=None)` | Uses `ResidualPlotData` |
| `field_comparison` | `plot_field_comparison(model, x_coords, t_coords, true_field, predicted_field, residual_field=None)` | Uses `FieldComparisonData` |

## Data contracts

`kd.viz._contracts` exposes the dataclasses the adapters populate. They are
available via `from kd.viz import ResidualPlotData, OptimizationHistoryData,
FieldComparisonData`.

- `ResidualPlotData`: normalized residual arrays; adapters can attach optional
  coordinates and metadata.
- `OptimizationHistoryData`: steps, objective values, optional complexity /
  population statistics.
- `FieldComparisonData`: grid coordinates with true/predicted/residual field
  data.

Adapters should create these contracts before calling legacy helpers to keep the
intent surface consistent across models.

## Current adapter status (2024-xx)

- **DLGA**: training, validation, search evolution, optimization, residual,
  field comparison, equation supported.
- **DSCV**: placeholder adapter only (returns warning).
- **SGA**: integration pending; continue using legacy plotting for now.

## Example script

`examples/kd_dlga_viz_api_example.py` showcases the full DLGA flow using these
helpers. Running it produces figures under `artifacts/dlga_viz/`.

## Next steps

- Add derivative relationship & parity intents via new contracts.
- Integrate DSCV & SGA adapters once contracts are agreed.
- Backfill README/usage docs whenever helpers expand.
