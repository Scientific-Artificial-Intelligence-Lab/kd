Visualization
=============

KD ships with a unified visualization facade in :mod:`kd.viz` that
provides high-level plotting helpers for all backends.

Basic usage
-----------

.. code-block:: python

   from kd.viz import configure, render_equation, plot_field_comparison, plot_parity

   configure(save_dir="artifacts/viz")

   render_equation(model)                      # equation as LaTeX-rendered PNG
   plot_field_comparison(model, x_coords=None,
                         t_coords=None,
                         true_field=None,
                         predicted_field=None)  # predicted vs true field heatmap
   plot_parity(model)                           # actual vs predicted scatter

Each helper returns a :class:`kd.viz.core.VizResult` containing:

* ``paths`` -- saved figure paths (if any),
* ``warnings`` -- diagnostics about missing data or unsupported intents,
* ``metadata`` -- a normalised data contract.

Available plot functions
------------------------

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - ``render_equation``
     - Render discovered equation as LaTeX PNG
   * - ``plot_field_comparison``
     - True vs predicted field heatmaps
   * - ``plot_parity``
     - Actual vs predicted scatter plot
   * - ``plot_residuals``
     - Residual analysis
   * - ``plot_training_curve``
     - Training loss curve (DLGA, DSCV)
   * - ``plot_search_evolution``
     - Search/evolution history (DLGA)
   * - ``plot_time_slices``
     - Field snapshots at selected times
   * - ``plot_derivative_relationships``
     - Top derivative term scatter plots

Adapter coverage
----------------

The current adapters (see :mod:`kd.viz.adapters`) provide intent coverage for:

* **SGA** -- equation rendering, field comparison, parity
* **DLGA** -- training/validation curves, search evolution, optimisation
  history, equation, residuals, field comparison, time slices, parity
* **DSCV / DSCV SPR** -- reward evolution, density plots, expression trees,
  equation, residuals, field comparison, parity, and PINN-specific
  diagnostics where available
