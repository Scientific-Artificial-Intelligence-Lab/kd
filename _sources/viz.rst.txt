Visualization
=============

KD ships with a unified visualization façade in :mod:`kd.viz.api` that
provides a small set of high-level plotting helpers.

Basic usage
-----------

.. code-block:: python

   configure(save_dir="artifacts/viz")
   plot_training_curve(model)
   plot_residuals(model, actual=y_true, predicted=y_pred, coordinates=X)

Each helper returns a :class:`kd.viz.core.VizResult` containing:

* ``paths`` – saved figure paths (if any),
* ``warnings`` – diagnostics about missing data or unsupported intents,
* ``metadata`` – a normalised data contract such as
  :class:`kd.viz._contracts.ResidualPlotData` or
  :class:`kd.viz._contracts.FieldComparisonData`.

Adapter coverage
----------------

The current adapters (see :mod:`kd.viz.adapters`) provide intent coverage for:

* SGA – equation rendering, residuals, field comparison, parity for the
  supported PDE benchmarks and custom ``u(x, t)`` fields.
* DLGA – training/validation curves, search evolution, optimisation history,
  equation, residuals, field comparison, time slices and parity.
* DSCV / DSCV SPR – reward evolution, density plots, expression trees,
  equation, residuals, field comparison, parity, and PINN-specific residual
  and field diagnostics where available.

For more internal design notes, see ``notes/viz/viz_helpers.md`` in the
repository.

