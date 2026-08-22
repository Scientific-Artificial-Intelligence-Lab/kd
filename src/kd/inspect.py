"""Sanity-check tools for PDE datasets.

Public surface:
- :func:`preview_report` — return a typed, JSON-safe :class:`DatasetReport`
  holding every fact the preview computes (per-axis spacing, per-field
  statistics, LHS spec, warnings), for programmatic callers such as an agent
  controller.
- :func:`preview` — pretty-print a one-shot summary of a ``PDEDataset``,
  flagging non-uniform grids, NaN/Inf, very small grids and dtype warnings.
  It builds the same report and renders it.

Designed for interactive use right before ``Model.fit()``. Output goes to
stdout (or any text stream) so users can spot grid-construction mistakes
without needing to wire up logging.
"""

from __future__ import annotations

import math
from typing import TextIO

import torch

from kd._inspect_report import (
    AxisReport,
    AxisSpacing,
    DatasetReport,
    FieldReport,
)
from kd.core.equation.rendering import render_lhs_label
from kd.core.equation.types import LhsSpec
from kd.data.derivatives.finite_diff import (
    UNIFORM_GRID_RTOL,
    is_uniform_grid,
)
from kd.data.schema import DataTopology, PDEDataset

__all__ = [
    "AxisReport",
    "DatasetReport",
    "FieldReport",
    "preview",
    "preview_report",
]



_SMALL_GRID_THRESHOLD = 16


_AXIS_NAME_W = 4
_FIELD_NAME_W = 4


def preview_report(dataset: PDEDataset) -> DatasetReport:
    """Build a typed report of a ``PDEDataset`` without printing anything.

    This is the computation behind :func:`preview`: the same per-axis, per-field
    and LHS facts, plus the same warning list in the same order, returned as a
    JSON-dumpable :class:`~kd._inspect_report.DatasetReport` instead of a
    formatted string.

    Topology matters. On a SCATTERED dataset each ``AxisInfo.values`` holds
    per-point coordinates rather than a grid vector, so no step exists: every
    axis reports ``spacing="not_applicable"`` with all four step statistics
    ``None``, and neither the spacing warnings nor the small-grid warning (whose
    meaning is "few grid points") is emitted. Field-level NaN/Inf/dtype and LHS
    warnings apply to both topologies.

    The report is answer-blind: it reads only name, topology, axes, axis_order,
    fields and the LHS spec. ``dataset.ground_truth`` and ``dataset.noise_level``
    are never touched.

    Args:
        dataset: The dataset to inspect. Not mutated; no RNG is consumed.

    Returns:
        A frozen ``DatasetReport``.
    """
    warnings: list[str] = []
    scattered = dataset.topology is DataTopology.SCATTERED


    axes: list[AxisReport] | None
    if dataset.axes is None or dataset.axis_order is None:
        axes = None
    else:
        axes = [
            _build_axis_report(
                axis_name,
                dataset.axes[axis_name].values,
                warnings,
                scattered=scattered,
            )
            for axis_name in dataset.axis_order
        ]


    fields: list[FieldReport] | None
    field_dtypes: set[torch.dtype] = set()
    if dataset.fields is None:
        fields = None
    else:
        fields = []
        for field_name, field in dataset.fields.items():
            field_dtypes.add(field.values.dtype)
            fields.append(_build_field_report(field_name, field.values, warnings))





    if len(field_dtypes) > 1:
        warnings.append(
            f"fields have mixed dtypes {sorted(str(d) for d in field_dtypes)} "
            f"— consider casting all fields to the same dtype"
        )


    lhs_label: str | None
    if dataset.topology is DataTopology.TABULAR:
        lhs_label = dataset.lhs_field
    elif dataset.lhs_field and dataset.lhs_axis:
        lhs_label = _build_lhs_label(
            dataset.lhs_field, dataset.lhs_axis, dataset.lhs_order
        )
    else:
        lhs_label = None
        warnings.append(
            "lhs_field / lhs_axis not set — Model.fit() will fall back to ('u', 't')"
        )

    return DatasetReport(
        name=dataset.name,
        topology=dataset.topology.value,
        axes=axes,
        fields=fields,
        lhs_field=dataset.lhs_field or None,
        lhs_axis=dataset.lhs_axis or None,
        lhs_label=lhs_label,
        lhs_order=dataset.lhs_order,
        warnings=warnings,
    )


def preview(dataset: PDEDataset, *, file: TextIO | None = None) -> None:
    """Print a sanity-check summary of a ``PDEDataset``.

    Outputs (in order):
    - Dataset name
    - Per-axis: name, length, range, step (with uniform-spacing check)
    - Per-field: name, dtype, shape, min/max/mean, NaN/Inf count
    - Auto-detected LHS (from ``dataset.lhs_field`` / ``lhs_axis``)
    - Status line + warnings (non-uniform grid, NaN/Inf, small grid,
      dtype mismatch).

    The facts come from :func:`preview_report`; this function only renders
    them. Call ``preview_report`` directly when a program, rather than a
    person, is the reader.

    Args:
        dataset: The dataset to inspect.
        file: Optional text stream (defaults to ``sys.stdout`` via ``print``).

    Returns:
        ``None`` — this is a side-effecting print tool.
    """
    print(_render_report(preview_report(dataset)), file=file)







def _build_lhs_label(field: str, axis: str, order: int) -> str:
    """Render the order-aware name of the LHS derivative.

    Delegates to the repository's single renderer for this string
    (``render_lhs_label``), so the report names the regression target the same
    way ``ExperimentResult.lhs_label`` does: ``u_t`` at order 1, ``u_tt`` at
    order 2. A first-order label read as if it were the target of a
    second-order dataset would point a caller at the wrong equation.

    ``build_derivative_name`` (behind the renderer) rejects an axis name
    containing ``"_"``, which a directly constructed ``PDEDataset`` permits.
    A report must describe such a dataset rather than raise, so that case
    degrades to the order-free ``"{field}_{axis}"`` form; ``lhs_order`` stays
    authoritative.
    """
    try:
        return render_lhs_label(LhsSpec(field=field, axis=axis, order=order))
    except ValueError:
        return f"{field}_{axis}"


def _build_axis_report(
    name: str,
    values: torch.Tensor,
    warnings: list[str],
    *,
    scattered: bool,
) -> AxisReport:
    """Measure one axis and append any axis-specific warnings.

    The uniformity verdict delegates to ``is_uniform_grid`` (the same
    predicate used by ``FiniteDiffProvider`` and the integrator) so the
    report never disagrees with the FD provider's accept/reject decision.
    Descending and inf-drift coords are surfaced as explicit warnings
    instead of being mis-blessed as ``uniform``.
    """
    n = values.numel()
    vmin = float(values.min().item())
    vmax = float(values.max().item())

    spacing: AxisSpacing
    step_first: float | None = None
    step_mean: float | None = None
    step_min: float | None = None
    step_max: float | None = None

    if scattered:



        spacing = "not_applicable"
    elif n < 2:
        spacing = "single_point"
    else:
        diffs = values[1:] - values[:-1]
        step_first = float(diffs[0].item())
        step_mean = float(diffs.mean().item())
        step_min = float(diffs.min().item())
        step_max = float(diffs.max().item())

        if not math.isfinite(step_first):
            spacing = "non_finite"
            warnings.append(
                f"axis '{name}' has non-finite spacing dx0={step_first:.4g} — "
                "finite-difference stencils require finite dx"
            )
        elif step_first < 0:

            spacing = "decreasing"
            warnings.append(
                f"axis '{name}' has decreasing spacing (dx0={step_first:.4g}) — "
                "finite-difference stencils require monotonic increasing "
                "coordinates (flip the array before fitting)"
            )
        elif is_uniform_grid(values, rtol=UNIFORM_GRID_RTOL):
            spacing = "uniform"
        else:
            spacing = "non_uniform"
            warnings.append(
                f"axis '{name}' is not uniformly spaced "
                f"(steps from {step_min:.4g} to {step_max:.4g}) — "
                "finite-difference derivatives assume uniform grids"
            )

    if not scattered and n < _SMALL_GRID_THRESHOLD:
        warnings.append(
            f"axis '{name}' has only {n} points (<{_SMALL_GRID_THRESHOLD}) "
            "— small grid (n<16) may make fits unstable"
        )

    return AxisReport(
        name=name,
        n=n,
        min=vmin,
        max=vmax,
        spacing=spacing,
        step_first=step_first,
        step_mean=step_mean,
        step_min=step_min,
        step_max=step_max,
    )


def _build_field_report(
    name: str,
    values: torch.Tensor,
    warnings: list[str],
) -> FieldReport:
    """Measure one field and append any field-specific warnings."""
    nan_count = int(torch.isnan(values).sum().item())
    inf_count = int(torch.isinf(values).sum().item())

    if nan_count > 0 or inf_count > 0:

        finite_mask = torch.isfinite(values)
        if finite_mask.any():
            finite_vals = values[finite_mask]
            vmin = float(finite_vals.min().item())
            vmax = float(finite_vals.max().item())
            vmean = float(finite_vals.mean().item())
        else:
            vmin = vmax = vmean = float("nan")
        if nan_count > 0:
            warnings.append(
                f"field '{name}' contains {nan_count} NaN value(s) — "
                "dataset will not fit until cleaned"
            )
        if inf_count > 0:
            warnings.append(
                f"field '{name}' contains {inf_count} Inf value(s) — "
                "dataset will not fit until cleaned"
            )
    else:
        vmin = float(values.min().item())
        vmax = float(values.max().item())
        vmean = float(values.mean().item())

    return FieldReport(
        name=name,
        dtype=str(values.dtype).replace("torch.", ""),
        shape=tuple(int(dim) for dim in values.shape),
        min=vmin,
        max=vmax,
        mean=vmean,
        nan_count=nan_count,
        inf_count=inf_count,
    )







def _render_report(report: DatasetReport) -> str:
    """Render a ``DatasetReport`` as the text ``preview`` prints."""
    lines: list[str] = [f"Dataset: {report.name}"]

    lines.append("Axes:")
    if report.axes is None:
        lines.append(" (none)")
    else:
        lines.extend(_format_axis_line(axis) for axis in report.axes)

    lines.append("Fields:")
    if report.fields is None:
        lines.append(" (none)")
    else:
        lines.extend(_format_field_line(field) for field in report.fields)

    if report.topology == DataTopology.TABULAR.value:



        lines.append(f"LHS: {report.lhs_label} (target column)")
    elif report.lhs_field is not None and report.lhs_axis is not None:




        lines.append(
            f"LHS: {report.lhs_field}_{report.lhs_axis} "
            f"(field='{report.lhs_field}', axis='{report.lhs_axis}')"
        )
    else:
        lines.append("LHS: (unset)")

    if report.warnings:
        lines.append(f"Status: {len(report.warnings)} warning(s)")
        lines.extend(f" - WARNING: {warn}" for warn in report.warnings)
    else:
        lines.append("Status: ready to fit")

    return "\n".join(lines)


def _format_axis_line(axis: AxisReport) -> str:
    """Render one axis row."""
    return (
        f" {axis.name:>{_AXIS_NAME_W}} | n={axis.n:<4d} | "
        f"range [{axis.min:.3f}, {axis.max:.3f}] | {_format_step(axis)}"
    )


def _format_step(axis: AxisReport) -> str:
    """Render the step part of an axis row from its spacing verdict."""
    if axis.spacing == "not_applicable":
        return "step n/a (scattered)"
    if axis.spacing == "single_point":
        return "step n/a (single point)"
    if axis.spacing == "non_finite":
        dx0 = _step_value(axis.step_first)
        return f"step {dx0:.4g} (NON-UNIFORM, non-finite spacing)"

    mean = _step_value(axis.step_mean)
    if axis.spacing == "decreasing":
        return f"step {mean:.4g} (NON-UNIFORM, decreasing)"
    if axis.spacing == "uniform":
        return f"step {mean:.4g} (uniform)"

    low = _step_value(axis.step_min)
    high = _step_value(axis.step_max)
    return f"step {mean:.4g} (NON-UNIFORM, range {low:.4g}-{high:.4g})"


def _step_value(value: float | None) -> float:
    """Return a step statistic that the spacing verdict promised is present."""
    if value is None:
        raise ValueError("axis spacing verdict requires step statistics")
    return value


def _format_field_line(field: FieldReport) -> str:
    """Render one field row."""
    nan_part = f"NaN={field.nan_count}"
    if field.nan_count > 0:
        nan_part += " (>0)"
    if field.inf_count > 0:
        nan_part += f" Inf={field.inf_count} (>0)"

    return (
        f" {field.name:>{_FIELD_NAME_W}} | dtype={field.dtype} "
        f"| shape={field.shape} | min={field.min:.3f} max={field.max:.3f} "
        f"mean={field.mean:.3f} ({nan_part})"
    )
