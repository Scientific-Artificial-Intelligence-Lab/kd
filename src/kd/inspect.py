"""Sanity-check tools for PDE datasets.

Public surface:
- :func:`preview` — pretty-print a one-shot summary of a ``PDEDataset``,
  flagging non-uniform grids, NaN/Inf, very small grids and dtype warnings.

Designed for interactive use right before ``Model.fit()``. Output goes to
stdout (or any text stream) so users can spot grid-construction mistakes
without needing to wire up logging.
"""

from __future__ import annotations

import math
from typing import TextIO

import torch

from kd.data.derivatives.finite_diff import (
    UNIFORM_GRID_RTOL,
    is_uniform_grid,
)
from kd.data.schema import PDEDataset



_SMALL_GRID_THRESHOLD = 16


_AXIS_NAME_W = 4
_FIELD_NAME_W = 4


def preview(dataset: PDEDataset, *, file: TextIO | None = None) -> None:
    """Print a sanity-check summary of a ``PDEDataset``.

    Outputs (in order):
    - Dataset name
    - Per-axis: name, length, range, step (with uniform-spacing check)
    - Per-field: name, dtype, shape, min/max/mean, NaN/Inf count
    - Auto-detected LHS (from ``dataset.lhs_field`` / ``lhs_axis``)
    - Status line + warnings (non-uniform grid, NaN/Inf, small grid,
      dtype mismatch).

    Args:
        dataset: The dataset to inspect.
        file: Optional text stream (defaults to ``sys.stdout`` via ``print``).

    Returns:
        ``None`` — this is a side-effecting print tool.
    """
    warnings: list[str] = []
    lines: list[str] = []

    lines.append(f"Dataset: {dataset.name}")


    lines.append("Axes:")
    if dataset.axes is None or dataset.axis_order is None:
        lines.append(" (none)")
    else:
        for axis_name in dataset.axis_order:
            axis = dataset.axes[axis_name]
            lines.append(_format_axis_line(axis_name, axis.values, warnings))


    lines.append("Fields:")
    field_dtypes: set[torch.dtype] = set()
    if dataset.fields is None:
        lines.append(" (none)")
    else:
        for field_name, field in dataset.fields.items():
            field_dtypes.add(field.values.dtype)
            lines.append(_format_field_line(field_name, field.values, warnings))


    if len(field_dtypes) > 1:
        warnings.append(
            f"fields have mixed dtypes {sorted(str(d) for d in field_dtypes)} "
            f"— consider casting all fields to the same dtype"
        )


    if dataset.lhs_field and dataset.lhs_axis:
        lines.append(
            f"LHS: {dataset.lhs_field}_{dataset.lhs_axis} "
            f"(field='{dataset.lhs_field}', axis='{dataset.lhs_axis}')"
        )
    else:
        lines.append("LHS: (unset)")
        warnings.append(
            "lhs_field / lhs_axis not set — Model.fit() will fall back to ('u', 't')"
        )


    if warnings:
        lines.append(f"Status: {len(warnings)} warning(s)")
        for warn in warnings:
            lines.append(f" - WARNING: {warn}")
    else:
        lines.append("Status: ready to fit")

    print("\n".join(lines), file=file)


def _format_axis_line(
    name: str,
    values: torch.Tensor,
    warnings: list[str],
) -> str:
    """Render one axis row and append any axis-specific warnings.

    The uniformity verdict delegates to ``is_uniform_grid`` (the same
    predicate used by ``FiniteDiffProvider`` and the integrator) so the
    preview never disagrees with the FD provider's accept/reject decision.
    Descending and inf-drift coords are surfaced as explicit warnings
    instead of being mis-blessed as ``uniform``.
    """
    n = values.numel()
    vmin = float(values.min().item())
    vmax = float(values.max().item())

    if n < 2:
        step_part = "step n/a (single point)"
    else:
        diffs = values[1:] - values[:-1]
        dx0 = float(diffs[0].item())
        diff_mean = float(diffs.mean().item())
        diff_min = float(diffs.min().item())
        diff_max = float(diffs.max().item())

        if not math.isfinite(dx0):
            step_part = f"step {dx0:.4g} (NON-UNIFORM, non-finite spacing)"
            warnings.append(
                f"axis '{name}' has non-finite spacing dx0={dx0:.4g} — "
                "finite-difference stencils require finite dx"
            )
        elif dx0 < 0:

            step_part = f"step {diff_mean:.4g} (NON-UNIFORM, decreasing)"
            warnings.append(
                f"axis '{name}' has decreasing spacing (dx0={dx0:.4g}) — "
                "finite-difference stencils require monotonic increasing "
                "coordinates (flip the array before fitting)"
            )
        elif is_uniform_grid(values, rtol=UNIFORM_GRID_RTOL):
            step_part = f"step {diff_mean:.4g} (uniform)"
        else:
            step_part = (
                f"step {diff_mean:.4g} "
                f"(NON-UNIFORM, range {diff_min:.4g}-{diff_max:.4g})"
            )
            warnings.append(
                f"axis '{name}' is not uniformly spaced "
                f"(steps from {diff_min:.4g} to {diff_max:.4g}) — "
                "finite-difference derivatives assume uniform grids"
            )

    if n < _SMALL_GRID_THRESHOLD:
        warnings.append(
            f"axis '{name}' has only {n} points (<{_SMALL_GRID_THRESHOLD}) "
            "— small grid (n<16) may make fits unstable"
        )

    return (
        f" {name:>{_AXIS_NAME_W}} | n={n:<4d} | "
        f"range [{vmin:.3f}, {vmax:.3f}] | {step_part}"
    )


def _format_field_line(
    name: str,
    values: torch.Tensor,
    warnings: list[str],
) -> str:
    """Render one field row and append any field-specific warnings."""
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
        nan_part = f"NaN={nan_count} (>0)" if nan_count > 0 else f"NaN={nan_count}"
        if inf_count > 0:
            nan_part += f" Inf={inf_count} (>0)"
    else:
        vmin = float(values.min().item())
        vmax = float(values.max().item())
        vmean = float(values.mean().item())
        nan_part = f"NaN={nan_count}"

    shape = tuple(values.shape)
    return (
        f" {name:>{_FIELD_NAME_W}} | dtype={str(values.dtype).replace('torch.', '')} "
        f"| shape={shape} | min={vmin:.3f} max={vmax:.3f} mean={vmean:.3f} ({nan_part})"
    )
