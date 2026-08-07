"""Typed dataset-inspection report returned by ``kd.preview_report``.

Public surface (re-exported from ``kd``):

- :class:`AxisReport` — per-axis length, range, step statistics and spacing verdict.
- :class:`FieldReport` — per-field dtype, shape, finite-subset statistics and
  NaN/Inf counts.
- :class:`DatasetReport` — the whole-dataset report, including the warning list
  ``kd.preview`` prints.

Every type is a frozen dataclass with a hand-written ``to_dict()`` returning
JSON-native primitives, so an external caller (an agent controller, an MCP
boundary) can consume the same facts ``kd.preview`` prints. Non-finite floats
are sanitized to ``None`` on serialization rather than rejected at
construction: describing dirty data is exactly this report's job, so a field of
all-NaN values is a legal input.

Answer-blind by construction: the report is built from an explicit field list
(name, topology, axes, fields, LHS spec) and never reads
``PDEDataset.ground_truth`` or ``PDEDataset.noise_level``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

from kd.core.jsonsafe import sanitize_float









AxisSpacing = Literal[
    "uniform",
    "non_uniform",
    "decreasing",
    "non_finite",
    "single_point",
    "not_applicable",
]

AXIS_SPACING_VALUES: Final[frozenset[str]] = frozenset(
    {
        "uniform",
        "non_uniform",
        "decreasing",
        "non_finite",
        "single_point",
        "not_applicable",
    }
)


@dataclass(frozen=True, kw_only=True)
class AxisReport:
    """Per-axis facts of one dataset axis.

    Attributes:
        name: Axis name as it appears in ``dataset.axis_order``.
        n: Number of coordinate values.
        min: Smallest coordinate value.
        max: Largest coordinate value.
        spacing: Spacing verdict (see :data:`AxisSpacing`). The ``"uniform"``
            verdict is the shared ``is_uniform_grid`` predicate the
            finite-difference provider uses, so the report never disagrees with
            the FD accept/reject decision.
        step_first: First difference ``values[1] - values[0]``; ``None`` when no
            step exists (single point, or SCATTERED topology).
        step_mean: Mean of all first differences; ``None`` as above.
        step_min: Smallest first difference; ``None`` as above.
        step_max: Largest first difference; ``None`` as above.
    """

    name: str
    n: int
    min: float
    max: float
    spacing: AxisSpacing
    step_first: float | None
    step_mean: float | None
    step_min: float | None
    step_max: float | None

    def __post_init__(self) -> None:
        """Validate the axis report domain."""
        if self.n < 0:
            raise ValueError(f"axis report n must be >= 0, got {self.n}")
        if self.spacing not in AXIS_SPACING_VALUES:
            raise ValueError(f"unknown axis spacing verdict: {self.spacing!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict of this axis report.

        Non-finite coordinates and steps sanitize to ``None`` (RFC 8259 has no
        NaN/Inf literal), so the value is indistinguishable from an absent
        step; ``spacing`` carries the verdict that tells them apart.
        """
        return {
            "name": self.name,
            "n": self.n,
            "min": sanitize_float(self.min),
            "max": sanitize_float(self.max),
            "spacing": self.spacing,
            "step_first": _optional_float(self.step_first),
            "step_mean": _optional_float(self.step_mean),
            "step_min": _optional_float(self.step_min),
            "step_max": _optional_float(self.step_max),
        }


@dataclass(frozen=True, kw_only=True)
class FieldReport:
    """Per-field facts of one dataset field.

    Attributes:
        name: Field name as it appears in ``dataset.fields``.
        dtype: Torch dtype without the ``torch.`` prefix (e.g. ``"float64"``).
        shape: Tensor shape as a tuple of ints.
        min: Smallest value over the FINITE entries; NaN when nothing is finite.
        max: Largest value over the finite entries; NaN when nothing is finite.
        mean: Mean over the finite entries; NaN when nothing is finite.
        nan_count: Number of NaN entries.
        inf_count: Number of Inf entries (either sign).
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    min: float
    max: float
    mean: float
    nan_count: int
    inf_count: int

    def __post_init__(self) -> None:
        """Validate the field report domain."""
        if self.nan_count < 0 or self.inf_count < 0:
            raise ValueError(
                "field report NaN/Inf counts must be >= 0, got "
                f"nan_count={self.nan_count}, inf_count={self.inf_count}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict of this field report.

        Statistics of an entirely non-finite field sanitize to ``None``; the
        ``nan_count`` / ``inf_count`` pair keeps the reason visible.
        """
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "min": sanitize_float(self.min),
            "max": sanitize_float(self.max),
            "mean": sanitize_float(self.mean),
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
        }


@dataclass(frozen=True, kw_only=True)
class DatasetReport:
    """Structured result of ``kd.preview_report`` (JSON-dumpable, answer-blind).

    Attributes:
        name: Dataset identifier.
        topology: ``DataTopology`` value string (``"grid"`` / ``"scattered"``).
        axes: Per-axis reports in ``axis_order``; ``None`` when the dataset
            carries no axis payload at all (metadata-only datasets).
        fields: Per-field reports in field insertion order; ``None`` when the
            dataset carries no field payload at all.
        lhs_field: LHS field name, or ``None`` when unset.
        lhs_axis: LHS axis name, or ``None`` when unset.
        lhs_label: Canonical name of the LHS derivative — the regression target
            a fit on this dataset predicts. ORDER-AWARE: ``"u_t"`` at
            ``lhs_order=1``, ``"u_tt"`` at ``lhs_order=2``, the same rendering
            ``ExperimentResult.lhs_label`` carries, so the repository's own
            decoder (``kd.core.expr.naming.parse_derivative_name``) reads back
            the field, axis and order this report states. ``None`` when either
            half of the LHS spec is unset. Two notes: ``kd.preview`` prints the
            order-free ``"{lhs_field}_{lhs_axis}"`` form instead (frozen output
            bytes), and an axis name the derivative grammar cannot encode (one
            containing ``"_"``) falls back to that same order-free form —
            ``lhs_order`` is authoritative in both cases.
        lhs_order: Order of the LHS derivative along ``lhs_axis``.
        warnings: Human-readable warnings in the order ``kd.preview`` prints
            them: per axis (spacing, then small grid) in ``axis_order``, then
            per field (NaN, then Inf) in insertion order, then mixed field
            dtypes, then unset LHS.
    """

    name: str
    topology: str
    axes: list[AxisReport] | None
    fields: list[FieldReport] | None
    lhs_field: str | None
    lhs_axis: str | None
    lhs_label: str | None
    lhs_order: int
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict of the full report (MCP-boundary contract)."""
        return {
            "name": self.name,
            "topology": self.topology,
            "axes": (
                None if self.axes is None else [axis.to_dict() for axis in self.axes]
            ),
            "fields": (
                None
                if self.fields is None
                else [field.to_dict() for field in self.fields]
            ),
            "lhs_field": self.lhs_field,
            "lhs_axis": self.lhs_axis,
            "lhs_label": self.lhs_label,
            "lhs_order": self.lhs_order,
            "warnings": list(self.warnings),
        }


def _optional_float(value: float | None) -> float | None:
    """Sanitize an optional float, keeping ``None`` for "no such step"."""
    return None if value is None else sanitize_float(value)
