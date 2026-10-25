
from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from numpy.typing import NDArray

from kd.core.equation._sketch_terms import lower_term
from kd.core.expr.executor import UNITY_TOKEN, _is_diff_operator, _parse_diff_name
from kd.core.expr.field_power import field_power
from kd.core.expr.naming import parse_compound_derivative
from kd.data.derivatives.finite_diff import UNIFORM_GRID_RTOL, is_uniform_grid
from kd.data.diagnostics import noise_estimate
from kd.data.schema import DataTopology

if TYPE_CHECKING:
    from kd.data.schema import PDEDataset

__all__ = [
    "METHOD_VERSION",
    "WeakDesignError",
    "WeakFormDiagnostics",
    "WeakFormOptions",
    "WeakFrame",
    "WeakRefusal",
    "WeakSolution",
    "WeakTerm",
    "check_weak_dataset",
    "classify_weak_terms",
    "raise_weak_refusals",
    "resolve_weak_frame",
    "solve_weak",
    "weak_diagnostics",
    "weak_zero_model",
]

METHOD_VERSION: Final[str] = "kd-weak-v1"
BUMP_DEGREE: Final[int] = 4
_MAX_ORDER: Final[int] = 3
_MAX_POWER: Final[int] = 3
_MIN_SEGMENT: Final[int] = 3
_FIRST_WIDTH: Final[int] = 4
_SPECTRUM_FLOOR: Final[float] = 1e-300


_DEGENERATE_VARIANCE_RTOL: Final[float] = 1e-12
_CLASS = (
    "one, the field, its square and cube as n2/n3 of the field or a raw mul "
    "product of the field with itself, pure spatial derivatives of order 1-3 "
    "and mul(field, first spatial derivative)"
)

Array = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class WeakFormOptions:

    half_widths: Mapping[str, int] | None = None
    spectrum_tau: float = 0.03
    noise_correction: Literal["none", "gaussian"] = "none"
    noise_sigma: float | None = None

    def __post_init__(self) -> None:
        if self.noise_correction not in ("none", "gaussian"):
            raise ValueError(
                "noise_correction must be 'none' or 'gaussian'; "
                f"got {self.noise_correction!r}"
            )
        if self.noise_sigma is not None:
            if self.noise_correction != "gaussian":
                raise ValueError("noise_sigma requires noise_correction='gaussian'")
            if not (math.isfinite(self.noise_sigma) and self.noise_sigma >= 0.0):
                raise ValueError(
                    "noise_sigma must be finite and nonnegative; "
                    f"got {self.noise_sigma}"
                )
        if not (math.isfinite(self.spectrum_tau) and self.spectrum_tau > 0.0):
            raise ValueError(
                f"spectrum_tau must be finite and positive; got {self.spectrum_tau}"
            )
        if self.half_widths is not None and any(
            m < 1 for m in self.half_widths.values()
        ):
            raise ValueError(f"half_widths must be at least 1; got {self.half_widths}")


@dataclass(frozen=True)
class WeakFormDiagnostics:

    method_version: str
    spatial_axis: str
    time_axis: str
    half_widths: Mapping[str, int]
    strides: Mapping[str, int]
    corner_indices: Mapping[str, int] | None
    spectrum_tau: float
    bump_degree: int
    n_windows: int
    rank: int
    n_columns: int
    target_variance: float
    noise_correction: str
    noise_sigma: float | None
    sigma_estimates: Mapping[str, float] | None
    condition_number_basis: str


class WeakDesignError(ValueError):
    pass


@dataclass(frozen=True)
class WeakTerm:

    kind: Literal["power", "derivative", "advection"]
    order: int


@dataclass(frozen=True)
class WeakRefusal:

    term: str
    reason: str
    unsupported: bool


@dataclass(frozen=True)
class WeakFrame:

    spatial_axis: str
    time_axis: str
    field: Array
    space_tests: tuple[Array, ...]
    time_tests: tuple[Array, ...]
    target: Array
    half_widths: dict[str, int]
    corner_indices: dict[str, int] | None
    spectrum_tau: float
    noise_correction: str
    noise_sigma: float | None
    sigma_estimates: dict[str, float] | None

    @property
    def n_windows(self) -> int:
        return int(self.target.size)


@dataclass(frozen=True)
class WeakSolution:

    coefficients: Array
    mse: float
    nmse: float
    r2: float
    condition_number: float | None
    rank: int


def check_weak_dataset(dataset: PDEDataset, lhs_order: int) -> tuple[str, str]:
    if dataset.topology is not DataTopology.GRID:
        raise NotImplementedError("the weak form integrates GRID datasets only")
    if lhs_order != 1:
        raise NotImplementedError(
            "the weak form supports first-order evolution only; got lhs_order="
            f"{lhs_order}"
        )
    fields = sorted(dataset.fields or {})
    if len(fields) != 1:
        raise NotImplementedError(
            f"the weak form supports one field; the dataset has fields {fields}"
        )
    if dataset.get_field(fields[0]).is_complex():
        raise NotImplementedError("the weak form supports a real field")
    spatial = list(dataset.spatial_axes)
    if len(spatial) != 1:
        raise NotImplementedError(
            f"the weak form supports one spatial axis; the dataset has {spatial}"
        )
    for axis in (spatial[0], dataset.lhs_axis):
        values = _axis_values(dataset, axis)
        if values.size < 2 or values[1] <= values[0]:
            raise NotImplementedError(f"axis '{axis}' must be increasing")
        if not is_uniform_grid(values, rtol=UNIFORM_GRID_RTOL):
            raise NotImplementedError(
                f"the weak form needs a uniform grid; axis '{axis}' is not uniform"
            )
    return spatial[0], dataset.lhs_axis


def _axis_values(dataset: PDEDataset, axis: str) -> Array:
    assert dataset.axes is not None
    return dataset.axes[axis].values.detach().cpu().numpy().astype(np.float64)


def classify_weak_terms(
    terms: Sequence[str],
    dataset: PDEDataset,
    *,
    spatial_axis: str,
    max_order: int,
) -> tuple[dict[str, WeakTerm], list[WeakRefusal]]:
    members: dict[str, WeakTerm] = {}
    refusals: list[WeakRefusal] = []
    for term in terms:
        member = _member(term, dataset, spatial_axis)
        if member is None:
            refusals.append(WeakRefusal(term, f"outside the class ({_CLASS})", True))
        elif member.kind == "derivative" and member.order > _MAX_ORDER:
            refusals.append(
                WeakRefusal(term, f"derivative order {member.order} above 3", True)
            )
        elif member.kind != "power" and member.order > max_order:
            refusals.append(
                WeakRefusal(
                    term,
                    f"derivative order {member.order} exceeds max_order={max_order}",
                    False,
                )
            )
        else:
            members[term] = member
    return members, refusals


def raise_weak_refusals(refusals: Sequence[WeakRefusal]) -> None:
    if not refusals:
        return
    summary = "; ".join(f"'{r.term}': {r.reason}" for r in refusals)
    message = f"the weak form refuses terms: {summary}"
    if any(r.unsupported for r in refusals):
        raise NotImplementedError(message)
    raise ValueError(message)


def _member(term: str, dataset: PDEDataset, spatial_axis: str) -> WeakTerm | None:
    try:
        node = ast.parse(lower_term(term).strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return None
    field = dataset.lhs_field
    if isinstance(node, ast.Name) and node.id == UNITY_TOKEN:
        return WeakTerm("power", 0)
    order = _derivative_order(node, dataset, spatial_axis)
    if order is not None:
        return WeakTerm("power", 1) if order == 0 else WeakTerm("derivative", order)
    power = field_power(node, (field,))
    if power is not None:
        return WeakTerm("power", power.degree) if power.degree <= _MAX_POWER else None
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
        return None
    head, args = node.func.id, node.args
    if head == "mul" and len(args) == 2:
        orders = sorted(
            -1 if o is None else o
            for o in (_derivative_order(a, dataset, spatial_axis) for a in args)
        )
        if orders == [0, 1]:
            return WeakTerm("advection", 1)
    return None


def _derivative_order(
    node: ast.expr, dataset: PDEDataset, spatial_axis: str
) -> int | None:
    field = dataset.lhs_field
    if isinstance(node, ast.Name):
        if node.id == field:
            return 0
        parsed = parse_compound_derivative(
            node.id,
            known_fields=set(dataset.fields or {}),
            known_axes=set(dataset.axes or {}),
        )
        if parsed is None or parsed[0] != field:
            return None
        if any(axis != spatial_axis for axis, _ in parsed[1]):
            return None
        return sum(order for _, order in parsed[1])
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and _is_diff_operator(node.func.id)
        and len(node.args) == 1
        and not node.keywords
    ):
        try:
            axis, order = _parse_diff_name(node.func.id)
        except ValueError:


            return None
        inner = _derivative_order(node.args[0], dataset, spatial_axis)
        if axis != spatial_axis or inner is None:
            return None
        return inner + order
    return None


def resolve_weak_frame(
    dataset: PDEDataset,
    options: WeakFormOptions,
    *,
    spatial_axis: str,
    time_axis: str,
) -> WeakFrame:
    axes = (spatial_axis, time_axis)
    field = _field_array(dataset, spatial_axis, time_axis)
    spacing = {
        axis: float(np.diff(_axis_values(dataset, axis)[:2])[0]) for axis in axes
    }
    corners: dict[str, int] | None = None
    if options.half_widths is None:
        corners = {a: spectral_corner(field, i, a) for i, a in enumerate(axes)}
        widths = {
            a: smallest_width(corners[a], field.shape[i], options.spectrum_tau, a)
            for i, a in enumerate(axes)
        }
    else:
        widths = _explicit_widths(options.half_widths, axes, field.shape)
    space_tests = tuple(
        _test_rows(field.shape[0], widths[spatial_axis], spacing[spatial_axis], r)
        for r in range(_MAX_ORDER + 1)
    )
    time_tests = tuple(
        _test_rows(field.shape[1], widths[time_axis], spacing[time_axis], r)
        for r in range(2)
    )
    target = -(space_tests[0] @ field @ time_tests[1].T).ravel()
    sigma, estimates = _noise_sigma(field, options, axes)
    return WeakFrame(
        spatial_axis=spatial_axis,
        time_axis=time_axis,
        field=field,
        space_tests=space_tests,
        time_tests=time_tests,
        target=target,
        half_widths=widths,
        corner_indices=corners,
        spectrum_tau=options.spectrum_tau,
        noise_correction=options.noise_correction,
        noise_sigma=sigma,
        sigma_estimates=estimates,
    )


def _field_array(dataset: PDEDataset, spatial_axis: str, time_axis: str) -> Array:
    values = dataset.get_field(dataset.lhs_field).detach().cpu().numpy()
    order = list(dataset.axis_order or [])
    return np.ascontiguousarray(
        np.transpose(values, (order.index(spatial_axis), order.index(time_axis))),
        dtype=np.float64,
    )


def _explicit_widths(
    half_widths: Mapping[str, int], axes: tuple[str, str], shape: tuple[int, ...]
) -> dict[str, int]:
    missing = [axis for axis in axes if axis not in half_widths]
    extra = sorted(set(half_widths) - set(axes))
    if missing or extra:
        raise ValueError(
            f"half_widths must name the spatial axis '{axes[0]}' and the time axis "
            f"'{axes[1]}'; missing {missing}, unknown {extra}"
        )
    for axis, n in zip(axes, shape, strict=True):
        if 2 * half_widths[axis] + 1 > n:
            raise ValueError(
                f"half-width {half_widths[axis]} along axis '{axis}' leaves no "
                f"window on {n} points"
            )
    return {axis: int(half_widths[axis]) for axis in axes}


def _bump(s: Array, order: int) -> Array:
    w = 1.0 - s**2
    if order == 0:
        return w**4
    if order == 1:
        return -8.0 * s * w**3
    if order == 2:
        return 8.0 * w**2 * (7.0 * s**2 - 1.0)
    return 48.0 * s * w * (3.0 - 7.0 * s**2)


def _test_rows(n: int, m: int, h: float, order: int) -> Array:
    values = _bump(np.arange(-m, m + 1) / m, order) / (m * h) ** order * h
    centres = range(m, n - m, max(1, m // 2))
    rows = np.zeros((len(centres), n), dtype=np.float64)
    for i, centre in enumerate(centres):
        rows[i, centre - m: centre + m + 1] = values
    return rows


def spectral_corner(field: Array, axis_index: int, axis: str) -> int:
    centred = field - field.mean(axis=axis_index, keepdims=True)
    power = np.abs(np.fft.rfft(centred, axis=axis_index)) ** 2
    spectrum = power.mean(axis=1 - axis_index)
    curve = np.cumsum(np.log(spectrum[1:] + _SPECTRUM_FLOOR))
    bins = np.arange(1, spectrum.size, dtype=np.float64)
    splits = range(_MIN_SEGMENT, bins.size - _MIN_SEGMENT + 1)
    if not splits:
        raise ValueError(
            f"axis '{axis}' has {bins.size} nonzero spectral bins; no split leaves "
            f"{_MIN_SEGMENT} on each side, so no spectral corner exists"
        )
    errors = [
        _affine_sse(bins[:j], curve[:j]) + _affine_sse(bins[j:], curve[j:])
        for j in splits
    ]
    return int(bins[splits[int(np.argmin(errors))]])


def _affine_sse(x: Array, y: Array) -> float:
    design = np.column_stack([x, np.ones_like(x)])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    return float(residual @ residual)


def smallest_width(corner: int, n: int, tau: float, axis: str) -> int:
    for m in range(_FIRST_WIDTH, n // 2):
        response = np.abs(np.fft.rfft(_bump(np.arange(-m, m + 1) / m, 0), n=n))
        if response[corner] <= tau * response[0]:
            return m
    raise ValueError(
        f"no half-width in [{_FIRST_WIDTH}, {n // 2}) along axis '{axis}' "
        f"attenuates corner bin {corner} to spectrum_tau={tau}"
    )


def _noise_sigma(
    field: Array, options: WeakFormOptions, axes: tuple[str, str]
) -> tuple[float | None, dict[str, float] | None]:
    if options.noise_correction == "none":
        return None, None
    if options.noise_sigma is not None:
        return options.noise_sigma, None
    std = float(np.std(field))
    estimates = {axis: noise_estimate(field, i) * std for i, axis in enumerate(axes)}
    return min(estimates.values()), estimates


def _columns(frame: WeakFrame, members: Sequence[WeakTerm]) -> Array:
    u = frame.field
    s2 = 0.0 if frame.noise_sigma is None else frame.noise_sigma**2
    moments = {0: np.ones_like(u), 1: u, 2: u**2 - s2, 3: u**3 - 3.0 * s2 * u}
    time_0 = frame.time_tests[0]

    def project(order: int, values: Array) -> Array:
        return (frame.space_tests[order] @ values @ time_0.T).ravel()

    columns = []
    for member in members:
        if member.kind == "power":
            columns.append(project(0, moments[member.order]))
        elif member.kind == "derivative":
            columns.append((-1.0) ** member.order * project(member.order, u))
        else:
            columns.append(-0.5 * project(1, moments[2]))
    return np.column_stack(columns)


def _target_variance(frame: WeakFrame) -> float:
    target = frame.target
    variance = float(np.var(target))
    if not variance > _DEGENERATE_VARIANCE_RTOL * float(np.mean(target**2)):
        raise WeakDesignError(
            f"the integrated weak target has degenerate variance {variance:.3g}; "
            "its NMSE would have no meaning"
        )
    return variance


def solve_weak(
    frame: WeakFrame,
    terms: Sequence[str],
    members: Sequence[WeakTerm],
    *,
    fixed: Mapping[str, float],
) -> WeakSolution:
    free = [j for j, term in enumerate(terms) if term not in fixed]
    if frame.n_windows < len(free):
        raise WeakDesignError(
            f"the weak design has {frame.n_windows} test-function windows for "
            f"{len(free)} columns; fewer windows than columns"
        )
    design = _columns(frame, members)
    norms = np.linalg.norm(design, axis=0)
    zero = [terms[int(j)] for j in np.flatnonzero(norms == 0.0)]
    if zero:
        raise WeakDesignError(f"weak columns are identically zero for terms {zero}")
    variance = _target_variance(frame)
    coefficients = np.array([fixed.get(term, 0.0) for term in terms])
    rank, condition_number = 0, None
    if free:
        target = frame.target - design @ coefficients
        unit, _, rank, singular = np.linalg.lstsq(
            design[:, free] / norms[free], target, rcond=None
        )
        coefficients[free] = unit / norms[free]
        condition_number = (
            float(singular[0] / singular[-1]) if singular[-1] > 0.0 else math.inf
        )
    residual = frame.target - design @ coefficients
    mse = float(np.mean(residual**2))
    return WeakSolution(
        coefficients=coefficients,
        mse=mse,
        nmse=mse / variance,
        r2=1.0 - mse / variance,
        condition_number=condition_number,
        rank=int(rank),
    )


def weak_zero_model(frame: WeakFrame) -> tuple[float, float]:
    mse = float(np.mean(frame.target**2))
    return mse, mse / _target_variance(frame)


def weak_diagnostics(
    frame: WeakFrame, *, rank: int, n_columns: int
) -> WeakFormDiagnostics:
    return WeakFormDiagnostics(
        method_version=METHOD_VERSION,
        spatial_axis=frame.spatial_axis,
        time_axis=frame.time_axis,
        half_widths=dict(frame.half_widths),
        strides={axis: max(1, m // 2) for axis, m in frame.half_widths.items()},
        corner_indices=None
        if frame.corner_indices is None
        else dict(frame.corner_indices),
        spectrum_tau=frame.spectrum_tau,
        bump_degree=BUMP_DEGREE,
        n_windows=frame.n_windows,
        rank=rank,
        n_columns=n_columns,
        target_variance=float(np.var(frame.target)),
        noise_correction=frame.noise_correction,
        noise_sigma=frame.noise_sigma,
        sigma_estimates=None
        if frame.sigma_estimates is None
        else dict(frame.sigma_estimates),
        condition_number_basis="unit_l2_columns",
    )
