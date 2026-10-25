
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from kd.data.schema import AxisInfo, FieldData, PDEDataset
from kd.data.subsample import _require_grid

_POLYORDER = 3


def lowpass_smooth(
    dataset: PDEDataset,
    *,
    modes: int | Mapping[str, int],
    axes: Sequence[str] | None = None,
    time_window: int = 0,
    polyorder: int = _POLYORDER,
) -> PDEDataset:
    return _smoothed_copy(
        dataset,
        modes=modes,
        axes=axes,
        time_window=time_window,
        polyorder=polyorder,
        ground_truth=dataset.ground_truth,
    )


def _smoothed_copy(
    dataset: PDEDataset,
    *,
    modes: int | Mapping[str, int],
    axes: Sequence[str] | None,
    time_window: int,
    polyorder: int,
    ground_truth: str | None,
) -> PDEDataset:
    _require_grid(dataset, what="lowpass_smooth")
    assert dataset.axes is not None and dataset.axis_order is not None
    assert dataset.fields is not None
    per_axis = _modes_per_axis(dataset, modes, axes)
    if time_window < 0 or (time_window > 0 and time_window % 2 == 0):
        raise ValueError(
            f"time_window must be 0 or a positive odd integer, got {time_window}"
        )
    axes_out = {
        name: AxisInfo(info.name, info.values.clone(), info.is_periodic, info.allow_nan)
        for name, info in dataset.axes.items()
    }
    fields: dict[str, FieldData] = {}
    for name, field in dataset.fields.items():
        values = field.values.detach().cpu().numpy()
        _refuse_nan(values, name)
        values = _smooth_values(
            values, dataset, per_axis, time_window=time_window, polyorder=polyorder
        )


        smoothed = torch.tensor(
            values, dtype=field.values.dtype, device=field.values.device
        )
        fields[name] = FieldData(field.name, smoothed, allow_nan=field.allow_nan)
    return PDEDataset(
        name=dataset.name,
        task_type=dataset.task_type,
        topology=dataset.topology,
        axes=axes_out,
        axis_order=list(dataset.axis_order),
        fields=fields,
        lhs_field=dataset.lhs_field,
        lhs_axis=dataset.lhs_axis,
        lhs_order=dataset.lhs_order,
        noise_level=dataset.noise_level,
        ground_truth=ground_truth,
        source=dataset.source,
    )


def _smooth_values(
    values: np.ndarray,
    dataset: PDEDataset,
    per_axis: Mapping[str, int],
    *,
    time_window: int,
    polyorder: int,
) -> np.ndarray:

    from scipy.signal import savgol_filter

    assert dataset.axes is not None and dataset.axis_order is not None
    axis_order = dataset.axis_order
    for axis, count in per_axis.items():
        values = _lowpass_axis(
            values,
            axis_order.index(axis),
            count,
            periodic=dataset.axes[axis].is_periodic,
        )
    if time_window > 0:
        values = savgol_filter(
            values,
            window_length=time_window,
            polyorder=polyorder,
            axis=axis_order.index(dataset.lhs_axis),
            mode="interp",
        )
    return values


def lowpass_spectrum(dataset: PDEDataset, field: str, axis: str) -> np.ndarray:
    from scipy.fft import dct, rfft

    _require_grid(dataset, what="lowpass_spectrum")
    assert dataset.axes is not None and dataset.axis_order is not None
    assert dataset.fields is not None
    if field not in dataset.fields:
        raise ValueError(
            f"lowpass_spectrum: unknown field {field!r}; the dataset has "
            f"{sorted(dataset.fields)}"
        )
    if axis not in dataset.axis_order:
        raise ValueError(
            f"lowpass_spectrum: unknown axis {axis!r}; the dataset has "
            f"{list(dataset.axis_order)}"
        )
    values = dataset.fields[field].values.detach().cpu().numpy()
    _refuse_nan(values, field)
    axis_index = dataset.axis_order.index(axis)
    n = values.shape[axis_index]
    if dataset.axes[axis].is_periodic:
        coefficients = np.abs(rfft(values, axis=axis_index)) ** 2
        weights = np.full(coefficients.shape[axis_index], 2.0)
        weights[0] = 1.0
        if n % 2 == 0:
            weights[-1] = 1.0
        coefficients = np.moveaxis(coefficients, axis_index, 0) * weights.reshape(
            -1, *([1] * (values.ndim - 1))
        )
    else:
        coefficients = np.moveaxis(
            dct(values, type=2, norm="ortho", axis=axis_index) ** 2, axis_index, 0
        )
    energy = coefficients.reshape(coefficients.shape[0], -1).sum(axis=1)
    total = float(energy.sum())
    if total == 0.0:
        raise ValueError(
            f"lowpass_spectrum: field {field!r} is identically zero along "
            f"{axis!r}; its energy fraction is undefined"
        )

    kept = np.minimum(np.cumsum(energy) / total, 1.0)
    fraction = np.ones(n, dtype=np.float64)
    fraction[: kept.shape[0]] = kept
    return fraction


_KNEE_FLOOR_DIVISOR = 8
_KNEE_FLOOR_FACTOR = 1.5
_CHANGE_FLOOR = 1e-12


class RefitOutcome(Protocol):

    @property
    def coefficients(self) -> Sequence[float]: ...


class LowpassRefit(Protocol):

    def __call__(
        self, dataset: PDEDataset, terms: Sequence[str], /
    ) -> RefitOutcome: ...


@dataclass(frozen=True)
class LowpassStep:

    modes: int
    energy_fraction: float
    coefficients: tuple[float, ...]
    change: float | None


@dataclass(frozen=True)
class LowpassSelection:

    modes: dict[str, int]
    knee: dict[str, int]
    trace: dict[str, tuple[LowpassStep, ...]]
    tolerance: float
    time_window: int


def select_lowpass_modes(
    dataset: PDEDataset,
    field: str,
    *,
    terms: Sequence[str],
    refit: LowpassRefit,
    axes: Sequence[str] | None = None,
    time_window: int = 0,
    tolerance: float = 0.02,
    max_counts: int = 8,
) -> LowpassSelection:
    if not (math.isfinite(tolerance) and tolerance > 0):
        raise ValueError(
            f"select_lowpass_modes: tolerance must be a positive finite number, "
            f"got {tolerance!r}"
        )
    if max_counts < 2:
        raise ValueError(
            f"select_lowpass_modes: max_counts must be at least 2 (a change "
            f"needs two refits), got {max_counts!r}"
        )
    _require_grid(dataset, what="select_lowpass_modes")
    assert dataset.axis_order is not None
    selected = (
        list(axes)
        if axes is not None
        else [a for a in dataset.axis_order if a != dataset.lhs_axis]
    )
    spectra = {axis: lowpass_spectrum(dataset, field, axis) for axis in selected}
    knee = {axis: _knee(fraction) for axis, fraction in spectra.items()}
    chosen = dict(knee)
    trace: dict[str, tuple[LowpassStep, ...]] = {}
    for axis in selected:
        trace[axis] = _scan_axis(
            dataset,
            axis,
            chosen,
            spectra[axis],
            terms=terms,
            refit=refit,
            time_window=time_window,
            tolerance=tolerance,
            max_counts=max_counts,
        )
        chosen[axis] = trace[axis][-2].modes
    return LowpassSelection(
        modes=chosen,
        knee=knee,
        trace=trace,
        tolerance=tolerance,
        time_window=time_window,
    )


def _knee(fraction: np.ndarray) -> int:
    remainder = 1.0 - fraction
    floor = remainder[max(1, remainder.size // _KNEE_FLOOR_DIVISOR) - 1]
    return int(np.argmax(remainder <= _KNEE_FLOOR_FACTOR * floor)) + 1


def _scan_axis(
    dataset: PDEDataset,
    axis: str,
    counts: Mapping[str, int],
    fraction: np.ndarray,
    *,
    terms: Sequence[str],
    refit: LowpassRefit,
    time_window: int,
    tolerance: float,
    max_counts: int,
) -> tuple[LowpassStep, ...]:
    start = counts[axis]
    stop = min(fraction.shape[0], start + max_counts - 1)
    steps: list[LowpassStep] = []
    for count in range(start, stop + 1):
        smoothed = _smoothed_copy(
            dataset,
            modes={**counts, axis: count},
            axes=None,
            time_window=time_window,
            polyorder=_POLYORDER,
            ground_truth=None,
        )
        coefficients = tuple(float(c) for c in refit(smoothed, terms).coefficients)
        change = (
            _relative_change(steps[-1].coefficients, coefficients) if steps else None
        )
        steps.append(
            LowpassStep(count, float(fraction[count - 1]), coefficients, change)
        )
        if change is not None and change <= tolerance:
            return tuple(steps)
    raise ValueError(
        f"select_lowpass_modes: the refitted coefficients along {axis!r} did not "
        f"settle within tolerance {tolerance} over the counts "
        f"{[step.modes for step in steps]} (changes "
        f"{[step.change for step in steps[1:]]}); raise max_counts or the "
        "tolerance, or prune near-zero terms"
    )


def _relative_change(previous: Sequence[float], current: Sequence[float]) -> float:
    before = np.asarray(previous, dtype=np.float64)
    after = np.asarray(current, dtype=np.float64)
    return float(
        np.max(np.abs(after - before) / np.maximum(np.abs(before), _CHANGE_FLOOR))
    )


def _modes_per_axis(
    dataset: PDEDataset, modes: int | Mapping[str, int], axes: Sequence[str] | None
) -> dict[str, int]:
    assert dataset.axes is not None and dataset.axis_order is not None
    if axes is None:
        if isinstance(modes, Mapping):
            selected = list(modes)
        else:
            selected = [a for a in dataset.axis_order if a != dataset.lhs_axis]
    else:
        selected = list(axes)
    unknown = [a for a in selected if a not in dataset.axis_order]
    if isinstance(modes, Mapping):
        unknown += [a for a in modes if a not in dataset.axis_order]
    if unknown:
        raise ValueError(
            f"lowpass_smooth: unknown axes {sorted(set(unknown))}; the dataset "
            f"has {list(dataset.axis_order)}"
        )
    if isinstance(modes, Mapping):
        if set(modes) != set(selected):
            raise ValueError(
                f"lowpass_smooth: modes names {sorted(modes)} but the selected "
                f"axes are {sorted(selected)}; give one count per selected axis"
            )
        per_axis = {a: modes[a] for a in selected}
    else:
        per_axis = dict.fromkeys(selected, modes)
    for axis, count in per_axis.items():
        length = dataset.axes[axis].values.numel()
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not (1 <= count <= length)
        ):
            raise ValueError(
                f"lowpass_smooth: modes for axis {axis!r} must be an integer in "
                f"1..{length}, got {count!r}"
            )
    return per_axis


def _refuse_nan(values: np.ndarray, field: str) -> None:
    if np.isnan(values).any():
        raise ValueError(
            f"field {field!r} holds NaN; a spectral transform would spread it "
            "over the whole field"
        )


def _lowpass_axis(
    values: np.ndarray, axis: int, modes: int, *, periodic: bool
) -> np.ndarray:
    from scipy.fft import dct, idct, irfft, rfft

    n = values.shape[axis]
    index = [slice(None)] * values.ndim
    index[axis] = slice(modes, None)
    if periodic:
        spectrum = rfft(values, axis=axis)
        spectrum[tuple(index)] = 0.0
        return np.asarray(irfft(spectrum, n=n, axis=axis))
    spectrum = dct(values, type=2, norm="ortho", axis=axis)
    spectrum[tuple(index)] = 0.0
    return np.asarray(idct(spectrum, type=2, norm="ortho", axis=axis))
