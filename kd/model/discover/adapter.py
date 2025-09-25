"""Adapters bridging `PDEDataset` objects with the DISCOVER data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from kd.model.discover.task.pde.utils_fd import FiniteDiff



@dataclass
class DSCVRegularAdapter:
    """Convert a :class:`~kd.dataset.PDEDataset` into DISCOVER regular-data format."""

    dataset: "PDEDataset"
    sym_true: Optional[str] = None
    n_input_dim: Optional[int] = None
    enforce_uniform_dt: bool = True
    _data: Dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from kd.dataset import PDEDataset  # local import to avoid circular deps

        if not isinstance(self.dataset, PDEDataset):
            raise TypeError("DSCVRegularAdapter expects a PDEDataset instance")

        self._data = self._prepare()

    def _prepare(self) -> Dict[str, Any]:
        dataset = self.dataset

        x = np.asarray(dataset.x, dtype=float)
        t = np.asarray(dataset.t, dtype=float)
        u = np.asarray(dataset.usol, dtype=float)

        if u.shape != (x.shape[0], t.shape[0]):
            raise ValueError(
                "PDEDataset usol shape {} does not match len(x)={} and len(t)={}".format(
                    u.shape, x.shape[0], t.shape[0]
                )
            )

        if t.size < 2:
            raise ValueError("PDEDataset requires at least two temporal samples for DSCV adapter")

        # Ensure evenly spaced time grid when requested.
        dt = np.diff(t)
        if self.enforce_uniform_dt and not np.allclose(dt, dt[0]):
            raise ValueError("Time grid must be evenly spaced for finite-difference evaluation")
        delta_t = float(dt[0])

        ut = np.zeros_like(u)
        for idx in range(u.shape[0]):
            ut[idx, :] = FiniteDiff(u[idx, :], delta_t)

        x_column = x.reshape(-1, 1)

        sym_true = self.sym_true
        if sym_true is None:
            name = getattr(dataset, "equation_name", None)
            if isinstance(name, str):
                try:
                    from kd.dataset import get_dataset_sym_true

                    sym_true = get_dataset_sym_true(name)
                except (ImportError, ValueError):
                    sym_true = None

        data: Dict[str, Any] = {
            "u": u,
            "ut": ut,
            "X": [x_column],
            "n_input_dim": self.n_input_dim or 1,
        }
        if sym_true is not None:
            data["sym_true"] = sym_true

        return data

    def get_data(self) -> Dict[str, Any]:
        """Return the DISCOVER-compatible data dictionary."""
        return self._data


@dataclass
class DSCVSparseAdapter:
    """Convert :class:`~kd.dataset.PDEDataset` samples into DISCOVER sparse-data format."""

    dataset: "PDEDataset"
    sample: Optional[int] = None
    sample_ratio: float = 0.1
    colloc_num: Optional[int] = None
    random_state: Optional[int] = None
    noise_level: float = 0.0
    data_ratio: Optional[float] = None
    spline_sample: bool = False
    cut_quantile: Optional[float] = None
    _data: Dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from kd.dataset import PDEDataset  # local import to avoid circular deps

        if not isinstance(self.dataset, PDEDataset):
            raise TypeError("DSCVSparseAdapter expects a PDEDataset instance")

        self._data = self._prepare()

    def _prepare(self) -> Dict[str, Any]:
        dataset = self.dataset

        x = np.asarray(dataset.x, dtype=float)
        t = np.asarray(dataset.t, dtype=float)
        u = np.asarray(dataset.usol, dtype=float)

        if u.shape != (x.shape[0], t.shape[0]):
            raise ValueError(
                "PDEDataset usol shape {} does not match len(x)={} and len(t)={}".format(
                    u.shape, x.shape[0], t.shape[0]
                )
            )

        x, t, u = self._apply_cut_quantile(x, t, u)

        if self.spline_sample:
            xt, values = self._spline_sample_points(x, t, u)
        else:
            xt, values = self._sample_points(x, t, u)

        if self.noise_level:
            std = float(self.noise_level) * np.std(values)
            rng = np.random.default_rng(self.random_state)
            values = values + rng.normal(scale=std, size=values.shape)

        from kd.data import SparseData as LegacySparseData

        sparse = LegacySparseData(xt, values)
        if self.colloc_num is not None:
            sparse.colloc_num = int(self.colloc_num)

        default_lb, default_ub = dataset.mesh_bounds()
        if self.cut_quantile is not None:
            lb = np.array([x.min(), t.min()])
            ub = np.array([x.max(), t.max()])
        else:
            lb, ub = default_lb, default_ub

        sparse.process_data((lb, ub))

        result = dict(sparse.get_data())
        if self.cut_quantile is not None:
            result['lb'] = lb
            result['ub'] = ub

        result['measured_points'] = xt
        result['sampling_strategy'] = 'spline' if self.spline_sample else 'random'

        if result['X_u_train'].shape[0]:
            base_colloc = result['X_f_train'][:-result['X_u_train'].shape[0]].copy()
        else:
            base_colloc = result['X_f_train']
        if self.data_ratio is not None:
            ratio = float(self.data_ratio)
            if not 0 < ratio <= 1:
                raise ValueError("data_ratio must be within (0, 1]")

            rng = np.random.default_rng(self.random_state)

            orig_train = result['X_u_train'].shape[0]
            target = max(1, int(orig_train * ratio))
            if target < orig_train:
                idx = rng.choice(orig_train, target, replace=False)
                result['X_u_train'] = result['X_u_train'][idx]
                result['u_train'] = result['u_train'][idx]

            orig_val = result['X_u_val'].shape[0]
            target_val = max(1, int(orig_val * ratio))
            if target_val < orig_val:
                idx_val = rng.choice(orig_val, target_val, replace=False)
                result['X_u_val'] = result['X_u_val'][idx_val]
                result['u_val'] = result['u_val'][idx_val]

            if result['X_u_train'].shape[0]:
                result['X_f_train'] = np.vstack((base_colloc, result['X_u_train']))

        mesh_x, mesh_t = np.meshgrid(x, t, indexing='ij')
        result['X_star'] = np.column_stack((mesh_x.ravel(), mesh_t.ravel()))
        result['u_star'] = u.reshape(-1, 1)
        result['shape'] = u.shape

        return result

    def _sample_points(self, x: np.ndarray, t: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mesh_x, mesh_t = np.meshgrid(x, t, indexing='ij')
        points = np.column_stack((mesh_x.ravel(), mesh_t.ravel()))
        values = u.reshape(-1, 1)

        total = values.shape[0]
        if self.sample is not None:
            n_samples = int(self.sample)
        else:
            ratio = float(self.sample_ratio)
            if not 0 < ratio <= 1:
                raise ValueError("sample_ratio must be within (0, 1]")
            n_samples = max(1, int(total * ratio))

        if n_samples > total:
            raise ValueError(f"Requested {n_samples} samples, but only {total} points available")

        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(total, n_samples, replace=False)
        return points[indices], values[indices]

    def _spline_sample_points(self, x: np.ndarray, t: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ratio = self.data_ratio if self.data_ratio is not None else self.sample_ratio
        if not 0 < ratio <= 1:
            raise ValueError("ratio for spline sampling must be within (0, 1]")

        total_time = len(t)
        total_space = len(x)

        if self.sample is not None:
            num_x = max(1, int(self.sample / total_time))
        else:
            num_x = max(1, int(total_space * ratio))
        num_x = min(num_x, total_space)

        rng = np.random.default_rng(self.random_state)
        idx_x = rng.choice(total_space, num_x, replace=False)
        idx_x.sort()

        x_selected = x[idx_x]
        points = np.column_stack((np.repeat(x_selected, total_time), np.tile(t, num_x)))
        values = u[idx_x, :].reshape(-1, 1)
        return points, values

    def _apply_cut_quantile(self, x: np.ndarray, t: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.cut_quantile is None:
            return x, t, u

        q = float(self.cut_quantile)
        if not 0 <= q < 0.5:
            raise ValueError("cut_quantile must be in [0, 0.5)")

        def trim(arr: np.ndarray, q: float) -> Tuple[np.ndarray, int, int]:
            length = len(arr)
            lower = int(np.floor(length * q))
            upper = length - int(np.floor(length * q))
            if upper - lower < 2:
                return arr, 0, length
            return arr[lower:upper], lower, upper

        x_trim, x_low, x_high = trim(x, q)
        t_trim, t_low, t_high = trim(t, q)

        u_trim = u
        if x_trim.shape[0] != x.shape[0]:
            u_trim = u_trim[x_low:x_high, :]
        if t_trim.shape[0] != t.shape[0]:
            u_trim = u_trim[:, t_low:t_high]

        return x_trim, t_trim, u_trim

    def get_data(self) -> Dict[str, Any]:
        return self._data


__all__ = ["DSCVRegularAdapter", "DSCVSparseAdapter"]
