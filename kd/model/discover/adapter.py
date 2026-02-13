"""Adapters bridging `PDEDataset` objects with the DISCOVER data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from kd.model.discover.task.pde.utils_fd import FiniteDiff

# Supported spatial dimensions: 2D and 3D (limited by existing Diff_2/Diff_3).
# 1D datasets must use the legacy path; the N-D path requires >= 2 spatial dims.
_MIN_SPATIAL_DIM = 2
_MAX_SPATIAL_DIM = 3
# Diff2 boundary stencil accesses indices 0..3 and n-4..n-1, requiring ≥4 points.
_MIN_SPATIAL_POINTS = 4


@dataclass
class DSCVRegularAdapter:
    """Convert a :class:`~kd.dataset.PDEDataset` into DISCOVER regular-data format.

    Supports both legacy 1D datasets (``x/t/usol``) and N-D structured grids
    (``fields_data/coords_1d/axis_order``).
    """

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

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _prepare(self) -> Dict[str, Any]:
        """Detect data mode and dispatch to the appropriate builder."""
        fields_data = getattr(self.dataset, "fields_data", None)
        coords_1d = getattr(self.dataset, "coords_1d", None)

        if fields_data is not None or coords_1d is not None:
            if fields_data is None or coords_1d is None:
                raise ValueError(
                    "N-D data requires both fields_data and coords_1d; "
                    "got only one. Use x/t/usol for legacy 1D data."
                )
            return self._prepare_nd()
        return self._prepare_legacy()

    # ------------------------------------------------------------------
    # Legacy 1D path (unchanged)
    # ------------------------------------------------------------------

    def _prepare_legacy(self) -> Dict[str, Any]:
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

        if not np.all(np.isfinite(u)):
            raise ValueError("Input field u contains NaN or Inf values")

        if t.size < 3:
            raise ValueError(
                "PDEDataset requires at least 3 temporal samples for DSCV adapter "
                "(FiniteDiff accesses u[2] and u[n-3])"
            )

        # Ensure evenly spaced time grid when requested.
        dt = np.diff(t)
        if self.enforce_uniform_dt and not np.allclose(dt, dt[0]):
            raise ValueError("Time grid must be evenly spaced for finite-difference evaluation")
        delta_t = float(dt[0])
        if delta_t == 0.0:
            raise ValueError("Time step delta_t is zero (degenerate time grid)")

        ut = np.zeros_like(u)
        for idx in range(u.shape[0]):
            ut[idx, :] = FiniteDiff(u[idx, :], delta_t)

        x_column = x.reshape(-1, 1)

        if self.n_input_dim is not None and self.n_input_dim != 1:
            raise ValueError(
                f"Legacy DSCV data is 1D spatial; n_input_dim must be 1, "
                f"got {self.n_input_dim}"
            )

        sym_true = self._resolve_sym_true()

        data: Dict[str, Any] = {
            "u": u,
            "ut": ut,
            "X": [x_column],
            "n_input_dim": self.n_input_dim or 1,
        }
        if sym_true is not None:
            data["sym_true"] = sym_true

        # Parameter fields — same shape as u, not differentiated.
        param_fields = getattr(self.dataset, "param_fields", None) or {}
        if param_fields:
            param_data_list: List[np.ndarray] = []
            param_names: List[str] = []
            for name, arr in param_fields.items():
                param_data_list.append(np.asarray(arr, dtype=float))
                param_names.append(name)
            data["param_data"] = param_data_list
            data["n_param_var"] = len(param_data_list)
            data["param_names"] = param_names

        return data

    # ------------------------------------------------------------------
    # N-D path (2D / 3D spatial)
    # ------------------------------------------------------------------

    def _prepare_nd(self) -> Dict[str, Any]:
        """Build DISCOVER data dict from an N-D :class:`PDEDataset`.

        The dso engine expects:
        - 2D spatial: ``u.shape = (nt, nx, ny)`` — time axis first
        - 3D spatial: ``u.shape = (nt, nx, ny, nz)`` — time axis first
        - ``ut`` same shape as ``u``
        - ``X`` follows spatial axis order from ``axis_order`` (minus ``lhs_axis``),
          each element shaped ``(n_i, 1)``
        """
        dataset = self.dataset
        axis_order: List[str] = list(dataset.axis_order)
        coords_1d: Dict[str, np.ndarray] = {
            k: np.asarray(v, dtype=float) for k, v in dataset.coords_1d.items()
        }
        target_field: str = getattr(dataset, "target_field", "u") or "u"
        lhs_axis: str = getattr(dataset, "lhs_axis", "t") or "t"

        u_raw = np.asarray(dataset.fields_data[target_field], dtype=float)

        if not np.all(np.isfinite(u_raw)):
            raise ValueError("Input field u contains NaN or Inf values")

        # Separate lhs (time) axis from spatial axes.
        if lhs_axis not in axis_order:
            raise ValueError(
                f"lhs_axis '{lhs_axis}' not found in axis_order {axis_order}"
            )
        spatial_axes = [a for a in axis_order if a != lhs_axis]
        n_spatial = len(spatial_axes)

        if n_spatial < _MIN_SPATIAL_DIM or n_spatial > _MAX_SPATIAL_DIM:
            raise ValueError(
                f"DSCV N-D mode supports {_MIN_SPATIAL_DIM}-{_MAX_SPATIAL_DIM} "
                f"spatial dimensions, got {n_spatial}. "
                "For 1D data, use the legacy path (x/t/usol attributes)."
            )

        # --- Permute u so that lhs_axis is at position 0 ---
        # Target axis order: [lhs_axis, *spatial_axes]
        target_order = [lhs_axis] + spatial_axes
        perm = [axis_order.index(a) for a in target_order]
        u = np.transpose(u_raw, perm)
        # u.shape is now (n_lhs, n_s1, n_s2, ...)

        # --- Validate spatial axis sizes and grid spacing ---
        for i, axis_name in enumerate(spatial_axes):
            axis_size = u.shape[i + 1]  # +1 because axis 0 is time
            if axis_size < _MIN_SPATIAL_POINTS:
                raise ValueError(
                    f"Spatial axis '{axis_name}' has {axis_size} points, "
                    f"but finite-difference stencils require at least "
                    f"{_MIN_SPATIAL_POINTS}"
                )
            coord = coords_1d[axis_name]
            dx_vals = np.diff(coord)
            if np.any(np.isclose(dx_vals, 0.0)):
                raise ValueError(
                    f"Spatial axis '{axis_name}' has zero spacing "
                    f"between consecutive points"
                )

        # --- Validate and compute time derivative ---
        t_coord = coords_1d[lhs_axis]
        if t_coord.size < 3:
            raise ValueError(
                f"PDEDataset requires at least 3 samples along "
                f"lhs_axis '{lhs_axis}' for DSCV adapter "
                "(FiniteDiff accesses u[2] and u[n-3])"
            )

        dt = np.diff(t_coord)
        if self.enforce_uniform_dt and not np.allclose(dt, dt[0]):
            raise ValueError(
                f"Grid along '{lhs_axis}' must be evenly spaced "
                "for finite-difference evaluation"
            )
        delta_t = float(dt[0])
        if delta_t == 0.0:
            raise ValueError(
                f"Time step along '{lhs_axis}' is zero (degenerate grid)"
            )

        # Compute ut along axis=0 (the lhs/time axis).
        # Flatten trailing dims → apply FiniteDiff per column → reshape back.
        n_t = u.shape[0]
        spatial_shape = u.shape[1:]
        n_spatial_pts = int(np.prod(spatial_shape))
        u_2d = u.reshape(n_t, n_spatial_pts)
        ut_2d = np.zeros_like(u_2d)
        for col in range(n_spatial_pts):
            ut_2d[:, col] = FiniteDiff(u_2d[:, col], delta_t)
        ut = ut_2d.reshape(u.shape)

        # --- Build X coordinate list ---
        x_list = [coords_1d[a].reshape(-1, 1) for a in spatial_axes]

        # --- Resolve n_input_dim ---
        if self.n_input_dim is not None and self.n_input_dim != n_spatial:
            raise ValueError(
                f"n_input_dim ({self.n_input_dim}) does not match actual "
                f"spatial dimension ({n_spatial})"
            )
        n_input_dim = self.n_input_dim if self.n_input_dim is not None else n_spatial

        sym_true = self._resolve_sym_true()

        data: Dict[str, Any] = {
            "u": u,
            "ut": ut,
            "X": x_list,
            "n_input_dim": n_input_dim,
        }
        if sym_true is not None:
            data["sym_true"] = sym_true

        # Parameter fields — apply same permute as u so shapes align.
        param_fields = getattr(self.dataset, "param_fields", None) or {}
        if param_fields:
            param_data_list: List[np.ndarray] = []
            param_names: List[str] = []
            for name, arr in param_fields.items():
                p_raw = np.asarray(arr, dtype=float)
                # Same transpose as u: axis_order → [lhs_axis, *spatial_axes]
                p = np.transpose(p_raw, perm)
                param_data_list.append(p)
                param_names.append(name)
            data["param_data"] = param_data_list
            data["n_param_var"] = len(param_data_list)
            data["param_names"] = param_names

        return data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_sym_true(self) -> Optional[str]:
        """Try to resolve the ground-truth symbolic expression."""
        sym_true = self.sym_true
        if sym_true is None:
            name = getattr(self.dataset, "equation_name", None)
            if isinstance(name, str):
                try:
                    from kd.dataset import get_dataset_sym_true

                    sym_true = get_dataset_sym_true(name)
                except (ImportError, ValueError):
                    sym_true = None
        return sym_true

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
        """Detect data mode and dispatch to the appropriate builder."""
        fields_data = getattr(self.dataset, "fields_data", None)
        coords_1d = getattr(self.dataset, "coords_1d", None)

        if fields_data is not None or coords_1d is not None:
            if fields_data is None or coords_1d is None:
                raise ValueError(
                    "N-D data requires both fields_data and coords_1d; "
                    "got only one. Use x/t/usol for legacy 1D data."
                )
            return self._prepare_nd()
        return self._prepare_legacy()

    def _prepare_legacy(self) -> Dict[str, Any]:
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

        # Parameter fields — raw arrays, not sampled.
        param_fields = getattr(self.dataset, "param_fields", None) or {}
        if param_fields:
            param_data_list: List[np.ndarray] = []
            param_names: List[str] = []
            for name, arr in param_fields.items():
                param_data_list.append(np.asarray(arr, dtype=float))
                param_names.append(name)
            result["param_data"] = param_data_list
            result["n_param_var"] = len(param_data_list)
            result["param_names"] = param_names

        return result

    # ------------------------------------------------------------------
    # N-D path (2D / 3D spatial)
    # ------------------------------------------------------------------

    def _prepare_nd(self) -> Dict[str, Any]:
        """Build sparse data dict from an N-D :class:`PDEDataset`.

        Produces the same output keys as the legacy path:
        ``X_u_train``, ``u_train``, ``X_f_train``, ``X_u_val``, ``u_val``,
        ``lb``, ``ub``, plus ``n_input_dim``.
        """
        dataset = self.dataset
        axis_order: List[str] = list(dataset.axis_order)
        coords_1d: Dict[str, np.ndarray] = {
            k: np.asarray(v, dtype=float) for k, v in dataset.coords_1d.items()
        }
        target_field: str = getattr(dataset, "target_field", "u") or "u"
        lhs_axis: str = getattr(dataset, "lhs_axis", "t") or "t"

        # N-D path does not support legacy 1D-only parameters.
        if self.spline_sample:
            raise NotImplementedError(
                "spline_sample is not supported for N-D data "
                "(scipy 1D spline is not applicable to multi-dimensional grids). "
                "Use random sampling (spline_sample=False)."
            )
        if self.cut_quantile is not None:
            raise NotImplementedError(
                "cut_quantile is not supported for N-D data yet. "
                "Remove cut_quantile or use the legacy 1D path."
            )
        if self.data_ratio is not None:
            raise NotImplementedError(
                "data_ratio is not supported for N-D data yet. "
                "Use sample_ratio to control sampling instead."
            )

        u_raw = np.asarray(dataset.fields_data[target_field], dtype=float)
        if not np.all(np.isfinite(u_raw)):
            raise ValueError("Input field data contains NaN or Inf values")

        # Separate spatial and time axes
        if lhs_axis not in axis_order:
            raise ValueError(
                f"lhs_axis '{lhs_axis}' not found in axis_order {axis_order}"
            )
        spatial_axes = [a for a in axis_order if a != lhs_axis]
        n_spatial = len(spatial_axes)

        # Build coordinate arrays
        spatial_coords = [coords_1d[a] for a in spatial_axes]
        t_coord = coords_1d[lhs_axis]
        all_coords = spatial_coords + [t_coord]
        axis_sizes = tuple(c.size for c in all_coords)
        total_points = int(np.prod(axis_sizes))

        # Reorder u to match (spatial_axes..., time) axis layout
        target_order = spatial_axes + [lhs_axis]
        perm = [axis_order.index(a) for a in target_order]
        u = np.transpose(u_raw, perm)

        # --- Index-based sampling (avoids full meshgrid materialization) ---
        rng = np.random.default_rng(self.random_state)
        if self.sample is not None:
            n_samples = int(self.sample)
        else:
            ratio = float(self.sample_ratio)
            if not 0 < ratio <= 1:
                raise ValueError("sample_ratio must be within (0, 1]")
            n_samples = max(1, int(total_points * ratio))

        if n_samples > total_points:
            raise ValueError(
                f"Requested {n_samples} samples, but only {total_points} points"
            )

        sample_idx = rng.choice(total_points, n_samples, replace=False)
        multi_idx = np.unravel_index(sample_idx, axis_sizes)
        xt = np.column_stack([
            all_coords[i][multi_idx[i]] for i in range(len(all_coords))
        ])
        sampled_values = u[multi_idx].reshape(-1, 1)

        if self.noise_level:
            std = float(self.noise_level) * np.std(sampled_values)
            sampled_values = sampled_values + rng.normal(
                scale=std, size=sampled_values.shape
            )

        # Compute lb/ub from coords directly (not from mesh_bounds)
        lb = np.array([c.min() for c in all_coords])
        ub = np.array([c.max() for c in all_coords])

        from kd.data import SparseData as LegacySparseData

        sparse = LegacySparseData(xt, sampled_values)
        if self.colloc_num is not None:
            sparse.colloc_num = int(self.colloc_num)
        sparse.process_data((lb, ub))

        result = dict(sparse.get_data())
        result['n_input_dim'] = n_spatial
        result['measured_points'] = xt
        result['sampling_strategy'] = 'random'

        # Build full X_star without meshgrid (stride-based coordinate expansion)
        n_dims = len(all_coords)
        X_star = np.empty((total_points, n_dims), dtype=float)
        stride = 1
        for i in range(n_dims - 1, -1, -1):
            coord = all_coords[i]
            n_tiles = total_points // (coord.size * stride)
            X_star[:, i] = np.tile(np.repeat(coord, stride), n_tiles)
            stride *= coord.size
        result['X_star'] = X_star
        result['u_star'] = u.ravel(order="C").reshape(-1, 1)
        result['shape'] = u.shape

        # Parameter fields — apply same perm as u (time-last for Sparse).
        param_fields = getattr(self.dataset, "param_fields", None) or {}
        if param_fields:
            param_data_list: List[np.ndarray] = []
            param_names: List[str] = []
            for name, arr in param_fields.items():
                p_raw = np.asarray(arr, dtype=float)
                p = np.transpose(p_raw, perm)
                param_data_list.append(p)
                param_names.append(name)
            result["param_data"] = param_data_list
            result["n_param_var"] = len(param_data_list)
            result["param_names"] = param_names

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
