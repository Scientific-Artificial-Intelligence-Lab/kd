"""Adapters bridging `PDEDataset` objects with the DISCOVER data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from discover.task.pde.utils_fd import FiniteDiff



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

        xt, values = self._sample_points(x, t, u)

        from kd.data import SparseData as LegacySparseData

        sparse = LegacySparseData(xt, values)
        if self.colloc_num is not None:
            sparse.colloc_num = int(self.colloc_num)

        domains = dataset.mesh_bounds()
        sparse.process_data(domains)
        return sparse.get_data()

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

    def get_data(self) -> Dict[str, Any]:
        return self._data


__all__ = ["DSCVRegularAdapter", "DSCVSparseAdapter"]
