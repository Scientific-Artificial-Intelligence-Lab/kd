"""Adapters bridging `PDEDataset` with the SGA solver stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - purely for type checking
    from kd.dataset import PDEDataset


@dataclass
class SGADataAdapter:
    """Translate a :class:`~kd.dataset.PDEDataset` into SolverConfig kwargs."""

    dataset: "PDEDataset"

    def to_solver_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments that prime ``SolverConfig`` for in-memory data."""
        # Delayed import keeps the dependency optional at runtime.
        from kd.dataset import PDEDataset  # pylint: disable=import-error, cyclic-import

        if not isinstance(self.dataset, PDEDataset):
            raise TypeError("SGADataAdapter expects a PDEDataset instance")

        # Prefer structured registry payloads when available (nD / multi-field ready).
        fields_data = getattr(self.dataset, "fields_data", None)
        coords_1d = getattr(self.dataset, "coords_1d", None)
        if fields_data is not None or coords_1d is not None:
            if fields_data is None or coords_1d is None:
                raise ValueError("Registry payload requires both fields_data and coords_1d.")
            axis_order = getattr(self.dataset, "axis_order", None) or list(coords_1d.keys())
            target_field = getattr(self.dataset, "target_field", "u")
            lhs_axis = getattr(self.dataset, "lhs_axis", "t")
            return {
                "fields_data": {k: np.asarray(v) for k, v in fields_data.items()},
                "coords_1d": {k: np.asarray(v) for k, v in coords_1d.items()},
                "axis_order": list(axis_order),
                "target_field": target_field,
                "lhs_axis": lhs_axis,
                "problem_name": getattr(self.dataset, "equation_name", "custom_dataset") or "custom_dataset",
            }

        x = np.asarray(self.dataset.x, dtype=float).flatten()
        t = np.asarray(self.dataset.t, dtype=float).flatten()
        u = np.asarray(self.dataset.usol, dtype=float)

        if u.shape != (len(x), len(t)):
            raise ValueError(
                "Inconsistent dataset dimensions: expected usol shape "
                f"({len(x)}, {len(t)}) but got {u.shape}"
            )

        return {
            "u_data": u,
            "x_data": x,
            "t_data": t,
            "problem_name": getattr(self.dataset, "equation_name", "custom_dataset") or "custom_dataset",
        }
