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
