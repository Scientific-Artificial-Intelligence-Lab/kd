
from __future__ import annotations

import pytest
import torch

import kd
from kd.data.schema import PDEDataset
from kd.search.dlga import DLGAConfig


def _order2_dataset() -> PDEDataset:
    x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    u = torch.outer(x, t)
    return PDEDataset.from_arrays(coords={"x": x, "t": t}, fields={"u": u}, lhs="u_tt")


@pytest.mark.integration
def test_facade_rejects_order2_dataset_with_default_dlga_before_build() -> None:
    dataset = _order2_dataset()
    model = kd.Model(algorithm="dlga", config=DLGAConfig())
    with pytest.raises(NotImplementedError, match="lhs_order"):
        model.fit(dataset)
