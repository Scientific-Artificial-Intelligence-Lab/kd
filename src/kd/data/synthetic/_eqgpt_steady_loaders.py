
from __future__ import annotations

from pathlib import Path

from kd.data.schema import PDEDataset

_ASSET_DIR = Path(__file__).resolve().parents[2] / "_assets" / "data"


def load_laplacian_eitech() -> PDEDataset:
    return PDEDataset.from_xlsx(
        _ASSET_DIR / "eqgpt_laplacian_eitech.xlsx",
        coords={"x": "x", "y": "y"},
        fields={"u": "values"},
        lhs="",
        name="eqgpt-laplacian-eitech",
        ground_truth="u_xx + u_yy + 1 = 0",
    )


def load_laplacian_smile() -> PDEDataset:
    return PDEDataset.from_xlsx(
        _ASSET_DIR / "eqgpt_laplacian_smile.xlsx",
        coords={"x": "x", "y": "y"},
        fields={"u": "values"},
        lhs="",
        name="eqgpt-laplacian-smile",
        ground_truth="u_xx + u_yy = 0",
    )


def load_poisson_disk() -> PDEDataset:
    return PDEDataset.from_xlsx(
        _ASSET_DIR / "eqgpt_poisson_disk.xlsx",
        coords={"x": "x", "y": "y"},
        fields={"u": "values"},
        lhs="",
        name="eqgpt-poisson-disk",
        ground_truth="u_xx + u_yy = 0",
    )


__all__ = [
    "load_laplacian_eitech",
    "load_laplacian_smile",
    "load_poisson_disk",
]
