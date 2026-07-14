
from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import kd
from kd.data import synthetic
from kd.data.schema import DataTopology, PDEDataset


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "loader_name",
        "point_count",
        "x_range",
        "y_range",
        "ground_truth",
        "range_atol",
    ),
    [
        (
            "load_laplacian_eitech",
            48367,
            (4.3, 141.0),
            (19.95, 54.6),
            "u_xx + u_yy + 1 = 0",
            1e-12,
        ),
        (
            "load_laplacian_smile",
            44711,
            (-4.0, 4.0),
            (-4.0, 4.0),
            "u_xx + u_yy = 0",
            1e-12,
        ),
        (
            "load_poisson_disk",
            40200,
            (-1.493, 1.493),
            (-1.493, 1.493),
            "u_xx + u_yy = 0",
            5e-4,
        ),
    ],
)
def test_steady_eqgpt_loader_returns_verified_scattered_dataset(
    loader_name: str,
    point_count: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    ground_truth: str,
    range_atol: float,
) -> None:
    loader: Callable[[], PDEDataset] = getattr(synthetic, loader_name)
    dataset = loader()

    assert dataset.topology == DataTopology.SCATTERED
    assert dataset.lhs_field == ""
    assert dataset.lhs_axis == ""
    assert dataset.lhs_order == 0
    assert dataset.get_shape() == (point_count,)
    assert dataset.ground_truth == ground_truth

    x = dataset.get_coords("x")
    y = dataset.get_coords("y")
    assert x.numel() == point_count
    assert y.numel() == point_count
    torch.testing.assert_close(
        x.min(), torch.tensor(x_range[0], dtype=x.dtype), rtol=0.0, atol=range_atol
    )
    torch.testing.assert_close(
        x.max(), torch.tensor(x_range[1], dtype=x.dtype), rtol=0.0, atol=range_atol
    )
    torch.testing.assert_close(
        y.min(), torch.tensor(y_range[0], dtype=y.dtype), rtol=0.0, atol=range_atol
    )
    torch.testing.assert_close(
        y.max(), torch.tensor(y_range[1], dtype=y.dtype), rtol=0.0, atol=range_atol
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dataset_id", "point_count", "u_range"),
    [
        ("eqgpt-laplacian-eitech", 48367, (1.40747e-05, 3.46053)),
        ("eqgpt-laplacian-smile", 44711, (-0.997922, 0.998564)),
        ("eqgpt-poisson-disk", 40200, (-0.89365, -0.00547498)),
    ],
)
def test_steady_eqgpt_public_api_loads_finite_u_field(
    dataset_id: str, point_count: int, u_range: tuple[float, float]
) -> None:
    spec = kd.get_dataset(dataset_id)
    dataset = spec.loader()

    assert dataset.name == dataset_id
    assert dataset.ground_truth == spec.equation

    u = dataset.get_field("u")
    assert u.numel() == point_count
    assert bool(torch.isfinite(u).all())
    torch.testing.assert_close(
        u.min(), torch.tensor(u_range[0], dtype=u.dtype), rtol=0.0, atol=1e-4
    )
    torch.testing.assert_close(
        u.max(), torch.tensor(u_range[1], dtype=u.dtype), rtol=0.0, atol=1e-4
    )
