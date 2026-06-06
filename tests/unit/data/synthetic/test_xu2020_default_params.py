
from __future__ import annotations

import pytest
import torch

from kd.data.synthetic import (
    generate_chaffee_infante_xu2020_data,
    generate_kdv_xu2020_data,
)


@pytest.mark.numerical
@pytest.mark.slow
def test_kdv_default_params_finite() -> None:
    dataset = generate_kdv_xu2020_data()
    u = dataset.get_field("u")
    assert torch.isfinite(u).all(), "KdV default-params produced NaN/Inf"


    assert u.abs().max() < 1.0, (
        f"KdV default field unreasonably large: max |u| = {u.abs().max().item()}"
    )





    norm0 = torch.linalg.norm(u[:, 0])
    norm_final = torch.linalg.norm(u[:, -1])
    rel_drift = abs(norm_final - norm0) / norm0
    assert rel_drift < 5e-2, (
        f"KdV L² norm drifted {rel_drift.item():.3%} (>5%); "
        f"CFL violation or RHS bug suspected"
    )


@pytest.mark.numerical
@pytest.mark.slow
def test_chaffee_infante_default_params_finite() -> None:
    dataset = generate_chaffee_infante_xu2020_data()
    u = dataset.get_field("u")
    assert torch.isfinite(u).all(), "CI default-params produced NaN/Inf"





    assert u.abs().max() < 1.5, (
        f"CI default field unreasonably large: max |u| = {u.abs().max().item()}"
    )



    final_max = u[:, -1].abs().max()
    initial_max = u[:, 0].abs().max()
    assert final_max < initial_max, (
        f"CI field grew from |u₀|.max={initial_max.item():.3f} to "
        f"|u_T|.max={final_max.item():.3f}; expected decay under -u term"
    )
