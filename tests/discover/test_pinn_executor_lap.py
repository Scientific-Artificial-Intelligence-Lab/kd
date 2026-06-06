
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from kd.core.expr import FunctionRegistry
from kd.data.schema import (
    DataTopology,
    PDEDataset,
    TaskType,
)
from kd.search.discover.pinn.executor import PINNExecutor, make_pinn_dataset

N_POINTS = 128
ATOL = 1e-4
EXP_CLAMP_MAX = 50.0
LAPLACIAN_2D_RESIDUAL_FACTOR = 1.0
NESTED_LAP_RESIDUAL_FACTOR = -3.0
ZERO_RESIDUAL = 0.0


def _safe_exp(value: Tensor) -> Tensor:
    return torch.exp(torch.clamp(value, max=EXP_CLAMP_MAX))


def _make_2d_coords() -> dict[str, Tensor]:
    return {
        "x": torch.linspace(-1.0, 1.0, N_POINTS).requires_grad_(),
        "y": torch.linspace(0.2, 1.2, N_POINTS).requires_grad_(),
        "t": torch.linspace(0.0, 0.8, N_POINTS).requires_grad_(),
    }


def _make_1d_coords() -> dict[str, Tensor]:
    return {
        "x": torch.linspace(-1.0, 1.0, N_POINTS).requires_grad_(),
        "t": torch.linspace(0.0, 0.8, N_POINTS).requires_grad_(),
    }


class _SinCosExpModel(nn.Module):

    def forward(self, **coords: Tensor) -> dict[str, Tensor]:
        x, y, t = coords["x"], coords["y"], coords["t"]
        return {"u": torch.sin(x) * torch.cos(y) * _safe_exp(-t)}


class _SinExpModel(nn.Module):

    def forward(self, **coords: Tensor) -> dict[str, Tensor]:
        x, t = coords["x"], coords["t"]
        return {"u": torch.sin(x) * _safe_exp(-t)}


def _executor() -> PINNExecutor:
    return PINNExecutor(registry=FunctionRegistry.create_default())


def _dataset_2d() -> PDEDataset:
    return make_pinn_dataset(
        axis_names=["x", "y", "t"],
        field_names=["u"],
        lhs_field="u",
        lhs_axis="t",
    )


def _dataset_1d() -> PDEDataset:
    return make_pinn_dataset(
        axis_names=["x", "t"],
        field_names=["u"],
        lhs_field="u",
        lhs_axis="t",
    )


def _dataset_without_lhs_axis() -> PDEDataset:
    return PDEDataset(
        name="missing_lhs_axis",
        task_type=TaskType.PDE,
        topology=DataTopology.SCATTERED,
        axis_order=["x", "t"],
        fields=None,
        lhs_field="u",
        lhs_axis="",
    )


class TestPINNExecutorLap:

    @pytest.mark.unit
    def test_lap_2d_matches_expanded_second_derivatives(self) -> None:
        model = _SinCosExpModel()
        dataset = _dataset_2d()

        lap_residual = _executor().compute_residual(
            model=model,
            terms=["lap(u)"],
            coefficients=[1.0],
            coords=_make_2d_coords(),
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        expanded_residual = _executor().compute_residual(
            model=model,
            terms=["diff2_x(u)", "diff2_y(u)"],
            coefficients=[1.0, 1.0],
            coords=_make_2d_coords(),
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        expected = (
            LAPLACIAN_2D_RESIDUAL_FACTOR
            * model(**_make_2d_coords())["u"]
        )

        assert torch.allclose(
            lap_residual.detach(),
            expanded_residual.detach(),
            atol=ATOL,
        )
        assert torch.allclose(lap_residual.detach(), expected.detach(), atol=ATOL)

    @pytest.mark.unit
    def test_lap_1d_reduces_to_diff2_x(self) -> None:
        model = _SinExpModel()
        dataset = _dataset_1d()

        lap_residual = _executor().compute_residual(
            model=model,
            terms=["lap(u)"],
            coefficients=[1.0],
            coords=_make_1d_coords(),
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        diff2_residual = _executor().compute_residual(
            model=model,
            terms=["diff2_x(u)"],
            coefficients=[1.0],
            coords=_make_1d_coords(),
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )

        assert torch.allclose(
            lap_residual.detach(),
            diff2_residual.detach(),
            atol=ATOL,
        )
        assert torch.allclose(
            lap_residual.detach(),
            torch.full_like(lap_residual, ZERO_RESIDUAL),
            atol=ATOL,
        )

    @pytest.mark.unit
    def test_nested_lap_diff2_dispatches(self) -> None:
        model = _SinCosExpModel()
        dataset = _dataset_2d()
        residual = _executor().compute_residual(
            model=model,
            terms=["lap(diff2_x(u))"],
            coefficients=[1.0],
            coords=_make_2d_coords(),
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        expected = NESTED_LAP_RESIDUAL_FACTOR * model(**_make_2d_coords())["u"]

        assert torch.allclose(residual.detach(), expected.detach(), atol=ATOL)

    @pytest.mark.unit
    def test_lap_without_lhs_axis_raises_kd_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty spatial_axes"):
            _executor().compute_residual(
                model=_SinExpModel(),
                terms=["lap(u)"],
                coefficients=[1.0],
                coords=_make_1d_coords(),
                dataset_metadata=_dataset_without_lhs_axis(),
                lhs_field="u",
                lhs_axis="t",
            )
