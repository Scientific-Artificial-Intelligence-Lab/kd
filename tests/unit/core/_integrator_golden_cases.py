
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)


EQUIVALENCE_RTOL = 1e-6


EQUIVALENCE_ATOL_SCALE = 1e-6











NESTED_VS_EXPANDED_RTOL = 2e-2
NESTED_VS_EXPANDED_ATOL_SCALE = 2e-2

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "integrator_golden"


def build_burgers_1d_periodic() -> PDEDataset:
    nx, nt = 32, 10
    x = torch.linspace(0.0, 2 * torch.pi, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.sin(x).unsqueeze(-1).expand(nx, nt).clone()
    return PDEDataset(
        name="golden-burgers-1d-periodic",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "t": AxisInfo(name="t", values=t, is_periodic=False),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def build_heat_1d_dirichlet() -> PDEDataset:
    nx, nt = 32, 10
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
    u = torch.zeros(nx, nt, dtype=torch.float64)
    u[:, 0] = torch.sin(torch.pi * x)
    return PDEDataset(
        name="golden-heat-1d-dirichlet",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=False),
            "t": AxisInfo(name="t", values=t, is_periodic=False),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def build_shifted_positive_1d_periodic() -> PDEDataset:
    nx, nt = 32, 8
    x = torch.linspace(0.0, 2 * torch.pi, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
    u = (1.5 + 0.5 * torch.sin(x)).unsqueeze(-1).expand(nx, nt).clone()
    return PDEDataset(
        name="golden-shifted-positive-1d-periodic",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "t": AxisInfo(name="t", values=t, is_periodic=False),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def build_advection_2d_periodic() -> PDEDataset:
    nx, ny, nt = 16, 16, 5
    x = torch.linspace(0.0, 2 * torch.pi, nx, dtype=torch.float64)
    y = torch.linspace(0.0, 2 * torch.pi, ny, dtype=torch.float64)
    t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
    x_grid, y_grid, _ = torch.meshgrid(x, y, t, indexing="ij")
    u = torch.sin(x_grid) * torch.cos(y_grid)
    return PDEDataset(
        name="golden-advection-2d-periodic",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "y": AxisInfo(name="y", values=y, is_periodic=True),
            "t": AxisInfo(name="t", values=t, is_periodic=False),
        },
        axis_order=["x", "y", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


@dataclass(frozen=True)
class GoldenCase:

    case_id: str
    rhs: str
    build_dataset: Callable[[], PDEDataset]
    method: str = "Radau"
    max_step: float | None = None

    def solve_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"method": self.method}
        if self.max_step is not None:
            kwargs["max_step"] = self.max_step
        return kwargs

    @property
    def fixture_path(self) -> Path:
        return FIXTURE_DIR / f"{self.case_id}.pt"


GOLDEN_CASES: tuple[GoldenCase, ...] = (
    GoldenCase(
        case_id="burgers_1d_periodic",
        rhs="u*u_x + u_xx",
        build_dataset=build_burgers_1d_periodic,
    ),
    GoldenCase(
        case_id="coordinate_1d_periodic",
        rhs="u + x",
        build_dataset=build_burgers_1d_periodic,
    ),
    GoldenCase(
        case_id="heat_1d_dirichlet",
        rhs="u_xx",
        build_dataset=build_heat_1d_dirichlet,
    ),
    GoldenCase(
        case_id="advection_2d_periodic",
        rhs="u_x + u_y",
        build_dataset=build_advection_2d_periodic,
    ),






    GoldenCase(
        case_id="coordinate_2d_periodic",
        rhs="u + x",
        build_dataset=build_advection_2d_periodic,
    ),





    GoldenCase(
        case_id="product_rule_1d_periodic",
        rhs="u*u_xx + u_x**2",
        build_dataset=build_shifted_positive_1d_periodic,
    ),
)


def golden_case(case_id: str) -> GoldenCase:
    for case in GOLDEN_CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(f"Unknown golden case id: {case_id}")


def load_golden_field(case_id: str) -> torch.Tensor:
    payload = torch.load(golden_case(case_id).fixture_path, weights_only=True)
    field = payload["predicted_field"]
    assert isinstance(field, torch.Tensor)
    return field


def golden_atol(golden_field: torch.Tensor) -> float:
    return EQUIVALENCE_ATOL_SCALE * float(golden_field.abs().max().item())
