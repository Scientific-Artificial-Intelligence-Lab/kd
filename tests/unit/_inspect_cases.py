
from __future__ import annotations

import torch

from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)


def _ramp(nx: int, nt: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.arange(nx * nt, dtype=dtype).reshape(nx, nt) / 100.0


def _grid_dataset(
    *,
    name: str,
    x: torch.Tensor,
    t: torch.Tensor,
    fields: dict[str, FieldData],
    lhs_field: str = "u",
    lhs_axis: str = "t",
    lhs_order: int = 1,
    ground_truth: str | None = None,
) -> PDEDataset:
    return PDEDataset(
        name=name,
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields=fields,
        lhs_field=lhs_field,
        lhs_axis=lhs_axis,
        lhs_order=lhs_order,
        ground_truth=ground_truth,
    )


def clean_uniform_dataset() -> PDEDataset:
    nx, nt = 32, 20
    return _grid_dataset(
        name="probe_clean",
        x=torch.linspace(0.0, 1.0, nx, dtype=torch.float64),
        t=torch.linspace(0.0, 2.0, nt, dtype=torch.float64),
        fields={"u": FieldData(name="u", values=_ramp(nx, nt))},
    )


def non_uniform_dataset() -> PDEDataset:
    nx, nt = 32, 20
    return _grid_dataset(
        name="probe_nonuniform",
        x=torch.linspace(0.0, 1.0, nx, dtype=torch.float64) ** 2,
        t=torch.linspace(0.0, 2.0, nt, dtype=torch.float64),
        fields={"u": FieldData(name="u", values=_ramp(nx, nt))},
    )


def descending_dataset() -> PDEDataset:
    nx, nt = 5, 8
    return _grid_dataset(
        name="probe_descending",
        x=torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0], dtype=torch.float64),
        t=torch.linspace(0.0, 1.0, nt, dtype=torch.float64),
        fields={"u": FieldData(name="u", values=_ramp(nx, nt))},
    )


def non_finite_dx_dataset() -> PDEDataset:
    nx, nt = 5, 8
    return _grid_dataset(
        name="probe_inf_dx",
        x=torch.tensor([-3.0e38, 3.0e38, 0.0, 1.0, 2.0], dtype=torch.float32),
        t=torch.linspace(0.0, 1.0, nt, dtype=torch.float64),
        fields={"u": FieldData(name="u", values=_ramp(nx, nt))},
    )


def nan_inf_field_dataset() -> PDEDataset:
    nx, nt = 20, 12
    dataset = _grid_dataset(
        name="probe_nan_inf",
        x=torch.linspace(0.0, 1.0, nx, dtype=torch.float64),
        t=torch.linspace(0.0, 1.0, nt, dtype=torch.float64),
        fields={"u": FieldData(name="u", values=_ramp(nx, nt))},
    )
    assert dataset.fields is not None
    values = dataset.fields["u"].values
    values[0, 0] = float("nan")
    values[1, 1] = float("inf")
    return dataset


def all_non_finite_field_dataset() -> PDEDataset:
    nx, nt = 20, 12
    dataset = _grid_dataset(
        name="probe_all_non_finite",
        x=torch.linspace(0.0, 1.0, nx, dtype=torch.float64),
        t=torch.linspace(0.0, 1.0, nt, dtype=torch.float64),
        fields={"u": FieldData(name="u", values=_ramp(nx, nt))},
    )
    assert dataset.fields is not None
    dataset.fields["u"].values.fill_(float("nan"))
    return dataset


def small_grid_dataset() -> PDEDataset:
    nx, nt = 8, 20
    return _grid_dataset(
        name="probe_small_grid",
        x=torch.linspace(0.0, 1.0, nx, dtype=torch.float64),
        t=torch.linspace(0.0, 2.0, nt, dtype=torch.float64),
        fields={"u": FieldData(name="u", values=_ramp(nx, nt))},
    )


def axis_probe_dataset(x: torch.Tensor, *, name: str = "probe_axis") -> PDEDataset:
    nx, nt = int(x.numel()), 8
    return _grid_dataset(
        name=name,
        x=x,
        t=torch.linspace(0.0, 1.0, nt, dtype=torch.float64),
        fields={"u": FieldData(name="u", values=_ramp(nx, nt))},
    )


def single_point_axis_dataset() -> PDEDataset:
    nx, nt = 1, 20
    return _grid_dataset(
        name="probe_single_point",
        x=torch.tensor([0.5], dtype=torch.float64),
        t=torch.linspace(0.0, 2.0, nt, dtype=torch.float64),
        fields={"u": FieldData(name="u", values=_ramp(nx, nt))},
    )


def lhs_unset_dataset() -> PDEDataset:
    nx, nt = 32, 20
    return _grid_dataset(
        name="probe_lhs_unset",
        x=torch.linspace(0.0, 1.0, nx, dtype=torch.float64),
        t=torch.linspace(0.0, 2.0, nt, dtype=torch.float64),
        fields={"u": FieldData(name="u", values=_ramp(nx, nt))},
        lhs_field="",
        lhs_axis="",
    )


def mixed_dtype_dataset() -> PDEDataset:
    nx, nt = 32, 20
    return _grid_dataset(
        name="probe_mixed_dtype",
        x=torch.linspace(0.0, 1.0, nx, dtype=torch.float64),
        t=torch.linspace(0.0, 2.0, nt, dtype=torch.float64),
        fields={
            "u": FieldData(name="u", values=_ramp(nx, nt)),
            "v": FieldData(name="v", values=_ramp(nx, nt, dtype=torch.float32)),
        },
    )


def wave_lhs_dataset() -> PDEDataset:
    nx, nt = 32, 20
    return _grid_dataset(
        name="probe_wave",
        x=torch.linspace(0.0, 1.0, nx, dtype=torch.float64),
        t=torch.linspace(0.0, 2.0, nt, dtype=torch.float64),
        fields={"u": FieldData(name="u", values=_ramp(nx, nt))},
        lhs_order=2,
    )


def underscore_axis_dataset() -> PDEDataset:
    nx, nt = 32, 20
    return PDEDataset(
        name="probe_underscore_axis",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(
                name="x", values=torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
            ),
            "x_1": AxisInfo(
                name="x_1", values=torch.linspace(0.0, 2.0, nt, dtype=torch.float64)
            ),
        },
        axis_order=["x", "x_1"],
        fields={"u": FieldData(name="u", values=_ramp(nx, nt))},
        lhs_field="u",
        lhs_axis="x_1",
        lhs_order=2,
    )


def ground_truth_dataset(secret: str) -> PDEDataset:
    nx, nt = 32, 20
    dataset = _grid_dataset(
        name="probe_ground_truth",
        x=torch.linspace(0.0, 1.0, nx, dtype=torch.float64),
        t=torch.linspace(0.0, 2.0, nt, dtype=torch.float64),
        fields={"u": FieldData(name="u", values=_ramp(nx, nt))},
        ground_truth=secret,
    )
    dataset.noise_level = 0.375
    return dataset


def scattered_dataset() -> PDEDataset:
    coords = {
        "x": torch.tensor([0.4, 0.1, 0.9, 0.2, 0.7], dtype=torch.float64),
        "t": torch.tensor([0.0, 0.5, 0.25, 0.75, 1.0], dtype=torch.float64),
    }
    fields = {"u": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)}
    return PDEDataset.from_scatter(coords, fields, name="probe_scattered")


def metadata_only_dataset() -> PDEDataset:
    return PDEDataset(
        name="probe_metadata_only",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
    )
