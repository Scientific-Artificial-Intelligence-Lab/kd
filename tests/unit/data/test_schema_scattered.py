
from __future__ import annotations

import pytest
import torch

from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.search.discover.pinn.executor import make_pinn_dataset






def _raw_scatter(
    *,
    axis_len: int = 6,
    field_len: int = 6,
) -> PDEDataset:
    t = torch.arange(axis_len, dtype=torch.float64)
    x = torch.arange(axis_len, dtype=torch.float64) + 0.5
    u_values = torch.arange(field_len, dtype=torch.float64) + 1.0
    return PDEDataset(
        name="scatter",
        task_type=TaskType.PDE,
        topology=DataTopology.SCATTERED,
        axes={"t": AxisInfo("t", t), "x": AxisInfo("x", x)},
        axis_order=["t", "x"],
        fields={"u": FieldData("u", u_values)},
        lhs_field="u",
        lhs_axis="t",
        lhs_order=1,
    )


def _grid_dataset(
    *, n_t: int = 5, n_x: int = 6, u_shape_bad: bool = False
) -> PDEDataset:
    t = torch.linspace(0.0, 1.0, n_t, dtype=torch.float64)
    x = torch.linspace(0.0, 1.0, n_x, dtype=torch.float64)
    shape = (n_t, n_t) if u_shape_bad else (n_t, n_x)
    u = torch.ones(shape, dtype=torch.float64)
    return PDEDataset(
        name="grid",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"t": AxisInfo("t", t), "x": AxisInfo("x", x)},
        axis_order=["t", "x"],
        fields={"u": FieldData("u", u)},
        lhs_field="u",
        lhs_axis="t",
        lhs_order=1,
    )


def _from_scatter(
    coords: dict[str, torch.Tensor], fields: dict[str, torch.Tensor], lhs: str
) -> PDEDataset:
    return PDEDataset.from_scatter(coords=coords, fields=fields, lhs=lhs)







class TestScatteredValidationBranch:

    @pytest.mark.unit
    def test_equal_length_scatter_is_valid(self) -> None:
        ds = _raw_scatter(axis_len=6, field_len=6)
        assert ds.topology == DataTopology.SCATTERED
        assert torch.equal(
            ds.get_field("u"), torch.arange(6, dtype=torch.float64) + 1.0
        )

    @pytest.mark.unit
    def test_grid_shaped_scatter_accepted_backward_compat(self) -> None:
        nx, nt = 8, 5
        x = torch.arange(nx, dtype=torch.float64)
        t = torch.arange(nt, dtype=torch.float64)
        u = torch.ones((nx, nt), dtype=torch.float64)
        ds = PDEDataset(
            name="grid-shaped-scatter",
            task_type=TaskType.PDE,
            topology=DataTopology.SCATTERED,
            axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
            axis_order=["x", "t"],
            fields={"u": FieldData("u", u)},
            lhs_field="u",
            lhs_axis="t",
        )
        assert ds.topology == DataTopology.SCATTERED
        assert ds.get_field("u").shape == (nx, nt)







class TestGridValidationUnchanged:

    @pytest.mark.unit
    def test_grid_matching_shape_valid(self) -> None:
        ds = _grid_dataset(n_t=5, n_x=6)
        assert ds.get_shape() == (5, 6)

    @pytest.mark.unit
    def test_grid_mismatched_shape_raises(self) -> None:
        with pytest.raises(ValueError, match=r"(?i)shape mismatch"):
            _grid_dataset(n_t=5, n_x=6, u_shape_bad=True)







class TestGetShape:

    @pytest.mark.unit
    def test_scattered_get_shape_returns_n(self) -> None:
        n = 7
        coords = {
            "t": torch.arange(n, dtype=torch.float64),
            "x": torch.arange(n, dtype=torch.float64) + 0.5,
        }
        fields = {"u": torch.arange(n, dtype=torch.float64) + 1.0}
        ds = _from_scatter(coords, fields, "u_t")
        assert ds.get_shape() == (7,)

    @pytest.mark.unit
    def test_grid_get_shape_unchanged(self) -> None:
        assert _grid_dataset(n_t=5, n_x=6).get_shape() == (5, 6)

    @pytest.mark.unit
    def test_pinn_metadata_get_shape_still_raises(self) -> None:
        ds = make_pinn_dataset(
            axis_names=["t", "x"],
            field_names=["u"],
            lhs_field="u",
            lhs_axis="t",
        )
        with pytest.raises(ValueError, match=r"(?i)missing axes/axis_order"):
            ds.get_shape()







class TestSpatialAxes:

    @pytest.mark.unit
    def test_grid_lhs_axis_empty_returns_empty(self) -> None:
        t = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
        x = torch.linspace(0.0, 1.0, 3, dtype=torch.float64)
        u = torch.ones((4, 3), dtype=torch.float64)
        ds = PDEDataset(
            name="grid-homog",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"t": AxisInfo("t", t), "x": AxisInfo("x", x)},
            axis_order=["t", "x"],
            fields={"u": FieldData("u", u)},
            lhs_field="",
            lhs_axis="",
            lhs_order=0,
        )
        assert ds.spatial_axes == []

    @pytest.mark.unit
    def test_grid_evolution_spatial_axes_unchanged(self) -> None:
        assert _grid_dataset(n_t=5, n_x=6).spatial_axes == ["x"]

    @pytest.mark.unit
    def test_scattered_evolution_spatial_axes(self) -> None:
        coords = {
            "t": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
            "x": torch.tensor([0.5, 1.5, 0.2], dtype=torch.float64),
        }
        fields = {"u": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)}
        ds = _from_scatter(coords, fields, "u_t")
        assert ds.spatial_axes == ["x"]

    @pytest.mark.unit
    def test_scattered_homogeneous_spatial_axes_all(self) -> None:
        coords = {
            "x": torch.tensor([0.1, 0.9, 0.4], dtype=torch.float64),
            "y": torch.tensor([2.0, 0.3, 1.1], dtype=torch.float64),
        }
        fields = {"u": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)}
        ds = _from_scatter(coords, fields, "")
        assert ds.spatial_axes == ["x", "y"]

    @pytest.mark.unit
    def test_pinn_evolution_spatial_axes_unchanged(self) -> None:
        ds = make_pinn_dataset(
            axis_names=["t", "x"],
            field_names=["u"],
            lhs_field="u",
            lhs_axis="t",
        )
        assert ds.spatial_axes == ["x"]
