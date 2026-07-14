
from __future__ import annotations

import pytest
import torch

from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType


def _dataset(*, lhs_order: int, lhs_axis: str, lhs_field: str) -> PDEDataset:
    x = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    u = torch.outer(x, t)
    return PDEDataset(
        name="steady",
        task_type=TaskType.PDE,
        axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
        axis_order=["x", "t"],
        fields={"u": FieldData("u", u)},
        lhs_field=lhs_field,
        lhs_axis=lhs_axis,
        lhs_order=lhs_order,
    )


class TestLhsOrderZero:

    @pytest.mark.unit
    def test_order_zero_empty_axis_is_valid(self) -> None:
        ds = _dataset(lhs_order=0, lhs_axis="", lhs_field="")
        assert ds.lhs_order == 0
        assert ds.lhs_axis == ""

    @pytest.mark.unit
    def test_order_zero_named_axis_raises_contradiction(self) -> None:
        with pytest.raises(ValueError, match=r"(?i)lhs_axis"):
            _dataset(lhs_order=0, lhs_axis="t", lhs_field="u")

    @pytest.mark.unit
    def test_order_zero_named_field_raises_contradiction(self) -> None:
        with pytest.raises(ValueError, match=r"(?i)lhs_field"):
            _dataset(lhs_order=0, lhs_axis="", lhs_field="u")


class TestLhsOrderBoundsRegressionGuards:

    @pytest.mark.unit
    def test_negative_order_raises(self) -> None:
        with pytest.raises(ValueError, match=r"(?i)lhs_order"):
            _dataset(lhs_order=-1, lhs_axis="t", lhs_field="u")

    @pytest.mark.unit
    def test_default_first_order_is_valid(self) -> None:
        ds = _dataset(lhs_order=1, lhs_axis="t", lhs_field="u")
        assert ds.lhs_order == 1

    @pytest.mark.unit
    def test_non_integer_order_raises(self) -> None:
        with pytest.raises(ValueError, match=r"(?i)lhs_order"):
            _dataset(lhs_order=0.5, lhs_axis="", lhs_field="")

    @pytest.mark.unit
    def test_bool_order_raises(self) -> None:
        with pytest.raises(ValueError, match=r"(?i)lhs_order"):
            _dataset(lhs_order=True, lhs_axis="", lhs_field="")
