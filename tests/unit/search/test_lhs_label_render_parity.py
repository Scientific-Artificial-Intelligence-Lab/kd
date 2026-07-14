
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from kd.core.equation import LhsSpec, render_lhs_label
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.search.runner import ExperimentRunner
from tests.unit.search._runner_mocks import RecordingAlgorithm






def _order_dataset(lhs: str) -> PDEDataset:
    nx, nt = 6, 5
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.outer(x, t)
    return PDEDataset.from_arrays(coords={"x": x, "t": t}, fields={"u": u}, lhs=lhs)


def _time_axis_dataset() -> PDEDataset:
    nx, nt = 6, 5
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    time = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.outer(x, time)
    return PDEDataset.from_arrays(
        coords={"x": x, "time": time},
        fields={"u": u},
        lhs="u_time",
    )


def _underscore_axis_dataset() -> PDEDataset:
    nx, nt = 6, 5
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.outer(x, t)
    return PDEDataset(
        name="underscore-axis",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x_1": AxisInfo("x_1", x), "t": AxisInfo("t", t)},
        axis_order=["x_1", "t"],
        fields={"u": FieldData("u", u)},
        lhs_field="u",
        lhs_axis="x_1",
    )


def _fake_final_eval(lhs_name: str) -> MagicMock:
    final_eval = MagicMock()
    final_eval.lhs_name = lhs_name
    final_eval.terms = ["u_x"]
    final_eval.coefficients = torch.tensor([1.0], dtype=torch.float64)
    final_eval.is_valid = True
    return final_eval


def _components(dataset: object) -> MagicMock:
    components = MagicMock()
    components.dataset = dataset
    return components







class TestRenderLhsLabel:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            (LhsSpec(field="u", axis="t", order=1), "u_t"),
            (LhsSpec(field="u", axis="t", order=2), "u_tt"),

            (LhsSpec(field="u", axis="x_1", order=1), "u_x_1"),

            (LhsSpec(field="phi", axis="x", order=3), "phi_xxx"),
        ],
    )
    def test_render_matches_legacy_label(self, spec: LhsSpec, expected: str) -> None:
        assert render_lhs_label(spec) == expected







_PARITY_CASES = [

    (_order_dataset("u_t"), "u_tt", "u_tt", True, "lhs_name_priority_order2"),
    (_order_dataset("u_t"), "u_t", "u_t", True, "lhs_name_priority_order1"),
    (_time_axis_dataset(), "u_time", "u_time", True, "lhs_name_multichar_axis"),
    (_time_axis_dataset(), "u_t", "u_t", False, "context_rejected_lhs_name"),
    (_order_dataset("u_t"), "u_x_y", "u_x_y", False, "unparseable_lhs_name"),
    (_order_dataset("u_t"), "", "u_t", True, "dataset_order1_legacy_fstring"),
    (_order_dataset("u_tt"), "", "u_tt", True, "dataset_order2_builder"),
    (_underscore_axis_dataset(), "", "u_x_1", True, "dataset_underscore_axis"),
    (MagicMock(), "", "u_t", False, "default_fallback"),
]


class TestLhsLabelDerivationParity:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("dataset", "lhs_name", "expected", "expects_equation"),
        [(ds, name, exp, expect) for ds, name, exp, expect, _cid in _PARITY_CASES],
        ids=[cid for _ds, _name, _exp, _expect, cid in _PARITY_CASES],
    )
    def test_derived_label_matches_runner_lhs_label(
        self,
        dataset: object,
        lhs_name: str,
        expected: str,
        expects_equation: bool,
    ) -> None:
        runner = ExperimentRunner(RecordingAlgorithm(), max_iterations=1)
        components = _components(dataset)
        final_eval = _fake_final_eval(lhs_name)


        old_label = runner._lhs_label(components, final_eval)
        assert old_label == expected


        equation = runner._build_equation(components, final_eval)
        if expects_equation:
            assert equation is not None
            assert equation.lhs_spec is not None
            new_label = render_lhs_label(equation.lhs_spec)
            assert new_label == old_label == expected
        else:
            assert equation is None
            assert old_label == expected
