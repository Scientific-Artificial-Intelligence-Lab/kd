
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kd.data.derivatives.finite_diff import (
    UNIFORM_GRID_RTOL,
    _check_uniform_grid,
)
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)


def _fd_accepts(coords: torch.Tensor) -> bool:
    try:
        _check_uniform_grid(coords, "x")
        return True
    except ValueError:
        return False


def _integrator_accepts(coords: torch.Tensor) -> bool:
    from kd.core.integrator import _check_spatial_uniformity




    nx = coords.shape[0]
    nt = 4
    t = torch.linspace(0.0, 1.0, nt, dtype=coords.dtype)
    u = torch.zeros(nx, nt, dtype=coords.dtype)
    dataset = PDEDataset(
        name="uniformity_consistency_check",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=coords),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )
    warning = _check_spatial_uniformity(dataset, ["x"])
    return warning is None







@pytest.mark.unit
class TestPredicateAgreement:

    def test_uniform_linspace_both_accept(self) -> None:
        coords = torch.linspace(0.0, 1.0, 100, dtype=torch.float64)
        assert _fd_accepts(coords) == _integrator_accepts(coords)
        assert _fd_accepts(coords), "uniform linspace should be accepted"

    def test_geometric_spacing_both_reject(self) -> None:
        coords = torch.tensor([2.0**i for i in range(10)], dtype=torch.float64)
        assert _fd_accepts(coords) == _integrator_accepts(coords)
        assert not _fd_accepts(coords), "geometric grid should be rejected"

    def test_log_spacing_both_reject(self) -> None:
        coords = torch.logspace(0.0, 2.0, 50, dtype=torch.float64)
        assert _fd_accepts(coords) == _integrator_accepts(coords)
        assert not _fd_accepts(coords), "log-spaced grid should be rejected"

    def test_symmetric_drift_predicates_agree(self) -> None:
        rtol = UNIFORM_GRID_RTOL
        dx0 = 0.1

        delta = 0.8 * rtol * abs(dx0)

        coords = torch.tensor(
            [
                0.0,
                dx0,
                dx0 + (dx0 + delta),
                dx0 + (dx0 + delta) + (dx0 - delta),
                dx0 + (dx0 + delta) + (dx0 - delta) + dx0,
            ],
            dtype=torch.float64,
        )


        diffs = (coords[1:] - coords[:-1]).numpy()
        max_minus_min = float(diffs.max() - diffs.min())
        max_abs_dev = float(np.max(np.abs(diffs - diffs[0])))
        assert max_minus_min > rtol * abs(dx0), (
            f"setup error: max-min={max_minus_min} should exceed {rtol * abs(dx0)}"
        )
        assert max_abs_dev <= rtol * abs(dx0), (
            f"setup error: max-abs-dev={max_abs_dev} should be <= {rtol * abs(dx0)}"
        )


        assert _fd_accepts(coords) == _integrator_accepts(coords), (
            f"FD and integrator predicates disagree on the same grid. "
            f"FD accepts: {_fd_accepts(coords)}, "
            f"integrator accepts: {_integrator_accepts(coords)}. "
            f"Both should use the same uniform-grid predicate."
        )

    def test_one_sided_drift_predicates_agree(self) -> None:
        rtol = UNIFORM_GRID_RTOL
        dx0 = 0.1

        delta = 0.5 * rtol * abs(dx0)
        diffs = [dx0, dx0 + delta, dx0 + delta, dx0]
        coord_list = [0.0]
        for d in diffs:
            coord_list.append(coord_list[-1] + d)
        coords = torch.tensor(coord_list, dtype=torch.float64)
        assert _fd_accepts(coords) == _integrator_accepts(coords)

    @pytest.mark.parametrize("scale", [1.0, 1e-6, 1e6])
    def test_scale_invariance(self, scale: float) -> None:
        coords = torch.linspace(0.0, 1.0, 100, dtype=torch.float64) * scale
        assert _fd_accepts(coords) == _integrator_accepts(coords)


@pytest.mark.unit
class TestSharedHelperContract:

    def test_shared_helper_exists(self) -> None:
        try:
            from kd.data.derivatives.finite_diff import (
                is_uniform_grid,
            )

            helper_exists = True
        except ImportError:
            helper_exists = False

        assert helper_exists, (
            "Expected ``is_uniform_grid`` helper to be exported from "
            "``kd.data.derivatives.finite_diff`` so both "
            "``_check_uniform_grid`` and ``integrator._check_spatial_uniformity``"
            " can share a single predicate. Until the helper exists the two "
            "predicates remain mathematically inequivalent."
        )

    def test_shared_helper_returns_bool(self) -> None:
        from kd.data.derivatives.finite_diff import (
            is_uniform_grid,
        )

        uniform = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
        nonuniform = torch.tensor([0.0, 0.1, 0.21, 0.33, 0.46], dtype=torch.float64)
        assert is_uniform_grid(uniform, rtol=UNIFORM_GRID_RTOL) is True
        assert is_uniform_grid(nonuniform, rtol=UNIFORM_GRID_RTOL) is False







@pytest.mark.unit
class TestPredicateNegativeCases:

    def test_constant_coordinate_axis_both_reject(self) -> None:
        coords = torch.zeros(10, dtype=torch.float64)

        with pytest.raises(ValueError, match="degenerate"):
            _check_uniform_grid(coords, "x")

        assert not _integrator_accepts(coords)

    def test_subnormal_dx_both_reject(self) -> None:
        coords = torch.linspace(0.0, 1e-31, 10, dtype=torch.float64)
        with pytest.raises(ValueError, match="degenerate"):
            _check_uniform_grid(coords, "x")
        assert not _integrator_accepts(coords)

    def test_two_point_grid_predicates_agree(self) -> None:
        coords = torch.tensor([0.0, 0.1], dtype=torch.float64)


        try:
            _check_uniform_grid(coords, "x")
            fd = True
        except ValueError:
            fd = False
        intg = _integrator_accepts(coords)
        assert fd == intg, f"Two-point grid disagreement: FD={fd}, integrator={intg}"

    @pytest.mark.parametrize(
        "coords_list",
        [
            [0.0, 0.1, 0.21, 0.33, 0.46],
            [0.0, math.pi, 6.28, 9.42],
        ],
    )
    def test_predicates_agree_on_parametric_inputs(
        self, coords_list: list[float]
    ) -> None:
        coords = torch.tensor(coords_list, dtype=torch.float64)
        assert _fd_accepts(coords) == _integrator_accepts(coords), (
            f"Predicates disagree on {coords_list}: "
            f"FD={_fd_accepts(coords)}, integrator={_integrator_accepts(coords)}"
        )








@pytest.mark.unit
class TestUniformGridRejectsInfiniteSpacing:

    def test_is_uniform_grid_rejects_inf_dx0(self) -> None:
        from kd.data.derivatives.finite_diff import is_uniform_grid

        coords = np.array([-1.7e308, 1.7e308], dtype=np.float64)
        assert is_uniform_grid(coords) is False, (
            "is_uniform_grid must reject coords whose diff overflows to inf; "
            "np.allclose([inf], inf) returns True, so the helper needs an "
            "explicit np.isfinite guard."
        )

    def test_check_uniform_grid_raises_on_inf_dx0(self) -> None:
        coords = torch.tensor([-1.7e308, 1.7e308], dtype=torch.float64)
        with pytest.raises(ValueError):
            _check_uniform_grid(coords, "x")








@pytest.mark.unit
class TestUniformGridRejectsDescendingAxes:

    def test_is_uniform_grid_rejects_descending(self) -> None:
        from kd.data.derivatives.finite_diff import is_uniform_grid

        coords = np.array([0.4, 0.3, 0.2, 0.1, 0.0], dtype=np.float64)
        assert is_uniform_grid(coords) is False, (
            "is_uniform_grid must reject descending coords (dx0 < 0). "
            "FD stencils require positive dx; the existing abs(dx0) check "
            "incorrectly accepts negative spacing."
        )

    def test_check_uniform_grid_raises_on_descending(self) -> None:
        coords = torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0], dtype=torch.float64)
        with pytest.raises(ValueError):
            _check_uniform_grid(coords, "x")
