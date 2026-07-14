
from __future__ import annotations

import re

import pytest
import torch

from kd.core.integrator import IntegrationResult, integrate_pde
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from tests.unit.core._integrator_golden_cases import (
    EQUIVALENCE_RTOL,
    GOLDEN_CASES,
    NESTED_VS_EXPANDED_ATOL_SCALE,
    NESTED_VS_EXPANDED_RTOL,
    GoldenCase,
    build_burgers_1d_periodic,
    build_heat_1d_dirichlet,
    build_shifted_positive_1d_periodic,
    golden_atol,
    load_golden_field,
)






@pytest.fixture
def periodic_1d_dataset() -> PDEDataset:
    return build_burgers_1d_periodic()


@pytest.fixture
def dirichlet_1d_dataset() -> PDEDataset:
    return build_heat_1d_dirichlet()


@pytest.fixture
def two_field_dataset() -> PDEDataset:
    nx, nt = 32, 10
    x = torch.linspace(0.0, 2 * torch.pi, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.sin(x).unsqueeze(-1).expand(nx, nt).clone()
    v = torch.cos(x).unsqueeze(-1).expand(nx, nt).clone()
    return PDEDataset(
        name="ir-native-two-field",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "t": AxisInfo(name="t", values=t, is_periodic=False),
        },
        axis_order=["x", "t"],
        fields={
            "u": FieldData(name="u", values=u),
            "v": FieldData(name="v", values=v),
        },
        lhs_field="u",
        lhs_axis="t",
    )


@pytest.fixture
def shifted_positive_dataset() -> PDEDataset:
    return build_shifted_positive_1d_periodic()


@pytest.fixture
def small_amplitude_dataset() -> PDEDataset:
    nx, nt = 32, 8
    x = torch.linspace(0.0, 2 * torch.pi, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
    u = (0.3 * torch.sin(x)).unsqueeze(-1).expand(nx, nt).clone()
    return PDEDataset(
        name="ir-native-small-amplitude",
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


def _assert_successful_field(result: IntegrationResult, dataset: PDEDataset) -> None:
    assert result.success is True, f"integration failed: {result.warning}"
    assert result.predicted_field is not None
    assert result.predicted_field.shape == dataset.get_shape()
    assert torch.isfinite(result.predicted_field).all()







class TestGoldenEquivalence:

    @pytest.mark.parametrize(
        "case", GOLDEN_CASES, ids=[c.case_id for c in GOLDEN_CASES]
    )
    def test_matches_old_path_golden_field(self, case: GoldenCase) -> None:
        dataset = case.build_dataset()
        result = integrate_pde(case.rhs, dataset, **case.solve_kwargs())

        assert result.success is True, f"integration failed: {result.warning}"
        assert result.predicted_field is not None
        golden = load_golden_field(case.case_id)
        torch.testing.assert_close(
            result.predicted_field,
            golden,
            rtol=EQUIVALENCE_RTOL,
            atol=golden_atol(golden),
        )







class TestNestedDerivativeCure:

    def test_nonlinear_nested_diff_integrates(
        self, shifted_positive_dataset: PDEDataset
    ) -> None:
        dataset = shifted_positive_dataset
        result = integrate_pde("diff_x(mul(u, u_x))", dataset)
        _assert_successful_field(result, dataset)
        assert result.predicted_field is not None

        golden = load_golden_field("product_rule_1d_periodic")
        torch.testing.assert_close(
            result.predicted_field,
            golden,
            rtol=NESTED_VS_EXPANDED_RTOL,
            atol=NESTED_VS_EXPANDED_ATOL_SCALE * float(golden.abs().max().item()),
        )

        assert dataset.fields is not None
        initial = dataset.fields["u"].values[:, 0]
        drift = (result.predicted_field[:, -1] - initial).abs().max().item()
        assert drift > 0.1, (
            f"field must evolve away from the initial condition, drift={drift:.3g}"
        )

    def test_nested_second_derivative_of_sum_integrates(
        self, dirichlet_1d_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("diff2_x(add(u, x))", dirichlet_1d_dataset)
        _assert_successful_field(result, dirichlet_1d_dataset)
        assert result.predicted_field is not None

        golden = load_golden_field("heat_1d_dirichlet")
        torch.testing.assert_close(
            result.predicted_field,
            golden,
            rtol=EQUIVALENCE_RTOL,
            atol=golden_atol(golden),
        )

    def test_nested_diff2_matches_terminal_uxx_golden(
        self, dirichlet_1d_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("diff2_x(u)", dirichlet_1d_dataset)

        assert result.success is True, f"integration failed: {result.warning}"
        assert result.predicted_field is not None
        golden = load_golden_field("heat_1d_dirichlet")
        torch.testing.assert_close(
            result.predicted_field,
            golden,
            rtol=EQUIVALENCE_RTOL,
            atol=golden_atol(golden),
        )







def _assert_clean_rejection(result: IntegrationResult) -> None:
    assert isinstance(result, IntegrationResult)
    assert result.success is False
    assert result.predicted_field is None
    assert result.warning, "rejection must carry a diagnostic warning"


class TestRejectionLocks:

    def test_time_axis_derivative_rejected(
        self, periodic_1d_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("diff_t(u)", periodic_1d_dataset)
        _assert_clean_rejection(result)
        assert "diff_t" in result.warning

    def test_time_axis_compound_name_rejected(
        self, periodic_1d_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("u_t", periodic_1d_dataset)
        _assert_clean_rejection(result)
        assert "u_t" in result.warning

    def test_unknown_axis_diff_call_rejected(
        self, periodic_1d_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("diff_y(u)", periodic_1d_dataset)
        _assert_clean_rejection(result)
        assert "diff_y" in result.warning

    def test_cross_field_derivative_rejected(
        self, two_field_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("v_x", two_field_dataset)
        _assert_clean_rejection(result)
        assert "v_x" in result.warning

    def test_cross_field_state_var_rejected(
        self, two_field_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("v", two_field_dataset)
        _assert_clean_rejection(result)
        assert re.search(r"\bv\b", result.warning), (
            f"warning must name the offending field 'v': {result.warning!r}"
        )

    def test_unknown_symbol_rejected(self, periodic_1d_dataset: PDEDataset) -> None:
        result = integrate_pde("u + foobar", periodic_1d_dataset)
        _assert_clean_rejection(result)
        assert "foobar" in result.warning

    def test_named_constant_pi_rejected(self, periodic_1d_dataset: PDEDataset) -> None:
        result = integrate_pde("pi*u", periodic_1d_dataset)
        _assert_clean_rejection(result)
        assert re.search(r"\bpi\b", result.warning), (
            f"warning must name the offending constant 'pi': {result.warning!r}"
        )

    def test_lap_operator_rejected(self, periodic_1d_dataset: PDEDataset) -> None:
        result = integrate_pde("lap(u)", periodic_1d_dataset)
        _assert_clean_rejection(result)
        assert "lap" in result.warning

    def test_non_ir_binary_operator_rejected_preflight(
        self, periodic_1d_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("u % 2", periodic_1d_dataset)
        _assert_clean_rejection(result)
        assert "Mod" in result.warning
        assert "solve_ivp" not in result.warning

    def test_malformed_diff_arity_rejected_preflight(
        self, periodic_1d_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("diff_x(u, u_x)", periodic_1d_dataset)
        _assert_clean_rejection(result)
        assert "diff_x" in result.warning
        assert "1" in result.warning and "2" in result.warning
        assert "solve_ivp" not in result.warning

    def test_zero_arg_diff_rejected_preflight(
        self, periodic_1d_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("diff_x()", periodic_1d_dataset)
        _assert_clean_rejection(result)
        assert "diff_x" in result.warning


class TestRejectionPositiveControls:

    def test_spatial_diff_call_accepted(self, periodic_1d_dataset: PDEDataset) -> None:
        result = integrate_pde("diff_x(u)", periodic_1d_dataset)
        _assert_successful_field(result, periodic_1d_dataset)

    def test_lhs_field_derivative_accepted(self, two_field_dataset: PDEDataset) -> None:
        result = integrate_pde("u_x", two_field_dataset)
        _assert_successful_field(result, two_field_dataset)

    def test_lhs_field_state_var_accepted(self, two_field_dataset: PDEDataset) -> None:
        result = integrate_pde("u", two_field_dataset)
        _assert_successful_field(result, two_field_dataset)

    def test_spatial_coordinate_symbol_accepted(
        self, periodic_1d_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("u + x", periodic_1d_dataset)
        _assert_successful_field(result, periodic_1d_dataset)

    def test_literal_constant_accepted(self, periodic_1d_dataset: PDEDataset) -> None:
        result = integrate_pde("3.141592653589793*u", periodic_1d_dataset)
        _assert_successful_field(result, periodic_1d_dataset)

    def test_explicit_second_derivative_call_accepted(
        self, periodic_1d_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("diff2_x(u)", periodic_1d_dataset)
        _assert_successful_field(result, periodic_1d_dataset)


class TestRegistryFunctionRHS:

    def test_registry_log_integrates_with_protected_semantics(
        self, periodic_1d_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("log(u)", periodic_1d_dataset)
        _assert_successful_field(result, periodic_1d_dataset)
        assert result.predicted_field is not None


        assert result.predicted_field.abs().max().item() <= 1.5







class TestDivergenceParityLocks:

    @pytest.mark.parametrize("method", ["Radau", "RK45"])
    def test_derivative_free_explosion_fails_gracefully(
        self, periodic_1d_dataset: PDEDataset, method: str
    ) -> None:
        result = integrate_pde("1000000.0*u", periodic_1d_dataset, method=method)

        assert isinstance(result, IntegrationResult)
        assert result.success is False
        assert result.predicted_field is None
        assert result.diverged_at_t is None
        assert result.warning, "failure must carry a diagnostic warning"

    def test_derivative_carrying_nan_aborts_cleanly(
        self, periodic_1d_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("1000000.0*u + u_xx", periodic_1d_dataset)

        assert isinstance(result, IntegrationResult)
        assert result.success is False
        assert result.predicted_field is None
        assert result.diverged_at_t is None
        assert result.warning, "abort path must carry a diagnostic warning"







class TestPowLocks:

    def test_pow_derivative_free_integrates(
        self, small_amplitude_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("u**3", small_amplitude_dataset)
        _assert_successful_field(result, small_amplitude_dataset)

    def test_pow_inside_nested_diff_executes(
        self, small_amplitude_dataset: PDEDataset
    ) -> None:
        result = integrate_pde("diff_x(u**2)", small_amplitude_dataset)
        _assert_successful_field(result, small_amplitude_dataset)







class TestKwargsParity:

    def test_method_kwarg_accepted(self, periodic_1d_dataset: PDEDataset) -> None:
        result = integrate_pde("u_x", periodic_1d_dataset, method="RK45")
        _assert_successful_field(result, periodic_1d_dataset)

    def test_max_step_kwarg_accepted(self, periodic_1d_dataset: PDEDataset) -> None:
        result = integrate_pde("u_x", periodic_1d_dataset, max_step=0.05)
        _assert_successful_field(result, periodic_1d_dataset)







def _build_scattered_dataset() -> PDEDataset:
    nx, nt = 16, 5
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
    u = torch.sin(x).unsqueeze(-1).expand(nx, nt).clone()
    return PDEDataset(
        name="ir-native-scattered",
        task_type=TaskType.PDE,
        topology=DataTopology.SCATTERED,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _build_tiny_grid_dataset() -> PDEDataset:
    nx, nt = 4, 6
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
    u = (1.0 + 0.25 * x).unsqueeze(-1).expand(nx, nt).clone()
    return PDEDataset(
        name="ir-native-tiny-grid",
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


class TestBoundaryLocks:

    def test_grid_only_guard_survives_signature_change(self) -> None:
        result = integrate_pde("u", _build_scattered_dataset())

        assert isinstance(result, IntegrationResult)
        assert result.success is False
        assert result.predicted_field is None
        assert "GRID" in result.warning

    def test_derivative_free_rhs_on_tiny_grid_integrates(self) -> None:
        dataset = _build_tiny_grid_dataset()
        result = integrate_pde("0.5*u", dataset)

        _assert_successful_field(result, dataset)
        assert result.predicted_field is not None



        assert result.predicted_field[1, -1] > result.predicted_field[1, 0]
