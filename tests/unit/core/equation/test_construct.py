
import logging

import pytest
import torch

import kd.core.equation as equation_pkg
from kd.core.equation import (
    Equation,
    Form,
    LhsSpec,
    Scalar,
    build_equation,
    make_evolution,
    make_homogeneous,
)

_LHS = LhsSpec(field="u", axis="t", order=1)
_TERMS = (("u_x", Scalar(1.0)), ("u_xx", Scalar(-0.5)))


class TestMakeEvolutionValid:
    @pytest.mark.unit
    def test_builds_evolution_equation(self) -> None:
        eq = make_evolution(_LHS, _TERMS)
        assert isinstance(eq, Equation)
        assert eq.form is Form.EVOLUTION
        assert eq.lhs_spec == _LHS
        assert eq.terms == _TERMS

    @pytest.mark.unit
    def test_terms_stored_as_tuple(self) -> None:
        eq = make_evolution(_LHS, [("u_x", Scalar(1.0)), ("u_xx", Scalar(2.0))])
        assert isinstance(eq.terms, tuple)
        assert eq.terms == (("u_x", Scalar(1.0)), ("u_xx", Scalar(2.0)))


class TestMakeEvolutionRejects:
    @pytest.mark.unit
    def test_rejects_none_lhs_spec(self) -> None:
        with pytest.raises(ValueError):
            make_evolution(None, _TERMS)

    @pytest.mark.unit
    def test_rejects_empty_terms(self) -> None:
        with pytest.raises(ValueError):
            make_evolution(_LHS, ())

    @pytest.mark.unit
    def test_rejects_empty_term_ir_string(self) -> None:
        with pytest.raises(ValueError):
            make_evolution(_LHS, (("u_x", Scalar(1.0)), ("", Scalar(2.0))))


class TestBuildEquationDegradation:
    @pytest.mark.unit
    def test_builds_from_term_irs_and_coefficients(self) -> None:
        eq = build_equation(["u_x", "u_xx"], torch.tensor([1.0, -0.5]), _LHS)

        assert eq == make_evolution(_LHS, _TERMS)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "coefficients",
        [
            None,
            torch.tensor([float("nan")]),
            torch.tensor([float("inf")]),
            torch.tensor([float("-inf")]),
        ],
    )
    def test_degrades_on_missing_or_non_finite_coefficients(
        self, coefficients: torch.Tensor | None
    ) -> None:
        assert build_equation(["u_x"], coefficients, _LHS) is None

    @pytest.mark.unit
    def test_degrades_on_length_mismatch_with_warning_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:





        caplog.set_level(logging.WARNING, logger="kd.core.equation.construct")

        assert build_equation(["u_x"], torch.tensor([1.0, 2.0]), _LHS) is None

        assert [
            record.levelno
            for record in caplog.records
            if "length mismatch" in record.getMessage()
        ] == [logging.WARNING]

    @pytest.mark.unit
    def test_degrades_on_non_finite_coefficient_with_warning_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:


        caplog.set_level(logging.WARNING, logger="kd.core.equation.construct")

        assert build_equation(["u_x"], torch.tensor([float("nan")]), _LHS) is None

        assert [
            record.levelno
            for record in caplog.records
            if "non-finite" in record.getMessage()
        ] == [logging.WARNING]

    @pytest.mark.unit
    def test_degrades_on_missing_lhs_spec(self) -> None:
        assert build_equation(["u_x"], torch.tensor([1.0]), None) is None

    @pytest.mark.unit
    def test_missing_lhs_spec_stays_at_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:




        caplog.set_level(logging.DEBUG, logger="kd.core.equation.construct")

        assert build_equation(["u_x"], torch.tensor([1.0]), None) is None

        assert [
            record.levelno
            for record in caplog.records
            if "missing lhs_spec" in record.getMessage()
        ] == [logging.DEBUG]
        assert [
            record.getMessage()
            for record in caplog.records
            if record.levelno >= logging.WARNING
        ] == []

    @pytest.mark.unit
    def test_degrades_on_empty_terms(self) -> None:
        assert build_equation([], torch.tensor([]), _LHS) is None

    @pytest.mark.unit
    def test_degrades_on_invalid_final_eval(self) -> None:
        assert (
            build_equation(["u_x"], torch.tensor([1.0]), _LHS, is_valid=False)
            is None
        )


class TestStillReservedFormConstructors:

    @pytest.mark.unit
    @pytest.mark.parametrize("ctor_name", ["make_parametric", "make_weak"])
    def test_reserved_form_constructor_absent_or_raises(self, ctor_name: str) -> None:
        ctor = getattr(equation_pkg, ctor_name, None)
        if ctor is None:
            pytest.skip(f"{ctor_name} reserved by absence (acceptable, D1-3)")
        with pytest.raises(NotImplementedError):
            ctor(_LHS, _TERMS)


class TestActiveIndices:

    @pytest.mark.unit
    def test_build_equation_threads_active_indices(self) -> None:
        eq = build_equation(
            ["a", "b", "c"], [1.0, 0.0, 2.0], _LHS, active_indices=[0, 2]
        )
        assert eq is not None
        assert eq.active_indices == (0, 2)

    @pytest.mark.unit
    def test_build_equation_active_indices_default_none(self) -> None:
        eq = build_equation(["u_x", "u_xx"], torch.tensor([1.0, -0.5]), _LHS)
        assert eq is not None
        assert eq.active_indices is None

    @pytest.mark.unit
    def test_build_equation_out_of_range_degrades_metadata_not_equation(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:

        caplog.set_level("WARNING", logger="kd.core.equation.construct")

        eq = build_equation(["u_x", "u_xx"], [1.0, -0.5], _LHS, active_indices=[5])

        assert eq is not None
        assert eq.active_indices is None
        assert "active_indices" in caplog.text

    @pytest.mark.unit
    def test_make_evolution_rejects_out_of_range_active_index(self) -> None:
        with pytest.raises(ValueError, match="active_indices"):
            make_evolution(_LHS, _TERMS, active_indices=[2])

    @pytest.mark.unit
    def test_make_homogeneous_rejects_out_of_range_active_index(self) -> None:
        with pytest.raises(ValueError, match="active_indices"):
            make_homogeneous(_TERMS, active_indices=[2])
