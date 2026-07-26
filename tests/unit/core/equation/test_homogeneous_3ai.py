
from __future__ import annotations

import pytest

from kd.core.equation import (
    Evolution,
    Form,
    Homogeneous,
    LhsSpec,
    RegressionForm,
    Scalar,
    build_homogeneous,
    from_dict,
    lower_to_regression,
    make_evolution,
    make_homogeneous,
    to_dict,
)
from kd.core.equation.lowering import PivotRegressionForm




_LAPLACE_TERMS: list[tuple[str, Scalar]] = [
    ("diff2_x(u)", Scalar(1.0)),
    ("diff2_y(u)", Scalar(1.0)),
]


class TestConstruct:
    @pytest.mark.unit
    def test_homogeneous_is_constructible_with_form_tag(self) -> None:
        eq = make_homogeneous(_LAPLACE_TERMS)
        assert isinstance(eq, Homogeneous)
        assert eq.form is Form.HOMOGENEOUS

    @pytest.mark.unit
    def test_single_term_homogeneous_is_legal(self) -> None:

        eq = make_homogeneous([("diff2_x(u)", Scalar(1.0))])
        assert len(eq.terms) == 1

    @pytest.mark.unit
    def test_empty_terms_rejected(self) -> None:
        with pytest.raises(ValueError, match="term"):
            make_homogeneous([])

    @pytest.mark.unit
    def test_empty_term_ir_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            make_homogeneous([("", Scalar(1.0))])


class TestBuildHomogeneous:
    @pytest.mark.unit
    def test_unnormalized_pivot_degrades_to_none(self) -> None:





        assert (
            build_homogeneous(["diff2_x(u)", "diff2_y(u)", "one"], [7.0, 1.0, 0.5])
            is None
        )

    @pytest.mark.unit
    def test_unnormalized_pivot_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            build_homogeneous(["diff2_x(u)", "one"], [0.7, 0.5])
        assert any("pivot" in message for message in caplog.messages)

    @pytest.mark.unit
    def test_pivot_within_tolerance_canonicalized_to_exact_one(self) -> None:



        eq = build_homogeneous(
            ["diff2_x(u)", "diff2_y(u)", "one"],
            [1.0 + 1e-7, 1.0, 0.5],
        )
        assert isinstance(eq, Homogeneous)
        assert eq.terms[0] == ("diff2_x(u)", Scalar(1.0))
        assert [ir for ir, _c in eq.terms] == ["diff2_x(u)", "diff2_y(u)", "one"]
        assert eq.terms[1][1] == Scalar(1.0)
        assert eq.terms[2][1] == Scalar(0.5)

    @pytest.mark.unit
    def test_degrades_to_none_on_invalid_eval(self) -> None:
        assert build_homogeneous(["diff2_x(u)"], [1.0], is_valid=False) is None

    @pytest.mark.unit
    def test_degrades_to_none_on_length_mismatch(self) -> None:
        assert build_homogeneous(["a", "b"], [1.0]) is None

    @pytest.mark.unit
    def test_degrades_to_none_on_non_finite_coefficient(self) -> None:
        assert build_homogeneous(["a", "b"], [1.0, float("nan")]) is None


class TestLowering:
    @pytest.mark.unit
    def test_homogeneous_lowers_to_pivot_form(self) -> None:
        eq = make_homogeneous([*_LAPLACE_TERMS, ("one", Scalar(0.5))])
        form = lower_to_regression(eq)
        assert isinstance(form, PivotRegressionForm)

        assert form.pivot_ir == "diff2_x(u)"
        assert form.rhs_irs == ("diff2_y(u)", "one")

    @pytest.mark.unit
    def test_evolution_arm_unchanged(self) -> None:
        eq = make_evolution(LhsSpec("u", "t", 1), [("diff2_x(u)", Scalar(0.1))])
        form = lower_to_regression(eq)
        assert isinstance(form, RegressionForm)
        assert not isinstance(form, PivotRegressionForm)


class TestSerialize:
    @pytest.mark.unit
    def test_homogeneous_round_trip(self) -> None:
        eq = make_homogeneous([*_LAPLACE_TERMS, ("one", Scalar(0.5))])
        assert from_dict(to_dict(eq)) == eq

    @pytest.mark.unit
    def test_homogeneous_wire_tag_and_null_lhs(self) -> None:
        payload = to_dict(make_homogeneous(_LAPLACE_TERMS))
        assert payload["form"] == "HOMOGENEOUS"
        assert payload["lhs_spec"] is None

    @pytest.mark.unit
    def test_evolution_round_trip_unchanged(self) -> None:
        eq = make_evolution(LhsSpec("u", "t", 1), [("diff2_x(u)", Scalar(0.1))])
        rebuilt = from_dict(to_dict(eq))
        assert isinstance(rebuilt, Evolution)
        assert rebuilt == eq

    @pytest.mark.unit
    def test_homogeneous_payload_with_lhs_spec_rejected(self) -> None:


        payload = to_dict(make_homogeneous(_LAPLACE_TERMS))
        payload["lhs_spec"] = {"field": "u", "axis": "t", "order": 1}
        with pytest.raises(ValueError, match="lhs_spec"):
            from_dict(payload)
