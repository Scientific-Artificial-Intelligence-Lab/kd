
import pytest

from kd.core.equation import LhsSpec, Scalar, lower_to_regression, make_evolution

_LHS = LhsSpec(field="u", axis="t", order=1)


class TestLowerToRegression:
    @pytest.mark.unit
    def test_term_irs_match_item_by_item(self) -> None:
        terms = (
            ("u_x", Scalar(1.0)),
            ("u*u_x", Scalar(-1.0)),
            ("u_xx", Scalar(0.1)),
        )
        eq = make_evolution(_LHS, terms)
        rf = lower_to_regression(eq)
        assert list(rf.term_irs) == ["u_x", "u*u_x", "u_xx"]

    @pytest.mark.unit
    def test_term_irs_preserve_order(self) -> None:
        terms = (("c", Scalar(1.0)), ("b", Scalar(1.0)), ("a", Scalar(1.0)))
        eq = make_evolution(_LHS, terms)
        rf = lower_to_regression(eq)

        assert list(rf.term_irs) == ["c", "b", "a"]

    @pytest.mark.unit
    def test_lhs_spec_is_identity(self) -> None:
        eq = make_evolution(_LHS, (("u_x", Scalar(1.0)),))
        rf = lower_to_regression(eq)
        assert rf.lhs_spec == _LHS
        assert rf.lhs_spec == eq.lhs_spec

    @pytest.mark.unit
    def test_lowering_is_pure(self) -> None:
        terms = (("u_x", Scalar(1.0)), ("u_xx", Scalar(-2.0)))
        eq = make_evolution(_LHS, terms)
        reference = make_evolution(_LHS, terms)
        lower_to_regression(eq)
        assert eq == reference
