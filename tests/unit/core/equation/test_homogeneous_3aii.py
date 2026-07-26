
from __future__ import annotations

import pytest

from kd.core.equation import (
    Form,
    Scalar,
    from_dict,
    make_evolution,
    make_homogeneous,
    render_homogeneous_label,
    residual_program,
    structure,
    term_diff,
    to_dict,
)
from kd.core.equation.types import LhsSpec


_LAPLACE_TERMS: list[tuple[str, Scalar]] = [
    ("diff2_x(u)", Scalar(1.0)),
    ("diff2_y(u)", Scalar(1.0)),
]


class TestStructure:
    @pytest.mark.unit
    def test_homogeneous_fingerprint_shape(self) -> None:
        fp = structure(make_homogeneous(_LAPLACE_TERMS))
        assert fp.form is Form.HOMOGENEOUS
        assert fp.lhs_spec is None
        assert fp.terms == frozenset({"diff2_x(u)", "diff2_y(u)"})

    @pytest.mark.unit
    def test_structure_is_permutation_invariant(self) -> None:


        forward = make_homogeneous(_LAPLACE_TERMS)
        reversed_ = make_homogeneous(list(reversed(_LAPLACE_TERMS)))
        assert structure(forward) == structure(reversed_)

    @pytest.mark.unit
    def test_one_term_canonicalizes_into_fingerprint(self) -> None:


        fp = structure(make_homogeneous([*_LAPLACE_TERMS, ("one", Scalar(0.5))]))
        assert "one" in fp.terms

    @pytest.mark.unit
    def test_cross_form_term_diff_sets_form_changed(self) -> None:
        homog = make_homogeneous(_LAPLACE_TERMS)
        evolution = make_evolution(LhsSpec("u", "t", 1), [("diff2_x(u)", Scalar(0.1))])
        diff = term_diff(evolution, homog)
        assert diff.form_changed is True


class TestRendering:
    @pytest.mark.unit
    def test_homogeneous_label_is_sum_equals_zero(self) -> None:
        label = render_homogeneous_label(make_homogeneous(_LAPLACE_TERMS))
        assert label == "diff2_x(u) + diff2_y(u) = 0"

    @pytest.mark.unit
    def test_one_term_renders_as_its_token(self) -> None:
        label = render_homogeneous_label(
            make_homogeneous([("diff2_x(u)", Scalar(1.0)), ("one", Scalar(0.5))])
        )
        assert label == "diff2_x(u) + one = 0"

    @pytest.mark.unit
    def test_single_term_label(self) -> None:
        label = render_homogeneous_label(
            make_homogeneous([("diff2_x(u)", Scalar(1.0))])
        )
        assert label == "diff2_x(u) = 0"


class TestResidualProgram:
    @pytest.mark.unit
    def test_single_term_residual_wraps_coefficient(self) -> None:


        program = residual_program(make_homogeneous([("diff2_x(u)", Scalar(1.0))]))
        assert program == "mul(1.0, diff2_x(u))"

    @pytest.mark.unit
    def test_multi_term_residual_folds_add_mul(self) -> None:

        program = residual_program(
            make_homogeneous([("diff2_x(u)", Scalar(1.0)), ("one", Scalar(2.0))])
        )
        assert program == "add(mul(1.0, diff2_x(u)), mul(2.0, one))"

    @pytest.mark.unit
    def test_residual_program_references_every_term(self) -> None:
        program = residual_program(
            make_homogeneous([*_LAPLACE_TERMS, ("one", Scalar(0.5))])
        )
        for term_ir, _c in [*_LAPLACE_TERMS, ("one", Scalar(0.5))]:
            assert term_ir in program

    @pytest.mark.unit
    def test_evolution_residual_arm_is_implemented(self) -> None:




        evolution = make_evolution(LhsSpec("u", "t", 1), [("diff2_x(u)", Scalar(0.1))])
        assert residual_program(evolution) == "sub(mul(0.1, diff2_x(u)), u_t)"

    @pytest.mark.unit
    def test_non_finite_coefficient_rejected_at_render(self) -> None:





        eq = make_homogeneous([("u", Scalar(float("nan")))])
        with pytest.raises(ValueError, match="non-finite"):
            residual_program(eq)


class TestSerializeWithUnity:
    @pytest.mark.unit
    def test_homogeneous_with_one_term_round_trips(self) -> None:



        eq = make_homogeneous([*_LAPLACE_TERMS, ("one", Scalar(0.5))])
        assert from_dict(to_dict(eq)) == eq
