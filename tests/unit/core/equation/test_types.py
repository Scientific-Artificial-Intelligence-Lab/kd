
from dataclasses import FrozenInstanceError

import pytest

import kd.core.equation as equation_pkg
from kd.core.equation import (
    Equation,
    EquationAttrs,
    Evolution,
    Form,
    LhsSpec,
    Scalar,
    make_evolution,
)


class TestForm:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "member", ["EVOLUTION", "HOMOGENEOUS", "PARAMETRIC", "WEAK"]
    )
    def test_form_has_member(self, member: str) -> None:
        assert hasattr(Form, member)
        assert isinstance(getattr(Form, member), Form)


class TestScalar:

    @pytest.mark.unit
    def test_scalar_carries_value(self) -> None:
        assert Scalar(1.5).value == 1.5

    @pytest.mark.unit
    def test_scalar_is_frozen(self) -> None:
        coef = Scalar(1.0)
        with pytest.raises(FrozenInstanceError):
            coef.value = 2.0

    @pytest.mark.unit
    def test_scalar_value_equality(self) -> None:
        assert Scalar(3.0) == Scalar(3.0)
        assert Scalar(3.0) != Scalar(4.0)


class TestReservedCoefficients:

    _RESERVED = {
        "Field": ("u_x",),
        "Hole": (),
        "Posterior": (0.0, 1.0),
    }

    @pytest.mark.unit
    @pytest.mark.parametrize("name,args", list(_RESERVED.items()))
    def test_reserved_coefficient_not_constructible(
        self, name: str, args: tuple[object, ...]
    ) -> None:
        variant = getattr(equation_pkg, name, None)
        if variant is None:
            pytest.skip(f"{name} reserved by absence (acceptable, D1-2)")
        with pytest.raises((NotImplementedError, TypeError)):
            variant(*args)


class TestEquationAttrs:

    @pytest.mark.unit
    def test_empty_attrs_constructible(self) -> None:
        attrs = EquationAttrs()
        assert attrs is not None

    @pytest.mark.unit
    def test_attrs_value_equality(self) -> None:
        assert EquationAttrs() == EquationAttrs()


class TestEquationValue:

    @staticmethod
    def _valid() -> Equation:
        return make_evolution(
            LhsSpec(field="u", axis="t", order=1),
            (("u_x", Scalar(1.0)),),
        )

    @pytest.mark.unit
    def test_equation_is_frozen(self) -> None:
        eq = self._valid()
        with pytest.raises(FrozenInstanceError):
            eq.form = Form.HOMOGENEOUS

    @pytest.mark.unit
    def test_equation_exposes_adt_fields(self) -> None:
        eq = self._valid()
        assert eq.form is Form.EVOLUTION
        assert eq.lhs_spec == LhsSpec(field="u", axis="t", order=1)
        assert eq.terms == (("u_x", Scalar(1.0)),)

    @pytest.mark.unit
    def test_post_init_does_not_enforce_semantic_invariants(self) -> None:
        eq = Evolution(
            lhs_spec=LhsSpec(field="u", axis="t", order=1),
            terms=(),
            attrs=EquationAttrs(),
        )
        assert eq.terms == ()



        eq_empty_ir = Evolution(
            lhs_spec=LhsSpec(field="u", axis="t", order=1),
            terms=(("", Scalar(1.0)),),
            attrs=EquationAttrs(),
        )
        assert eq_empty_ir.terms == (("", Scalar(1.0)),)
