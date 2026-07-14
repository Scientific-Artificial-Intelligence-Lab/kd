
from __future__ import annotations

import dataclasses
import json

import pytest

import kd.core.equation as equation_pkg
import kd.core.equation.types as equation_types
from kd.core.equation import (
    Equation,
    EquationAttrs,
    Evolution,
    Form,
    LhsSpec,
    Scalar,
    build_equation,
    from_dict,
    lower_to_regression,
    make_evolution,
    to_dict,
)
from kd.core.equation.types import Homogeneous

_LHS = LhsSpec(field="u", axis="t", order=1)
_TERMS = (("u_x", Scalar(1.0)), ("u_xx", Scalar(-0.5)))

def _evolution() -> Evolution:
    return make_evolution(_LHS, _TERMS)







class TestEvolutionTypeIdentity:
    @pytest.mark.unit
    def test_make_evolution_returns_evolution_instance(self) -> None:
        assert isinstance(make_evolution(_LHS, _TERMS), Evolution)

    @pytest.mark.unit
    def test_evolution_exported_from_package_root(self) -> None:
        assert equation_pkg.Evolution is Evolution
        assert isinstance(Evolution, type)
        assert dataclasses.is_dataclass(Evolution)

    @pytest.mark.unit
    def test_build_equation_returns_evolution_instance(self) -> None:
        eq = build_equation(["u_x", "u_xx"], [1.0, -0.5], _LHS)
        assert isinstance(eq, Evolution)

    @pytest.mark.unit
    def test_from_dict_returns_evolution_instance(self) -> None:
        rebuilt = from_dict(to_dict(_evolution()))
        assert isinstance(rebuilt, Evolution)

    @pytest.mark.unit
    def test_evolution_instance_is_frozen(self) -> None:
        eq = _evolution()
        with pytest.raises(dataclasses.FrozenInstanceError):
            eq.lhs_spec = None







class TestEvolutionFields:
    @pytest.mark.unit
    def test_evolution_fields_are_lhs_terms_attrs(self) -> None:
        names = {f.name for f in dataclasses.fields(Evolution)}
        assert names == {"lhs_spec", "terms", "attrs"}

    @pytest.mark.unit
    def test_evolution_has_no_form_field(self) -> None:
        names = {f.name for f in dataclasses.fields(Evolution)}
        assert "form" not in names

    @pytest.mark.unit
    def test_to_dict_still_emits_form_wire_tag(self) -> None:
        assert to_dict(_evolution())["form"] == "EVOLUTION"







class TestF11IllegalStatesUnrepresentable:
    @pytest.mark.unit
    def test_homogeneous_has_no_lhs_spec_field(self) -> None:
        names = {f.name for f in dataclasses.fields(Homogeneous)}
        assert "lhs_spec" not in names

    @pytest.mark.unit
    def test_evolution_lhs_spec_field_is_required(self) -> None:
        by_name = {f.name: f for f in dataclasses.fields(Evolution)}
        lhs_field = by_name["lhs_spec"]
        assert lhs_field.default is dataclasses.MISSING
        assert lhs_field.default_factory is dataclasses.MISSING

    @pytest.mark.unit
    def test_evolution_lhs_spec_annotation_is_non_optional(self) -> None:


        annotation = str(Evolution.__annotations__["lhs_spec"])
        assert "None" not in annotation
        assert "Optional" not in annotation

    @pytest.mark.unit
    def test_make_evolution_rejects_none_lhs_spec(self) -> None:
        with pytest.raises(ValueError):
            make_evolution(None, _TERMS)







class TestHomogeneousActivation:
    @pytest.mark.unit
    def test_direct_construction_does_not_enforce_nonempty_terms(self) -> None:
        equation = Homogeneous(terms=(), attrs=EquationAttrs())
        assert equation.form is Form.HOMOGENEOUS
        assert equation.terms == ()

    @pytest.mark.unit
    @pytest.mark.parametrize("name", ["Parametric", "Weak"])
    def test_parametric_weak_reserved_by_absence(self, name: str) -> None:
        assert not hasattr(equation_types, name)
        assert not hasattr(equation_pkg, name)







class TestEquationIsUnionAlias:
    @pytest.mark.unit
    def test_equation_is_not_a_concrete_dataclass(self) -> None:
        assert not dataclasses.is_dataclass(Equation)

    @pytest.mark.unit
    def test_evolution_is_a_member_of_the_equation_union(self) -> None:
        assert isinstance(_evolution(), Equation)







class TestExhaustiveDispatch:
    @pytest.mark.unit
    def test_lower_to_regression_accepts_evolution(self) -> None:
        rf = lower_to_regression(_evolution())
        assert rf.lhs_spec == _LHS
        assert list(rf.term_irs) == ["u_x", "u_xx"]

    @pytest.mark.unit
    @pytest.mark.parametrize("form", ["PARAMETRIC", "WEAK"])
    def test_from_dict_reserved_form_raises(self, form: str) -> None:
        payload = {
            "form": form,
            "lhs_spec": {"field": "u", "axis": "t", "order": 1},
            "terms": [["u_x", {"kind": "Scalar", "value": 1.0}]],
            "attrs": None,
        }
        with pytest.raises(NotImplementedError):
            from_dict(payload)

    @pytest.mark.unit
    def test_form_enum_retains_all_four_tags(self) -> None:
        assert {m.name for m in Form} == {
            "EVOLUTION",
            "HOMOGENEOUS",
            "PARAMETRIC",
            "WEAK",
        }







class TestRoundTripAcrossNewType:
    @pytest.mark.unit
    def test_roundtrip_simple_evolution(self) -> None:
        eq = make_evolution(_LHS, _TERMS)
        rebuilt = from_dict(to_dict(eq))
        assert isinstance(rebuilt, Evolution)
        assert rebuilt == eq

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "lhs",
        [
            LhsSpec(field="u", axis="t", order=2),
            LhsSpec(field="u", axis="x_1", order=1),
        ],
    )
    def test_roundtrip_order_and_underscore_axis(self, lhs: LhsSpec) -> None:
        eq = make_evolution(lhs, (("u*u_x", Scalar(3.14)),))
        rebuilt = from_dict(json.loads(json.dumps(to_dict(eq))))
        assert isinstance(rebuilt, Evolution)
        assert rebuilt == eq
