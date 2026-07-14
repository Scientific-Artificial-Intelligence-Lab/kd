
from __future__ import annotations

import json

import pytest

from kd.core.equation import (
    EquationAttrs,
    Homogeneous,
    Scalar,
    build_homogeneous,
    from_dict,
    make_homogeneous,
    to_dict,
)


class TestStructuralVsSemanticInvariants:
    @pytest.mark.unit
    def test_direct_construction_allows_empty_term_ir(self) -> None:
        equation = Homogeneous(
            terms=(("", Scalar(1.0)),),
            attrs=EquationAttrs(),
        )

        assert equation.terms == (("", Scalar(1.0)),)


class TestBuildHomogeneousDegradation:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("term_irs", "coefficients"),
        [
            (None, [1.0]),
            ([], []),
            (["u_x"], None),
            (["u_x"], [float("inf")]),
            (["u_x"], [float("-inf")]),
        ],
    )
    def test_remaining_build_guards_degrade_to_none(
        self,
        term_irs: list[str] | None,
        coefficients: list[float] | None,
    ) -> None:
        assert build_homogeneous(term_irs, coefficients) is None


class TestHomogeneousJsonWire:
    @pytest.mark.unit
    def test_round_trip_through_json_preserves_exact_wire_shape(self) -> None:
        equation = make_homogeneous(
            (("diff2_x(u)", Scalar(1.0)), ("diff2_y(u)", Scalar(-0.5)))
        )

        payload = to_dict(equation)

        assert payload == {
            "form": "HOMOGENEOUS",
            "lhs_spec": None,
            "terms": [
                ["diff2_x(u)", {"kind": "Scalar", "value": 1.0}],
                ["diff2_y(u)", {"kind": "Scalar", "value": -0.5}],
            ],
            "attrs": None,
        }
        assert from_dict(json.loads(json.dumps(payload))) == equation
