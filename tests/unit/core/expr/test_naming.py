
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kd.core.expr.naming import (
    build_derivative_name,
    parse_compound_derivative,
    parse_derivative_name,
)






_axis_st = st.sampled_from(list("xyzt"))

_field_st = st.sampled_from(["u", "v", "w", "p", "phi", "psi", "vel", "rho"])

_order_st = st.integers(min_value=1, max_value=5)







class TestBuildDerivativeNameSmoke:

    @pytest.mark.smoke
    def test_callable(self) -> None:
        assert callable(build_derivative_name)

    @pytest.mark.smoke
    def test_returns_string(self) -> None:
        result = build_derivative_name("u", "x", 1)
        assert isinstance(result, str)


class TestBuildDerivativeNameBasic:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("field", "axis", "order", "expected"),
        [
            ("u", "x", 1, "u_x"),
            ("u", "x", 2, "u_xx"),
            ("u", "x", 3, "u_xxx"),
            ("u", "t", 1, "u_t"),
            ("u", "t", 2, "u_tt"),
            ("v", "y", 1, "v_y"),
            ("v", "y", 4, "v_yyyy"),
        ],
    )
    def test_basic_cases(
        self, field: str, axis: str, order: int, expected: str
    ) -> None:
        assert build_derivative_name(field, axis, order) == expected

    @pytest.mark.unit
    def test_default_order_is_one(self) -> None:
        assert build_derivative_name("u", "x") == "u_x"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("field", "axis", "order", "expected"),
        [
            ("phi", "x", 1, "phi_x"),
            ("phi", "x", 2, "phi_xx"),
            ("psi", "t", 3, "psi_ttt"),
            ("vel", "z", 1, "vel_z"),
            ("rho", "y", 2, "rho_yy"),
        ],
    )
    def test_multi_char_field(
        self, field: str, axis: str, order: int, expected: str
    ) -> None:
        assert build_derivative_name(field, axis, order) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("field", "axis", "order", "expected"),
        [

            ("u_x", "y", 1, "u_x_y"),
            ("u_x", "x", 1, "u_x_x"),
            ("u_xx", "y", 2, "u_xx_yy"),
            ("phi_t", "x", 1, "phi_t_x"),
        ],
    )
    def test_compound_building(
        self, field: str, axis: str, order: int, expected: str
    ) -> None:
        assert build_derivative_name(field, axis, order) == expected


class TestBuildDerivativeNameValidation:

    @pytest.mark.unit
    def test_empty_field_raises(self) -> None:
        with pytest.raises(ValueError):
            build_derivative_name("", "x", 1)

    @pytest.mark.unit
    def test_empty_axis_raises(self) -> None:
        with pytest.raises(ValueError):
            build_derivative_name("u", "", 1)

    @pytest.mark.unit
    def test_order_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            build_derivative_name("u", "x", 0)

    @pytest.mark.unit
    def test_negative_order_raises(self) -> None:
        with pytest.raises(ValueError):
            build_derivative_name("u", "x", -1)

    @pytest.mark.unit
    def test_negative_large_order_raises(self) -> None:
        with pytest.raises(ValueError):
            build_derivative_name("u", "x", -10)


class TestBuildDerivativeNameProperties:

    @pytest.mark.unit
    @given(field=_field_st, axis=_axis_st, order=_order_st)
    @settings(max_examples=50)
    def test_result_contains_field_prefix(
        self, field: str, axis: str, order: int
    ) -> None:
        result = build_derivative_name(field, axis, order)
        assert result.startswith(field)

    @pytest.mark.unit
    @given(field=_field_st, axis=_axis_st, order=_order_st)
    @settings(max_examples=50)
    def test_result_contains_underscore_separator(
        self, field: str, axis: str, order: int
    ) -> None:
        result = build_derivative_name(field, axis, order)

        suffix = result[len(field):]
        assert suffix.startswith("_")

    @pytest.mark.unit
    @given(field=_field_st, axis=_axis_st, order=_order_st)
    @settings(max_examples=50)
    def test_result_ends_with_axis_repeated(
        self, field: str, axis: str, order: int
    ) -> None:
        result = build_derivative_name(field, axis, order)
        expected_suffix = axis * order
        assert result.endswith(expected_suffix)

    @pytest.mark.unit
    @given(field=_field_st, axis=_axis_st, order=_order_st)
    @settings(max_examples=50)
    def test_higher_order_produces_longer_name(
        self, field: str, axis: str, order: int
    ) -> None:
        r1 = build_derivative_name(field, axis, order)
        r2 = build_derivative_name(field, axis, order + 1)
        assert len(r2) == len(r1) + 1







class TestParseDerivativeNameSmoke:

    @pytest.mark.smoke
    def test_callable(self) -> None:
        assert callable(parse_derivative_name)

    @pytest.mark.smoke
    def test_returns_tuple_or_none(self) -> None:
        result = parse_derivative_name("u_x")
        assert result is None or (isinstance(result, tuple) and len(result) == 3)


class TestParseDerivativeNameBasic:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("u_x", ("u", "x", 1)),
            ("u_xx", ("u", "x", 2)),
            ("u_xxx", ("u", "x", 3)),
            ("u_t", ("u", "t", 1)),
            ("u_tt", ("u", "t", 2)),
            ("v_y", ("v", "y", 1)),
            ("v_yy", ("v", "y", 2)),
            ("w_z", ("w", "z", 1)),
        ],
    )
    def test_single_char_field(self, name: str, expected: tuple[str, str, int]) -> None:
        assert parse_derivative_name(name) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("phi_x", ("phi", "x", 1)),
            ("phi_xx", ("phi", "x", 2)),
            ("phi_tt", ("phi", "t", 2)),
            ("vel_t", ("vel", "t", 1)),
            ("rho_yy", ("rho", "y", 2)),
            ("psi_zzz", ("psi", "z", 3)),
        ],
    )
    def test_multi_char_field(self, name: str, expected: tuple[str, str, int]) -> None:
        assert parse_derivative_name(name) == expected


class TestParseDerivativeNameRejects:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "name",
        [
            "u_xy",
            "u_xt",
            "u_x_y",
            "u_xx_y",
            "u",
            "x",
            "",
            "123",
            "42",
            "_x",
            "u_",
        ],
    )
    def test_rejects_non_derivatives(self, name: str) -> None:
        assert parse_derivative_name(name) is None


class TestParseDerivativeNameDisambiguation:

    @pytest.mark.unit
    def test_known_axes_single_char(self) -> None:
        result = parse_derivative_name("u_xx", known_axes={"x"})
        assert result == ("u", "x", 2)

    @pytest.mark.unit
    def test_known_axes_multi_char_axis(self) -> None:
        result = parse_derivative_name("u_xx", known_axes={"xx"})
        assert result == ("u", "xx", 1)

    @pytest.mark.unit
    def test_known_fields_helps_boundary(self) -> None:
        result = parse_derivative_name("phi_x", known_fields={"phi"})
        assert result == ("phi", "x", 1)

    @pytest.mark.unit
    def test_known_fields_changes_parse(self) -> None:

        result = parse_derivative_name("phi_x_y")
        assert result is None

    @pytest.mark.unit
    def test_known_axes_no_match(self) -> None:
        result = parse_derivative_name("u_xx", known_axes={"y"})
        assert result is None

    @pytest.mark.unit
    def test_known_fields_changes_field_boundary(self) -> None:

        default = parse_derivative_name("phi_x")
        assert default == ("phi", "x", 1)



        shifted = parse_derivative_name("phi_x", known_fields={"ph"})
        assert shifted is None

    @pytest.mark.unit
    def test_known_fields_confirms_default(self) -> None:
        result = parse_derivative_name("phi_x", known_fields={"phi"})
        assert result == ("phi", "x", 1)


class TestParseDerivativeNameRoundTrip:

    @pytest.mark.unit
    @given(field=_field_st, axis=_axis_st, order=_order_st)
    @settings(max_examples=50)
    def test_roundtrip_build_then_parse(
        self, field: str, axis: str, order: int
    ) -> None:
        name = build_derivative_name(field, axis, order)
        result = parse_derivative_name(name)
        assert result == (field, axis, order)







class TestParseCompoundDerivativeSmoke:

    @pytest.mark.smoke
    def test_callable(self) -> None:
        assert callable(parse_compound_derivative)

    @pytest.mark.smoke
    def test_returns_tuple_or_none(self) -> None:
        result = parse_compound_derivative("u_x")
        assert result is None or isinstance(result, tuple)


class TestParseCompoundDerivativeSameAxis:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("u_x", ("u", [("x", 1)])),
            ("u_xx", ("u", [("x", 2)])),
            ("u_xxx", ("u", [("x", 3)])),
            ("v_t", ("v", [("t", 1)])),
            ("v_tt", ("v", [("t", 2)])),
            ("phi_yy", ("phi", [("y", 2)])),
        ],
    )
    def test_same_axis_derivatives(
        self, name: str, expected: tuple[str, list[tuple[str, int]]]
    ) -> None:
        assert parse_compound_derivative(name) == expected


class TestParseCompoundDerivativeMixed:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("u_x_y", ("u", [("x", 1), ("y", 1)])),
            ("u_xx_y", ("u", [("x", 2), ("y", 1)])),
            ("u_x_yy", ("u", [("x", 1), ("y", 2)])),
            ("u_xx_yy", ("u", [("x", 2), ("y", 2)])),
            ("u_xxx_yy", ("u", [("x", 3), ("y", 2)])),
        ],
    )
    def test_two_axis_mixed(
        self, name: str, expected: tuple[str, list[tuple[str, int]]]
    ) -> None:
        assert parse_compound_derivative(name) == expected

    @pytest.mark.unit
    def test_same_axis_repeated_segments_not_merged(self) -> None:

        result_separate = parse_compound_derivative("u_x_x")
        assert result_separate == ("u", [("x", 1), ("x", 1)])


        result_merged = parse_compound_derivative("u_xx")
        assert result_merged == ("u", [("x", 2)])

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("u_x_y_z", ("u", [("x", 1), ("y", 1), ("z", 1)])),
            ("u_xx_y_z", ("u", [("x", 2), ("y", 1), ("z", 1)])),
            ("u_x_yy_zz", ("u", [("x", 1), ("y", 2), ("z", 2)])),
        ],
    )
    def test_three_axis_mixed(
        self, name: str, expected: tuple[str, list[tuple[str, int]]]
    ) -> None:
        assert parse_compound_derivative(name) == expected

    @pytest.mark.unit
    def test_multi_char_field_mixed(self) -> None:
        result = parse_compound_derivative("phi_x_y")
        assert result == ("phi", [("x", 1), ("y", 1)])

    @pytest.mark.unit
    def test_multi_char_field_higher_order_mixed(self) -> None:
        result = parse_compound_derivative("phi_xx_tt")
        assert result == ("phi", [("x", 2), ("t", 2)])


class TestParseCompoundDerivativeRejects:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "name",
        [
            "u",
            "x",
            "",
            "123",
            "42",
            "_x",
            "u_",
            "u__x",
            "u_x_",
        ],
    )
    def test_rejects_non_derivatives(self, name: str) -> None:
        assert parse_compound_derivative(name) is None


class TestParseCompoundDerivativeSingleSegmentMixed:

    @pytest.mark.unit
    def test_single_segment_mixed_axis_rejected(self) -> None:
        assert parse_compound_derivative("u_xy") is None

    @pytest.mark.unit
    def test_single_segment_mixed_axis_with_known_axes_still_rejected(self) -> None:
        result = parse_compound_derivative("u_xy", known_axes={"x", "y"})
        assert result is None

    @pytest.mark.unit
    def test_single_segment_mixed_axis_partial_known(self) -> None:

        assert parse_compound_derivative("u_xy", known_axes={"x"}) is None


class TestParseCompoundDerivativeDisambiguation:

    @pytest.mark.unit
    def test_known_axes_affects_segment_parsing(self) -> None:
        result = parse_compound_derivative("u_xx", known_axes={"x"})
        assert result == ("u", [("x", 2)])

    @pytest.mark.unit
    def test_known_axes_multi_char(self) -> None:
        result = parse_compound_derivative("u_xx", known_axes={"xx"})
        assert result == ("u", [("xx", 1)])

    @pytest.mark.unit
    def test_known_fields_compound(self) -> None:
        result = parse_compound_derivative("phi_x_y", known_fields={"phi"})
        assert result == ("phi", [("x", 1), ("y", 1)])


class TestParseCompoundDerivativeRoundTrip:

    @pytest.mark.unit
    def test_roundtrip_single_axis(self) -> None:
        name = build_derivative_name("u", "x", 2)
        result = parse_compound_derivative(name)
        assert result is not None
        field, derivs = result
        assert field == "u"
        assert derivs == [("x", 2)]

    @pytest.mark.unit
    def test_roundtrip_two_axis_sequential_build(self) -> None:
        step1 = build_derivative_name("u", "x", 1)
        step2 = build_derivative_name(step1, "y", 1)
        result = parse_compound_derivative(step2)
        assert result is not None
        field, derivs = result
        assert field == "u"
        assert derivs == [("x", 1), ("y", 1)]

    @pytest.mark.unit
    def test_roundtrip_higher_order_compound(self) -> None:
        step1 = build_derivative_name("u", "x", 2)
        step2 = build_derivative_name(step1, "y", 2)
        result = parse_compound_derivative(step2)
        assert result is not None
        field, derivs = result
        assert field == "u"
        assert derivs == [("x", 2), ("y", 2)]







class TestCrossFunctionConsistency:

    @pytest.mark.unit
    @given(field=_field_st, axis=_axis_st, order=_order_st)
    @settings(max_examples=50)
    def test_same_axis_parse_agrees_with_compound(
        self, field: str, axis: str, order: int
    ) -> None:
        name = build_derivative_name(field, axis, order)
        simple = parse_derivative_name(name)
        compound = parse_compound_derivative(name)

        assert simple is not None
        assert compound is not None

        s_field, s_axis, s_order = simple
        c_field, c_derivs = compound

        assert s_field == c_field

        assert len(c_derivs) == 1
        assert c_derivs[0] == (s_axis, s_order)

    @pytest.mark.unit
    def test_compound_rejects_implies_simple_rejects(self) -> None:
        for name in ["u", "x", "", "123"]:
            assert parse_derivative_name(name) is None
            assert parse_compound_derivative(name) is None

    @pytest.mark.unit
    def test_mixed_partial_simple_rejects_compound_accepts(self) -> None:
        name = "u_x_y"
        assert parse_derivative_name(name) is None
        result = parse_compound_derivative(name)
        assert result is not None
        assert result == ("u", [("x", 1), ("y", 1)])







class TestExecutorConventionCompatibility:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("name", "field", "axis", "order"),
        [

            ("u_x", "u", "x", 1),
            ("u_xx", "u", "x", 2),
            ("u_xxx", "u", "x", 3),
            ("v_t", "v", "t", 1),
            ("v_tt", "v", "t", 2),
            ("phi_yy", "phi", "y", 2),
        ],
    )
    def test_matches_executor_terminal_derivatives(
        self, name: str, field: str, axis: str, order: int
    ) -> None:
        result = parse_derivative_name(name)
        assert result is not None
        assert result == (field, axis, order)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "name",
        [

            "u_xy",
            "u_xt",
            "v_yz",
        ],
    )
    def test_rejects_what_executor_rejects(self, name: str) -> None:
        assert parse_derivative_name(name) is None


class TestSympyBridgeCompatibility:

    @pytest.mark.unit
    def test_sympy_bridge_simple_derivative(self) -> None:
        result = parse_compound_derivative("u_xx")
        assert result == ("u", [("x", 2)])

    @pytest.mark.unit
    def test_sympy_bridge_compound_derivative(self) -> None:
        result = parse_compound_derivative("u_x_y")
        assert result == ("u", [("x", 1), ("y", 1)])

    @pytest.mark.unit
    def test_sympy_bridge_higher_compound(self) -> None:
        result = parse_compound_derivative("u_xx_yy")
        assert result == ("u", [("x", 2), ("y", 2)])
