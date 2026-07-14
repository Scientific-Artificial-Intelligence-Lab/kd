
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kd.core.equation import LhsSpec, Scalar, from_dict, make_evolution, to_dict



_ident = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
    max_size=8,
)


_finite = st.floats(allow_nan=False, allow_infinity=False)


@st.composite
def _evolution_equations(draw: st.DrawFn):
    lhs = LhsSpec(
        field=draw(_ident),
        axis=draw(_ident),
        order=draw(st.integers(min_value=1, max_value=5)),
    )
    n = draw(st.integers(min_value=1, max_value=5))
    terms = tuple((draw(_ident), Scalar(draw(_finite))) for _ in range(n))
    return make_evolution(lhs, terms)


class TestSerializeRoundTrip:
    @pytest.mark.unit
    def test_roundtrip_simple(self) -> None:
        eq = make_evolution(
            LhsSpec(field="u", axis="t", order=1),
            (("u_x", Scalar(1.0)), ("u_xx", Scalar(-0.5))),
        )
        assert from_dict(to_dict(eq)) == eq

    @pytest.mark.unit
    def test_roundtrip_through_json(self) -> None:
        eq = make_evolution(
            LhsSpec(field="u", axis="t", order=2),
            (("u*u_x", Scalar(3.14)),),
        )
        payload = json.loads(json.dumps(to_dict(eq)))
        assert from_dict(payload) == eq

    @pytest.mark.unit
    @given(eq=_evolution_equations())
    @settings(max_examples=200)
    def test_roundtrip_property(self, eq) -> None:
        payload = to_dict(eq)
        assert from_dict(payload) == eq

        assert from_dict(json.loads(json.dumps(payload))) == eq

    @pytest.mark.unit
    def test_attrs_serializes_as_null_in_step1(self) -> None:
        eq = make_evolution(
            LhsSpec(field="u", axis="t", order=1),
            (("u_x", Scalar(1.0)),),
        )

        assert to_dict(eq)["attrs"] is None


class TestDeserializeValidation:
    @staticmethod
    def _payload(
        *,
        form: str = "EVOLUTION",
        lhs_field: str = "u",
        lhs_axis: str = "t",
        terms: list[object] | None = None,
        attrs: object = None,
    ) -> dict[str, object]:
        return {
            "form": form,
            "lhs_spec": {"field": lhs_field, "axis": lhs_axis, "order": 1},
            "terms": (
                terms
                if terms is not None
                else [["u_x", {"kind": "Scalar", "value": 1.0}]]
            ),
            "attrs": attrs,
        }

    @pytest.mark.unit
    @pytest.mark.parametrize("form", ["WEAK", "PARAMETRIC"])
    def test_reserved_forms_raise(self, form: str) -> None:
        with pytest.raises(NotImplementedError, match=f"{form} equations are reserved"):
            from_dict(self._payload(form=form))

    @pytest.mark.unit
    def test_evolution_empty_terms_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one term"):
            from_dict(self._payload(terms=[]))

    @pytest.mark.unit
    def test_evolution_null_lhs_spec_raises(self) -> None:
        payload = self._payload()
        payload["lhs_spec"] = None

        with pytest.raises(ValueError, match="lhs_spec"):
            from_dict(payload)

    @pytest.mark.unit
    def test_empty_lhs_field_raises(self) -> None:
        with pytest.raises(ValueError, match="lhs_spec.field"):
            from_dict(self._payload(lhs_field=""))

    @pytest.mark.unit
    def test_empty_lhs_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="lhs_spec.axis"):
            from_dict(self._payload(lhs_axis=""))

    @pytest.mark.unit
    def test_non_null_attrs_raise(self) -> None:
        with pytest.raises(ValueError, match="attrs must be null"):
            from_dict(self._payload(attrs={"provenance": "legacy"}))
