
from __future__ import annotations

import pytest

from kd.core.expr import TermFeatures, TermVocabulary, analyze_term

VOCAB = TermVocabulary(
    fields=frozenset({"u", "v"}),
    coordinates=frozenset({"t", "x", "y"}),
)


def _features(term_ir: str) -> TermFeatures:
    return analyze_term(term_ir, VOCAB)


class TestTermVocabularyConstruction:

    def test_disjoint_nonempty_names_construct(self) -> None:
        vocabulary = TermVocabulary(
            fields=frozenset({"u"}), coordinates=frozenset({"x"})
        )
        assert vocabulary.fields == frozenset({"u"})
        assert vocabulary.coordinates == frozenset({"x"})

    def test_empty_fields_rejected(self) -> None:
        with pytest.raises(ValueError, match="fields"):
            TermVocabulary(fields=frozenset(), coordinates=frozenset({"x"}))

    def test_empty_coordinates_rejected(self) -> None:
        with pytest.raises(ValueError, match="coordinates"):
            TermVocabulary(fields=frozenset({"u"}), coordinates=frozenset())

    @pytest.mark.parametrize(
        "fields,coordinates",
        [
            (frozenset({"u_x"}), frozenset({"x"})),
            (frozenset({"u"}), frozenset({"x_y"})),
        ],
    )
    def test_underscore_name_rejected(
        self, fields: frozenset[str], coordinates: frozenset[str]
    ) -> None:

        with pytest.raises(ValueError, match="underscore"):
            TermVocabulary(fields=fields, coordinates=coordinates)

    @pytest.mark.parametrize(
        "fields,coordinates",
        [
            (frozenset({"u", ""}), frozenset({"x"})),
            (frozenset({"u"}), frozenset({"x", ""})),
        ],
    )
    def test_empty_name_rejected(
        self, fields: frozenset[str], coordinates: frozenset[str]
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            TermVocabulary(fields=fields, coordinates=coordinates)

    def test_field_coordinate_overlap_rejected(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            TermVocabulary(
                fields=frozenset({"u", "x"}), coordinates=frozenset({"x"})
            )


class TestDerivativeOrder:

    def test_bare_field_has_no_derivative_path(self) -> None:
        features = _features("u")
        assert features.base_fields == frozenset({"u"})
        assert features.derivative_multiindices == frozenset()
        assert features.max_total_derivative_order == 0
        assert features.coordinate_dependencies == frozenset()

    def test_terminal_derivative_yields_its_own_multiindex(self) -> None:
        features = _features("u_xx")
        assert features.base_fields == frozenset({"u"})
        assert features.derivative_multiindices == frozenset(
            {("u", (("x", 2),))}
        )
        assert features.max_total_derivative_order == 2

    def test_nested_diff_accumulates_on_the_same_axis(self) -> None:

        features = _features("diff_x(u_xx)")
        assert features.derivative_multiindices == frozenset(
            {("u", (("x", 3),))}
        )
        assert features.max_total_derivative_order == 3

    def test_nested_diff_on_a_new_axis_extends_the_multiindex(self) -> None:
        features = _features("diff_y(u_xx)")
        assert features.derivative_multiindices == frozenset(
            {("u", (("x", 2), ("y", 1)))}
        )
        assert features.max_total_derivative_order == 3

    def test_diff_order_digits_are_read_from_the_head(self) -> None:
        features = _features("diff2_x(u_x)")
        assert features.derivative_multiindices == frozenset(
            {("u", (("x", 3),))}
        )
        assert features.max_total_derivative_order == 3

    def test_diff_reaches_every_occurrence_inside_a_product(self) -> None:
        features = _features("diff_x(mul(u,u_x))")
        assert features.derivative_multiindices == frozenset(
            {("u", (("x", 1),)), ("u", (("x", 2),))}
        )
        assert features.max_total_derivative_order == 2

    def test_coordinate_factor_contributes_no_derivative(self) -> None:
        features = _features("mul(x,u)")
        assert features.base_fields == frozenset({"u"})
        assert features.coordinate_dependencies == frozenset({"x"})
        assert features.derivative_multiindices == frozenset()
        assert features.max_total_derivative_order == 0

    def test_coordinate_under_a_diff_path_yields_no_field_derivative(
        self,
    ) -> None:
        features = _features("diff_x(x)")
        assert features.base_fields == frozenset()
        assert features.derivative_multiindices == frozenset()
        assert features.max_total_derivative_order == 0
        assert features.coordinate_dependencies == frozenset({"x"})

    def test_derivative_occurrence_also_reports_its_base_field(self) -> None:
        features = _features("mul(u_x,v_yy)")
        assert features.base_fields == frozenset({"u", "v"})


class TestOperatorSet:

    @pytest.mark.parametrize("term_ir", ["diff_x(u)", "diff2_x(u)"])
    def test_diff_heads_are_not_operators(self, term_ir: str) -> None:
        assert _features(term_ir).operators == frozenset()

    def test_non_diff_head_around_a_diff_is_an_operator(self) -> None:
        assert _features("mul(u,diff_x(u))").operators == frozenset({"mul"})

    def test_alias_folding_makes_div_and_mul_recip_identical(self) -> None:
        divided = _features("div(u,x)")
        assert divided == _features("mul(u,recip(x))")
        assert divided.operators == frozenset({"mul", "recip"})

    def test_head_neg_is_folded_away_before_features(self) -> None:

        assert _features("neg(mul(u,u_x))") == _features("mul(u,u_x)")

    def test_inner_neg_survives_as_an_operator(self) -> None:

        assert "neg" in _features("mul(u,neg(u_x))").operators


class TestUnresolvableInput:

    def test_symbol_outside_the_vocabulary_is_named_in_the_error(self) -> None:
        with pytest.raises(ValueError, match="psi"):
            _features("mul(u,psi)")

    def test_resolvable_twin_of_the_unresolvable_case_succeeds(self) -> None:

        assert _features("mul(u,u_x)").base_fields == frozenset({"u"})

    def test_diff_axis_outside_the_vocabulary_rejected(self) -> None:
        with pytest.raises(ValueError, match="axis"):
            _features("diff_z(u)")

    def test_diff_with_more_than_one_argument_rejected(self) -> None:
        with pytest.raises(ValueError, match="argument"):
            _features("diff_x(u,v)")

    def test_numeric_constant_rejected(self) -> None:

        with pytest.raises(ValueError, match="constant"):
            _features("mul(2.0,u)")


class TestUndecomposableHeads:

    def test_special_derivative_operator_rejected(self) -> None:



        with pytest.raises(ValueError, match="lap"):
            _features("lap(u)")

    def test_zero_order_diff_rejected(self) -> None:

        with pytest.raises(ValueError, match="order"):
            _features("diff0_x(u)")


_EMPTY_FEATURES = TermFeatures(
    base_fields=frozenset(),
    coordinate_dependencies=frozenset(),
    derivative_multiindices=frozenset(),
    max_total_derivative_order=0,
    operators=frozenset(),
)


class TestUnityToken:

    def test_unity_token_has_no_structural_features(self) -> None:
        assert _features("one") == _EMPTY_FEATURES

    def test_head_neg_folds_before_the_token_is_collected(self) -> None:


        assert _features("neg(one)") == _EMPTY_FEATURES

    @pytest.mark.parametrize(
        "term", ["diff_x(one)", "diff3_x(one)", "diff_x(mul(one,u))"]
    )
    def test_unity_under_a_diff_path_rejected(self, term: str) -> None:



        with pytest.raises(ValueError, match="unity"):
            _features(term)

    def test_nested_unity_contributes_nothing_to_a_product(self) -> None:
        features = _features("mul(one,u_x)")
        assert features.base_fields == frozenset({"u"})
        assert features.coordinate_dependencies == frozenset()
        assert features.operators == frozenset({"mul"})
        assert features.derivative_multiindices == frozenset(
            {("u", (("x", 1),))}
        )
        assert features.max_total_derivative_order == 1

    def test_reserved_name_as_a_field_rejected(self) -> None:


        with pytest.raises(ValueError, match="reserved"):
            TermVocabulary(
                fields=frozenset({"u", "one"}), coordinates=frozenset({"x"})
            )

    def test_reserved_name_as_a_coordinate_rejected(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            TermVocabulary(
                fields=frozenset({"u"}),
                coordinates=frozenset({"t", "x", "one"}),
            )
