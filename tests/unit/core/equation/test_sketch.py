
from __future__ import annotations

from typing import Any

import pytest

from kd.core.equation import (
    SKETCH_SCHEMA_TAG,
    AnchoredTerm,
    Evolution,
    LhsSpec,
    PinnedTerm,
    Scalar,
    Sketch,
    SketchMatchPolicy,
    TermConstraint,
    TermHole,
    constraint_admits,
    law_term_entry,
    law_term_key,
    make_evolution,
    make_homogeneous,
    sketch_from_dict,
    sketch_to_dict,
)
from kd.core.expr import TermVocabulary, analyze_term

VOCAB = TermVocabulary(
    fields=frozenset({"u", "v"}), coordinates=frozenset({"t", "x"})
)
LHS = LhsSpec(field="u", axis="t", order=1)
EXACT = SketchMatchPolicy(
    coeff_atol=0.0, coeff_rtol=0.0, support_threshold=0.0
)
U_ONLY = TermConstraint(fields=frozenset({"u"}))
V_ONLY = TermConstraint(fields=frozenset({"v"}))


def _sketch(
    *,
    pinned: tuple[PinnedTerm, ...] = (),
    anchored: tuple[AnchoredTerm, ...] = (),
    holes: tuple[TermHole, ...] = (),
    match_policy: SketchMatchPolicy = EXACT,
    lhs_spec: LhsSpec = LHS,
    vocabulary: TermVocabulary = VOCAB,
) -> Sketch:
    return Sketch(
        lhs_spec=lhs_spec,
        vocabulary=vocabulary,
        pinned=pinned,
        anchored=anchored,
        holes=holes,
        match_policy=match_policy,
    )


def _hole(
    hole_id: str,
    constraint: TermConstraint,
    *,
    min_count: int = 0,
    max_count: int = 4,
) -> TermHole:
    return TermHole(
        id=hole_id,
        min_count=min_count,
        max_count=max_count,
        constraint=constraint,
    )


def _equation(
    *terms: tuple[str, float],
    lhs_spec: LhsSpec = LHS,
    active_indices: tuple[int, ...] | None = None,
) -> Evolution:
    return make_evolution(
        lhs_spec,
        tuple((term_ir, Scalar(value)) for term_ir, value in terms),
        active_indices=active_indices,
    )


class TestLawTermIdentity:

    def test_key_is_the_entry_at_unit_coefficient(self) -> None:
        assert law_term_key("mul(u,u_x)") == law_term_entry("mul(u,u_x)", 1.0)[0]

    def test_alias_spellings_share_one_key(self) -> None:
        assert law_term_key("div(u,x)") == law_term_key("mul(u,recip(x))")

    def test_commutative_operand_order_shares_one_key(self) -> None:
        assert law_term_key("mul(u,u_x)") == law_term_key("mul(u_x,u)")

    def test_distinct_terms_keep_distinct_keys(self) -> None:
        assert law_term_key("u_x") != law_term_key("u_xx")

    def test_head_neg_folds_into_the_coefficient_sign(self) -> None:
        key, value = law_term_entry("neg(mul(u,u_x))", 1.0)
        assert key == law_term_key("mul(u,u_x)")
        assert value == -1.0

    def test_open_form_and_terminal_spellings_stay_distinct(self) -> None:


        assert law_term_key("diff_x(u)") != law_term_key("u_x")


class TestClauseConstruction:

    def test_pinned_term_constructs(self) -> None:
        assert PinnedTerm("u_xx", 0.1).term_ir == "u_xx"

    def test_zero_pin_rejected(self) -> None:

        with pytest.raises(ValueError, match="zero"):
            PinnedTerm("u_xx", 0.0)

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_non_finite_pin_rejected(self, value: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            PinnedTerm("u_xx", value)

    def test_pin_value_is_coerced_to_float(self) -> None:
        assert isinstance(PinnedTerm("u_xx", 2).value, float)

    @pytest.mark.parametrize("clause", [PinnedTerm, AnchoredTerm])
    def test_unparseable_ir_rejected(self, clause: type) -> None:
        args = ("mul(u,", 1.0) if clause is PinnedTerm else ("mul(u,",)
        with pytest.raises(ValueError, match="syntax"):
            clause(*args)

    def test_anchored_term_constructs(self) -> None:
        assert AnchoredTerm("u_xx").term_ir == "u_xx"


class TestTermConstraintConstruction:

    def test_unconstrained_constraint_leaves_every_dimension_open(self) -> None:
        constraint = TermConstraint()
        assert constraint.max_deriv_order is None
        assert constraint.operators is None
        assert constraint.fields is None
        assert constraint.axes is None

    def test_negative_max_deriv_order_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_deriv_order"):
            TermConstraint(max_deriv_order=-1)

    def test_empty_fields_set_rejected(self) -> None:

        with pytest.raises(ValueError, match="fields"):
            TermConstraint(fields=frozenset())

    def test_empty_operator_and_axis_sets_are_legal(self) -> None:
        constraint = TermConstraint(operators=frozenset(), axes=frozenset())
        assert constraint.operators == frozenset()
        assert constraint.axes == frozenset()

    def test_empty_name_inside_a_set_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            TermConstraint(operators=frozenset({""}))


class TestTermHoleConstruction:

    def test_hole_constructs(self) -> None:
        assert _hole("rest", U_ONLY, min_count=1, max_count=2).id == "rest"

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="id"):
            _hole("", U_ONLY)

    @pytest.mark.parametrize("min_count", [-1, 3])
    def test_out_of_band_min_count_rejected(self, min_count: int) -> None:
        with pytest.raises(ValueError, match="min_count"):
            _hole("rest", U_ONLY, min_count=min_count, max_count=2)

    def test_zero_max_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_count"):
            _hole("rest", U_ONLY, min_count=0, max_count=0)


class TestMatchPolicyConstruction:

    def test_default_labels_are_the_v1_schemes(self) -> None:
        assert EXACT.term_identity == "kd-lawsig-v1"
        assert EXACT.hole_assignment == "disjoint"
        assert EXACT.derivative_order == "total-effective"

    @pytest.mark.parametrize(
        "label",
        ["term_identity", "hole_assignment", "derivative_order"],
    )
    def test_foreign_scheme_label_rejected(self, label: str) -> None:
        with pytest.raises(ValueError, match=label):
            SketchMatchPolicy(
                coeff_atol=0.0,
                coeff_rtol=0.0,
                support_threshold=0.0,
                **{label: "kd-something-else"},
            )

    @pytest.mark.parametrize(
        "name", ["coeff_atol", "coeff_rtol", "support_threshold"]
    )
    @pytest.mark.parametrize("value", [-1.0, float("nan")])
    def test_negative_or_non_finite_tolerance_rejected(
        self, name: str, value: float
    ) -> None:
        defaults = {
            "coeff_atol": 0.0,
            "coeff_rtol": 0.0,
            "support_threshold": 0.0,
        }
        with pytest.raises(ValueError, match=name):
            SketchMatchPolicy(**{**defaults, name: value})

    def test_tolerances_are_coerced_to_float(self) -> None:
        policy = SketchMatchPolicy(
            coeff_atol=0, coeff_rtol=0, support_threshold=1
        )
        assert isinstance(policy.support_threshold, float)


class TestSketchConstruction:

    def test_minimal_sketch_constructs(self) -> None:
        sketch = _sketch(pinned=(PinnedTerm("u_xx", 0.1),))
        assert sketch.vocabulary == VOCAB

    def test_lhs_field_outside_the_vocabulary_rejected(self) -> None:
        with pytest.raises(ValueError, match="field"):
            _sketch(
                pinned=(PinnedTerm("u_xx", 0.1),),
                lhs_spec=LhsSpec(field="w", axis="t", order=1),
            )

    def test_lhs_axis_outside_the_vocabulary_rejected(self) -> None:
        with pytest.raises(ValueError, match="axis"):
            _sketch(
                pinned=(PinnedTerm("u_xx", 0.1),),
                lhs_spec=LhsSpec(field="u", axis="z", order=1),
            )

    def test_multi_letter_lhs_axis_rejected(self) -> None:


        vocabulary = TermVocabulary(
            fields=frozenset({"u"}), coordinates=frozenset({"x", "tt"})
        )
        with pytest.raises(ValueError, match="axis"):
            _sketch(
                pinned=(PinnedTerm("u_x", 0.1),),
                lhs_spec=LhsSpec(field="u", axis="tt", order=1),
                vocabulary=vocabulary,
            )

    def test_zero_lhs_order_rejected(self) -> None:
        with pytest.raises(ValueError, match="order"):
            _sketch(
                pinned=(PinnedTerm("u_xx", 0.1),),
                lhs_spec=LhsSpec(field="u", axis="t", order=0),
            )

    def test_alias_pair_occupying_two_clauses_rejected(self) -> None:

        with pytest.raises(ValueError, match="law-identity"):
            _sketch(
                pinned=(PinnedTerm("div(u,x)", 1.0),),
                anchored=(AnchoredTerm("mul(u,recip(x))"),),
            )

    def test_alias_pair_inside_one_clause_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="law-identity"):
            _sketch(
                pinned=(
                    PinnedTerm("div(u,x)", 1.0),
                    PinnedTerm("mul(u,recip(x))", 2.0),
                ),
            )

    def test_distinct_keys_across_clauses_construct(self) -> None:
        sketch = _sketch(
            pinned=(PinnedTerm("div(u,x)", 1.0),),
            anchored=(AnchoredTerm("u_xx"),),
        )
        assert len(sketch.pinned) == 1

    def test_clause_term_outside_the_vocabulary_rejected(self) -> None:
        with pytest.raises(ValueError, match="psi"):
            _sketch(anchored=(AnchoredTerm("mul(u,psi)"),))

    def test_duplicate_hole_ids_rejected(self) -> None:
        with pytest.raises(ValueError, match="id"):
            _sketch(holes=(_hole("rest", U_ONLY), _hole("rest", V_ONLY)))

    def test_constraint_fields_outside_the_vocabulary_rejected(self) -> None:
        with pytest.raises(ValueError, match="psi"):
            _sketch(holes=(_hole("rest", TermConstraint(fields=frozenset({"psi"}))),))

    def test_constraint_axes_outside_the_vocabulary_rejected(self) -> None:
        with pytest.raises(ValueError, match="zeta"):
            _sketch(holes=(_hole("rest", TermConstraint(axes=frozenset({"zeta"}))),))

    def test_unregistered_operator_name_is_accepted(self) -> None:

        sketch = _sketch(
            holes=(_hole("rest", TermConstraint(operators=frozenset({"wibble"}))),)
        )
        assert len(sketch.holes) == 1

    @pytest.mark.parametrize(
        "first,second",
        [
            (TermConstraint(), TermConstraint()),
            (TermConstraint(), U_ONLY),
            (U_ONLY, TermConstraint(fields=frozenset({"u", "v"}))),
        ],
    )
    def test_holes_without_provable_disjointness_rejected(
        self, first: TermConstraint, second: TermConstraint
    ) -> None:

        with pytest.raises(ValueError, match="disjoint"):
            _sketch(holes=(_hole("a", first), _hole("b", second)))

    def test_disjoint_field_holes_construct(self) -> None:
        sketch = _sketch(holes=(_hole("a", U_ONLY), _hole("b", V_ONLY)))
        assert len(sketch.holes) == 2

    def test_single_unconstrained_hole_skips_the_disjointness_rule(self) -> None:
        sketch = _sketch(holes=(_hole("only", TermConstraint()),))
        assert sketch.holes[0].constraint.fields is None

    def test_fully_empty_sketch_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _sketch()

    def test_support_threshold_above_a_pin_band_rejected(self) -> None:

        with pytest.raises(ValueError, match="support_threshold"):
            _sketch(
                pinned=(PinnedTerm("u_xx", 0.5),),
                match_policy=SketchMatchPolicy(
                    coeff_atol=0.0, coeff_rtol=0.0, support_threshold=0.75
                ),
            )

    def test_support_threshold_equal_to_the_pin_band_constructs(self) -> None:
        sketch = _sketch(
            pinned=(PinnedTerm("u_xx", 0.5),),
            match_policy=SketchMatchPolicy(
                coeff_atol=0.0, coeff_rtol=0.0, support_threshold=0.5
            ),
        )
        assert sketch.match_policy.support_threshold == 0.5


class TestConstraintAdmits:

    def test_open_constraint_admits_a_high_order_term(self) -> None:
        features = analyze_term("diff_x(u_xx)", VOCAB)
        assert constraint_admits(TermConstraint(), features) is True

    @pytest.mark.parametrize(
        "max_deriv_order,expected", [(2, False), (3, True)]
    )
    def test_max_deriv_order_uses_the_total_effective_order(
        self, max_deriv_order: int, expected: bool
    ) -> None:
        features = analyze_term("diff_x(u_xx)", VOCAB)
        constraint = TermConstraint(max_deriv_order=max_deriv_order)
        assert constraint_admits(constraint, features) is expected

    @pytest.mark.parametrize(
        "term_ir,operators,expected",
        [
            ("mul(u,u_x)", frozenset({"mul"}), True),
            ("mul(u,u_x)", frozenset({"add"}), False),
            ("mul(u,u_x)", frozenset(), False),
            ("u_x", frozenset(), True),
        ],
    )
    def test_operators_are_a_subset_test(
        self, term_ir: str, operators: frozenset[str], expected: bool
    ) -> None:
        features = analyze_term(term_ir, VOCAB)
        constraint = TermConstraint(operators=operators)
        assert constraint_admits(constraint, features) is expected

    @pytest.mark.parametrize(
        "term_ir,fields,expected",
        [
            ("x", frozenset({"u"}), False),
            ("x", None, True),
            ("mul(u,v)", frozenset({"u"}), False),
            ("mul(u,v)", frozenset({"u", "v"}), True),
        ],
    )
    def test_fields_are_existential_and_subset(
        self, term_ir: str, fields: frozenset[str] | None, expected: bool
    ) -> None:


        features = analyze_term(term_ir, VOCAB)
        constraint = TermConstraint(fields=fields)
        assert constraint_admits(constraint, features) is expected

    @pytest.mark.parametrize(
        "term_ir,axes,expected",
        [
            ("mul(x,u)", frozenset({"t"}), False),
            ("mul(x,u)", frozenset({"x"}), True),
            ("u_xx", frozenset(), False),
            ("u", frozenset(), True),
        ],
    )
    def test_axes_cover_derivative_and_explicit_coordinate_usage(
        self, term_ir: str, axes: frozenset[str], expected: bool
    ) -> None:
        features = analyze_term(term_ir, VOCAB)
        constraint = TermConstraint(axes=axes)
        assert constraint_admits(constraint, features) is expected


class TestMatchingPipeline:

    def test_homogeneous_candidate_rejected(self) -> None:
        sketch = _sketch(anchored=(AnchoredTerm("u_xx"),))
        homogeneous = make_homogeneous((("u_xx", Scalar(1.0)),))
        with pytest.raises(TypeError, match="EVOLUTION"):
            sketch.matches(homogeneous)

    def test_lhs_mismatch_fails_overall_without_hiding_clause_detail(
        self,
    ) -> None:
        sketch = _sketch(pinned=(PinnedTerm("u_xx", 1.0),))
        verdict = sketch.matches(
            _equation(("u_xx", 1.0), lhs_spec=LhsSpec("u", "t", 2))
        )
        assert verdict.lhs_matched is False
        assert verdict.pinned[0].matched is True
        assert verdict.overall is False

    def test_pin_hits_across_spellings(self) -> None:
        sketch = _sketch(pinned=(PinnedTerm("div(u,x)", 2.0),))
        verdict = sketch.matches(_equation(("mul(u,recip(x))", 2.0)))
        assert verdict.pinned[0].law_key == law_term_key("div(u,x)")
        assert verdict.pinned[0].observed == 2.0
        assert verdict.pinned[0].expected == 2.0
        assert verdict.overall is True

    def test_head_neg_in_the_candidate_folds_into_the_observed_sign(
        self,
    ) -> None:
        sketch = _sketch(pinned=(PinnedTerm("mul(u,diff_x(u))", -1.0),))
        verdict = sketch.matches(_equation(("neg(mul(u,diff_x(u)))", 1.0)))
        assert verdict.pinned[0].observed == -1.0
        assert verdict.pinned[0].expected == -1.0
        assert verdict.overall is True

    def test_same_key_entries_sum_for_the_pin_comparison(self) -> None:
        sketch = _sketch(pinned=(PinnedTerm("u_xx", 0.1),))
        verdict = sketch.matches(_equation(("u_xx", 0.05), ("u_xx", 0.05)))
        assert verdict.pinned[0].observed == 0.1
        assert verdict.overall is True

    def test_same_key_entries_are_one_hole_filling(self) -> None:

        sketch = _sketch(
            holes=(_hole("rest", U_ONLY, min_count=1, max_count=1),)
        )
        verdict = sketch.matches(_equation(("u_xx", 0.05), ("u_xx", 0.05)))
        assert verdict.holes[0].assigned == (law_term_key("u_xx"),)
        assert verdict.overall is True

    def test_below_support_threshold_term_is_absent_everywhere(self) -> None:
        sketch = _sketch(
            anchored=(AnchoredTerm("u_xx"),),
            match_policy=SketchMatchPolicy(
                coeff_atol=0.0, coeff_rtol=0.0, support_threshold=0.5
            ),
        )
        verdict = sketch.matches(_equation(("u_xx", 0.1)))
        assert verdict.anchored[0].observed is None
        assert verdict.anchored[0].matched is False
        assert verdict.unassigned == ()
        assert verdict.overall is False

    def test_above_support_threshold_term_is_present(self) -> None:
        sketch = _sketch(
            anchored=(AnchoredTerm("u_xx"),),
            match_policy=SketchMatchPolicy(
                coeff_atol=0.0, coeff_rtol=0.0, support_threshold=0.5
            ),
        )
        verdict = sketch.matches(_equation(("u_xx", 0.6)))
        assert verdict.anchored[0].observed == 0.6
        assert verdict.overall is True

    @pytest.mark.parametrize("observed,expected", [(1.5, True), (1.75, False)])
    def test_relative_tolerance_boundary(
        self, observed: float, expected: bool
    ) -> None:
        sketch = _sketch(
            pinned=(PinnedTerm("u_xx", 1.0),),
            match_policy=SketchMatchPolicy(
                coeff_atol=0.0, coeff_rtol=0.5, support_threshold=0.0
            ),
        )
        verdict = sketch.matches(_equation(("u_xx", observed)))
        assert verdict.pinned[0].matched is expected

    @pytest.mark.parametrize("observed,expected", [(1.25, True), (1.5, False)])
    def test_absolute_tolerance_boundary(
        self, observed: float, expected: bool
    ) -> None:
        sketch = _sketch(
            pinned=(PinnedTerm("u_xx", 1.0),),
            match_policy=SketchMatchPolicy(
                coeff_atol=0.25, coeff_rtol=0.0, support_threshold=0.0
            ),
        )
        verdict = sketch.matches(_equation(("u_xx", observed)))
        assert verdict.pinned[0].matched is expected

    def test_absent_pin_reports_no_observation(self) -> None:
        sketch = _sketch(
            pinned=(PinnedTerm("u_xx", 1.0),), holes=(_hole("rest", U_ONLY),)
        )
        verdict = sketch.matches(_equation(("u_x", 1.0)))
        assert verdict.pinned[0].observed is None
        assert verdict.pinned[0].matched is False

    def test_absent_anchor_fails_its_clause(self) -> None:
        sketch = _sketch(
            anchored=(AnchoredTerm("u_xx"),), holes=(_hole("rest", U_ONLY),)
        )
        verdict = sketch.matches(_equation(("u_x", 1.0)))
        assert verdict.anchored[0].observed is None
        assert verdict.anchored[0].matched is False
        assert verdict.overall is False

    def test_hole_below_its_min_count_fails(self) -> None:
        sketch = _sketch(
            holes=(
                _hole("h_u", U_ONLY, min_count=0, max_count=2),
                _hole("h_v", V_ONLY, min_count=1, max_count=1),
            )
        )
        verdict = sketch.matches(_equation(("u_x", 1.0)))
        assert verdict.holes[0].matched is True
        assert verdict.holes[1].assigned == ()
        assert verdict.holes[1].matched is False
        assert verdict.overall is False

    def test_hole_above_its_max_count_fails(self) -> None:
        sketch = _sketch(
            holes=(_hole("h_u", U_ONLY, min_count=0, max_count=1),)
        )
        verdict = sketch.matches(_equation(("u_x", 1.0), ("u_xx", 1.0)))
        assert verdict.holes[0].assigned == tuple(
            sorted((law_term_key("u_x"), law_term_key("u_xx")))
        )
        assert verdict.holes[0].matched is False

    def test_pinned_and_anchored_keys_never_reach_the_holes(self) -> None:
        sketch = _sketch(
            pinned=(PinnedTerm("u_xx", 1.0),),
            anchored=(AnchoredTerm("u_x"),),
            holes=(_hole("h_u", U_ONLY, min_count=0, max_count=1),),
        )
        verdict = sketch.matches(
            _equation(("u_xx", 1.0), ("u_x", 1.0), ("mul(u,u_x)", 1.0))
        )
        assert verdict.holes[0].assigned == (law_term_key("mul(u,u_x)"),)
        assert verdict.overall is True

    def test_out_of_space_terms_are_reported_not_raised(self) -> None:
        sketch = _sketch(anchored=(AnchoredTerm("u_xx"),))
        verdict = sketch.matches(
            _equation(("u_xx", 1.0), ("mul(u,psi)", 1.0), ("v_x", 1.0))
        )
        keys = tuple(entry.law_key for entry in verdict.unassigned)
        assert set(keys) == {law_term_key("mul(u,psi)"), law_term_key("v_x")}
        assert keys == tuple(sorted(keys))
        reasons = {entry.law_key: entry.reason for entry in verdict.unassigned}


        assert reasons[law_term_key("mul(u,psi)")] != reasons[law_term_key("v_x")]
        assert "psi" in reasons[law_term_key("mul(u,psi)")]
        assert verdict.overall is False

    def test_inactive_catalog_terms_are_invisible(self) -> None:

        sketch = _sketch(pinned=(PinnedTerm("u_xx", 0.1),))
        verdict = sketch.matches(
            _equation(("u_xx", 0.1), ("u_xxx", 5.0), active_indices=(0,))
        )
        assert verdict.unassigned == ()
        assert verdict.overall is True

    def test_dense_catalog_exposes_every_term(self) -> None:
        sketch = _sketch(pinned=(PinnedTerm("u_xx", 0.1),))
        verdict = sketch.matches(_equation(("u_xx", 0.1), ("u_xxx", 5.0)))
        assert tuple(entry.law_key for entry in verdict.unassigned) == (
            law_term_key("u_xxx"),
        )
        assert verdict.overall is False

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_candidate_coefficient_fails_loud(self, bad: float) -> None:


        sketch = _sketch(pinned=(PinnedTerm("u_xx", 0.1),))
        with pytest.raises(ValueError, match="finite"):
            sketch.matches(_equation(("u_xx", 0.1), ("mul(u,u_xx)", bad)))

    def test_exact_zero_aggregate_is_absent(self) -> None:


        sketch = _sketch(
            anchored=(AnchoredTerm("u_xx"),), holes=(_hole("rest", U_ONLY),)
        )
        verdict = sketch.matches(
            _equation(("u_x", 1.0), ("u_xx", 3.0), ("u_xx", -3.0))
        )
        assert verdict.anchored[0].observed is None
        assert verdict.anchored[0].matched is False
        assert verdict.unassigned == ()

    def test_undecomposable_candidate_term_reports_unassigned(self) -> None:


        sketch = _sketch(anchored=(AnchoredTerm("u_xx"),))
        verdict = sketch.matches(_equation(("u_xx", 1.0), ("lap(u)", 2.0)))
        assert verdict.overall is False
        assert "lap" in verdict.unassigned[0].reason

    def test_undecomposable_clause_term_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="lap"):
            _sketch(anchored=(AnchoredTerm("lap(u)"),))

    def test_verdict_keeps_sketch_clause_order_and_policy(self) -> None:
        sketch = _sketch(
            pinned=(PinnedTerm("u_xx", 0.1), PinnedTerm("u_x", 0.2)),
            anchored=(AnchoredTerm("mul(u,u_x)"), AnchoredTerm("v_xx")),
            holes=(_hole("zed", U_ONLY), _hole("abe", V_ONLY)),
        )
        verdict = sketch.matches(
            _equation(
                ("u_xx", 0.1),
                ("u_x", 0.2),
                ("mul(u,u_x)", 3.0),
                ("v_xx", 4.0),
            )
        )
        assert tuple(entry.term_ir for entry in verdict.pinned) == (
            "u_xx",
            "u_x",
        )
        assert tuple(entry.term_ir for entry in verdict.anchored) == (
            "mul(u,u_x)",
            "v_xx",
        )
        assert tuple(entry.hole_id for entry in verdict.holes) == ("zed", "abe")
        assert verdict.policy == EXACT
        assert verdict.overall is True


def _representative_sketch() -> Sketch:
    return _sketch(
        pinned=(PinnedTerm("mul(u,diff_x(u))", -1.0),),
        anchored=(AnchoredTerm("u_xx"),),
        holes=(
            _hole(
                "rest",
                TermConstraint(
                    max_deriv_order=2,
                    operators=frozenset({"mul"}),
                    fields=frozenset({"u"}),
                    axes=frozenset({"x"}),
                ),
                min_count=0,
                max_count=2,
            ),
            _hole("other", V_ONLY, min_count=1, max_count=1),
        ),
        match_policy=SketchMatchPolicy(
            coeff_atol=1e-8, coeff_rtol=0.05, support_threshold=0.0
        ),
    )


def _payload() -> dict[str, Any]:
    return sketch_to_dict(_representative_sketch())


class TestSerialization:

    def test_round_trip_returns_an_equal_sketch(self) -> None:
        sketch = _representative_sketch()
        assert sketch_from_dict(sketch_to_dict(sketch)) == sketch

    def test_wire_shape_carries_the_schema_tag(self) -> None:
        assert SKETCH_SCHEMA_TAG == "kd-sketch-v1"
        payload = sketch_to_dict(_representative_sketch())
        assert payload["schema"] == SKETCH_SCHEMA_TAG

    def test_frozensets_serialize_as_sorted_lists(self) -> None:
        payload = _payload()
        assert payload["vocabulary"] == {
            "fields": ["u", "v"],
            "coordinates": ["t", "x"],
        }
        assert payload["lhs_spec"] == {"field": "u", "axis": "t", "order": 1}

    def test_unconstrained_constraint_dimensions_serialize_as_null(
        self,
    ) -> None:
        payload = sketch_to_dict(_sketch(holes=(_hole("rest", U_ONLY),)))
        holes: Any = payload["holes"]
        assert holes[0]["constraint"] == {
            "max_deriv_order": None,
            "operators": None,
            "fields": ["u"],
            "axes": None,
        }

    def test_unknown_top_level_key_rejected(self) -> None:
        payload = _payload()
        payload["surprise"] = 1
        with pytest.raises(ValueError, match="surprise"):
            sketch_from_dict(payload)

    def test_missing_top_level_key_rejected(self) -> None:
        payload = _payload()
        del payload["pinned"]
        with pytest.raises(ValueError, match="pinned"):
            sketch_from_dict(payload)

    def test_unknown_policy_key_rejected(self) -> None:
        payload = _payload()
        payload["match_policy"]["surprise"] = 1
        with pytest.raises(ValueError, match="surprise"):
            sketch_from_dict(payload)

    def test_unknown_constraint_key_rejected(self) -> None:
        payload = _payload()
        payload["holes"][0]["constraint"]["surprise"] = 1
        with pytest.raises(ValueError, match="surprise"):
            sketch_from_dict(payload)

    def test_missing_nested_key_rejected(self) -> None:
        payload = _payload()
        del payload["lhs_spec"]["field"]


        with pytest.raises(ValueError, match="'field'"):
            sketch_from_dict(payload)

    def test_foreign_schema_tag_rejected(self) -> None:
        payload = _payload()
        payload["schema"] = "kd-sketch-v2"
        with pytest.raises(ValueError, match="schema"):
            sketch_from_dict(payload)

    def test_decoding_re_runs_construction_validation(self) -> None:
        payload = _payload()
        payload["pinned"][0]["value"] = 0.0
        with pytest.raises(ValueError, match="zero"):
            sketch_from_dict(payload)


class TestConstantTermSketches:

    def test_unconstrained_hole_admits_the_constant(self) -> None:
        sketch = _sketch(holes=(_hole("rest", TermConstraint()),))
        verdict = sketch.matches(_equation(("one", 0.5), ("u_xx", 1.0)))
        assert verdict.holes[0].assigned == tuple(
            sorted((law_term_key("one"), law_term_key("u_xx")))
        )
        assert verdict.unassigned == ()
        assert verdict.overall is True

    def test_pinned_constant_matches_its_coefficient(self) -> None:
        sketch = _sketch(pinned=(PinnedTerm("one", 0.5),))
        verdict = sketch.matches(_equation(("one", 0.5)))
        assert verdict.pinned[0].law_key == law_term_key("one")
        assert verdict.pinned[0].observed == 0.5
        assert verdict.pinned[0].matched is True
        assert verdict.overall is True

    def test_pinned_constant_rejects_a_wrong_coefficient(self) -> None:
        sketch = _sketch(
            pinned=(PinnedTerm("one", 0.5),),
            match_policy=SketchMatchPolicy(
                coeff_atol=0.01, coeff_rtol=0.0, support_threshold=0.0
            ),
        )
        verdict = sketch.matches(_equation(("one", 0.9)))
        assert verdict.pinned[0].observed == 0.9
        assert verdict.pinned[0].matched is False
        assert verdict.overall is False

    def test_anchored_constant_is_satisfied_when_present(self) -> None:
        sketch = _sketch(anchored=(AnchoredTerm("one"),))
        verdict = sketch.matches(_equation(("one", 0.5)))
        assert verdict.anchored[0].law_key == law_term_key("one")
        assert verdict.anchored[0].observed == 0.5
        assert verdict.anchored[0].matched is True
        assert verdict.overall is True

    def test_anchored_constant_fails_when_absent(self) -> None:
        sketch = _sketch(
            anchored=(AnchoredTerm("one"),), holes=(_hole("rest", U_ONLY),)
        )
        verdict = sketch.matches(_equation(("u_xx", 1.0)))
        assert verdict.anchored[0].observed is None
        assert verdict.anchored[0].matched is False
        assert verdict.overall is False

    def test_fields_constrained_hole_never_admits_the_constant(self) -> None:


        sketch = _sketch(holes=(_hole("rest", U_ONLY),))
        verdict = sketch.matches(_equation(("one", 0.5), ("u_xx", 1.0)))
        assert verdict.holes[0].assigned == (law_term_key("u_xx"),)
        assert tuple(entry.law_key for entry in verdict.unassigned) == (
            law_term_key("one"),
        )


        assert "no sketch hole admits" in verdict.unassigned[0].reason
        assert verdict.overall is False

    def test_multi_hole_sketch_never_admits_the_constant(self) -> None:



        sketch = _sketch(holes=(_hole("h_u", U_ONLY), _hole("h_v", V_ONLY)))
        verdict = sketch.matches(_equation(("one", 0.5), ("u_x", 1.0)))
        assert verdict.holes[0].assigned == (law_term_key("u_x"),)
        assert verdict.holes[1].assigned == ()
        assert tuple(entry.law_key for entry in verdict.unassigned) == (
            law_term_key("one"),
        )
        assert "no sketch hole admits" in verdict.unassigned[0].reason

    def test_admitted_constant_counts_against_the_hole_budget(self) -> None:
        sketch = _sketch(
            holes=(_hole("rest", TermConstraint(), min_count=0, max_count=1),)
        )
        verdict = sketch.matches(_equation(("one", 0.5), ("u_xx", 1.0)))
        assert verdict.holes[0].assigned == tuple(
            sorted((law_term_key("one"), law_term_key("u_xx")))
        )
        assert verdict.holes[0].matched is False
        assert verdict.overall is False

    def test_diff_wrapped_unity_clause_rejected_at_construction(self) -> None:

        with pytest.raises(ValueError, match="unity"):
            _sketch(pinned=(PinnedTerm("diff_x(one)", 0.5),))

    def test_diff_wrapped_unity_candidate_reports_unassigned(self) -> None:


        sketch = _sketch(holes=(_hole("rest", TermConstraint()),))
        verdict = sketch.matches(
            _equation(("diff3_x(one)", 7.0), ("u_xx", 1.0))
        )
        assert tuple(entry.law_key for entry in verdict.unassigned) == (
            law_term_key("diff3_x(one)"),
        )
        assert "unity" in verdict.unassigned[0].reason
        assert verdict.overall is False

    def test_constant_pin_survives_the_round_trip(self) -> None:
        sketch = _sketch(pinned=(PinnedTerm("one", 0.5),))
        payload = sketch_to_dict(sketch)

        assert payload["pinned"] == [{"term_ir": "one", "value": 0.5}]
        assert sketch_from_dict(payload) == sketch
