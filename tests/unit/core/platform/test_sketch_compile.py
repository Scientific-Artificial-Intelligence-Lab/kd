
from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from kd.core.platform.sketch_compile import (
    SKETCH_CLAUSES,
    SKETCH_CONFIG_KEY,
    SketchClauseLevels,
    compile_sketch,
    used_clauses,
)

from kd.core.equation.signature import law_signature, law_term_entry, law_term_key
from kd.core.equation.sketch import (
    AnchoredTerm,
    PinnedTerm,
    Sketch,
    TermConstraint,
    TermHole,
    sketch_to_dict,
)
from kd.core.equation.types import Equation, Evolution, LhsSpec, Scalar
from kd.core.evaluator import EvaluationResult
from tests.unit.search._sketch_fakes import (
    PINNED_ADVECTION,
    PINNED_ADVECTION_VALUE,
    burgers_sketch,
    default_hole,
    match_policy,
    sketch_vocabulary,
)


def hole_only_sketch(constraint: TermConstraint) -> Sketch:
    return Sketch(
        lhs_spec=LhsSpec("u", "t", 1),
        vocabulary=sketch_vocabulary(),
        pinned=(),
        anchored=(),
        holes=(TermHole(id="h", min_count=0, max_count=2, constraint=constraint),),
        match_policy=match_policy(),
    )


def candidate_eval(
    terms: Sequence[str] | None,
    coefficients: Sequence[float] | None,
    *,
    selected_indices: list[int] | None = None,
    is_valid: bool = True,
) -> EvaluationResult:
    return EvaluationResult(
        mse=0.0,
        nmse=0.0,
        r2=1.0,
        terms=None if terms is None else list(terms),
        coefficients=(
            None
            if coefficients is None
            else torch.tensor(list(coefficients), dtype=torch.float64)
        ),
        selected_indices=selected_indices,
        is_valid=is_valid,
    )


def law_entries(eq: Equation) -> list[tuple[str, float]]:
    assert isinstance(eq, Evolution)
    entries: list[tuple[str, float]] = []
    for term_ir, coefficient in eq.terms:
        assert isinstance(coefficient, Scalar)
        entries.append((term_ir, coefficient.value))
    return entries


def test_config_key_and_clause_vocabulary_are_the_frozen_v1_set() -> None:
    assert SKETCH_CONFIG_KEY == "sketch"
    assert SKETCH_CLAUSES == (
        "fixed_terms",
        "anchors",
        "hole_count",
        "derivative_order",
        "operator_set",
        "field_axis_set",
    )


def test_clause_levels_default_to_unsupported_and_report_per_clause() -> None:
    levels = SketchClauseLevels()
    assert all(levels.level(clause) == "unsupported" for clause in SKETCH_CLAUSES)
    declared = SketchClauseLevels(fixed_terms="lowered", anchors="fit_enforced")
    assert declared.level("fixed_terms") == "lowered"
    assert declared.level("anchors") == "fit_enforced"
    assert declared.level("hole_count") == "unsupported"


def test_clause_levels_reject_a_level_outside_the_vocabulary() -> None:
    with pytest.raises(ValueError, match="fixed_terms"):
        SketchClauseLevels(fixed_terms="mostly_enforced")


@pytest.mark.parametrize(
    ("sketch", "expected"),
    [
        (
            Sketch(
                lhs_spec=LhsSpec("u", "t", 1),
                vocabulary=sketch_vocabulary(),
                pinned=(PinnedTerm(PINNED_ADVECTION, PINNED_ADVECTION_VALUE),),
                anchored=(),
                holes=(),
                match_policy=match_policy(),
            ),
            {"fixed_terms"},
        ),
        (
            Sketch(
                lhs_spec=LhsSpec("u", "t", 1),
                vocabulary=sketch_vocabulary(),
                pinned=(),
                anchored=(AnchoredTerm("u_xx"),),
                holes=(),
                match_policy=match_policy(),
            ),
            {"anchors"},
        ),
        (hole_only_sketch(TermConstraint()), {"hole_count"}),
        (
            hole_only_sketch(TermConstraint(max_deriv_order=2)),
            {"hole_count", "derivative_order"},
        ),
        (
            hole_only_sketch(TermConstraint(operators=frozenset({"mul"}))),
            {"hole_count", "operator_set"},
        ),
        (
            hole_only_sketch(TermConstraint(fields=frozenset({"u"}))),
            {"hole_count", "field_axis_set"},
        ),
        (
            hole_only_sketch(TermConstraint(axes=frozenset({"x"}))),
            {"hole_count", "field_axis_set"},
        ),
    ],
)
def test_used_clauses_reports_only_the_dimensions_the_sketch_constrains(
    sketch: Sketch, expected: set[str]
) -> None:
    assert used_clauses(sketch) == frozenset(expected)


def test_platform_compile_lowers_fixed_terms_and_exit_checks_the_rest() -> None:
    compiled = compile_sketch(burgers_sketch())
    levels = compiled.report.levels
    assert levels.fixed_terms == "lowered"
    assert [
        levels.level(clause) for clause in SKETCH_CLAUSES if clause != "fixed_terms"
    ] == ["exit_checked"] * 5
    assert compiled.sketch == burgers_sketch()


def test_closed_sketch_lifts_without_touching_a_solver() -> None:
    sketch = burgers_sketch(holes=())
    compiled = compile_sketch(sketch)
    assert compiled.closed is True

    lifted = compiled.lift(None)

    assert lifted is not None
    key, value = law_term_entry(PINNED_ADVECTION, PINNED_ADVECTION_VALUE)
    assert law_entries(lifted) == [(key, value)]
    assert isinstance(lifted, Evolution)
    assert lifted.lhs_spec == sketch.lhs_spec


def test_open_sketch_without_a_final_evaluation_is_a_caller_bug() -> None:
    compiled = compile_sketch(burgers_sketch())
    assert compiled.closed is False

    with pytest.raises(ValueError, match="final_eval"):
        compiled.lift(None)


def test_invalid_final_evaluation_yields_no_liftable_law() -> None:
    compiled = compile_sketch(burgers_sketch())
    assert compiled.lift(candidate_eval(["u_xx"], [0.1], is_valid=False)) is None


def test_lift_restricts_candidates_to_the_active_support() -> None:
    compiled = compile_sketch(burgers_sketch())

    lifted = compiled.lift(
        candidate_eval(["u", "u_xx", "u_x"], [5.0, 0.1, 7.0], selected_indices=[1])
    )

    assert lifted is not None
    assert law_entries(lifted)[1:] == [(law_term_key("u_xx"), 0.1)]


def test_lift_aggregates_alias_identical_terms_and_drops_exact_zero_sums() -> None:
    compiled = compile_sketch(burgers_sketch())

    lifted = compiled.lift(
        candidate_eval(["u_xx", "u_xx", "u", "neg(u)"], [0.04, 0.06, 0.5, 0.5])
    )

    assert lifted is not None
    candidates = law_entries(lifted)[1:]
    assert [key for key, _value in candidates] == [law_term_key("u_xx")]
    torch.testing.assert_close(candidates[0][1], 0.1, rtol=0.0, atol=1e-12)


def test_lift_emits_law_key_spellings_with_entry_signed_values() -> None:
    sketch = burgers_sketch(pinned=(PinnedTerm("neg(u)", 2.0),))
    compiled = compile_sketch(sketch)

    lifted = compiled.lift(candidate_eval(["u_xx"], [0.1]))

    assert lifted is not None
    assert law_entries(lifted)[0] == ("u", -2.0)


def test_lift_puts_pinned_entries_first_then_sorted_candidate_keys() -> None:
    sketch = burgers_sketch(
        pinned=(
            PinnedTerm(PINNED_ADVECTION, PINNED_ADVECTION_VALUE),
            PinnedTerm("u", 3.0),
        ),
        holes=(default_hole(),),
    )
    compiled = compile_sketch(sketch)

    lifted = compiled.lift(candidate_eval(["u_xx", "u_x"], [0.1, 0.2]))

    assert lifted is not None
    keys = [key for key, _value in law_entries(lifted)]
    assert keys[:2] == [law_term_key(PINNED_ADVECTION), law_term_key("u")]
    assert keys[2:] == sorted(keys[2:])
    assert set(keys[2:]) == {law_term_key("u_x"), law_term_key("u_xx")}


def test_sub_band_candidate_copy_of_a_pin_merges_to_the_exact_pin() -> None:
    sketch = burgers_sketch()
    compiled = compile_sketch(sketch)
    pinned_key = law_term_key(PINNED_ADVECTION)

    lifted = compiled.lift(
        candidate_eval(["u_xx", PINNED_ADVECTION], [0.1, 1e-12])
    )

    assert lifted is not None
    entries = law_entries(lifted)
    assert [key for key, _value in entries].count(pinned_key) == 1
    assert entries[0] == (pinned_key, PINNED_ADVECTION_VALUE)
    law_signature(lifted)


def test_material_candidate_copy_of_a_pin_fails_the_verdict() -> None:
    sketch = burgers_sketch()
    compiled = compile_sketch(sketch)
    pinned_key = law_term_key(PINNED_ADVECTION)

    lifted = compiled.lift(
        candidate_eval(["u_xx", PINNED_ADVECTION], [0.1, 0.25])
    )

    assert lifted is not None
    entries = law_entries(lifted)
    assert [key for key, _value in entries].count(pinned_key) == 1
    torch.testing.assert_close(
        entries[0][1], PINNED_ADVECTION_VALUE + 0.25, rtol=0.0, atol=1e-15
    )
    verdict = sketch.matches(lifted)
    assert verdict.overall is False
    assert [entry.matched for entry in verdict.pinned] == [False]
    law_signature(lifted)


@pytest.mark.parametrize(
    "final_eval",
    [
        candidate_eval(None, [0.1]),
        candidate_eval(["u_xx"], None),
        candidate_eval(["u_xx"], [0.1, 0.2]),
        candidate_eval(["u_xx"], [0.1], selected_indices=[3]),
        candidate_eval(["u_xx"], [float("inf")]),
        candidate_eval(["u_xx", "u_xx"], [float("inf"), float("-inf")]),
        candidate_eval(["u_xx", "u_xx"], [1e308, 1e308]),
        candidate_eval(["mul(0.5,u_xx)"], [0.1]),
        candidate_eval(["u*u_x"], [0.1]),
        candidate_eval(["u_xx("], [0.1]),
        candidate_eval([""], [0.1]),
    ],
    ids=[
        "no-terms",
        "no-coefficients",
        "length-mismatch",
        "out-of-range",
        "non-finite",
        "alias-inf-cancel",
        "alias-overflow",
        "numeric-constant-term",
        "python-operator-term",
        "unclosed-paren-term",
        "empty-term",
    ],
)
def test_lift_degrades_to_none_on_unusable_candidate_data(
    final_eval: EvaluationResult,
) -> None:
    assert compile_sketch(burgers_sketch()).lift(final_eval) is None


def test_sketch_payload_is_deterministic_across_independent_construction() -> None:
    assert sketch_to_dict(burgers_sketch()) == sketch_to_dict(burgers_sketch())
