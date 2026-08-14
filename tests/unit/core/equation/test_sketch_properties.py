
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from kd.core.equation import (
    AnchoredTerm,
    Evolution,
    LhsSpec,
    PinnedTerm,
    Scalar,
    Sketch,
    SketchMatchPolicy,
    TermConstraint,
    TermHole,
    make_evolution,
    sketch_from_dict,
    sketch_to_dict,
)
from kd.core.expr import TermVocabulary

VOCAB = TermVocabulary(
    fields=frozenset({"u", "v"}), coordinates=frozenset({"t", "x"})
)
LHS = LhsSpec(field="u", axis="t", order=1)
TERM_POOL = ("u_x", "u_xx", "mul(u,u_x)", "v_x", "v_xx", "diff_x(u_xx)")


VALUES = (-2.0, -0.5, 0.25, 1.0, 3.0)
TOLERANCES = (0.0, 0.125, 0.25)
HOLE_FIELDS = (frozenset({"u"}), frozenset({"v"}))

Candidate = tuple[list[tuple[str, float]], list[int] | None, list[int]]


@st.composite
def _sketches(draw: st.DrawFn) -> Sketch:
    hole_count = draw(st.integers(min_value=0, max_value=2))
    term_irs = draw(
        st.lists(
            st.sampled_from(TERM_POOL),
            min_size=1 if hole_count == 0 else 0,
            max_size=3,
            unique=True,
        )
    )
    split = draw(st.integers(min_value=0, max_value=len(term_irs)))
    pinned = tuple(
        PinnedTerm(term_ir, draw(st.sampled_from(VALUES)))
        for term_ir in term_irs[:split]
    )
    anchored = tuple(AnchoredTerm(term_ir) for term_ir in term_irs[split:])

    holes: list[TermHole] = []
    for index in range(hole_count):
        fields = (
            HOLE_FIELDS[index]
            if hole_count == 2
            else draw(st.sampled_from((None, *HOLE_FIELDS)))
        )
        min_count = draw(st.integers(min_value=0, max_value=2))
        max_count = max(1, min_count + draw(st.integers(min_value=0, max_value=2)))
        holes.append(
            TermHole(
                id=f"h{index}",
                min_count=min_count,
                max_count=max_count,
                constraint=TermConstraint(
                    max_deriv_order=draw(st.sampled_from((None, 0, 2))),
                    operators=draw(
                        st.sampled_from((None, frozenset(), frozenset({"mul"})))
                    ),
                    fields=fields,
                    axes=draw(
                        st.sampled_from(
                            (None, frozenset(), frozenset({"x"}), frozenset({"t", "x"}))
                        )
                    ),
                ),
            )
        )

    return Sketch(
        lhs_spec=LhsSpec(
            field=draw(st.sampled_from(("u", "v"))),
            axis=draw(st.sampled_from(("t", "x"))),
            order=draw(st.integers(min_value=1, max_value=2)),
        ),
        vocabulary=VOCAB,
        pinned=pinned,
        anchored=anchored,
        holes=tuple(holes),
        match_policy=SketchMatchPolicy(
            coeff_atol=draw(st.sampled_from(TOLERANCES)),
            coeff_rtol=draw(st.sampled_from(TOLERANCES)),
            support_threshold=draw(st.sampled_from(TOLERANCES)),
        ),
    )


@st.composite
def _candidates(draw: st.DrawFn) -> Candidate:
    terms = draw(
        st.lists(
            st.tuples(st.sampled_from(TERM_POOL), st.sampled_from(VALUES)),
            min_size=1,
            max_size=4,
        )
    )
    count = len(terms)
    active = draw(
        st.one_of(
            st.none(),
            st.lists(
                st.integers(min_value=0, max_value=count - 1),
                min_size=1,
                max_size=count,
                unique=True,
            ),
        )
    )
    return terms, active, draw(st.permutations(list(range(count))))


ORDER_SKETCH = Sketch(
    lhs_spec=LHS,
    vocabulary=VOCAB,
    pinned=(PinnedTerm("u_xx", 1.0),),
    anchored=(AnchoredTerm("v_x"),),
    holes=(
        TermHole(
            id="h_u",
            min_count=0,
            max_count=3,
            constraint=TermConstraint(fields=frozenset({"u"})),
        ),
        TermHole(
            id="h_v",
            min_count=0,
            max_count=3,
            constraint=TermConstraint(fields=frozenset({"v"})),
        ),
    ),
    match_policy=SketchMatchPolicy(
        coeff_atol=0.0, coeff_rtol=0.0, support_threshold=0.0
    ),
)


def _evolution(
    terms: list[tuple[str, float]], active: list[int] | None
) -> Evolution:
    return make_evolution(
        LHS,
        tuple((term_ir, Scalar(value)) for term_ir, value in terms),
        active_indices=None if active is None else tuple(active),
    )


@settings(max_examples=50, deadline=None)
@given(sketch=_sketches())
def test_wire_round_trip_is_lossless(sketch: Sketch) -> None:
    assert sketch_from_dict(sketch_to_dict(sketch)) == sketch


@settings(max_examples=50, deadline=None)
@given(candidate=_candidates())
def test_matching_ignores_candidate_term_order(candidate: Candidate) -> None:
    terms, active, permutation = candidate
    original = _evolution(terms, active)
    position = {old: new for new, old in enumerate(permutation)}
    reordered = _evolution(
        [terms[index] for index in permutation],
        None if active is None else [position[index] for index in active],
    )
    assert ORDER_SKETCH.matches(original) == ORDER_SKETCH.matches(reordered)
