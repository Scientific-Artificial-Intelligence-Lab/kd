
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from kd.core.equation import LhsSpec, Scalar, from_dict, make_evolution, to_dict
from kd.core.equation.structure import structure, term_diff
from kd.core.equation.types import Evolution

_NAMES = st.sampled_from(["u", "x", "t", "u_x", "v"])
_UNARY_OPS = st.sampled_from(["diff_x", "diff2_x", "n2"])
_BINARY_OPS = st.sampled_from(["add", "mul", "sub", "div"])


def _term_irs() -> st.SearchStrategy[str]:
    return st.recursive(
        _NAMES,
        lambda children: st.one_of(
            st.tuples(_UNARY_OPS, children).map(lambda t: f"{t[0]}({t[1]})"),
            st.tuples(_BINARY_OPS, children, children).map(
                lambda t: f"{t[0]}({t[1]}, {t[2]})"
            ),
        ),
        max_leaves=8,
    )


_COEFFS = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
_LHS = st.builds(
    LhsSpec,
    field=st.sampled_from(["u", "v"]),
    axis=st.sampled_from(["t", "tau"]),
    order=st.sampled_from([1, 2]),
)


def _evolutions() -> st.SearchStrategy[Evolution]:
    return st.builds(
        make_evolution,
        _LHS,
        st.lists(st.tuples(_term_irs(), st.builds(Scalar, _COEFFS)), min_size=1, max_size=4).map(
            tuple
        ),
    )


@settings(deadline=None)
@given(eq=_evolutions())
def test_structure_invariant_under_serialize_round_trip(eq: Evolution) -> None:
    assert structure(from_dict(to_dict(eq))) == structure(eq)


@settings(deadline=None)
@given(eq=_evolutions())
def test_structure_invariant_under_term_permutation(eq: Evolution) -> None:
    reversed_eq = make_evolution(eq.lhs_spec, tuple(reversed(eq.terms)))
    assert structure(reversed_eq) == structure(eq)


@settings(deadline=None)
@given(eq=_evolutions(), value=_COEFFS)
def test_structure_invariant_under_coefficient_change(
    eq: Evolution, value: float
) -> None:
    rescaled = make_evolution(
        eq.lhs_spec, tuple((term_ir, Scalar(value)) for term_ir, _ in eq.terms)
    )
    assert structure(rescaled) == structure(eq)


@settings(deadline=None)
@given(eq=_evolutions())
def test_term_diff_self_is_empty(eq: Evolution) -> None:
    d = term_diff(eq, eq)
    assert d.added == frozenset()
    assert d.removed == frozenset()
    assert d.lhs_changed is False
    assert d.form_changed is False
    assert d.common == structure(eq).terms


@settings(deadline=None)
@given(x=_term_irs(), y=_term_irs(), op=st.sampled_from(["add", "mul"]))
def test_commutative_flip_gives_same_fingerprint(x: str, y: str, op: str) -> None:
    lhs = LhsSpec(field="u", axis="t", order=1)
    flipped = make_evolution(lhs, ((f"{op}({y}, {x})", Scalar(1.0)),))
    straight = make_evolution(lhs, ((f"{op}({x}, {y})", Scalar(1.0)),))
    assert structure(flipped) == structure(straight)


@settings(deadline=None)
@given(old=_evolutions(), new=_evolutions())
def test_term_diff_partitions_both_fingerprints(
    old: Evolution, new: Evolution
) -> None:
    d = term_diff(old, new)
    assert d.common | d.removed == structure(old).terms
    assert d.common | d.added == structure(new).terms
    assert d.common & d.added == frozenset()
    assert d.common & d.removed == frozenset()
    assert d.added & d.removed == frozenset()


@settings(deadline=None)
@given(old=_evolutions(), new=_evolutions())
def test_term_diff_antisymmetric(old: Evolution, new: Evolution) -> None:
    forward = term_diff(old, new)
    backward = term_diff(new, old)
    assert forward.added == backward.removed
    assert forward.removed == backward.added
    assert forward.common == backward.common
    assert forward.lhs_changed == backward.lhs_changed
    assert forward.form_changed == backward.form_changed
