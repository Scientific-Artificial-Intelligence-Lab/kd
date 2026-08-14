
from __future__ import annotations

import pytest

from kd.core.equation.sketch import (
    AnchoredTerm,
    PinnedTerm,
    Sketch,
    TermConstraint,
    TermHole,
)
from kd.search.pysindy.sketch_backend import compile_for_pysindy
from tests.unit.search._sketch_fakes import (
    HOLE_TERM,
    PINNED_ADVECTION,
    PYSINDY_DECLARED_LEVELS,
    burgers_sketch,
)



_ORDER_3_TERM = "u_xxx"
_FOREIGN_TERM = "v"

_CATALOG = ("u", "u_x", HOLE_TERM, PINNED_ADVECTION)


def _hole(
    *,
    hole_id: str = "diffusion",
    min_count: int = 1,
    max_count: int = 1,
    max_deriv_order: int = 2,
) -> TermHole:
    return TermHole(
        id=hole_id,
        min_count=min_count,
        max_count=max_count,
        constraint=TermConstraint(max_deriv_order=max_deriv_order),
    )







def test_a_closed_sketch_returns_the_configured_library_unfiltered() -> None:
    compiled = compile_for_pysindy(burgers_sketch(holes=()), _CATALOG)

    assert compiled.effective_terms == _CATALOG
    assert compiled.dropped == ()
    assert any("closed" in note for note in compiled.report.notes)







def test_b_anchored_representability_is_decided_by_law_key() -> None:
    sketch = burgers_sketch(pinned=(), anchored=(AnchoredTerm("mul(u_x,u)"),))

    compiled = compile_for_pysindy(sketch, ("u", "mul(u, u_x)", HOLE_TERM))

    assert "mul(u, u_x)" in compiled.effective_terms


def test_c_an_unrepresentable_anchor_is_refused_before_any_fit() -> None:
    sketch = burgers_sketch(pinned=(), anchored=(AnchoredTerm(_ORDER_3_TERM),))

    with pytest.raises(ValueError, match="anchors") as excinfo:
        compile_for_pysindy(sketch, ("u", "u_x", HOLE_TERM))

    assert _ORDER_3_TERM in str(excinfo.value)







def test_d_pinned_columns_leave_the_effective_library() -> None:
    compiled = compile_for_pysindy(burgers_sketch(), _CATALOG)

    assert compiled.effective_terms == ("u", "u_x", HOLE_TERM)
    assert (PINNED_ADVECTION, "pinned") in compiled.dropped


@pytest.mark.parametrize(
    ("pin", "excluded"),
    [
        (PinnedTerm("mul(u_x,u)", -1.0), "mul(u, u_x)"),
        (PinnedTerm("neg(u)", -1.0), "u"),
    ],
    ids=["swapped-operands", "head-neg"],
)
def test_e_pinned_exclusion_compares_law_keys_not_spellings(
    pin: PinnedTerm, excluded: str
) -> None:
    compiled = compile_for_pysindy(
        burgers_sketch(pinned=(pin,)), ("u", "mul(u, u_x)", HOLE_TERM)
    )

    assert excluded not in compiled.effective_terms







def test_f_a_hole_constraint_filters_the_library_at_compile_time() -> None:
    compiled = compile_for_pysindy(
        burgers_sketch(), ("u", HOLE_TERM, _ORDER_3_TERM, PINNED_ADVECTION)
    )

    assert compiled.effective_terms == ("u", HOLE_TERM)
    assert _ORDER_3_TERM in [term for term, _reason in compiled.dropped]


def test_g_a_term_outside_the_sketch_vocabulary_is_dropped_by_name() -> None:
    compiled = compile_for_pysindy(
        burgers_sketch(), ("u", HOLE_TERM, _FOREIGN_TERM, PINNED_ADVECTION)
    )

    assert _FOREIGN_TERM not in compiled.effective_terms
    assert (_FOREIGN_TERM, "outside-vocabulary") in compiled.dropped


def test_h_an_anchored_column_survives_a_hole_that_excludes_it() -> None:
    sketch = burgers_sketch(pinned=(), anchored=(AnchoredTerm(_ORDER_3_TERM),))

    compiled = compile_for_pysindy(sketch, ("u", HOLE_TERM, _ORDER_3_TERM))

    assert _ORDER_3_TERM in compiled.effective_terms


def test_i_a_sketch_without_holes_keeps_only_its_anchored_columns() -> None:
    compiled = compile_for_pysindy(
        burgers_sketch(anchored=(AnchoredTerm(HOLE_TERM),), holes=()), _CATALOG
    )

    assert compiled.effective_terms == (HOLE_TERM,)







def test_j_an_empty_effective_library_is_refused() -> None:
    sketch = burgers_sketch(holes=(_hole(hole_id="constant", max_deriv_order=0),))

    with pytest.raises(ValueError, match="(?i)empty"):
        compile_for_pysindy(sketch, ("u_x", HOLE_TERM, PINNED_ADVECTION))


def test_k_a_structurally_unsatisfiable_hole_count_names_the_hole() -> None:
    sketch = burgers_sketch(
        holes=(_hole(hole_id="diffusion-pair", min_count=2, max_count=2),)
    )

    with pytest.raises(ValueError, match="diffusion-pair"):
        compile_for_pysindy(sketch, (HOLE_TERM, PINNED_ADVECTION))


def test_o_an_anchor_reserved_column_cannot_feed_a_hole() -> None:
    sketch = burgers_sketch(pinned=(), anchored=(AnchoredTerm(HOLE_TERM),))

    with pytest.raises(ValueError, match="diffusion"):
        compile_for_pysindy(sketch, (HOLE_TERM,))


def test_p_alias_duplicate_columns_count_once_toward_a_hole() -> None:
    sketch = burgers_sketch(
        pinned=(), holes=(_hole(hole_id="pair", min_count=2, max_count=2),)
    )

    with pytest.raises(ValueError, match="pair"):
        compile_for_pysindy(sketch, ("mul(u, u_x)", "mul(u_x,u)"))







@pytest.mark.parametrize(
    "holes", [(), (_hole(),)], ids=["closed", "open"]
)
def test_l_the_report_declares_the_frozen_enforcement_vector(
    holes: tuple[TermHole, ...],
) -> None:
    compiled = compile_for_pysindy(burgers_sketch(holes=holes), _CATALOG)

    assert compiled.report.levels == PYSINDY_DECLARED_LEVELS


def test_m_the_report_accounts_for_every_dropped_column() -> None:
    compiled = compile_for_pysindy(
        burgers_sketch(), ("u", HOLE_TERM, _ORDER_3_TERM, PINNED_ADVECTION)
    )

    assert {term for term, _reason in compiled.dropped} == {
        _ORDER_3_TERM,
        PINNED_ADVECTION,
    }
    notes = " ".join(compiled.report.notes)
    assert all(term in notes for term, _reason in compiled.dropped)


def test_n_a_pinned_sketch_alone_still_compiles_against_its_catalog() -> None:
    open_sketch: Sketch = burgers_sketch()
    closed_sketch: Sketch = burgers_sketch(holes=())

    assert PINNED_ADVECTION not in compile_for_pysindy(
        open_sketch, _CATALOG
    ).effective_terms
    assert PINNED_ADVECTION in compile_for_pysindy(
        closed_sketch, _CATALOG
    ).effective_terms
