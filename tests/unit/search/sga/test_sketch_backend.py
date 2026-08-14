
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from kd.core.equation.signature import law_term_key
from kd.core.equation.sketch import (
    AnchoredTerm,
    PinnedTerm,
    Sketch,
    TermConstraint,
    TermHole,
)
from kd.core.platform.sketch_compile import SketchClauseLevels
from kd.search.sga.config import OP1, OP2, OPS, ROOT, OperatorPool, SGAConfig
from kd.search.sga.convert import tree_to_kd_expr
from kd.search.sga.evaluate import DiffContext, execute_tree
from kd.search.sga.sketch_backend import SGACompiled, compile_for_sga
from kd.search.sga.tree import Node, Tree
from tests.unit.search._sketch_fakes import PINNED_ADVECTION, burgers_sketch




SGA_DECLARED_LEVELS = SketchClauseLevels(
    fixed_terms="lowered",
    anchors="exit_checked",
    hole_count="exit_checked",
    derivative_order="generation_enforced",
    operator_set="generation_enforced",
    field_axis_set="generation_enforced",
)




_VARS: list[str] = ["u", "u_x", "x"]
_DEN: OperatorPool = (("x", 0),)
_DEFAULT_TERM = "u"

_MUL = ("*", 2)
_DIV = ("/", 2)
_SQUARE = ("^2", 1)
_CUBE = ("^3", 1)
_DIFF = ("d", 2)
_DIFF2 = ("d^2", 2)


def _hole(
    *,
    hole_id: str = "diffusion",
    min_count: int = 1,
    max_count: int = 2,
    max_deriv_order: int | None = 2,
    operators: frozenset[str] | None = None,
    fields: frozenset[str] | None = None,
    axes: frozenset[str] | None = None,
) -> TermHole:
    return TermHole(
        id=hole_id,
        min_count=min_count,
        max_count=max_count,
        constraint=TermConstraint(
            max_deriv_order=max_deriv_order,
            operators=operators,
            fields=fields,
            axes=axes,
        ),
    )


def _compile(
    sketch: Sketch,
    *,
    pool: list[str] | None = None,
    den: OperatorPool = _DEN,
    config: SGAConfig | None = None,
    default_term_name: str | None = _DEFAULT_TERM,
) -> SGACompiled:
    return compile_for_sga(
        sketch,
        vars=list(_VARS if pool is None else pool),
        den=den,
        ops=OPS,
        root=ROOT,
        op1=OP1,
        op2=OP2,
        default_term_name=default_term_name,
        config=SGAConfig() if config is None else config,
    )


def _sga_data() -> tuple[dict[str, Tensor], DiffContext]:
    nx, nt = 6, 4
    x_axis = torch.linspace(0.5, 1.5, nx, dtype=torch.float64)
    t_axis = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    x_grid = x_axis.unsqueeze(1).expand(nx, nt)
    t_grid = t_axis.unsqueeze(0).expand(nx, nt)
    field = torch.sin(x_grid) + 0.25 * t_grid + 1.0
    data: dict[str, Tensor] = {
        "u": field.flatten().clone(),
        "u_x": torch.cos(x_grid).flatten().clone(),
        "x": x_grid.flatten().clone(),
    }
    context = DiffContext(
        field_shape=(nx, nt),
        axis_map={"x": 0, "t": 1},
        delta={
            "x": float(x_axis[1] - x_axis[0]),
            "t": float(t_axis[1] - t_axis[0]),
        },
        lhs_axis="t",
    )
    return data, context


def _column(tree: Tree) -> Tensor:
    data, context = _sga_data()
    return execute_tree(tree, data, context)


def _pinned_column(compiled: SGACompiled) -> Tensor:
    assert len(compiled.pinned) == 1
    _key, _value, tree = compiled.pinned[0]
    return _column(tree)


def _dropped_elements(compiled: SGACompiled) -> set[str]:
    return {element for element, _reason in compiled.dropped}







def test_a_closed_sketch_leaves_every_pool_and_the_default_untouched() -> None:
    compiled = _compile(burgers_sketch(holes=()))

    assert compiled.vars == tuple(_VARS)
    assert compiled.den == _DEN
    assert compiled.ops == OPS
    assert compiled.root == ROOT
    assert compiled.op1 == OP1
    assert compiled.op2 == OP2
    assert compiled.default_kept is True
    assert compiled.dropped == ()
    assert any("closed" in note for note in compiled.report.notes)


def test_b_a_closed_sketch_admits_a_pin_outside_the_tree_language() -> None:
    sketch = burgers_sketch(pinned=(PinnedTerm("mul(u,neg(u_x))", -1.0),), holes=())

    compiled = _compile(sketch)

    assert compiled.pinned == ()
    assert compiled.report.levels == SGA_DECLARED_LEVELS







def test_c_a_terminal_spelled_anchor_is_refused_with_a_generable_hint() -> None:
    sketch = burgers_sketch(pinned=(), anchored=(AnchoredTerm("u_x"),))

    with pytest.raises(ValueError, match="u_x") as excinfo:
        _compile(sketch)

    assert "diff_x" in str(excinfo.value)


def test_d_an_open_form_anchor_round_trips_and_compiles() -> None:
    sketch = burgers_sketch(pinned=(), anchored=(AnchoredTerm("diff_x(u)"),))

    compiled = _compile(sketch)

    assert law_term_key("diff_x(u)") in compiled.anchored_keys


def test_e_an_operator_rooted_anchor_is_reachable() -> None:
    sketch = burgers_sketch(pinned=(), anchored=(AnchoredTerm(PINNED_ADVECTION),))

    compiled = _compile(sketch)

    assert law_term_key(PINNED_ADVECTION) in compiled.anchored_keys


def test_f_an_anchor_on_the_retained_default_column_is_reachable() -> None:
    sketch = burgers_sketch(pinned=(), anchored=(AnchoredTerm("u"),))

    compiled = _compile(sketch)

    assert compiled.default_kept is True
    assert law_term_key("u") in compiled.anchored_keys







def test_g_a_hole_demanding_more_terms_than_width_is_refused() -> None:
    sketch = burgers_sketch(holes=(_hole(hole_id="quartet", min_count=4, max_count=4),))

    with pytest.raises(ValueError, match="width"):
        _compile(sketch, config=SGAConfig(width=2))


def test_h_the_retained_default_column_counts_toward_the_width_budget() -> None:
    sketch = burgers_sketch(holes=(_hole(hole_id="pair", min_count=2, max_count=2),))

    compiled = _compile(sketch, config=SGAConfig(width=2))

    assert compiled.default_kept is True







def test_i_a_derivative_terminal_survives_a_field_hole() -> None:
    sketch = burgers_sketch(
        holes=(
            _hole(
                fields=frozenset({"u"}),
                axes=frozenset({"x"}),
                max_deriv_order=2,
            ),
        )
    )

    compiled = _compile(sketch)

    assert "u_x" in compiled.vars


def test_j_a_bare_coordinate_survives_for_composition() -> None:
    sketch = burgers_sketch(
        holes=(_hole(fields=frozenset({"u"}), axes=frozenset({"x"})),)
    )

    compiled = _compile(sketch)

    assert "x" in compiled.vars


def test_k_a_symbol_outside_the_vocabulary_is_dropped_by_name() -> None:
    compiled = _compile(burgers_sketch(), pool=[*_VARS, "v"])

    assert "v" not in compiled.vars
    assert "v" in _dropped_elements(compiled)


@pytest.mark.parametrize(
    ("operators", "kept", "dropped"),
    [
        (frozenset({"mul"}), _MUL, _DIV),
        (frozenset({"n2"}), _SQUARE, _CUBE),
    ],
    ids=["div-needs-recip", "square-is-n2"],
)
def test_l_the_operator_pools_are_filtered_by_law_features(
    operators: frozenset[str],
    kept: tuple[str, int],
    dropped: tuple[str, int],
) -> None:
    sketch = burgers_sketch(holes=(_hole(operators=operators),))

    compiled = _compile(sketch)

    assert kept in compiled.ops
    assert dropped not in compiled.ops


def test_m_the_division_operator_survives_a_clause_naming_both_features() -> None:
    sketch = burgers_sketch(holes=(_hole(operators=frozenset({"mul", "recip"})),))

    compiled = _compile(sketch)

    assert _DIV in compiled.ops
    assert _MUL in compiled.ops


def test_n_the_paired_derivative_operator_is_dropped_at_order_cap_one() -> None:
    sketch = burgers_sketch(holes=(_hole(max_deriv_order=1),))

    compiled = _compile(sketch)

    assert _DIFF2 not in compiled.ops
    assert _DIFF2 not in compiled.root
    assert _DIFF2 not in compiled.op2
    assert _DIFF in compiled.ops


def test_o_an_anchor_keeps_what_the_holes_alone_would_narrow_away() -> None:
    sketch = burgers_sketch(
        pinned=(),
        anchored=(AnchoredTerm("diff_x(u_x)"),),
        holes=(_hole(max_deriv_order=0),),
    )

    compiled = _compile(sketch)

    assert "u_x" in compiled.vars
    assert _DIFF in compiled.ops
    assert _DEN[0] in compiled.den


def test_p_the_unary_pool_stays_consistent_with_the_operator_pool() -> None:
    sketch = burgers_sketch(holes=(_hole(operators=frozenset({"n2"})),))

    compiled = _compile(sketch)

    assert compiled.op1 == (_SQUARE,)
    assert set(compiled.op1) <= set(compiled.ops)


def test_q_an_algebraic_sketch_may_legally_narrow_the_axis_pool_to_empty() -> None:
    sketch = burgers_sketch(
        holes=(
            _hole(
                max_deriv_order=0,
                operators=frozenset({"mul"}),
                axes=frozenset(),
            ),
        )
    )

    compiled = _compile(sketch)

    assert compiled.den == ()
    assert _DIFF not in compiled.ops
    assert _DIFF2 not in compiled.ops
    assert compiled.vars


def test_r_an_empty_axis_pool_is_refused_while_derivatives_survive() -> None:
    sketch = burgers_sketch(holes=(_hole(max_deriv_order=2, axes=frozenset()),))

    with pytest.raises(ValueError, match="den"):
        _compile(sketch)


def test_s_a_hole_admitting_no_terminal_is_refused() -> None:
    sketch = burgers_sketch(
        fields=("u", "v"),
        holes=(_hole(fields=frozenset({"v"}), axes=frozenset(), max_deriv_order=0),),
    )

    with pytest.raises(ValueError, match="vars|variable"):
        _compile(sketch)


def test_t_a_sketch_that_narrows_every_operator_away_is_refused() -> None:
    sketch = burgers_sketch(holes=(_hole(max_deriv_order=0, operators=frozenset()),))

    with pytest.raises(ValueError, match="ops|root"):
        _compile(sketch)







def test_u_a_hole_admitted_default_column_is_kept_and_takes_its_seat() -> None:
    compiled = _compile(burgers_sketch(holes=(_hole(hole_id="diffusion"),)))

    assert compiled.default_kept is True
    assert compiled.default_law_key == law_term_key(_DEFAULT_TERM)
    assert compiled.default_hole_id == "diffusion"


def test_v_a_pinned_default_column_is_dropped_for_the_run() -> None:
    sketch = burgers_sketch(pinned=(PinnedTerm("u", 0.5),))

    compiled = _compile(sketch)

    assert compiled.default_kept is False
    assert _DEFAULT_TERM in _dropped_elements(compiled)


def test_w_a_default_column_no_hole_admits_is_dropped() -> None:
    sketch = burgers_sketch(fields=("u", "v"), holes=(_hole(fields=frozenset({"v"})),))

    compiled = _compile(sketch)

    assert compiled.default_kept is False
    assert compiled.default_hole_id is None







@pytest.mark.parametrize(
    "holes",
    [(), (_hole(),)],
    ids=["closed", "open"],
)
def test_x_the_report_declares_the_frozen_enforcement_vector(
    holes: tuple[TermHole, ...],
) -> None:
    compiled = _compile(burgers_sketch(holes=holes))

    assert compiled.report.levels == SGA_DECLARED_LEVELS


def test_y_the_report_accounts_for_every_dropped_element() -> None:
    compiled = _compile(burgers_sketch(holes=(_hole(),)), pool=[*_VARS, "v"])

    assert _dropped_elements(compiled)
    notes = " ".join(compiled.report.notes)
    assert all(element in notes for element in _dropped_elements(compiled))







def test_z_a_first_order_terminal_pin_stays_a_leaf() -> None:
    sketch = burgers_sketch(pinned=(PinnedTerm("u_x", 0.5),))

    compiled = _compile(sketch)

    data, _context = _sga_data()
    assert compiled.pinned[0][0] == law_term_key("u_x")
    assert compiled.pinned[0][1] == pytest.approx(0.5)
    torch.testing.assert_close(
        _pinned_column(compiled), data["u_x"], rtol=0.0, atol=0.0
    )


def test_aa_a_compound_second_order_pin_becomes_the_paired_operator() -> None:
    sketch = burgers_sketch(pinned=(PinnedTerm("u_xx", 0.3),))

    compiled = _compile(sketch)

    expected = _column(Tree(Node("d^2", 2, [Node("u", 0), Node("x", 0)])))
    torch.testing.assert_close(_pinned_column(compiled), expected, rtol=0.0, atol=0.0)


def test_ab_an_open_form_pin_uses_the_tree_derivative_operator() -> None:
    sketch = burgers_sketch(pinned=(PinnedTerm("diff_x(u)", 0.3),))

    compiled = _compile(sketch)

    expected = _column(Tree(Node("d", 2, [Node("u", 0), Node("x", 0)])))
    torch.testing.assert_close(_pinned_column(compiled), expected, rtol=0.0, atol=0.0)


def test_ac_a_pin_is_parsed_from_its_raw_spelling_not_its_law_key() -> None:
    sketch = burgers_sketch(pinned=(PinnedTerm("div(u,x)", 0.5),))

    compiled = _compile(sketch)

    expected = _column(Tree(Node("/", 2, [Node("u", 0), Node("x", 0)])))
    assert compiled.pinned[0][0] == law_term_key("div(u,x)")
    torch.testing.assert_close(_pinned_column(compiled), expected, rtol=0.0, atol=0.0)


def test_ad_a_head_negation_folds_into_the_deducted_sign() -> None:
    sketch = burgers_sketch(pinned=(PinnedTerm("neg(u_x)", 0.5),))

    compiled = _compile(sketch)

    data, _context = _sga_data()
    assert compiled.pinned[0][0] == law_term_key("u_x")
    assert compiled.pinned[0][1] == pytest.approx(-0.5)
    torch.testing.assert_close(
        _pinned_column(compiled), data["u_x"], rtol=0.0, atol=0.0
    )


def test_ae_an_arithmetic_pin_composes_the_native_operators() -> None:
    compiled = _compile(burgers_sketch())

    data, _context = _sga_data()
    assert compiled.pinned_keys == frozenset({law_term_key(PINNED_ADVECTION)})
    torch.testing.assert_close(
        _pinned_column(compiled), data["u"] * data["u_x"], rtol=0.0, atol=0.0
    )


def test_af_a_power_pin_composes_the_unary_operator() -> None:
    sketch = burgers_sketch(pinned=(PinnedTerm("n2(u_x)", 0.5),))

    compiled = _compile(sketch)

    expected = _column(Tree(Node("^2", 1, [Node("u_x", 0)])))
    torch.testing.assert_close(_pinned_column(compiled), expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "term_ir",
    ["mul(u,neg(u_x))", "mul(u,recip(x))"],
    ids=["inner-neg", "recip"],
)
def test_ag_a_pin_outside_the_tree_language_is_refused_by_name(
    term_ir: str,
) -> None:
    sketch = burgers_sketch(pinned=(PinnedTerm(term_ir, 0.5),))

    with pytest.raises(ValueError) as excinfo:
        _compile(sketch)

    assert term_ir in str(excinfo.value)


def test_ah_a_pin_on_a_symbol_sga_cannot_terminate_is_refused() -> None:
    sketch = burgers_sketch(fields=("u", "v"), pinned=(PinnedTerm("v_xx", 0.5),))

    with pytest.raises(ValueError, match="v_xx"):
        _compile(sketch)







def test_ai_an_operator_free_anchor_does_not_whitelist_arithmetic() -> None:
    sketch = burgers_sketch(
        pinned=(),
        anchored=(AnchoredTerm("diff_x(u)"),),
        holes=(_hole(operators=frozenset({"mul"})),),
    )

    compiled = _compile(sketch)

    assert {name for name, _arity in compiled.ops} == {"*", "d", "d^2"}
    assert compiled.op1 == ()


def test_aj_two_pins_on_one_column_family_are_refused() -> None:
    _compile(
        burgers_sketch(
            pinned=(
                PinnedTerm(PINNED_ADVECTION, -1.0),
                PinnedTerm("u_xx", 0.1),
            )
        )
    )

    with pytest.raises(ValueError, match="alias"):
        _compile(
            burgers_sketch(
                pinned=(
                    PinnedTerm("u_xx", 0.1),
                    PinnedTerm("diff2_x(u)", 0.2),
                )
            )
        )


def test_ak_an_anchor_on_a_pinned_column_family_is_refused() -> None:
    _compile(
        burgers_sketch(
            pinned=(PinnedTerm("u_xx", 0.1),),
            anchored=(AnchoredTerm("diff_x(u)"),),
        )
    )

    with pytest.raises(ValueError, match="alias"):
        _compile(
            burgers_sketch(
                pinned=(PinnedTerm("u_xx", 0.1),),
                anchored=(AnchoredTerm("diff2_x(u)"),),
            )
        )


def test_al_an_odd_order_pin_nests_the_paired_operator_inside() -> None:
    compiled = _compile(burgers_sketch(pinned=(PinnedTerm("u_xxx", 0.3),)))

    _key, _value, tree = compiled.pinned[0]
    frozen = _column(tree)
    other = _column(
        Tree(
            Node(
                "d^2",
                2,
                [
                    Node("d", 2, [Node("u", 0), Node("x", 0)]),
                    Node("x", 0),
                ],
            )
        )
    )

    assert tree_to_kd_expr(tree) == "diff_x(diff2_x(u))"
    assert not torch.allclose(frozen, other)
