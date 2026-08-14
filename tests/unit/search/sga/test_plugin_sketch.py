
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from kd.core.equation.signature import law_term_entry, law_term_key
from kd.core.equation.sketch import (
    AnchoredTerm,
    PinnedTerm,
    Sketch,
    TermConstraint,
    TermHole,
    constraint_admits,
)
from kd.core.evaluator import EvaluationResult
from kd.core.expr.term_features import analyze_term
from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.core.platform.sketch_compile import CompileReport, SketchClauseLevels
from kd.data.schema import PDEDataset
from kd.search.protocol import DiscoveryTask, PlatformComponents
from kd.search.sga.config import OP1, OP2, OPS, ROOT, SGAConfig
from kd.search.sga.convert import tree_to_kd_expr
from kd.search.sga.evaluate import execute_tree
from kd.search.sga.pde import PDE
from kd.search.sga.plugin import SGAPlugin
from kd.search.sga.tree import Node, Tree
from tests.unit.search._sketch_fakes import (
    PINNED_ADVECTION,
    PINNED_ADVECTION_VALUE,
    build_components,
    burgers_sketch,
    constant_field_dataset,
    default_hole,
    tiny_burgers_dataset,
)




SGA_DECLARED_LEVELS = SketchClauseLevels(
    fixed_terms="lowered",
    anchors="exit_checked",
    hole_count="exit_checked",
    derivative_order="generation_enforced",
    operator_set="generation_enforced",
    field_axis_set="generation_enforced",
)



_CONFIG = SGAConfig(num=4, width=2, depth=2, seed=0)







@pytest.fixture(scope="module")
def burgers() -> PDEDataset:
    return tiny_burgers_dataset()


def _hole(
    *, min_count: int = 1, max_count: int = 2, max_deriv_order: int | None = 2
) -> TermHole:
    return TermHole(
        id="diffusion",
        min_count=min_count,
        max_count=max_count,
        constraint=TermConstraint(max_deriv_order=max_deriv_order),
    )


def _sketch(
    *,
    pinned: tuple[PinnedTerm, ...] = (
        PinnedTerm(PINNED_ADVECTION, PINNED_ADVECTION_VALUE),
    ),
    anchored: tuple[AnchoredTerm, ...] = (),
    hole: TermHole | None = None,
) -> Sketch:
    return burgers_sketch(
        pinned=pinned,
        anchored=anchored,
        holes=(_hole() if hole is None else hole,),
    )


def _leaf(name: str) -> Node:
    return Node(name, 0)


def _op(name: str, *children: Node) -> Node:
    return Node(name, len(children), children=list(children))


def _pde(*roots: Node) -> PDE:
    return PDE(terms=[Tree(root=root) for root in roots])


def _prepared(
    dataset: PDEDataset,
    sketch: Sketch | None = None,
    *,
    config: SGAConfig | None = None,
) -> SGAPlugin:
    plugin = SGAPlugin(_CONFIG if config is None else config)
    task = None if sketch is None else DiscoveryTask.from_sketch(sketch)
    plugin.prepare(build_components(dataset, task=task, sketch_lower_owner="native"))
    return plugin


def _restored(
    dataset: PDEDataset, sketch: Sketch, *, config: SGAConfig | None = None
) -> SGAPlugin:
    cfg = _CONFIG if config is None else config
    seed = SGAPlugin(cfg)
    seed.prepare(build_components(dataset, sketch_lower_owner="native"))
    plugin = SGAPlugin(cfg)
    plugin.state = seed.state
    plugin.prepare(
        build_components(
            dataset,
            task=DiscoveryTask.from_sketch(sketch),
            sketch_lower_owner="native",
        )
    )
    return plugin


def _native_components(dataset: PDEDataset, sketch: Sketch) -> PlatformComponents:
    reqs = DerivativeReqs(provider_kind="finite_diff", max_atomic_order=2, lhs_order=1)
    return PlatformBuilder(
        dataset,
        reqs,
        task=DiscoveryTask.from_sketch(sketch),
        sketch_lower_owner="native",
    ).build()


def _offer(plugin: SGAPlugin, pde: PDE) -> tuple[object | None, int, set[str]]:
    duplicates = 0

    def bump() -> None:
        nonlocal duplicates
        duplicates += 1

    scored = plugin._dedup_and_score(pde, bump)
    return scored, duplicates, set(plugin._pde_lib)


def _column(plugin: SGAPlugin, root: Node) -> Tensor:
    return execute_tree(Tree(root=root), plugin._data_dict, plugin._diff_ctx)


def _target(plugin: SGAPlugin) -> Tensor:
    assert plugin._y is not None
    return plugin._y


def _admits(sketch: Sketch, pde: PDE) -> bool:
    pinned = {law_term_key(pin.term_ir) for pin in sketch.pinned}
    pin_families = set()
    for pin in sketch.pinned:
        features = analyze_term(pin.term_ir, sketch.vocabulary)
        pin_families.add(
            (
                features.base_fields,
                features.coordinate_dependencies,
                features.derivative_multiindices,
                features.operators,
            )
        )
    anchored = {law_term_key(anchor.term_ir) for anchor in sketch.anchored}
    for tree in pde.terms:
        try:
            key = law_term_key(tree_to_kd_expr(tree))
            features = analyze_term(key, sketch.vocabulary)
        except ValueError:
            return False
        if key in pinned:
            return False
        family = (
            features.base_fields,
            features.coordinate_dependencies,
            features.derivative_multiindices,
            features.operators,
        )
        if family in pin_families:
            return False
        if key in anchored:
            continue
        if not any(
            constraint_admits(hole.constraint, features) for hole in sketch.holes
        ):
            return False
    return True







def test_the_plugin_claims_ownership_of_its_own_lower() -> None:
    assert SGAPlugin.sketch_lower_owner == "native"


def test_the_compile_report_is_published_on_the_runner_seam(
    burgers: PDEDataset,
) -> None:
    report = _prepared(burgers, _sketch()).sketch_compile_report

    assert isinstance(report, CompileReport)
    assert report.levels == SGA_DECLARED_LEVELS


def test_the_report_seam_is_empty_before_prepare() -> None:
    assert SGAPlugin(_CONFIG).sketch_compile_report is None


def test_a_task_free_run_publishes_no_report(burgers: PDEDataset) -> None:
    assert _prepared(burgers).sketch_compile_report is None







def test_without_a_task_the_pools_stay_the_module_constants(
    burgers: PDEDataset,
) -> None:
    plugin = _prepared(burgers)

    assert plugin._ops is OPS
    assert plugin._root is ROOT
    assert plugin._op1 is OP1
    assert plugin._op2 is OP2


def test_a_task_rebinds_the_pools_to_the_narrowed_ones(
    burgers: PDEDataset,
) -> None:
    plugin = _prepared(burgers, _sketch(hole=_hole(max_deriv_order=1)))

    assert ("d^2", 2) not in plugin._ops
    assert ("d", 2) in plugin._ops


def test_the_task_free_search_is_unchanged_under_a_fixed_seed(
    burgers: PDEDataset,
) -> None:
    first = _prepared(burgers)
    second = _prepared(burgers)

    assert first.propose(4) == second.propose(4)
    assert first.best_expression == second.best_expression







def test_the_pins_are_deducted_from_sgas_own_target(
    burgers: PDEDataset,
) -> None:
    raw = _prepared(burgers)
    lowered = _prepared(burgers, _sketch())

    column = _column(lowered, _op("*", _leaf("u"), _leaf("u_x")))

    torch.testing.assert_close(
        _target(lowered),
        _target(raw) - PINNED_ADVECTION_VALUE * column,
        rtol=0.0,
        atol=0.0,
    )


def test_re_preparing_does_not_deduct_the_pins_twice(
    burgers: PDEDataset,
) -> None:
    plugin = _prepared(burgers, _sketch())
    once = _target(plugin).detach().clone()

    plugin.prepare(
        build_components(
            burgers,
            task=DiscoveryTask.from_sketch(_sketch()),
            sketch_lower_owner="native",
        )
    )

    torch.testing.assert_close(_target(plugin), once, rtol=0.0, atol=0.0)


def test_the_result_target_is_the_deducted_one(burgers: PDEDataset) -> None:
    plugin = _prepared(burgers, _sketch())

    torch.testing.assert_close(
        plugin.build_result_target(), _target(plugin), rtol=0.0, atol=0.0
    )


def test_a_head_negated_pin_deducts_the_same_column(
    burgers: PDEDataset,
) -> None:
    folded = _prepared(
        burgers,
        _sketch(pinned=(PinnedTerm(f"neg({PINNED_ADVECTION})", 1.0),)),
    )
    plain = _prepared(burgers, _sketch())

    assert law_term_entry(f"neg({PINNED_ADVECTION})", 1.0)[1] == pytest.approx(
        PINNED_ADVECTION_VALUE
    )
    torch.testing.assert_close(_target(folded), _target(plain), rtol=0.0, atol=0.0)


def test_a_closed_sketch_leaves_the_target_untouched(
    burgers: PDEDataset,
) -> None:
    raw = _prepared(burgers)
    closed = _prepared(burgers, burgers_sketch(holes=()))

    torch.testing.assert_close(_target(closed), _target(raw), rtol=0.0, atol=0.0)


def test_an_effectively_closed_open_sketch_is_refused() -> None:
    sketch = burgers_sketch(pinned=(PinnedTerm("u", 1.0),), holes=(default_hole(),))
    components = _native_components(constant_field_dataset(), sketch)

    with pytest.raises(ValueError, match="variance"):
        SGAPlugin(_CONFIG).prepare(components)


def test_a_pinned_default_column_is_dropped_for_the_run(
    burgers: PDEDataset,
) -> None:
    plugin = _prepared(burgers, _sketch(pinned=(PinnedTerm("u", 0.5),)))

    assert plugin._default_terms is None
    assert plugin._default_term_name is None


def test_a_hole_admitted_default_column_is_kept(burgers: PDEDataset) -> None:
    plugin = _prepared(burgers, _sketch())

    assert plugin._default_terms is not None
    assert plugin._default_term_name == "u"







def test_an_over_order_candidate_is_rejected_at_the_offspring_sink(
    burgers: PDEDataset,
) -> None:
    plugin = _prepared(burgers, _sketch())

    scored, duplicates, library = _offer(
        plugin,
        _pde(_op("d", _op("d^2", _leaf("u"), _leaf("x")), _leaf("x"))),
    )

    assert scored is None
    assert duplicates == 0
    assert library == set()


def test_an_in_order_candidate_passes_the_offspring_sink(
    burgers: PDEDataset,
) -> None:
    plugin = _prepared(burgers, _sketch())

    scored, _duplicates, _library = _offer(
        plugin, _pde(_op("d^2", _leaf("u"), _leaf("x")))
    )

    assert scored is not None


def test_regenerating_a_pinned_law_is_rejected(burgers: PDEDataset) -> None:
    plugin = _prepared(burgers, _sketch())

    scored, duplicates, library = _offer(
        plugin, _pde(_op("*", _leaf("u"), _leaf("u_x")))
    )

    assert scored is None
    assert duplicates == 0
    assert library == set()


def test_a_free_product_term_is_not_mistaken_for_the_pin(
    burgers: PDEDataset,
) -> None:
    plugin = _prepared(burgers, _sketch())

    scored, _duplicates, _library = _offer(
        plugin, _pde(_op("*", _leaf("u"), _leaf("x")))
    )

    assert scored is not None


def test_alias_spellings_count_as_one_law_key(burgers: PDEDataset) -> None:
    plugin = _prepared(burgers, _sketch())

    scored, _duplicates, _library = _offer(
        plugin,
        _pde(
            _op("*", _leaf("u"), _leaf("x")),
            _op("*", _leaf("x"), _leaf("u")),
        ),
    )

    assert scored is not None


def test_capacity_counts_distinct_selected_keys_per_hole(
    burgers: PDEDataset,
) -> None:
    plugin = _prepared(burgers, _sketch(hole=_hole(max_count=1)))
    basis = ["diff2_x(u)", "mul(u, x)"]

    def result(selected: list[int]) -> EvaluationResult:
        return EvaluationResult(
            mse=0.0, nmse=0.0, r2=1.0, selected_indices=selected
        )

    assert plugin._selected_within_capacity(result([0]), terms=basis)
    assert not plugin._selected_within_capacity(result([0, 1]), terms=basis)


def test_the_capacity_gate_reads_the_aligned_basis_not_the_payload(
    burgers: PDEDataset,
) -> None:
    plugin = _prepared(burgers, _sketch(hole=_hole(max_count=1)))
    overflow = EvaluationResult(
        mse=0.0,
        nmse=0.0,
        r2=1.0,
        terms=None,
        selected_indices=[0, 1],
    )

    assert not plugin._selected_within_capacity(
        overflow, terms=["diff2_x(u)", "mul(u, x)"]
    )


def test_the_default_column_counts_only_when_selected(
    burgers: PDEDataset,
) -> None:
    plugin = _prepared(burgers, _sketch(hole=_hole(max_count=1)))
    basis = ["u", "diff2_x(u)"]

    def result(selected: list[int]) -> EvaluationResult:
        return EvaluationResult(
            mse=0.0, nmse=0.0, r2=1.0, selected_indices=selected
        )

    assert plugin._selected_within_capacity(result([1]), terms=basis)
    assert not plugin._selected_within_capacity(result([0, 1]), terms=basis)


def test_an_anchored_key_needs_no_hole_seat(burgers: PDEDataset) -> None:
    plugin = _prepared(
        burgers,
        _sketch(hole=_hole(max_count=1), anchored=(AnchoredTerm("diff2_x(u)"),)),
    )

    scored, _duplicates, _library = _offer(
        plugin, _pde(_op("d^2", _leaf("u"), _leaf("x")))
    )

    assert scored is not None


def test_only_anchored_keys_are_admissible_without_holes(
    burgers: PDEDataset,
) -> None:
    sketch = burgers_sketch(anchored=(AnchoredTerm("diff2_x(u)"),), holes=())
    plugin = _restored(burgers, sketch)

    anchored_scored, _dup, library_after_accept = _offer(
        plugin, _pde(_op("d^2", _leaf("u"), _leaf("x")))
    )
    free_scored, duplicates, library_after_reject = _offer(
        plugin, _pde(_op("*", _leaf("u"), _leaf("x")))
    )

    assert anchored_scored is not None
    assert free_scored is None
    assert duplicates == 0


    assert library_after_reject == library_after_accept


def test_a_pinned_alias_spelling_is_rejected_at_the_sink(
    burgers: PDEDataset,
) -> None:
    plugin = _prepared(
        burgers,
        burgers_sketch(
            pinned=(PinnedTerm("u_xx", 0.05),),
            holes=(
                TermHole(
                    id="advection",
                    min_count=1,
                    max_count=2,
                    constraint=TermConstraint(max_deriv_order=1),
                ),
            ),
        ),
    )

    scored, duplicates, library = _offer(
        plugin, _pde(_op("d^2", _leaf("u"), _leaf("x")))
    )

    assert scored is None
    assert duplicates == 0
    assert library == set()


def test_a_generation_the_sketch_rejects_wholesale_degrades_to_empty(
    burgers: PDEDataset,
) -> None:
    plugin = _restored(
        burgers,
        burgers_sketch(anchored=(AnchoredTerm("mul(u, diff2_x(u))"),), holes=()),
        config=SGAConfig(num=4, width=2, depth=2, seed=0, p_mute=0.0),
    )

    batch = plugin.propose(4)
    plugin.update(plugin.evaluate(batch))

    assert batch == []
    assert plugin._population







def test_every_seeded_individual_satisfies_the_sketch(
    burgers: PDEDataset,
) -> None:
    sketch = _sketch()
    plugin = _prepared(burgers, sketch)

    population = plugin._population
    assert population
    assert all(_admits(sketch, pde) for pde in population)
