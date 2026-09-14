"""Example 21 - Discover with a sketch: pin what you know, search the rest.

Examples 01-20 search blind: every fit starts from "any equation could be
here". Real problems rarely look like that - part of the law is often
settled physics. ``Model.fit(dataset, sketch=...)`` turns that partial
knowledge into a first-class task object:

    PinnedTerm a term you know WITH its coefficient. It is subtracted
                    from the regression target before the search and restored
                    exactly in the solution - never re-fitted, and the search
                    cannot spend budget rediscovering it.
    AnchoredTerm a term you know structurally; its coefficient stays free,
                    and the exit checks the final equation carries it.
    TermHole the unknown remainder: how many terms may fill it, and
                    what shapes they may take (derivative-order cap, allowed
                    operators, fields, axes).

The exit is certified: ``result_.sketch_outcome.solution`` is only published
when the discovered law satisfies every clause (see ``.verdict``) AND fits the
data on the platform's own footing within the fit's ``verify`` policy, which
defaults to ``kd.SKETCH_EXIT_VERIFY`` (NMSE <= 0.05); otherwise the run reports
the best candidate and the reason instead of overclaiming.

Here we pin the advection term of Burgers u_t = -1.0*u*u_x + 0.1*u_xx and
leave one order-2 hole, on SGA - one of the two backends with their own sketch
compiler (PySINDy is the other): SGA narrows the search's variable and operator
pools at the source, and a structural predicate rejects the composite shapes
that still slip past the narrowed pools, so the search budget concentrates on
the admissible space. The same Sketch object runs on PySINDy unchanged - it
enforces the same three generation-level clauses (derivative order, operator
set, field/axis set) by filtering its explicit term library, provided that
library already lists the pinned term and the candidates that may fill the hole.
narrows the search's variable and operator pools at the source, and a
structural predicate rejects the composite shapes that still slip past the
narrowed pools, so the search budget concentrates on the admissible space.

Run: python examples/21_sketch_discovery.py
"""

import kd
from kd.core.expr import to_sympy
from kd.core.equation import (
    LhsSpec,
    PinnedTerm,
    Sketch,
    SketchMatchPolicy,
    TermConstraint,
    TermHole,
)
from kd.core.expr import TermVocabulary
from kd.data.synthetic import generate_burgers_data

dataset = generate_burgers_data(nx=64, nt=51, nu=0.1, seed=0)

sketch = Sketch(
    lhs_spec=LhsSpec("u", "t", 1),
    vocabulary=TermVocabulary(
        fields=frozenset({"u"}), coordinates=frozenset({"x", "t"})
    ),
    pinned=(PinnedTerm("mul(u,u_x)", -1.0),),
    anchored=(),
    holes=(
        TermHole(
            id="diffusion",
            min_count=1,
            max_count=2,
            constraint=TermConstraint(max_deriv_order=2),
        ),
    ),
    match_policy=SketchMatchPolicy(
        coeff_atol=1e-9, coeff_rtol=1e-9, support_threshold=0.0
    ),
)

model = kd.Model(
    algorithm="sga",
    config=kd.SGAConfig(num=6, depth=2, width=3, seed=0),
    generations=2,
    verbose=False,
)
model.fit(dataset, sketch=sketch)

result = model.result_
assert result is not None
outcome = result.sketch_outcome
assert outcome is not None

print("Sketch: u_t = -1.0*u*u_x + [1..2 terms, order <= 2]")
print("Truth: u_t = -1.0*u*u_x + 0.1*u_xx")
if outcome.solution is not None:
    print(f"Discovered: {outcome.solution}")
    print(f"Certified: {outcome.solution is not None}")
    print(" (= every sketch clause was honoured AND the whole law")
    print(" fits the data on the platform footing, within the")
    print(" SKETCH_EXIT_VERIFY threshold; it is still NOT a claim")
    print(" that the pinned coefficient was the right physics)")
    if outcome.full_verify is not None:
        print(f"Fit NMSE: {outcome.full_verify.nmse:.3e}")
    print("Pinned coefficient restored exactly:")
    for term_ir, coefficient in outcome.solution.terms:
        print(f" {to_sympy(term_ir)}: {coefficient.value:+.6f}")
else:
    print(f"Not certified: {outcome.failure or 'verdict False'}")
    print("Best candidate and per-clause verdict stay available:")
    print(f" {outcome.verdict}")
