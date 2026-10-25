"""Example 21 - Discover with a sketch: pin what you know, search the rest.

Examples 01-20 search blind: every fit starts from "any equation could be
here". Real problems rarely look like that - part of the law is often
settled physics. ``Model.fit(dataset, sketch=...)`` turns that partial
knowledge into a first-class task object, written with ``kd.Sketch``:

    pinned terms you know WITH their coefficients, as a dict such as
                {"u*u_x": -1.0}. Each is subtracted from the regression
                target before the search and restored exactly in the
                solution - never re-fitted, and the search cannot spend
                budget rediscovering it.
    anchored terms you know structurally, such as ["u_xx"]; their
                coefficients stay free, and the exit checks the final
                equation carries them.
    holes the unknown remainder, as kd.Hole objects: how many terms
                may fill it, and what shapes they may take (derivative-order
                cap, allowed operators, fields).

A term is a name (u, u_xx), a product (u*u_x), a positive integer power
(u**2) or kd's funcall IR (mul(u,u_x)). The fit takes the left-hand side, the
vocabulary and an exact match policy from the dataset;
``sketch.resolve(dataset)`` returns the complete Sketch the fit runs under,
and ``kd.core.equation`` holds the typed clauses (PinnedTerm, AnchoredTerm,
TermHole) for constraints the shorthand does not cover, such as axes.

The exit is certified: ``result_.sketch_outcome.solution`` is only published
when the discovered law satisfies every clause (see ``.verdict``) AND fits the
data on the platform's own footing within the fit's ``verify`` policy, which
defaults to ``kd.SKETCH_EXIT_VERIFY`` (NMSE <= 0.05); otherwise the run reports
the best candidate and the reason instead of overclaiming.

Here we pin the advection term of Burgers u_t = -1.0*u*u_x + 0.1*u_xx and
leave one hole of one or two terms of derivative order at most 2, on SGA,
which compiles the sketch into its own generation grammar: SGA narrows the
search's variable and operator pools at the source, and a structural
predicate rejects the composite shapes that still slip past the narrowed
pools, so the search budget concentrates on the admissible space. The same
sketch runs on PySINDy unchanged - it enforces the same three
generation-level clauses (derivative order, operator set, field/axis set) by
filtering its explicit term library, provided that library already lists the
pinned term and the candidates that may fill the hole.

Run: python examples/21_sketch_discovery.py
"""

import kd
from kd.core.expr import to_sympy
from kd.data.synthetic import generate_burgers_data

dataset = generate_burgers_data(nx=64, nt=51, nu=0.1, seed=0)

sketch = kd.Sketch(
    pinned={"u*u_x": -1.0},
    holes=[kd.Hole(max_terms=2, max_deriv_order=2)],
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
