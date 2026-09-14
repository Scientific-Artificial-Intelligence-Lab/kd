## Search space

PySR is a third-party symbolic-regression package whose search is documented
by its authors. Install the optional backend with `pip install "sail-kd[pysr]"`;
it uses Julia. The KD wrapper supplies numeric columns and converts the
result into KD's equation representation.

For PDE data, `PySRConfig.terms` names the candidate columns in KD's
function-call notation. PySR combines those columns but does not differentiate
or invent another input column. Use the field and axis names in the dataset;
adding an axis does not expand the configured library. An unresolvable term
fails the fit. Compound derivatives use `u_x_t` or `diff_t(u_x)`, not `u_xt`.

A `kd.TabularDataset` selects tabular mode, where the feature columns supply
the inputs automatically. Both modes read `binary_operators` and
`unary_operators` from the configuration. These operators let the search form
nonlinear combinations beyond a linear fit of the original columns.

## Method

KD executes the term columns into a Theta matrix and gives PySR generated
placeholder names such as `c0` and `c1` that do not collide with dataset fields
or term symbols. PySR performs its own multi-population search and returns a
hall of fame with complexity, loss and expressions. One KD run contains one
batch search, rather than one KD iteration per internal PySR generation.
`generations` sets PySR's internal iteration budget.

For PDEs, conversion replaces placeholders with the terms' SymPy expressions,
expands the sum, removes each additive term's numeric factor and removes
pure constants, then serializes the remaining structure to KD IR. A fitted
constant inside a unary operation, as in `sin(mul(2, u))`, remains part of
that structure. For tables the expression stays whole, including fitted
constants and additive intercepts. An expression with no feature symbols is
rejected as a constant model.

A resumed `Model.fit(resume_from=...)` restores PySR populations and the hall
of fame and runs the new model's `generations` as additional internal
iterations. It incurs another Julia startup. The hall of fame retains its
best internal loss, while the headline expression is selected again and its
KD score can move in either direction. The parameter table gives the declared
resume tiers; changes to the input columns or operator vocabulary change the
search identity.

## Result interpretation

`model.best_score_` is KD's least-squares refit NMSE, not PySR's selection loss.
For PDEs, the refit supplies coefficients for the converted structural terms;
`best_expr_` itself is structural. For tables, PySR's fitted constants remain
inside the expression and the refit supplies one outer scale. That scale is
often near one, but the score measures the scaled expression's shape.
Refit coefficients are available on `result_.final_eval.coefficients` and
`result_.equation`.

`model.result_.pareto_front()` returns each convertible hall-of-fame entry
with its expression, PySR `complexity` and `loss`, and, for tables, its fitted
outer `scale`. Recorder series include `pareto_complexity`, `pareto_loss`,
`pareto_nmse` and `pareto_expressions`, plus `pareto_scale` for tables.
`pareto_nmse` is KD's independent re-score of each entry. PySR selects the
headline expression before KD re-scores it; the KD re-score does not select a
new headline. Unconvertible front rows are omitted, but an unconvertible
selected expression fails the run.

## References

Cranmer (2023). "Interpretable machine learning for science with PySR and SymbolicRegression.jl". arXiv:2305.01582. [Paper](https://arxiv.org/abs/2305.01582)

Documentation: [ai.damtp.cam.ac.uk/pysr](https://ai.damtp.cam.ac.uk/pysr/)

Code: [MilesCranmer/PySR](https://github.com/MilesCranmer/PySR)

<div style="font-size: 0.85em" markdown="1">

Implementation notes. KD calls the published PySR package and targets its 1.5 API: the `complexity`,
`loss` and `sympy_format` columns in `equations_`, and `model.sympy()` for the
selected expression. It does not reimplement the search. The wrapper changes
the input columns, reported score and, for PDEs, additive coefficients.
Configured operators must have a KD IR representation; a custom operator or
`^` can make conversion fail.

Additional backend settings pass unchanged through `extra_pysr_kwargs`.
A seed sets `random_state`; bitwise repeatability also needs PySR's
`deterministic=True` and serial execution. A resumed search retains the
backend's state rather than starting a fresh seeded population.

</div>
