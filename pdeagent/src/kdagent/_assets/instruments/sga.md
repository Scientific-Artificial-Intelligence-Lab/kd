## Search space

In its PDE modes, SGA constructs an equation from expression trees over field values, spatial
coordinates and derivatives. It fits coefficients during the search and needs
no candidate-term library. A candidate has at most `width` searched trees plus
the field itself, which is always included as a default column; the fitted law
can therefore contain `width + 1` terms. A law containing only that field is a
valid outcome, while an empty law is excluded.

The left-hand-side axis is excluded from derivative denominators, precomputed
terminals and coordinate leaves. The search space contains spatial derivatives
and explicit spatial dependence. Mixed space-time derivatives, higher time
derivatives and explicit time dependence require a different grammar. A term
that still differentiates along the left-hand-side axis is discarded during
execution. A derivative terminal that cannot be computed is removed from the
run's vocabulary with a warning.

With autograd enabled, a field surrogate is trained before the search. It
supplies the first-order derivative terminals; raw field leaves still come
from the observations, and derivative operators inside trees still use SGA's
finite differences. Surrogate training adds a separate cost before the first
generation. SGA needs no optional package, pretrained weights or endpoint.

The tabular mode searches expressions over the feature columns, the constant
`one` and two optional vocabulary fields. `functions` adds unary functions,
any of `sin`, `cos`, `exp`, `log` and `sqrt`; `constants` adds numeric leaves,
which must be finite, distinct and different from 0 and 1. Both are empty by
default, which leaves arithmetic, squares, cubes and rational expressions; the
PDE modes refuse a non-empty value of either. The target column is excluded
from candidates, and no field column is prepended automatically. Bare features
and `one` can be complete terms; spatial derivative operators are absent.

A candidate containing an algebraically identically zero term or
subexpression, including a zero denominator, is rejected. This also excludes
zero expressions whose protected numerical values could be amplified by fitted
coefficients. The rule concerns the expression itself, not a variable
denominator that happens to vanish at an observed sample. A function whose
argument holds no input feature after rational reduction, such as `log(one)`,
`sin(6.283185307179586)` or `cos(div(x, x))`, makes the whole tree
inadmissible. Its value is a number: `log(one)` is zero, and the `sin` example
is about 1e-16, which protected division turns into a large column. Offspring
refused by these rules are scored invalid without a fit. Initial draws refused by
them are drawn again; when 500 draws for one population slot are refused,
initialization stops with an error that names the vocabulary. A vocabulary
with many constants and few features can reach that bound.

Before fitting a table, SGA cancels common algebraic factors and expands
polynomial roots into independently fitted monomial columns. Reduced rational
roots remain whole columns, with their numerator and denominator scales
normalized. Exact duplicate or linearly dependent functions are removed without
using correlations between sampled values. The existing `width` bounds both
the number of searched trees and the number of independent fitted columns;
a candidate exceeding the latter bound is rejected before fitting.

A function node enters this reduction as one symbol named by its reduced
argument. The argument is brought to lowest terms, with a primitive integer
denominator whose first term is positive, and keeps its rational multiple,
because no fitted coefficient outside the function can absorb it:
`sin(mul(x, x))` and `sin(n2(x))` are one column, while `sin(add(x, x))` and
`sin(x)` are two, as are `exp(sub(one, x))` and `exp(sub(x, one))`. Repeated
symbols are recognized, so `sub(sin(x), sin(x))` is a zero subexpression and
`mul(sin(x), sin(x))` is the square of one symbol. Identities between
functions are not recognized: `sin(x)**2 + cos(x)**2` keeps both squares as
columns, and `mul(exp(x), exp(y))` and `exp(add(x, y))` are different symbols.
Columns that agree in value without being algebraically equal are left to
STRidge's column normalization and threshold.

Each table constant enters the reduction as the exact rational value of its
float, so equal floats give one column. In the polynomial part of a term a
constant folds into a monomial's coefficient, which the fitted coefficient
absorbs: `mul(2.5, x)` and `x` are one column. A constant changes the basis
only inside a function argument or in a denominator that does not cancel, as in
`sin(mul(2.5, x))` or `div(one, add(one, mul(2.5, x)))`. Constants are not
fitted; a value outside the table is reachable only through arithmetic of
table values and `one` within the depth bound. pi and e enter as their float
values.

The canonical columns define both numerical fitting and the returned equation.
Cancellation extends removable factors to their simplified values: `x/x`
becomes `one`, including at `x=0`. Denominators that remain after cancellation
use protected division, which adds 1e-10 with the sign of the denominator, so
they stay finite at their zeros; squares and cubes clamp their argument to
[-1e6, 1e6]. The functions use their ordinary definitions, not the protected
`exp` and `log` of KD's default registry: a column that is not finite on the
table, such as `log` or `sqrt` of a negative value or an `exp` that overflows,
is removed and the candidate's other columns are fitted. Genetic operations
continue to act on the original trees.

For ODE discovery, prepare time derivatives before fitting the table. KD's
trajectory preprocessing computes finite differences separately on each
trajectory, or fits one multi-output neural network per trajectory and obtains
time derivatives by automatic differentiation. Neural preprocessing can keep
observed states as the features or use the network's state predictions; both
choices share the same derivative targets. Each state equation is a separate
scalar fit using all state features and any supplied trajectory parameters.
Prepared tables can be reused across targets, algorithms and search seeds.
Neural training belongs to preprocessing and is not repeated by a tabular
search, checkpoint resume or restart. PDE surrogate settings are therefore
refused when fitting a table; configure neural derivatives in preprocessing.

## Method

A population contains candidate equations, each a set of up to `width` trees
of depth at most `depth`. In the PDE modes, leaves include every field, spatial coordinates and each field's first-order
derivatives along every non-target axis. Only the left-hand-side field is
prepended as the default regression column. Internal nodes are addition, subtraction,
multiplication, division, square, cube and the first- and second-derivative
operators. Tree roots exclude addition and subtraction because the term list
already represents a sum. In tabular mode, leaves are the feature columns,
`one` and the configured constants, and internal nodes are the same
operators without the derivative operators. The configured functions join the
operator, root and unary-mutation pools and the constants join the leaf pool,
each in the order given, so reordering `functions` or `constants` changes the
random draws. A constant is never a tree root.

Each generation applies crossover, evaluation and truncation, then mutation
and replacement followed by another evaluation and truncation. Duplicate
expressions are filtered by their expression keys. The evaluator uses searched
trees plus the default field column in PDE modes, and canonical columns in
tabular mode. It removes non-finite and all-zero columns and fits coefficients
with an internal STRidge tolerance sweep. An OLS
baseline is followed by adaptive `d_tol` steps, bounded by `maxit`, with
`str_iters` ridge iterations and column normalization of order 2. The fit's AIC
ranks candidates for selection. One generation is one KD iteration, controlled
by `generations`.

In tabular mode, every tolerance comparison and candidate score uses only the
selected terms. The same coefficients define the returned equation, prediction
and residuals, including selected terms with small coefficients. This mode uses
the existing STRidge solver and activity threshold. Its AIC term count is the
number of selected canonical columns, including a selected constant; table
constants are structural choices and are not counted. The AIC formula takes
`max(MSE, F)` in place of the MSE, with
`F = max(tiny, (64 * n * eps)**2 * mean(y**2))` for `n` rows, target column `y`
and the float64 `tiny` and `eps`, so an exact fit on noise-free data has a
finite score. There is no lower AIC bound in this mode, and a support with at
least as many selected columns as rows is invalid. A support in which two
selected columns differ only in numeric literals, such as `cos(mul(1.5, x))`
and `cos(mul(0.5, x))`, is also invalid: the REGRESSION equation identifies a
term by its structure without literals and cannot hold both. Without functions
or constants this concerns rational columns such as `div(one, add(2, x))` and
`div(one, add(3, x))`. Exact algebraic compilation adds search cost and is
reused through a bounded structural cache.

With `restart_interval` set to k, the population is re-drawn once k
generations have been committed since the last draw. The re-draw happens when
the next generation's proposal starts, so no draw follows the last generation
and a value of at least `generations` leaves the run unchanged; `generations`
remains the total across segments, and initial draws are not counted as
evaluations. Restart j (j = 1, 2, ...) re-seeds the random generator with
`seed + 1000 * j`, clears the history of expressions already seen, and draws a
new population as at the start of a run. Before each re-draw, the best-ranked
candidate of the finished segment replaces the stored best of earlier segments
only if its AIC is strictly lower. The reported equation is the better of the
stored best and the current population's best-ranked candidate, with ties
going to the earlier segment. With finite differences, each segment reproduces
an independent run of k generations at its seed. With autograd, the field
surrogate trained before the first generation is shared by all segments, so
the segments are not independent runs at those seeds.

## Result interpretation

`model.best_score_` combines fit error and support size through
`2 * k * aic_ratio + 2 * ln(MSE)`, with `max(MSE, F)` in tabular mode (see
Method). NMSE alone therefore does not determine which
candidate is preferred. AIC ranks candidates on the same dataset; NMSE is
measured on SGA's own fitting domain. For a PDE this is its grid and derivative
basis, which can differ from a refit using another derivative provider. For a
table it is the fixed target column and the selected observed or surrogate
state features. Low training residuals alone do not establish ODE recovery;
the coefficients and trajectories generated by the complete discovered system
also need evaluation on independent initial conditions.

`model.best_expr_` carries SGA's sparse-fit coefficients. The structured terms
and coefficients are also available on `model.result_.equation`. KD exports
function-call notation: `add`, `sub`, `mul`, `div`, `n2`, `n3`,
`diff_<axis>(...)`, `diff2_<axis>(...)` and terminals such as `u_x`. Repeated
and composed derivatives may have equivalent spellings, such as `diff2_x(u)`,
`diff_x(u_x)` and `diff_x(diff_x(u))`; compare their mathematical meaning rather
than their text. Compound derivative terminals use underscores, such as
`u_x_t`, rather than the ambiguous `u_xt`. Tabular terms can also contain the
configured function names and numeric literals from the canonical reduction,
written as an integer or a float.

Tabular results carry a regression equation whose left-hand side is the target
column name. An ODE derivative target is still a regression column at this
interface; its time-derivative provenance belongs to the prepared data artifact.
The search does not infer a differential-equation form from the target's name.
Checkpoints bind the mode and feature vocabulary as well as the input identity.
Tabular checkpoints also bind feature order, the canonical scoring policy and
the vocabulary; cached scores from an older policy cannot be resumed under a
different fitted basis. Preserve the feature order when resuming a table
search. A different `functions` or `constants` starts a new lineage, so a run
cannot be resumed under another vocabulary.

The recorder exposes `best_aic`, `gen_best_aic`, `gen_best_nmse`,
`gen_mean_aic`, `gen_mean_complexity`, `n_valid` and `n_unique`. The `gen_*`
series describe newly evaluated offspring; `pop_mean_aic` describes the
surviving population. A flat best-score curve records a lack of improvement
on that run, and does not establish whether the structure is correct.

## References

Chen et al. (2022). "Symbolic genetic algorithm for discovering open-form partial differential equations (SGA-PDE)". *Phys. Rev. Research* 4, 023174. [Paper](https://doi.org/10.1103/PhysRevResearch.4.023174) · [arXiv:2106.11927](https://arxiv.org/abs/2106.11927)

Code: [YuntianChen/SGA-PDE](https://github.com/YuntianChen/SGA-PDE)

<div style="font-size: 0.85em" markdown="1">

Implementation notes. KD's SGA was ported from an earlier implementation of SGA-PDE. For PDE data, it preserves
the paper's three-tree construction at the standard width and STRidge column
normalization (`normalize=2`), with crossover pairing and the two-stage
selection order checked against that implementation. Protected division keeps
mixed-zero divisors finite. The tree finite-difference implementation uses a
five-point fourth-order stencil, with a three-point stencil on shorter arrays,
where the reference used three-point second-order differences. Non-finite MSE
and fits with empty support receive infinite AIC, and so does zero MSE in the
PDE modes. The tabular mode, with its function vocabulary, constant table and
MSE floor, is a KD addition; the reference implementation fits PDEs only.

Only first-order derivative terminals are precomputed; higher orders come
from tree composition. There is no literal-zero terminal or `param_fields`
injection for non-differentiable parameter fields. The term executor and data
layer have different boundary stencils, so a precomputed derivative and its
tree-composed equivalent can differ at boundaries.

</div>
