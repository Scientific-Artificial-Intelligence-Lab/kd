## Search space

SGA-PDE constructs an equation from expression trees over field values, spatial
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

## Method

A population contains candidate equations, each a set of up to `width` trees
of depth at most `depth`. Leaves are the field, spatial coordinates and
first-order spatial derivatives. Internal nodes are addition, subtraction,
multiplication, division, square, cube and the first- and second-derivative
operators. Tree roots exclude addition and subtraction because the term list
already represents a sum.

Each generation applies crossover, evaluation and truncation, then mutation
and replacement followed by another evaluation and truncation. Duplicate
expressions are filtered by their expression keys. The evaluator executes each
term into a column, removes non-finite and all-zero columns, prepends the field
column, and fits coefficients with an internal STRidge tolerance sweep. An OLS
baseline is followed by adaptive `d_tol` steps, bounded by `maxit`, with
`str_iters` ridge iterations and column normalization of order 2. The fit's AIC
ranks candidates for selection. One generation is one KD iteration, controlled
by `generations`.

## Result interpretation

`model.best_score_` combines fit error and support size through
`2 * k * aic_ratio + 2 * ln(MSE)`. NMSE alone therefore does not determine which
candidate is preferred. AIC ranks candidates on the same dataset; NMSE is
measured on SGA's own grid and derivative basis, which can differ from a refit
using another derivative provider.

`model.best_expr_` carries SGA's sparse-fit coefficients. The structured terms
and coefficients are also available on `model.result_.equation`. KD exports
function-call notation: `add`, `sub`, `mul`, `div`, `n2`, `n3`,
`diff_<axis>(...)`, `diff2_<axis>(...)` and terminals such as `u_x`. Repeated
and composed derivatives may have equivalent spellings, such as `diff2_x(u)`,
`diff_x(u_x)` and `diff_x(diff_x(u))`; compare their mathematical meaning rather
than their text. Compound derivative terminals use underscores, such as
`u_x_t`, rather than the ambiguous `u_xt`.

The recorder exposes `best_aic`, `gen_best_aic`, `gen_best_nmse`,
`gen_mean_aic`, `gen_mean_complexity`, `n_valid` and `n_unique`. The `gen_*`
series describe newly evaluated offspring; `pop_mean_aic` describes the
surviving population. A flat best-score curve records a lack of improvement
on that run, and does not establish whether the structure is correct.

## References

Chen et al. (2022). "Symbolic genetic algorithm for discovering open-form partial differential equations (SGA-PDE)". *Phys. Rev. Research* 4, 023174. [Paper](https://doi.org/10.1103/PhysRevResearch.4.023174) · [arXiv:2106.11927](https://arxiv.org/abs/2106.11927)

Code: [YuntianChen/SGA-PDE](https://github.com/YuntianChen/SGA-PDE)

<div style="font-size: 0.85em" markdown="1">

Implementation notes. KD's SGA was ported from an earlier implementation of SGA-PDE. It preserves
the paper's three-tree construction at the standard width and STRidge column
normalization (`normalize=2`), with crossover pairing and the two-stage
selection order checked against that implementation. Protected division keeps
mixed-zero divisors finite. The tree finite-difference implementation uses a
five-point fourth-order stencil, with a three-point stencil on shorter arrays,
where the reference used three-point second-order differences. Zero or
non-finite MSE, and fits with empty support, receive infinite AIC.

Only first-order derivative terminals are precomputed; higher orders come
from tree composition. There is no literal-zero terminal or `param_fields`
injection for non-differentiable parameter fields. The term executor and data
layer have different boundary stencils, so a precomputed derivative and its
tree-composed equivalent can differ at boundaries.

</div>
