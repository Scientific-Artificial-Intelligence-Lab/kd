## Search space

The KD wrapper uses PySINDy's sparse optimizer over a supplied term library.
`PySINDyConfig.terms` contains KD function-call strings using the dataset's
field and axis names. Terms can contain coordinates, products such as
`mul(u, u_x)`, and operators such as `sin`, `exp` or binary `div`. A compound
derivative uses `u_x_t` or `diff_t(u_x)` rather than `u_xt`. Supply terms for
all relevant axes and fields; dataset dimensions do not expand the library.

Use uniformly spaced axes with at least five points per axis for the
finite-difference stencil. Any number of spatial axes can supply columns. Atomic derivative names support orders through three per axis;
compound and open-form derivatives are computed during execution. For example,
`u_xxxx` exceeds the atomic-order cap, while `diff_x(u_xxx)` uses composition.
`kd.validate_terms(dataset, terms)` checks the library before the solve. A
non-uniform axis or an unresolvable term fails explicitly. Smoothing noisy
observations before differentiation can improve the columns used for support
selection.

Numeric literals are rejected in the term catalog and no intercept column is
added. The fitted combination is restricted to the supplied columns, so a
missing governing term can still produce a valid but incomplete fit.

## Method

KD supplies the derivatives, candidate columns and single regression target;
PySINDy supplies `pysindy.optimizers.STLSQ`. Atomic derivative names determine
the precomputed orders on each axis. Compound derivatives and open-form calls
are evaluated as expressions. The finite-difference provider uses fourth-order
interior stencils for first and second derivatives, with separate boundary
and third-derivative stencils.

STLSQ alternates a ridge solve with hard thresholding. Coefficients below
`threshold` in magnitude are zeroed and their columns removed, until support
stabilizes or `max_iter` rounds have run. With `unbias` enabled, an
unregularized least-squares solve on the retained support removes ridge
shrinkage. `normalize_columns` changes the scale at which thresholding acts,
so choose normalization and threshold together.

One sparse-regression solve answers the run; `generations` does not create
additional solves and the seed does not randomize the optimizer. Equivalent
catalog spellings can still produce collinear numeric columns: canonical
syntax catches duplicate terms but does not equate `u_xx` and `diff2_x(u)`.
Collinear columns can split a coefficient and change threshold selection.

## Result interpretation

The returned coefficients are the optimizer's own, with no intervening KD
refit. `model.best_score_` is NMSE evaluated with those coefficients.
`model.best_expr_` lists the structural terms; read the coefficient vector on
`model.result_.final_eval.coefficients` or the terms and coefficients on
`model.result_.equation`. The full evaluated catalog retains zero coefficients
for dropped terms and records its active support and fitted left-hand side.

Interpret NMSE with the candidate library and threshold. A small value does
not establish that the library contains the governing equation. If all
coefficients are removed, the fit raises an empty-support error naming the
threshold; reduce the threshold to retain a support. Repeating an otherwise
unchanged deterministic solve does not explore another structure. A near-zero
residual obtained by putting the target derivative itself in the library is
a tautology.

## References

de Silva et al. (2020). "PySINDy: A Python package for the sparse identification of nonlinear dynamical systems from data". *J. Open Source Softw.* 5(49), 2104. [Paper](https://doi.org/10.21105/joss.02104)

Kaptanoglu et al. (2022). "PySINDy: A comprehensive Python package for robust sparse system identification". *J. Open Source Softw.* 7(69), 3994. [Paper](https://doi.org/10.21105/joss.03994)

Documentation: [pysindy.readthedocs.io/en/latest](https://pysindy.readthedocs.io/en/latest/)

Code: [dynamicslab/pysindy](https://github.com/dynamicslab/pysindy)

<div style="font-size: 0.85em" markdown="1">

Implementation notes. KD uses the native STLSQ optimizer. It does not use PySINDy's `SINDy` model,
`PDELibrary`, differentiation, weak formulation, ensembling or multi-target
fit. A nonzero optimizer intercept is rejected because the KD term matrix has
no intercept column. The typed settings follow STLSQ's defaults; other settings,
such as ridge `alpha`, pass through `extra_optimizer_kwargs`. That mapping
rejects names already represented by typed fields rather than overriding them.
Threshold and iteration bounds are validated before computation. Integration
checks compare the plugin with a direct STLSQ solve on identical Theta and
target arrays, including their coefficient vectors.

</div>
