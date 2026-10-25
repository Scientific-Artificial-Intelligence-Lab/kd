## Search space

LLM4ED asks a language model for coefficient-free equations over a fixed symbol
library and fits their coefficients by sparse regression. Its PDE mode
operates on one scalar field with one time axis and spatial axes `x`, `y` or
both. The symbols are the field `u` and, for each spatial axis present, its
coordinate and the derivatives along it: `x`, `u_x`, `u_xx`, `u_xxx` and `y`,
`u_y`, `u_yy`, `u_yyy`.
The prompts specify `+`, `-`, `*`, `/`, squares and cubes; the parser also
accepts integer powers through five. Explicit spatial dependence is
expressible. Extra fields, other-axis derivatives, mixed space-time derivatives,
higher spatial derivatives, trigonometric functions, exponentials, square roots
and standalone constants are outside this route's symbol library.
Extra fields and spatial axes other than `x` and `y` are refused before proposal
generation.

Operand construction needs at least four spatial and three time points.
Spacing is taken from `x[2] - x[1]` and `t[1] - t[0]`, so use uniform axes.
The method itself does not smooth or train a surrogate; the platform's
`savgol_window` applies before differentiation. Higher derivatives therefore
depend on measurement noise and resolution.

### Tabular regression

Tabular data selects the tabular mode, without a `mode` parameter. The symbols
are the table's feature columns, and the target column is the left-hand side;
no derivatives are computed. The prompts offer `+`, `-`, `*`, `/`, squares and
cubes and the functions named in `functions`, any of `sin`, `cos`, `exp`, `log`
and `sqrt`; the default is none, and the PDE mode refuses a non-empty value.
The parser accepts integer powers through five, and with `sqrt` a half-integer
power of magnitude at most five, such as `x**(3/2)`, reads as an integer power
of `sqrt(x)`. A feature or target named `c1`, `c2`, ... is refused, and so is a
name the SymPy parser does not read back as one plain symbol.

The prompts ask for each constant inside a term as a placeholder `c1`, `c2`,
..., pi and e included; integers may be written directly, and the prompts ask
for no decimal numbers. The parser keeps a placeholder wherever it appears and
reads a decimal like any other number. A nonzero numeric term, or a term of
placeholders alone (`c1`, `2*c1`, `sin(c1)`), is an intercept column on `one`
whose value is the fitted coefficient; several such terms merge into one
intercept. A term whose value is a non-numeric constant, such as `pi` or
`log(2)`, makes the candidate invalid, and so does a constant
term of zero, which is what `x - x` expands to. SymPy expands each proposal
before it is split into terms, so `exp(x + 1)` becomes `E*exp(x)` and
`log(2*x)` becomes `log(2) + log(x)`; both are invalid for their non-numeric
constant.

## Method

The language model receives the symbol library and scored proposal history,
not the observed arrays. The first round requests a numbered batch of
coefficient-free equations. Later rounds alternate an optimization prompt,
which lists earlier equations in ascending score order and asks for improved
ones, with an evolution prompt that requests selection, crossover and mutation
of term sets.

Each proposal is parsed with SymPy, expanded into additive terms, and executed
as design-matrix columns over second-order finite-difference operands. Division
allows reciprocals, but a zero denominator that makes a column non-finite
rejects that candidate. STRidge sweeps a tolerance ladder to fit coefficients;
fitted coefficient magnitudes outside `[1e-4, 1e4]` also reject a candidate.

In the tabular mode the operands are the table's columns. Division and the
functions use their ordinary definitions, so a zero denominator, `log` or
`sqrt` of a negative value, or an `exp` that overflows gives a non-finite
column and rejects the candidate. Before the columns are built, the
placeholders of one equation are fitted together; a name used in several terms
is one constant. For given values every term is executed into a column, the
outer coefficients are the least-squares solution over all terms, and the
objective is the residual sum of squares divided by `y @ y` for the target
column `y` (the plain sum when `y` is all zero). BFGS with gradient tolerance
1e-10 runs once from each of the fixed starts 1.0, -1.0 and 0.1, with every
placeholder at the start value, and the start with the smallest objective
wins. The terms then carry the fitted values as numeric literals, and STRidge
fits the outer coefficients as in the PDE mode; terms it sets to zero do not
cause a refit of the constants. The fit is local: a constant that BFGS does
not reach from any of the three starts is not found, and whether it is found
can depend on the range of the data. A placeholder that only scales its term,
as in `c1*x`, is not identified apart from the outer coefficient; its fitted
value, and with it the size of the outer coefficient, depends on the start
that wins, and an outer coefficient outside `[1e-4, 1e4]` rejects the
candidate.

Within one tabular equation, two columns that are equal or opposite
elementwise, within 1e-9 of the larger column's maximum magnitude, are one
term; the PDE mode keeps the reference's rule based on absolute sums. After the
fit, two terms with nonzero coefficients that differ only in numeric
constants, such as `sin(0.5*x1)` and `sin(1.5*x1)`, make the candidate invalid
(counted as `inexpressible`): the REGRESSION equation identifies a term by its
structure without literals and cannot hold both.

A round resamples until enough distinct-scoring survivors arrive or its call
budget is exhausted. Scores at or below `reward_limit` and scores already kept
in the round are discarded. The best `samples_per_epoch` survivors are offered
to an elite pool of `pool_size`; after filling, it admits only a strict
improvement over its minimum with an unseen score. The next prompt receives
that batch and the pool as it stood before the update. One round is one KD
iteration, controlled by `generations`. Per-round and per-run call limits bound
endpoint requests. Once the run limit is exhausted, subsequent rounds can
return no proposals; a stop threshold above the attainable reward cannot end
the search early.

## Result interpretation

The sparse reward is `(1 - 0.01 * k) / (1 + RMSE / std(target))`, where `k`
counts fitted nonzero coefficients and is capped at five. It is rounded to
four decimals before ranking and duplicate filtering. A one-term perfect fit
reaches 0.99, three terms reach 0.97 and four terms reach 0.96. The same reward
therefore implies different errors for different support sizes. Before
rounding, the inverse is
`nmse = ((1 - 0.01 * k) / reward - 1)**2`. In the tabular mode a nonzero
intercept counts toward `k` like any other column, and fitted placeholder
constants do not count; the reference has neither, so these counts have no
reference value.

The returned law contains only surviving structural terms and their native
sparse-fit coefficients. Terms whose coefficients were zeroed are removed from
both lists. `model.result_.equation` carries the aligned values, and NMSE and
MSE use the finite-difference domain that supplied the candidate columns, or
the table's rows in the tabular mode. A tabular law is a REGRESSION equation
with the target column on the left; its terms carry the fitted constants as
numeric literals, while the proposal history shown to the language model keeps
the placeholders. A tabular checkpoint records the mode and `functions`; a
restore into the other mode or under different `functions` is refused.

The per-round recorder includes `pool_best`, `pool_median`, `pool_worst`,
`n_valid`, `n_invalid` and `n_llm_calls`. The admitted count `n_valid` is
measured after truncation and saturates at `samples_per_epoch`. `n_invalid`
counts dropped scoring events, broken down into eight series:
`n_invalid_undefined_operands`, `n_invalid_undefined_operators`,
`n_invalid_inexpressible`, `n_invalid_parse`, `n_invalid_non_finite`,
`n_invalid_lstsq`, `n_invalid_abnormal_coef` and `n_invalid_other`.
They sum to the total in each round; unused causes record zero, and `other`
collects labels outside those named categories. Pool progress and admissions
help distinguish proposal failures from valid but poorly fitting equations.

## Python setup

### Endpoint configuration

Set an OpenAI-compatible `base_url` and `OPENAI_API_KEY` for the default
client; the OpenAI SDK is installed with KD. This is the algorithm that calls
an external language-model service during search. Alternatively, pass a
`provider=` implementing `kd.llm.LLMProvider`: a prompt goes in and text comes
out. A custom client can set timeouts and headers, use `BudgetedProvider` for
metering, or use `TapeRecordingProvider` to save a JSONL tape for offline replay.
The authors' mirrored datasets download on first load.

## References

Du et al. (2024). "Large language models for automatic equation discovery of nonlinear dynamics". *Phys. Fluids* 36, 097121. [Paper](https://doi.org/10.1063/5.0224297) · [arXiv:2405.07761](https://arxiv.org/abs/2405.07761)

Code: [menggedu/EDL](https://github.com/menggedu/EDL)

<div style="font-size: 0.85em" markdown="1">

Implementation notes. KD ports the reference's finite-difference templates, STRidge tolerance sweep,
coefficient gate, reward formula and rounding. The NumPy STRidge port matches
its reference on identical design matrices. Prompt term permutation uses a
plugin-owned seeded generator, including on resume. Columns use KD's executor
instead of NumPy lambdify, so floating-point differences can affect candidates
near the coefficient gate. Single-character candidates are scored rather than
skipped. In the PDE mode, constructs outside the prompt vocabulary, such as
trigonometric functions and non-integer exponents, are rejected explicitly.
The call budgets are KD additions, and so is the tabular mode with its prompt
templates, placeholder fit, intercept and elementwise duplicate rule; the
reference's three PDE templates are unchanged. The paper used GPT-3.5-turbo; the KD client accepts
OpenAI-compatible endpoints. The method's contribution is the prompting loop
that generates equations and evaluates them by sparse regression.

</div>
