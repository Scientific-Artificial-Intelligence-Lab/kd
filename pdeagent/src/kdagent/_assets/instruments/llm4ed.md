## Search space

LLM4ED asks a language model for coefficient-free equations over a fixed symbol
library and fits their coefficients by sparse regression. It operates on one
scalar field with exactly one spatial axis and one time axis. The symbols are the
field `u`, derivatives `u_x`, `u_xx`, `u_xxx`, and spatial coordinate `x`.
The prompts specify `+`, `-`, `*`, `/`, squares and cubes; the parser also
accepts integer powers through five. Explicit spatial dependence is
expressible. Extra fields, other-axis derivatives, mixed space-time derivatives,
higher spatial derivatives, trigonometric functions, exponentials, square roots
and standalone constants require a different symbol library.

Operand construction needs at least four spatial and three time points.
Spacing is taken from `x[2] - x[1]` and `t[1] - t[0]`, so use uniform axes.
The method differentiates the observations directly without smoothing or a
surrogate; the accuracy of higher derivatives therefore depends on measurement
noise and resolution.

### Endpoint configuration

Set an OpenAI-compatible `base_url` and `OPENAI_API_KEY` for the default
client; the OpenAI SDK is installed with KD. This is the algorithm that calls
an external language-model service during search. Alternatively, pass a
`provider=` implementing `kd.llm.LLMProvider`: a prompt goes in and text comes
out. A custom client can set timeouts and headers, use `BudgetedProvider` for
metering, or use `TapeRecordingProvider` to save a JSONL tape for offline replay.
The authors' mirrored datasets download on first load.

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
out-of-range fitted coefficients also reject a candidate.

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
`nmse = ((1 - 0.01 * k) / reward - 1)**2`.

The returned law contains only surviving structural terms and their native
sparse-fit coefficients. Terms whose coefficients were zeroed are removed from
both lists. `model.result_.equation` carries the aligned values, and NMSE and
MSE use the finite-difference domain that supplied the candidate columns.

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
skipped. Constructs outside the prompt vocabulary, such as trigonometric
functions and non-integer exponents, are rejected explicitly. The call budgets
are KD additions. The paper used GPT-3.5-turbo; the KD client accepts
OpenAI-compatible endpoints. The method's contribution is the prompting loop
that generates equations and evaluates them by sparse regression.

</div>
