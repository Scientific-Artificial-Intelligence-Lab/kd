## Search space

DISCOVER samples expression trees from operator and variable tokens, without
a fixed candidate-term library. In the standard PDE vocabulary, products,
quotients and trigonometric expressions can use `u`, `x` and `t`, with spatial
derivatives represented by `diff_x(u)` and `diff2_x(u)`. Repeated multiplication
forms powers. Priors restrict nesting of derivatives and trigonometric
functions. Coordinate tokens permit coordinate-dependent fits.

The PDE vocabulary is fixed to those names unless a typed `DiscoverConfig`
supplies a different `library`. An additional dataset axis does not add tokens
automatically. The standard vocabulary contains no mixed derivative or
third-order derivative token. `max_diff_order` can further restrict the
vocabulary but cannot add an absent token or override another structural prior.
The facade controls iteration count with `generations`; `n_iterations` does
not set its loop length. The PINN cycle and its stability-selection path belong
to the separate typed workflow. The facade's PDE path evaluates derivatives
from the observed grid and requires no pretrained weights or optional backend.

### Tabular regression

A `kd.TabularDataset` selects tabular mode in `Model.fit()`. Its vocabulary
is derived from the feature columns plus `add`, `sub`, `mul`, `div` and a
`const` terminal. No derivative tokens or configured term library are used.
Each distinct candidate's constants are fitted before scoring. The recurrent
controller and training rule remain the same, while complexity counts tokens
instead of additive terms.

## Method

An LSTM samples token sequences in pre-order, conditioned on the parent and
left sibling of each slot. Structural priors mask illegal choices: expression
length and token-repetition bounds, addition or subtraction below a derivative,
derivative arguments other than fields or permitted derivative expressions,
unary operators directly below their inverse, and excessive cumulative
derivative order. The standard priors also prevent nested derivatives and
trigonometric nesting. Sequences that fail length or legality checks are
dropped and surviving expressions are deduplicated before scoring.

In PDE mode the platform splits top-level additions into terms, fits their
coefficients by dense least squares against the time derivative, and computes
NMSE. Subtraction is carried by signed terms. In tabular mode, all-constant
candidates are rejected; unique traversals fit constants with a scale-free
BFGS objective before serialization as numeric literals. The platform then
scores the whole expression as one regression term with a fitted outer scale.

Risk-seeking policy gradient retains the top `epsilon` fraction of valid
rewards, subtracts a scalar baseline, and takes an Adam step on the policy loss
plus an entropy bonus. Invalid fits receive zero reward and are excluded from
training. The champion changes only on strict improvement, so ties retain the
incumbent. The seed controls both initial network weights and token sampling.
One controller iteration is one KD iteration, controlled by `generations`.

## Result interpretation

The reward combines fit and complexity as
`(1 - reward_alpha * complexity) / (1 + sqrt(nmse))`, clipped to `[0, 1]`.
Complexity is the term count for PDEs and token count for tables. NMSE alone
does not rank expressions. A surviving coordinate term indicates dependence
on that coordinate and should be interpreted according to the physical model
being sought.

For PDEs, `model.best_expr_` is the sampled structure and
`model.result_.equation` carries the coefficients from the same least-squares
fit used in scoring. The recorder's `best_reward` is non-decreasing;
`n_eval_valid` distinguishes invalid sampling from valid candidates with poor
fits. An improving champion at the final iteration suggests additional search
may still change the result; a flat curve alone does not identify the cause.

Tabular results carry a `REGRESSION` equation with the target column on the
left. The expression retains its fitted numeric literals and one outer
least-squares scale. `model.result_.pareto_front()` exposes nondominated
expressions with their token counts, loss and outer scale; the corresponding
series are `pareto_expressions`, `pareto_complexity`, `pareto_loss` and
`pareto_scale`. The inner constants and outer scale are distinct fitted values.

## References

Du et al. (2024). "DISCOVER: Deep identification of symbolically concise open-form partial differential equations via enhanced reinforcement learning". *Phys. Rev. Research* 6, 013182. [Paper](https://doi.org/10.1103/PhysRevResearch.6.013182) · [arXiv:2210.02181](https://arxiv.org/abs/2210.02181)

Code: [menggedu/DISCOVER](https://github.com/menggedu/DISCOVER)

<div style="font-size: 0.85em" markdown="1">

Implementation notes. KD implements the reinforcement-learning search, structural priors and the
separate optional stability-selection code. The paper's genetic operators and
dynamic subtree bank are not implemented. Risk-seeking policy gradient is the
training strategy; PQT has zero weight in the reference's shipped configuration,
and PPO is not implemented. The configured quantile follows DSO-PyTorch, while
the paper used 0.02. Unary `diff_x(u)` replaces the paper's binary `diff(u, x)`
for one-dimensional spatial differentiation.

Dense least squares and the stability-selection statistic `std / mean` follow
the reference code; the paper prints the reciprocal statistic. The optional
coefficient-magnitude rejection is disabled unless configured. A follow-up
method trains a neural surrogate and differentiates it by autograd; that PINN
workflow is separate from the facade path documented here.

</div>
