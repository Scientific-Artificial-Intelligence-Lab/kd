## Search space

DISCOVER samples expression trees from operator and variable tokens, without
a fixed candidate-term library. In the PDE vocabulary, products, quotients
and trigonometric expressions can use the dataset's fields and coordinates,
with spatial derivatives represented by `diff_<axis>(...)` and
`diff2_<axis>(...)` for each spatial axis (`diff_x(u)` and `diff2_x(u)` for a
field `u` on an axis `x`). Repeated multiplication forms powers. Priors restrict nesting of derivatives and trigonometric
functions. Coordinate tokens permit coordinate-dependent fits.

The facade builds the grid vocabulary from the dataset: every field is a leaf,
sorted by name (the target is the left-hand side, not a token); the spatial
axes, in the dataset's order, and `t` are coordinate leaves; and each spatial
axis adds `diff_<axis>` and `diff2_<axis>`. Spatial axes other than `x` and `y`
are refused. A `library` set in the Python configuration is used as given and
refuses a field or axis it does not name. The vocabulary contains no mixed
derivative or third-order derivative token. `max_diff_order` can further restrict the
vocabulary but cannot add an absent token or override another structural prior.
The vocabulary a run searched is sealed in its result config as
`searched_vocabulary` (operators, state and coordinate variables, and the
engine's token names and arities; on a sketch run, the library narrowed to
the sketch), so it is read there rather than reconstructed from the
requested configuration.
The facade controls iteration count with `generations`; `n_iterations` does
not set its loop length. Without `pinn`, the PDE path evaluates derivatives
from the observed grid by finite differences and requires no pretrained weights
or optional backend. With `pinn` set, the same vocabulary is searched in PINN
mode, where a neural network is trained on the data and derivatives are taken
from the network by automatic differentiation.

### Tabular regression

Tabular mode is selected by the input data, without a `mode` parameter.
Its vocabulary is derived from the feature columns plus `add`, `sub`, `mul`,
`div`, `sin`, `cos`, `exp`, `log` and a fitted `const` terminal. No derivative tokens or configured term library are used.
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

### PINN mode

Setting `pinn` on grid data selects PINN mode. On a fresh call, a fully connected
network is first pretrained on the observations: every grid point, or the share
given by `obs_ratio`. A resumed call loads the stored network without pretraining.
Candidates are then scored on collocation points sampled inside
the data domain, with the time derivative on the left-hand side and every
spatial derivative taken from the network by automatic differentiation.

After `n_iterations` generations, and then after every `cycle_n_iterations`
generations, the network is retrained on the data loss plus `coef_pde` times
the residual of the best equation found so far, and the search continues on
the derivatives of the retrained network. The champion is re-scored on each
new network before the search resumes. On a fresh call, `generations` is the
total number of search generations, so a retraining phase runs only when
`n_iterations` is smaller than `generations`. On resume, the phase boundaries
use the generation count over all segments of the run; the resumed call's
`generations` add to that count. A phase on a call's last generation is not run in that call. If
the earlier segment ended on such a boundary, the resumed call runs that phase
first, before its first generation. When `stability_selection` is positive, the final
equation is chosen among the top candidates of the last cycle by refitting
each on bootstrap resamples.

The time of the pretraining and of each retraining phase grows with the number
of epochs times the number of training points. The library defaults of
`pretrain_epoch`, `pinn_epoch` and `n_collocation` are the research budget of
the reference configuration.

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

In PINN mode the final equation is fitted on the collocation points, so the
target and the residuals of the result have one entry per collocation point,
and the result config records `provider_kind` as `autograd`. On a fresh call,
the recorder carries the pretraining outcome in the `pinn_pretrain_*` series;
a resumed segment's recorder has no such series. Each segment records one entry
per retraining phase it executes in `pinn_cycle_idx`, `pinn_phase_ran`, `pinn_data_loss`,
`pinn_physics_loss`, `pinn_total_loss`, `pinn_epochs_run` and
`pinn_train_seconds`. A phase with no valid best equation is recorded with
`pinn_phase_ran` equal to 0 and leaves the network unchanged.

Tabular results carry a `REGRESSION` equation with the target column on the
left. The expression retains its fitted numeric literals and one outer
least-squares scale. `model.result_.pareto_front()` exposes nondominated
expressions with their token counts, loss and outer scale; the corresponding
series are `pareto_expressions`, `pareto_complexity`, `pareto_loss` and
`pareto_scale`. The inner constants and outer scale are distinct fitted values.

## Python setup

A `kd.TabularDataset` selects tabular mode in `Model.fit()`.
A typed `DiscoverConfig` can supply a different `library`, which is Python-only
and unavailable through JSON `params`. `pinn` takes a `PINNConfig`, exported
from `kd.search`, or a JSON object of its fields, in which every omitted key
keeps its default. `DiscoverConfig.burgers_preset(pinn=PINNConfig(...))`
combines a reference preset with PINN mode through `config=`. A sketch, an
`EarlyStoppingCallback` and a `kd.TabularDataset` are refused together with
`pinn`.

## References

Du et al. (2024). "DISCOVER: Deep identification of symbolically concise open-form partial differential equations via enhanced reinforcement learning". *Phys. Rev. Research* 6, 013182. [Paper](https://doi.org/10.1103/PhysRevResearch.6.013182) · [arXiv:2210.02181](https://arxiv.org/abs/2210.02181)

Code: [menggedu/DISCOVER](https://github.com/menggedu/DISCOVER)

<div style="font-size: 0.85em" markdown="1">

Implementation notes. KD implements the reinforcement-learning search, structural priors, PINN mode
and its optional stability selection. The paper's genetic operators and
dynamic subtree bank are not implemented. Risk-seeking policy gradient is the
training strategy; PQT has zero weight in the reference's shipped configuration,
and PPO is not implemented. The configured quantile follows DSO-PyTorch, while
the paper used 0.02. Unary `diff_x(u)` replaces the paper's binary `diff(u, x)`
for one-dimensional spatial differentiation.

Dense least squares and the stability-selection statistic `std / mean` follow
the reference code; the paper prints the reciprocal statistic. The optional
coefficient-magnitude rejection is disabled unless configured. A follow-up
method trains a neural surrogate and differentiates it by autograd; KD runs it
as PINN mode. The retraining phases are placed inside the `generations` loop
by `n_iterations` and `cycle_n_iterations`, and the reference's cycle count
`n_cycles` is read only by the standalone cycle runner.

</div>
