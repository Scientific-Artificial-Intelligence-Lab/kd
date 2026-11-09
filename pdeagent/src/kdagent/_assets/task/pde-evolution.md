## Task: PDE evolution

Fit the named time-derivative target {target}. Fields: {fields}. Axes: {axes}.
Use this input's field and axis names to construct `terms` or `library`.
Grid term grammar: `mul` and `add` are binary; nest them for more factors or
summands. Atomic derivative terminals support at most third order per axis;
use `diff_<axis>(...)` for composition and `u_x_t` for a mixed partial.
The constant column is `one`; numeric constants are not accepted in a term
catalog. These example spellings must use this input's actual names.

For a second time derivative, dlga, pysr, pysindy, both sga PDE modes, and
discover MODE1 and MODE2 accept the input; set `target_lhs_order=2` to match
the dataset. The table below states how additional fields and spatial axes
affect each mode.

dlga fits the time derivative of the order `target_lhs_order` names and no
other. With `lhs_auto_select=true` it fits the first and the second time
derivative and keeps the branch with the lower NMSE, so it can return a law
for the other order; `primary.law.lhs` names the fitted order. A law whose
`lhs` is not this input's target is not an answer for this input: do not
submit or select it.

Of the modes for this task kind, dlga (always) and sga (with
`derivatives="autograd"`) train a neural surrogate of the field before the
search, on every fresh call without an injected surrogate. For dlga, set
`surrogate_max_epochs` explicitly. `train_surrogate` trains and registers a
surrogate once; `run_discovery(surrogate=<id>)` injects it, so later fresh
lineages of that algorithm do not retrain. `smooth_input(method="network")`
trains a network once per input, recipe and seed and writes its field as a
derived input that every mode searches like the input itself; it is no
mode's surrogate. Weak-form measurements (`method="weak"`) take this task's
first- or second-order time derivative, one field and one spatial axis.
Fourth-order spatial terms take `max_order=4`, `weak={{"bump_degree": 6}}`;
`u_xxxx`, `diff_x(u_xxx)` and `diff2_x(u_xx)` are accepted there. On
finite-difference frames keep `max_order=3` and use the nested spellings;
kd's finite-difference terminal/stencil cap remains third order.

discover with `pinn` also trains a network before the search: pretraining on
a fresh call, or one owed retraining phase on a resumed call whose parent
stopped at a retraining boundary. For all three, the training runs before
the search and its time comes out of the search's cap: a search whose cap
the training spent still runs one generation and returns its candidate,
and a worker still running 120 seconds after the cap is stopped and
returns none.

LLM4ED calls a language-model endpoint that the operator configures for the
run; the run supplies its `base_url`, `model` and `extra_body`. It is
available only when this run has one (see the Availability table and the
LLM4ED card).

### Availability

{availability}

### Required settings

{required}
