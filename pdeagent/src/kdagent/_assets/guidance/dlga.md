Set `surrogate_max_epochs` explicitly. Training precedes every search on a
fresh call and counts against the search's cap: a cap reached during training
cuts the network short (`stopped_at_deadline` true) and the search then runs
one generation on it, so its result is sealed but rests on an incomplete
training. Training time grows with epochs and points; the derivative library
also affects memory.
Use `train_surrogate` and `surrogate=` to share a trained model across lineages.
`smooth_input(method="network", surrogate_id=<id>)` samples that model's
field on the input it was trained on as a derived input, without training,
for other algorithms' searches and measurements; the model itself stays a
DLGA surrogate, injectable into DLGA only.
At relative noise 1e-2, DLGA with surrogates trained for 20000 epochs found
the equation on 3 of 3 seeds on two benchmark problems, where eight
finite-difference pipelines found it on none of seven problems. On one of
those problems at that noise, 5000 epochs gave 0 of 3 and 20000 or 50000
epochs 3 of 3. These 2026-09-20 measurements share one network per cell
across the three GA seeds. They predate trainer RNG isolation (2026-09-25)
and elitism restoration (2026-10-02), and were not remeasured on current code.
The mode description's 5000-epoch calibration was a fixed-law coefficient-fit
headroom check on one problem at noise 0 and 0.05, requiring each coefficient
within 5%; it was not search recovery. Its acceptance search at noise 0.05
returned one extra term.

Historically, with four CPU trainings in parallel, 5000 epochs took 2638 s.
In a separate CPU pilot of 42 agent runs, ten DLGA searches exhausted their
60-minute cap and no final answer came from DLGA. Two separate trainings at
50000 epochs, the default, ended with `training_timeout` after 51 and 57
minutes, retaining no network. These are budget-terminated
trainings, not completed-training timings.

For a second-order target set `target_lhs_order=2`. Every candidate is fitted
against the time derivative of that order only: the first for a first-order
target, the second for a second-order one. `lhs_auto_select=true` fits both
orders and keeps the branch with the lower NMSE, which can return a law for
the order this input does not ask for; leave it off when the target is given.
A sketch overrides that selection and fits only its own LHS order. If its
admitted holes allow no products, each searched module has one token.
The default library covers `u` and its derivatives along spatial axes `x`,
`y` or both; for other axes, build `library` with those axes' derivative
names. Only the
left-hand-side field is modeled; tokens from other fields are refused before
training. `mode` accepts only `constant`.
`max_modules` and `max_module_length` bound the genome; `epsilon` multiplies
the expression-length penalty. A null law can require a different library or
better surrogate fit; repeated poor fits alone do not prove inexpressibility.
