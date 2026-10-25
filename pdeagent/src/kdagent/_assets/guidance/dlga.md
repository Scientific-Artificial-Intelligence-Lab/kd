Set `surrogate_max_epochs` explicitly. Training precedes every search on a
fresh call, so a limit reached during training yields no candidate. Training
time grows with epochs and points; the derivative library also affects memory.
Use `train_surrogate` and `surrogate=` to share a trained model across lineages.
`smooth_input(method="network", surrogate_id=<id>)` samples that model's
field on the input it was trained on as a derived input, without training,
for other algorithms' searches and measurements; the model itself stays a
DLGA surrogate, injectable into DLGA only.
At relative noise 1e-2, DLGA with surrogates trained for 20000 epochs found
the equation on 3 of 3 seeds on two benchmark problems, where eight
finite-difference pipelines found it on none of seven problems. On one of
those problems at that noise, 5000 epochs gave 0 of 3 and 20000 or 50000
epochs 3 of 3. On the CPU, with four trainings in parallel, 5000 epochs
took 2638 s, and in one CPU pilot of 42 agent runs ten DLGA searches used their whole 60-minute cap and two separate
trainings took 51 and 57 minutes.

For a second-order target set `target_lhs_order=2`. Every candidate is fitted
against the time derivative of that order only: the first for a first-order
target, the second for a second-order one. `lhs_auto_select=true` fits both
orders and keeps the branch with the lower NMSE, which can return a law for
the order this input does not ask for; leave it off when the target is given.
The default library covers `u` and its derivatives along spatial axes `x`,
`y` or both; for other axes, build `library` with those axes' derivative
names. Only the
left-hand-side field is modeled; tokens from other fields are refused before
training. `mode` accepts only `constant`.
`max_modules` and `max_module_length` bound the genome; `epsilon` multiplies
the expression-length penalty. A null law can require a different library or
better surrogate fit; repeated poor fits alone do not prove inexpressibility.
