Set `surrogate_max_epochs` explicitly. Training precedes every search on a
fresh call, so a limit reached during training yields no candidate. Training
time grows with epochs and points; the derivative library also affects memory.
Use `train_surrogate` and `surrogate=` to share a trained model across lineages.

For a second-order target set `target_lhs_order=2`. Every candidate is fitted
against the time derivative of that order only: the first for a first-order
target, the second for a second-order one. `lhs_auto_select=true` fits both
orders and keeps the branch with the lower NMSE, which can return a law for
the order this input does not ask for; leave it off when the target is given.
For additional spatial axes, build `library` with those axes' derivative
names; the default library uses only `u` and its x derivatives. Only the
left-hand-side field is modeled; tokens from other fields are refused before
training. `mode` accepts only `constant`.
`max_modules` and `max_module_length` bound the genome; `epsilon` multiplies
the expression-length penalty. A null law can require a different library or
better surrogate fit; repeated poor fits alone do not prove inexpressibility.
