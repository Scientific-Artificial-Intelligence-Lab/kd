Supply `terms` for grids other than one-field `u` with spatial axes `x`, `y` or
both. Tabular mode derives
its inputs from feature columns, preserves constants inside the whole
expression and fits one outer coefficient without a separate outer intercept;
a pure-constant expression is refused. The default tabular unary operators
are `sin`, `cos`, `exp`, `log`.

`generations` sets `niterations`; a resumed run adds that budget to restored
populations and the hall of fame. Every call and resume incurs Julia startup
before search timing. With a wall limit, the backend gets that limit minus
50 seconds, with a 15-second minimum. `max_evaluations` does not bound PySR's
search; reported evaluations count re-scored hall-of-fame entries.

A seed alone gives weak reproducibility. For deterministic execution use
`extra_pysr_kwargs={"deterministic": true, "parallelism": "serial"}`; serial
search is slower. Do not repeat typed configuration keys inside that mapping.
The selected expression is in the envelope; inspect `recorder.json` for the
front. A linear fit over the supplied columns does not test all nonlinear
combinations PySR can form.

### Narrowing with a sketch

With a sketch hole constrained by `fields`, a
candidate's constant additive tail currently prevents a law from being
returned; account for this limitation when narrowing.
