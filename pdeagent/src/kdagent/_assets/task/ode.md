## Task: ODE regression

Fit target column {target} from feature columns {features}.
All columns: {fields}. Axes: {axes}.
This target is a time derivative estimated from trajectories: by the provider
for a table given as an ODE table (for a benchmark row the method is recorded
with the row), by the controller for a trajectory file (the input's
`projection` says how). The answer is the law of this target. On a
trajectory file, `run_discovery(target_state=...)` searches another state's
derivative on the same trajectories, and `verify_system` integrates one search
per state together and compares the result with the observations; once a
system has a result, the default answer is this target's law in the best
system (the kd-tools skill gives the rule).
The discover, pysr, pysindy and sga tabular modes derive their inputs from
the feature columns; `terms` is not accepted. Numeric constants inside
regression expressions are allowed. Discover and PySR report a whole expression
and its outer coefficient; PySINDy reports selected polynomial terms and
coefficients; SGA reports a sum of searched expression terms over the feature
columns with one coefficient each. PySINDy's library contains only polynomials.

### Availability

{availability}

### Required settings

{required}
