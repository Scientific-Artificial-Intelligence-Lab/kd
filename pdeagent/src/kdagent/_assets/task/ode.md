## Task: ODE regression

Fit target column {target} from state feature columns {features}.
All columns: {fields}. Axes: {axes}.
The provider estimated this target time derivative from trajectories; its
estimation method is recorded with the benchmark row. Each row asks for the
derivative of one state. The controller receives the resulting table.
The discover, pysr and pysindy tabular modes derive their inputs from the
feature columns; `terms` is not accepted. Numeric constants inside regression
expressions are allowed. Discover and PySR report a whole expression and its
outer coefficient; PySINDy reports selected polynomial terms and coefficients.
PySINDy's library contains only polynomials.

### Availability

{availability}

### Required settings

{required}
