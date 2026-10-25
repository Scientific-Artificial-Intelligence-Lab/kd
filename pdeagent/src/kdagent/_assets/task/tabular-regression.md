## Task: tabular regression

Fit target column {target} from feature columns {features}.
All columns: {fields}. Axes: {axes}.
The discover, pysr, pysindy and sga tabular modes derive their inputs from
these columns; `terms` is not accepted. Numeric constants inside regression
expressions are allowed. Discover and PySR report one whole-expression term
and its outer coefficient; PySINDy reports selected polynomial terms and
coefficients; SGA reports a sum of searched expression terms over the feature
columns with one coefficient each. PySINDy's library is polynomial, so low
residual on a nonpolynomial target may be a polynomial approximation.

### Availability

{availability}

### Required settings

{required}
