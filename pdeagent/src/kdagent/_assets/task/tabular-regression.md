## Task: tabular regression

Fit target column {target} from feature columns {features}.
All columns: {fields}. Axes: {axes}.
The discover, pysr, pysindy, sga and llm4ed tabular modes derive their inputs from
these columns; `terms` is not accepted. Numeric constants inside regression
expressions are allowed. Discover and PySR report one whole-expression term
and its outer coefficient; PySINDy reports selected polynomial terms and
coefficients; SGA reports a sum of searched expression terms over the feature
columns with one coefficient each. LLM4ED also reports several fitted terms
and fits numeric constants inside them; its unary functions are those in
`functions`. It is available only when this run has an operator-configured
LLM4ED endpoint. PySINDy's library is polynomial, so low
residual on a nonpolynomial target may be a polynomial approximation.

### Availability

{availability}

### Required settings

{required}
