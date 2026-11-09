PySR requires the separate optional `sail-kd[pysr]` installation, including
its Julia backend. Availability declarations do not establish that the backend
is installed; a missing dependency fails the fit.

Supply `terms` for grids other than one-field `u` with spatial axes `x`, `y` or
both. Tabular mode derives
its inputs from feature columns, preserves constants inside the whole
expression and fits one outer coefficient without a separate outer intercept;
a pure-constant expression is refused. The default tabular unary operators
are `sin`, `cos`, `exp`, `log`; PDE mode defaults to no unary operators.
Conversion also supports `sqrt` and integer or half-integer powers when
these occur in the selected expression. `binary_operators` and
`unary_operators` take PySR's operator names (`+`, `-`, `*`, `/`, `square`,
`cube`, `inv`, `sin`, `cos`, `exp`, `log`, `sqrt`), not KD's term functions:
`add`, `mul`, `n2`, `n3` and `recip` are refused, and so is an operator
written as a definition (`"inv(x) = 1/x"`).

`generations` sets `niterations`; a resumed run adds that budget to restored
populations and the hall of fame. Every call and resume incurs Julia startup
before search timing. With a wall limit (`max_search_seconds`, or the run's
remaining time when that is omitted), the backend gets that limit minus
50 seconds, with a 15-second minimum; without one there is no timeout.
`max_evaluations` does not bound PySR's
search; reported evaluations count re-scored hall-of-fame entries.

A seed alone gives weak reproducibility. For deterministic execution use
`extra_pysr_kwargs={"deterministic": true, "parallelism": "serial"}`; serial
search is slower. That mapping accepts only `deterministic` (true or false)
and `parallelism` (`"serial"`, `"multithreading"` or `"multiprocessing"`);
any other key, `timeout_in_seconds` included, is refused before the search
starts.
The selected expression is in the envelope; inspect `recorder.json` for the
front. A linear fit over the supplied columns does not test all nonlinear
combinations PySR can form.

### Narrowing with a sketch

Under an open sketch, each converted term passes a gate before KD scoring: it
must use the vocabulary, avoid reserved pins and their monomials, and match
an anchor column or a hole, including its operator clause. Thus the
`exit_checked` operator clause also acts before scoring; hole counts remain
exit checks. PySR owns the selected expression: if it fails this gate, KD
does not replace it with a compliant hall-of-fame alternative. A constant
additive tail becomes `one`, which a hole constrained by `fields` does not
admit because it contains no field.

An anchor needs an equivalent configured column, matched by column
fingerprint rather than literal spelling. A composition of other configured
columns does not count, and the search is not forced to include the anchor;
its presence is judged at exit.
