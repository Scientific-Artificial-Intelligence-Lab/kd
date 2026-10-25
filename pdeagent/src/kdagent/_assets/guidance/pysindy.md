Supply `terms` for any grid other than one-field `u` with spatial axes `x`, `y`
or both. Include every field
and axis the law may need; extra data dimensions do not expand the library.
`mul` and `add` are binary; nest multiplication for a cube. Atomic derivatives
stop at order three; use `diff_x(u_xxx)` to compose a higher derivative.
Tabular input uses `degree` and refuses `terms`; its library is polynomial only.

Choose `threshold` together with `normalize_columns` and `relative_threshold`.
An absolute threshold can remove small but necessary coefficients, and it is
in the data's units: rescaling the field or an axis changes which terms the
same threshold keeps. Relative thresholding uses the target norm and
requires normalized columns. The
default absolute threshold 0.1 is larger than the smallest true coefficient,
expanded into monomials, on 3 of 7 benchmark problems; on one of them a
noise-free probe gave empty support at 0.1, a wrong law at 0.01 and the
right law at 0.001. Empty
support raises an error naming the active threshold. One solve gives one
evaluation; changing `generations` cannot add search, and repeating the same
settings adds no independent evidence. Read `primary.law` for coefficients.
