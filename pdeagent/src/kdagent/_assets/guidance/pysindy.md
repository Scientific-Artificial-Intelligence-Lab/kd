Supply `terms` for any grid other than one-field `u(x,t)`. Include every field
and axis the law may need; extra data dimensions do not expand the library.
`mul` and `add` are binary; nest multiplication for a cube. Atomic derivatives
stop at order three; use `diff_x(u_xxx)` to compose a higher derivative.
Tabular input uses `degree` and refuses `terms`; its library is polynomial only.

Choose `threshold` together with `normalize_columns` and `relative_threshold`.
An absolute threshold can remove small but necessary coefficients; relative
thresholding uses the target norm and requires normalized columns. Empty
support raises an error naming the active threshold. One solve gives one
evaluation; changing `generations` cannot add search, and repeating the same
settings adds no independent evidence. Read `primary.law` for coefficients.
