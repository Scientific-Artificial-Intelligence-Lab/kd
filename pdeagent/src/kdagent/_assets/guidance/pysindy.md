### PySINDy observations

The tool's display contains structural terms without coefficients; read
primary.law. The sealed catalog fit retains all candidate columns, zero
coefficients for dropped ones, active_indices and lhs_spec. Empty support
raises an error naming the active threshold. The old card recommended
threshold=0 as a least-squares floor. That probe removes threshold pruning,
but an OLS interpretation also depends on unbias or the ridge setting; use
`kd.evaluate_terms` for a direct platform fit instead of assuming all optimizer
settings produce OLS. Repeating a deterministic solve with the same settings
adds no independent search evidence. The four typed defaults were checked
against PySINDy 2.1.0, and direct/plugin coefficients were bit-identical in the
integration test. The atomic derivative precompute rejects u_xxxx as order 4;
the open form diff_x(u_xxx) differentiates during execution.
