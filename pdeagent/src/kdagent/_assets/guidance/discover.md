### DISCOVER tool configuration

A JSON value for `library` or `pinn` is rejected with
`TypeError: ... are Python-only ... Pass a typed discover config through config= instead`.
That `config=` means KD's `Model(config=...)` Python route; `run_discovery`
does not accept a typed `config=`, so neither field is available through this
tool. The default PDE vocabulary does not grow when the input adds an axis.
`max_diff_order` at its then-default 4 did not override the priors' order-2 cap;
only lowering it below 2 tightened that cap. `n_iterations` does not control
the facade loop. `stability_selection` belongs to the PINN path.
`diagnostic_scaffold` inserts an assumed equation shape and requires
`DISCOVER_ENABLE_DIAGNOSTICS=1` at fit time; it is not ordinary discovery.

### DISCOVER observations

Seventeen grid runs with batch_size=16, 10-50 iterations and 2e3-2.5e5 points
finished in under two seconds. This does not redefine the schema's cost class;
read actual search_seconds. Recorded reward 0.728 at complexity 4 corresponded
to NMSE 0.101; reward near 0.5 corresponded to NMSE near 1. The best recorded
fit reached 3e-7. A bare coordinate or coordinate factor can fit observations
without recovering a physical law; assess whether it belongs in the target
model. The default quantile was 0.05 versus the paper's 0.02, and the default
prior allowed at most five add tokens.

The display can be a raw nested genotype, such as
sub(add(mul(sub(t,diff_x(u)),u),add(u,diff_x(u))),diff2_x(u)); use the split law
and aligned coefficients. A best_reward fixed from the first iteration with
n_eval_valid near batch_size records valid but unimproved proposals. An
improving final iteration records a search stopped while still improving.
Neither curve alone proves representability or recovery.
