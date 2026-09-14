## Agent tools

The common card describes KD's Python API. In this workspace use
`run_discovery`; pass JSON-settable fields as `params`, with `seed` as a
separate tool argument. The generated table below describes this tool's
parameters and defaults. When omitted, `generations` is {DEFAULT_GENERATIONS}
where applicable and a fresh run uses seed {DEFAULT_SEED}. A resumed run
inherits its seed. A different seed requires `reseed`, which restores the
checkpoint's search state and derives a new random stream. The new segment's
generation budget is additional work, not a cumulative total.

Read the input summary from `list_datasets` or the prepared file reference
before selecting an algorithm; it includes axes, fields and the left-hand side.
The generated capabilities and parameters describe the available modes.
The controller accepts PDE inputs; the cards' tabular Python examples do not
make tabular inputs available to `run_discovery`.

Read `primary.law.support` and its aligned `coefficients` rather than extracting
numbers from `primary.display`. Coefficients can be absent or null when they
were not measured. `provenance.headline_coefficient_source` reports whether
they are native or a platform refit. `diagnostics.score_kind`,
`score_direction`, `nmse` and `mse` describe the recorded fit; native scores and
derivative domains do not establish comparisons across algorithms.
`diagnostics.platform_nmse` and `platform_coefficients` are separate neutral
refit measurements when available, not replacements for the headline values.

`provenance.run_dir` points to `record.json` (sealed evidence), `recorder.json`
(iteration series) and `phases.jsonl` (phase boundaries). Read
`evidence.catalog_fit.lhs_spec` to identify the fitted field, axis and order.
`diagnostics.segment_report` gives segment facts; `provenance.run_id` identifies
it for selection or resume. Failure status and `failures` must be read before
using a law. A `partial` result is not a successful discovery.

Historical observations below describe the recorded conditions, not guaranteed
recovery rates or current runtime limits. Training now runs under the worker's
wall-clock supervision, and trained surrogates can be reused where the schema
provides that channel. Old measurements without those controls do not describe
an uninterruptible current tool call.
