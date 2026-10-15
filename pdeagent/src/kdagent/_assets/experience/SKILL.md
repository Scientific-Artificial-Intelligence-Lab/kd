---
name: experience
description: Interpretation of retrieved training results, conditions and reproducible configs.
---

## Retrieved rows

This block aggregates runs on training problems. It is evidence about those
problems, not a measurement of this input or a guarantee of recovery here.
`pipeline.algorithm` and `pipeline.mode` identify the search route;
`pipeline.config` is complete for this kd version and can be passed directly
as `run_discovery(params=...)`. Supply `dataset_id`, `algorithm` and your own
`seed` separately. Conditions record device, BLAS threads, Python hash seed
and soft wall-clock budget; different devices may have different costs.

## Results, cost and support

`result.structure_rate` is the fraction of all recorded runs judged to have
the correct structure, including unsuccessful runs in the denominator.
`median_max_relative_error` is the median available maximum relative
coefficient error; `median_heldout_nrmse` is the median available held-out
NRMSE. Each median excludes missing measurements; null means not measured,
not zero. These are different denominators from the structure rate.
`cost.median_wall_seconds` and `median_evaluations` are medians of the reported
wall times and candidate-evaluation counts; null means not reported.
`support.n_runs` counts runs, and `n_problems` counts distinct training
problems. A row covering fewer problems provides less evidence about transfer.

## Features and matching

The vocabulary depends on the task kind. `n_fields` on a table counts feature
columns and the target. Numeric diagnostics are assigned to bins using the
reported edges; the top bin includes infinity. `not_computed` and `undefined`
are distinct and cannot match a template that includes that diagnostic.
Each row lists its indexed keys; other features were not used for matching.
The template list and minimum support describe the library's coverage rules.
Usage advice is a separate versioned document, labeled with its `advice_id`;
it can change independently of the measured rows.
