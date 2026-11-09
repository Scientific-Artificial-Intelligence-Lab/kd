---
name: experience
description: Interpretation of retrieved training results, conditions and reproducible configs.
---

## Retrieved rows

This block aggregates runs on training problems, which may include this input.
It is evidence about those problems, not a measurement of this input or a
guarantee of recovery here.
`pipeline.algorithm` and `pipeline.mode` identify the search route;
`pipeline.config` contains ordinary settings for this kd version. Add the full
`conditioning.terms` and `conditioning.library` to `run_discovery(params=...)`;
pass `conditioning.sketch` separately as `sketch=`. These are recorded scientific
conditions, including complete anchors, holes and constraints, not recovered
answers or a claim that the same knowledge is available on the current input. Supply `dataset_id`, `algorithm` and your own
`seed` separately. Conditions record device, BLAS threads, Python hash seed
and soft wall-clock budget; different devices may have different costs.
A row may say what its searches ran on: the input as given, a low-passed
or Savitzky-Golay copy made with `smooth_input`, or a PINN-refined field,
each with or without a derivative surrogate from `train_surrogate`. Such a row is matched on the
input as given, so to follow it take the same step first and run the search
on the derived id. A row that says nothing ran on the input as given.
`input_steps` records the ordered transformations and their full scientific
parameters, including the Savitzky-Golay window and polynomial order.
PINN terms/laws are supplied conditioning. Network sampling and
compositions retain their recorded recipes; private dataset IDs and artifact
paths are not instructions to reuse the training input. `continuation` lists
incremental segment configs and conditions, in order. Reproduce that schedule
instead of treating the terminal config as one fresh search.
A row whose input steps and refinement were recorded as tool calls ends with
those calls numbered in order: each input step, the search on the id the last
input step returned, then each scoring, refinement and submission step. Each
call lists its recorded fields; its instructions (Then, Choose, Note) and
measurements follow it as sentences. A quoted description stands for a value
of this run, such as the pinned input or an id an earlier call returned.

## Results, cost and support

`result.structure_rate` is the fraction of all recorded runs judged to have
the correct structure. Its denominator includes unsuccessful runs, runs that
failed after their search started and runs whose grade could not be read.
`support.n_refused` counts the last two kinds when the row records it; a row
built before that count existed carries none. For a search row the rate grades
the support the search itself returned, not a later edited or refitted answer.
Source records the build could not assign to a pipeline, among them calls kd
refused before any search started, lack a trustworthy recipe: the build's
admission summary counts them, and no pipeline rate includes them. Duplicate
source lines do not add another attempt. A continuation chain counts as one
logical search.
`median_max_relative_error` is the median available maximum relative
coefficient error; `median_heldout_nrmse` is the median available held-out
NRMSE. Each median excludes missing measurements; null means not measured,
not zero. These are different denominators from the structure rate.
`cost.median_wall_seconds` and `median_evaluations` are medians of the reported
wall times and candidate-evaluation counts; null means not reported. These are
search costs; external smoothing, PINN refinement and separate surrogate
training are not included. Input labels describe historical process categories; use `input_steps` for
recorded preprocessing settings. A `refinement` row grades the refinement's
resulting law separately from its source search. It lists the method, supplied
parameters, support origin and resulting support, or the recorded calls
described above. Displayed search cost excludes
refinement expenditure, whose absence means unknown, never zero.
`support.n_runs` counts runs, and `n_problems` counts distinct training
problems. A row covering fewer problems provides less evidence about transfer.

## Features and matching

The vocabulary depends on the task kind. `n_fields` on a table counts feature
columns and the target. Numeric diagnostics are assigned to bins using the
reported edges; the top bin includes infinity. `not_computed` and `undefined`
are distinct and cannot match a template that includes that diagnostic.
Each row lists its indexed keys; other features were not used for matching.
A row not matched on `noise_bin` does not record the noise levels of its
training runs.
The template list and minimum support describe the library's coverage rules.
Usage advice is a separate versioned document, labeled with its `advice_id`;
it can change independently of the measured rows.

An evaluation document's build withholds the target independent problem at
source, across its instances and seeds, and other problems from the same family
remain eligible; rows added by hand afterwards may include runs on that problem.
Retrieval cannot identify a current problem from its coarse features; a usage
document makes no same-problem exclusion claim.
