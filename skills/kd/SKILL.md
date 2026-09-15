---
name: kd
description: Discover equations from user data with KD's public Python API, from data inspection and algorithm selection through fitting, verification, and reporting.
---

# Equation discovery with KD

Use this workflow when writing Python against an installed `kd` package.
Start with `import kd`; preprocessing also uses `import kd.data`, and batches
use `import kd.harness` and `from pathlib import Path`.
Read current parameters in the schema and the
[KD documentation](https://scientific-artificial-intelligence-lab.github.io/kd/).
The calls below are templates: supply the user's arrays, names and run budget.

1. **Inspect and load the data.** Inspect array keys, shapes, dtypes and
   non-finite counts before choosing a layout or mapping.
   `inventory = kd.inspect_file(path)`
   `dataset = kd.load(path, periodic=periodic_axes)`
   A recognized layout loads directly; otherwise pass `coords=` and `fields=`
   mappings, or `loader=` returning a dataset. Use `field_axes=` for storage
   order and `select=` for a sample axis. Custom loaders own construction.
   For arrays, use `kd.PDEDataset.from_arrays(coords=..., fields=..., lhs=...)`
   on a grid or `kd.PDEDataset.from_scatter(coords=..., fields=..., lhs=...)`
   on points; scalar X-to-y data uses `kd.TabularDataset(...)`.
   Grid field dimensions follow coordinate mapping order; scattered arrays
   share one point count. Periodicity is never inferred from a file: declare
   it with `periodic={...}` when constructing a grid, without a repeated endpoint.
   Unsupported inspection containers need their own reader before loading.

2. **Preview before fitting.** For each PDE dataset, inspect the axes, fields
   and LHS before any fit, and resolve warnings until `Status: ready to fit`.
   `kd.preview(dataset)`
   This checks data readiness; algorithm compatibility is the next step.
   Preview accepts `PDEDataset`, not `TabularDataset`; for a table inspect
   `X`, `y`, feature names and target name before fitting.

3. **Select a compatible algorithm.** Read the declarations rather than
   guessing from an algorithm's name.
   `schemas = kd.instrument_schemas()`
   Each mode declares `forms`, `topologies`, `provider_kind` and `lhs_orders`.
   Match all of them to the data and target. `fit()` refuses a mismatch and
   names what the algorithm supports. Use the schema and website for parameters.

4. **Choose derivatives and preprocessing.** Use finite differences for clean
   regular grids. For noisy fields consider SGA with `derivatives="autograd"`
   or DLGA's trained surrogate. SGA still uses finite differences for derivative
   operators inside its trees; autograd does not remove its grid requirements.
   `model = kd.Model(algorithm="sga", derivatives="autograd", seed=seed)`
   For controlled noise studies use `kd.add_noise(dataset, level, seed=seed)`.
   `kd.data.stride_subsample(dataset, stride)` keeps a grid;
   `kd.data.ratio_subsample(dataset, ratio, seed=seed)` produces scattered points;
   `kd.data.resample_to_grid(dataset, axes=axes)` trains a field model and samples
   it on a grid. Preview and check compatibility again after preprocessing.

5. **Validate explicit term libraries.** Before fitting a PDE term library on
   a grid, check it with the platform's finite-difference evaluator.
   `validation = kd.validate_terms(dataset, terms)`
   It executes terms but trains no model and fits no coefficients; inspect
   `validation.rejected` for offending terms and reasons. Use function-call
   syntax and separate additive terms. Set `max_order=` when the terminal
   derivatives require it, consistently in validation and later evaluation.

6. **Fit and read the equation.** Supply the selected algorithm, seed and budget,
   plus its derivative and library settings chosen above.
   `model = kd.Model(algorithm=algorithm, seed=seed, generations=generations).fit(dataset)`
   Read `model.result_.equation` for terms, coefficients and LHS;
   `model.best_expr_` is a display string. `model.best_score_` is the algorithm's
   own score, qualified by `model.result_.score_kind` and `score_direction`.
   Do not compare native scores across algorithms.

7. **Compare in a shared frame.** For grid PDE term sets, compare separate refits:
   `evaluation = kd.evaluate_terms(dataset, terms)`
   Use identical data, LHS, derivative settings and solver across comparisons
   so `nmse` and `r2` share a frame; these refits have their own coefficients.
   `structure_key = kd.law_signature(eq).structure_key`
   This identifies canonical structure independently of display order, without
   general algebraic simplification. Keep one derivative spelling throughout:
   a terminal such as `u_xx` and a call such as `diff2_x(u)` can have different keys.
   A re-check of reported coefficients without refitting runs inside
   `kd.harness.build_consensus` when `datasets=` is supplied (step 8); the
   underlying `kd.verify_equation` needs an executor and context that the harness builds.

8. **Check stability across runs.** Use several seeds or algorithms rather than
   conclude from one fit. Repeating a deterministic solve does not explore.
   `plan = kd.harness.ExperimentPlan(name=study_name, entries=entries)`
   Build `entries` as a tuple of `kd.harness.PlanEntry` objects naming
   `instrument`, `dataset_ref`, `seed` and `model_kwargs`.
   `batch = kd.harness.run_plan(plan, datasets=datasets, store_root=Path(out_dir))`
   The mapping `datasets` resolves each `dataset_ref`; the store must be fresh.
   `store = kd.harness.EvidenceStore.load(batch.store_root)`
   `consensus = kd.harness.build_consensus(store, datasets=datasets)`
   Report structure, support, coefficient and empirical axes separately,
   including unavailable measurements and the breadth of corroboration.
   Do not invent a combined confidence number.

9. **Checkpoint long runs.** Set `checkpoint_dir=` and `checkpoint_every=` on
   the model before fitting, using a fresh checkpoint directory for each run.
   `point = kd.resolve_checkpoint(checkpoint_dir)`
   Select from the manifest through this call, never by filename sorting.
   `resumed.fit(dataset, resume_from=point.path)`
   The resumed model's `generations` is an additional segment budget.
   Parameters have resume-safe, init-only or identity-breaking tiers; consult
   their declarations. A refused configuration change names the field and both
   values. Restore that setting or start a fresh experiment as instructed.

10. **Write the report.** Render every applicable figure and one HTML page.
    `report = model.report(out_dir)`
    Include the paths and `report.warnings`, which lists skipped or degraded
    figure families, alongside the equation and comparison evidence.

11. **Act on errors.** Read the remedy in an exception, such as a shape fix,
    LHS spelling, missing extra or supported topology. Apply that correction
    once before changing approach. If it still fails, preserve the error and
    input details in the report rather than claiming a completed discovery.
