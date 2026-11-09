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
   mappings, or `loader=` returning a dataset. Mapping order sets the axis
   order and each field's storage order is read from its dimension lengths.
   Pass `field_axes=` in storage order when two coordinates have the same
   length (the error lists the calls), and with `select=` for a sample axis.
   Custom loaders own construction.
   For arrays, use `kd.PDEDataset.from_arrays(coords=..., fields=..., lhs=...)`
   on a grid or `kd.PDEDataset.from_scatter(coords=..., fields=..., lhs=...)`
   on points; scalar X-to-y data uses `kd.TabularDataset(...)`.
   `table = kd.load_tabular(path)` reads a `kd-tabular-npz-v1` file.
   `table = kd.data.trajectories_to_tabular(trajectories, target_state)`
   converts a `kd.data.TrajectorySet` to one state's time-derivative table,
   using finite differences separately along each trajectory.
   Grid field dimensions follow coordinate mapping order; scattered arrays
   share one point count. Periodicity is never inferred from a file: declare
   it with `periodic={...}` when constructing a grid, without a repeated endpoint.
   Unsupported inspection containers need their own reader before loading.

2. **Preview before fitting.** For each dataset, inspect the axes, fields
   and LHS before any fit, and resolve warnings until `Status: ready to fit`.
   `kd.preview(dataset)`
   This checks data readiness; algorithm compatibility is the next step.
   A `TabularDataset` previews its feature and target columns, with the target
   as the LHS and no axes.

3. **Select a compatible algorithm.** Read the declarations rather than
   guessing from an algorithm's name.
   `inputs = kd.search.mode_inputs(kd.preview_report(dataset).to_dict())`
   `verdicts = {(s["algorithm"], m["name"]): kd.mode_accepts(m, **inputs) for s in kd.instrument_schemas() for m in s["modes"]}`
   `kd.mode_accepts` checks every declaration that decides acceptance: forms,
   topologies, LHS orders, whether the mode is selectable from data, field
   names and spatial axes. Read each verdict's `accepts`, `reason` and
   `params`: the parameters that select the mode or that you must set
   (`"<set>"` marks a value you choose; on fields or axes other than the
   defaults, a mode that builds its catalog from them asks for `terms` or
   `library`). `fit()` refuses a mismatch and names what the algorithm
   supports. Use the schema and website for parameters.

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
   A smoothed grid is another derivative source:
   `smoothed = kd.lowpass_smooth(dataset, modes=modes, time_window=window)`
   keeps the lowest spectral modes, `kd.select_lowpass_modes(...)` chooses the
   counts from refits of candidate terms, and
   `prepared = kd.network_smooth(dataset, recipe=recipe, seed=seed)` fits a
   network to the data (the recipe `DLGA_NN_1` of kd's models package
   reproduces DLGA's network) whose field `prepared.dataset` replaces the
   noisy one;
   `kd.surrogate_field(dataset, model)` samples an already trained field
   model on the dataset's grid instead.
   A network field is an input like any other dataset, not a surrogate an
   algorithm injects, and its training report (`prepared.quality`) does not
   say its derivatives are accurate.
   Measured on 2026-09-20 on seven benchmark problems with three noise draws
   per level: the
   true equation refitted on finite differences of the raw field passed (NMSE
   below 0.05 and every coefficient within 5%) on all seven without noise, on
   two at relative noise 1e-4 plus one more on 1 of 3 draws, and on none at
   1e-3 and above. After Savitzky-Golay smoothing with window 11
   (`savgol_window=11` on `kd.Model`) it passed up to 1e-3 on three problems
   and up to 1e-4 on three, and at no level, including no noise, on the
   problem with a shock. DLGA with surrogates trained for 20000 epochs found
   the equation on 3 of 3 GA seeds on two problems at 1e-2; each cell shared
   one network. These measurements predate trainer RNG isolation
   (2026-09-25) and elitism restoration (2026-10-02), and were not repeated
   on current code. Along x, the median
   `noise_estimate` from `kd.preview_report(dataset)` was within 5% of the
   relative noise at 1e-2 and 5e-2 on all seven problems.

5. **Validate explicit term libraries.** Before fitting a PDE term library on
   a grid, check it with the platform's finite-difference evaluator.
   `validation = kd.validate_terms(dataset, terms, max_order=3)`
   It executes terms but trains no model and fits no coefficients; inspect
   `validation.rejected` for offending terms and reasons. Use function-call
   syntax and separate additive terms. Set `max_order=` when the terminal
   derivatives require it, consistently in validation and later evaluation.
   The default 2 refuses third-order terminals such as `u_xxx`; it does not
   gate open-form derivatives.

6. **Fit and read the equation.** Supply the selected algorithm, seed and budget,
   plus its derivative and library settings chosen above.
   `model = kd.Model(algorithm=algorithm, seed=seed, generations=generations).fit(dataset)`
   Read `model.result_.equation` for terms, coefficients and LHS;
   `model.best_expr_` is a display string. `model.best_score_` is the algorithm's
   own score, qualified by `model.result_.score_kind` and `score_direction`.
   Do not compare native scores across algorithms.

7. **Narrow the search with known structure.** When part of the equation is
   known, pass it as a sketch: terms fixed with their coefficient (`pinned`),
   terms required with a fitted coefficient (`anchored`), and holes the search
   fills from a constrained pool.
   `model = kd.Model(algorithm=algorithm, seed=seed).fit(dataset, sketch=kd.sketch_from_dict(payload))`
   This payload requires one term and leaves one or two terms of derivative
   order at most 2 to the search; `"pinned": [{"term_ir": "mul(u,u_x)", "value": -0.5}]`
   with an empty `anchored` list fixes the coefficient instead. The term and
   the value only show the format.
   ```json
   {"schema": "kd-sketch-v2",
    "lhs_spec": {"field": "u", "axis": "t", "order": 1},
    "vocabulary": {"fields": ["u"], "coordinates": ["t", "x"]},
    "pinned": [],
    "anchored": [{"term_ir": "mul(u,u_x)"}],
    "holes": [{"id": "diffusion", "min_count": 1, "max_count": 2,
               "constraint": {"max_deriv_order": 2, "operators": null,
                              "fields": null, "axes": null}}],
    "match_policy": {"coeff_atol": 1e-09, "coeff_rtol": 1e-09,
                     "support_threshold": 0.0, "term_identity": "kd-lawsig-v2",
                     "hole_assignment": "disjoint",
                     "derivative_order": "total-effective"}}
   ```
   Each mode in `kd.instrument_schemas()` lists the clause levels it supports
   under `sketch`. On tabular data a sketch's LHS is the target at order 0
   with no axis and no coordinates, and the law's terms are split at its
   additions, its constant being `one`. Every clause is judged at
   exit; a backend may also act on an `exit_checked` clause before exit.
   `model.result_.sketch_outcome.solution` is the certified law, or None when
   the result misses a clause or the residual gate (NMSE at most 0.05 by
   default). `best_candidate` retains the lifted candidate; `verdict` reports
   clause matches and `full_verify` the residual evidence. `failure` names
   lift, matching or verification errors and can be None when only a clause
   misses. `compile_report.notes` describe the backend's configured rules.
   Pin only a term and coefficient that agreed across several of your own
   fits (seeds or algorithms), not one taken from the task text, and record
   which fits justified the pin. A pin with a wrong coefficient is not
   repaired by the search: a hole candidate whose expansion shares a monomial
   with a pinned term is refused before scoring, but this does not cover every
   rational or joint-term respelling; pin reservation alone does not establish
   the physical coefficient. On 2026-09-25, a wrong-sign pin on a noise-free
   128 by 51 Burgers grid gave no certified law in 8 of 8 DISCOVER and 10 of
   10 SGA runs. The SGA results predate its 2026-10-07 STRidge correction
   and have not been remeasured on current code.
   No hole clause excludes an explicit coordinate factor along an axis the
   hole differentiates: `fields` limits which data fields a term contains, and
   `axes` limits derivative axes and coordinate factors together, so
   `"axes": ["x"]` refuses any dependence on t but admits a factor of x.
   With several fits in hand, `kd.vote_terms(supports, min_count=k)` counts
   how many independent fits found each term (by law-term identity), and
   `kd.sketch_from_anchors(dataset, anchors, min_additional_terms=...,
   max_additional_terms=..., max_derivative_order=...)` builds the Sketch
   that requires the anchors you keep. Votes, then ablations (step 9), then a
   revised Sketch are a sequence of hypotheses: a recurring term or a large
   drop-one rise is not evidence that the term is physically present.
   `small_term_rtol` is not applied on a sketch fit. The residual gate uses
   the actual verification frame and input; pass
   `verify=kd.VerifyPolicy(nmse_max=...)` to `fit` to set its threshold
   (default 0.05). Least-squares
   fits of the true terms on finite differences of unsmoothed noisy benchmark
   fields had median NMSE above 0.05 on 4 of 7 problems at relative noise
   1e-4, 6 of 7 at 1e-3 and 7 of 7 at 1e-2 (after window-11 smoothing: 1, 2
   and 6 of 7). These medians over three noise draws were measured on
   2026-09-20 using the benchmark judge's finite differences, not through
   the current exit gate. An earlier comparison of sketch fits with blind
   search ran on code of 2026-09-12, including that day's executor fix but
   before the 2026-09-15 periodic-boundary fix and 2026-10-07 STRidge change,
   on that Burgers grid
   with relative noise added, with SGA (population 20, 100 generations, five
   seeds) and coefficients judged within 40% on SGA's own derivatives: blind
   search found the equation on 3 of 5 seeds at noise 1e-4 and 4 of 5 at
   2e-4, and a constrained hole on 5 of 5 at both; at 3e-4 and above every
   arm found it on 0 of 5. That comparison covers one problem and one
   algorithm, not current-code performance; the later fixes' effect on that
   generated grid was not measured. On 2026-10-18, DISCOVER on KdV with a
   pinned term and a constrained hole, in three sketch configurations
   (pinned, hole only, wrong pin) over three rounds and two parameter
   settings, certified 0 of 27 runs. Six blind runs did not find the equation
   either; 18 of the 27 sketch runs had hole candidates with coordinate
   factors the sketch could not exclude.

8. **Compare in a shared frame.** For grid PDE term sets, compare separate refits:
   `evaluation = kd.evaluate_terms(dataset, terms)`
   Use identical data, LHS, derivative settings and solver across comparisons
   so `nmse` and `r2` share a frame; these refits have their own coefficients.
   `structure_key = kd.law_signature(eq).structure_key`
   This identifies canonical structure independently of display order, without
   general algebraic simplification. Keep one derivative spelling throughout:
   a terminal such as `u_xx` and a call such as `diff2_x(u)` can have different keys.
   A re-check of reported coefficients without refitting runs inside
   `kd.harness.build_consensus` when `datasets=` is supplied (step 11); the
   underlying `kd.verify_equation` needs an executor and context that the harness builds.

9. **Refine a candidate.** Refit an equation's coefficients with its terms
   held fixed, optionally dropping small terms first:
   `refit = kd.refit_terms(dataset, terms, coefficients, small_term_rtol=small_term_rtol)`
   `terms` and `coefficients` are the equation's active terms
   (`model.result_.equation.terms` at `active_indices`). `small_term_rtol`
   drops terms whose coefficient is below that fraction of the largest one,
   and 0 drops nothing; the rest is then refitted. The measured gain from
   dropping came from `kd.Model(small_term_rtol=1e-4)`, which drops those
   terms from the algorithm's result without a refit: it turned 73 of 1641
   noise-free answers right and none wrong in a retrospective check of
   2026-09 records, including SGA results before the 2026-10-07 STRidge
   correction. This was not a new search on current code. To refit on a
   smoothed field, smooth first with
   `smoothed = kd.data.savgol_smooth(dataset, savgol_window)`.
   Read `refit.terms`, `refit.coefficients`, `refit.dropped`, `refit.nmse`,
   `refit.r2` and `refit.condition_number`; `refit.frame` is the resolved
   measurement frame. `method="weak"` with
   `weak=kd.WeakFormOptions(spectrum_tau=..., noise_correction=...)` fits
   integrals against smooth test windows instead of pointwise derivatives
   (one field, one spatial axis, a first- or second-order time derivative, and only
   `one`, the field, its square and cube (`n2(u)`/`n3(u)` or the raw products
   `mul(u,u)` and `mul(mul(u,u),u)`), pure spatial derivatives up to fourth
   order and the field times its first derivative; other terms are refused).
   Fourth-order terms require `max_order=4` and `WeakFormOptions(bump_degree=6)`;
   defaults remain 3 and 4. Weak frames accept `u_xxxx`, `diff_x(u_xxx)` and
   `diff2_x(u_xx)`; finite-difference frames keep `max_order=3` and use the
   nested spellings under kd's existing third-order terminal/stencil cap;
   `noise_correction="gaussian"` assumes additive zero-mean Gaussian noise of
   one level everywhere, so report the uncorrected fit beside it.
   `method="autograd"` with `surrogate=` differentiates a stored surrogate
   trained on that dataset, and `trim=` drops boundary rows after the
   finite differences are taken. `kd.ablate_terms(dataset, terms)` refits
   without each term in one frame (`relative_rise` per dropped term) and
   `kd.score_term_sets(dataset, term_sets)` fits several supports in one
   frame; numbers from different frames do not compare. In the controller,
   `select_support` scores the same sets once and chooses the fewest terms
   whose NMSE is at most `ratio` times the lowest (default 2.0), breaking
   ties by lower NMSE, with no absolute floor. It returns the ranked table
   and chosen entry with a measurement receipt (`law`, `refinement_id`).
   Submit on the same cited search, or refit with `refine_candidate` first.
   The controller admits the cited search's terms or an exact term set
   measured on this run's registered input by a receipt; the sealed LHS
   stays unchanged. `answer_origin` distinguishes search terms from
   `receipt_measured`; edited coefficients borrow no measurement. Coefficient cutoffs
   such as `small_term_rtol` or a thresholding solver's threshold depend on
   the data's units and scale: state them with the result. On a table, refit the numeric
   constants inside the terms: `refit = kd.refit_constants(dataset, terms)`,
   then read `refit.terms`, `refit.coefficients`, `refit.constants` and
   `refit.nmse`. The default grid refit uses finite differences. In the
   measured raw noisy-data cases it was not better than the algorithm's
   own coefficients, and replacing DLGA's network-derived coefficients
   worsened the result. That observation does not cover a refit on a
   separately prepared network field: derive it with `smooth_input(method="network")`
   and pass its id as `evaluation_input_id` to `refine_candidate`, stating
   the method and trim used.

10. **Check a candidate against the data.** Three separate discrepancy
    measurements: `kd.validate_terms(dataset, terms, max_order=3)` (can each term be
    evaluated), `kd.block_dispersion(dataset, terms, max_order=3)` (how much the
    coefficients move across grid blocks) and
    `kd.forward_error(dataset, terms, coefficients, max_seconds=seconds)` (the
    relative error of the law integrated from the first time slice; a
    first-order law on a grid). `max_seconds` stops the integration at a
    cooperative deadline with `reason="timeout"`, which is unmeasured, not an
    infinite error; a single long solver step is not interrupted, so run it
    in a process with a hard limit when that matters. None of the three
    establishes a law on its own.

11. **Check stability across runs.** Use several seeds or algorithms rather than
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

12. **Checkpoint long runs.** Set `checkpoint_dir=` and `checkpoint_every=` on
    the model before fitting, using a fresh checkpoint directory for each run.
    `point = kd.resolve_checkpoint(checkpoint_dir)`
    Select from the manifest through this call, never by filename sorting.
    `resumed.fit(dataset, resume_from=point.path)`
    The resumed model's `generations` is an additional segment budget.
    Parameters have resume-safe, init-only or identity-breaking tiers; consult
    their declarations. A refused configuration change names the field and both
    values. Restore that setting or start a fresh experiment as instructed.

13. **Write the report.** Render every applicable figure and one HTML page.
    `report = model.report(out_dir)`
    Include the paths and `report.warnings`, which lists skipped or degraded
    figure families, alongside the equation and comparison evidence. Name
    the frame of every measurement, and the operation that produced the
    reported coefficients (a fit, a refit and its method). When no candidate
    passes the checks you declared, report the result as unresolved, with the
    checks and their outcomes, rather than the closest law.

14. **Act on errors.** Read the remedy in an exception, such as a shape fix,
    LHS spelling, missing extra or supported topology. Apply that correction
    once before changing approach. If it still fails, preserve the error and
    input details in the report rather than claiming a completed discovery.
