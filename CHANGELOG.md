# Changelog

All notable changes to KD are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versioning policy: KD is in 0.x development. A patch release may contain a
breaking change; every breaking entry is marked **Breaking** and names what a
caller must change. The seven bundled instruments are updated in the same
release and are never the party that breaks. From 0.8.0 on, each release is
also published on PyPI and as a GitHub Release carrying longer notes.

## [Unreleased]

## [0.8.0] - 2026-09-15

The first release on PyPI. The distribution is named `sail-kd` (`pip install sail-kd`; PyPI does not allow the name `kd`), the import name stays `kd`, and the console script stays `kd-agent`.

### Added

- `kd.add_noise(dataset, level, *, seed)`: a copy of a `PDEDataset` with
  relative Gaussian noise on every field (the `xu2020_relative` recipe; field
  `k` draws from `seed + k`), `noise_level` recorded, the input untouched.
- `kd.search.used_clauses` and `kd.search.SKETCH_CLAUSES`: inspect the clauses a sketch uses before checking an instrument's per-mode declarations.
- `kd.load(source, ...)`: one entry for every dataset source. A catalog id
  returns that entry (`kd.load("kdv")` is `kd.load_kdv()`); a file path is
  read on evidence and never on a guess: a registered layout that recognizes
  the file's arrays builds it (`pdebench-1d` and `pdebench-cfd` for PDEBench
  HDF5 files, `kd-npz` for the self-describing npz convention the DISCOVER
  loader already used), otherwise the caller names the arrays with
  `coords=` / `fields=` (plus `field_axes=` and `select=` for a stored axis
  order or a sample axis), or passes `loader=`, a function of their own.
  A file no layout recognizes raises with the file inventory and the mapping
  form spelled out; a multi-sample PDEBench file without `select` is refused
  rather than defaulted. `.xlsx` routes to `PDEDataset.from_xlsx`. The layout
  and mapping routes end in `from_arrays` (xlsx in `from_scatter`), so
  validation runs once.
- `kd.inspect_file(path)` and `kd.SourceReport` / `kd.ArrayReport`: what a
  data file holds before any interpretation (per array: key, shape, dtype,
  finite min/max, NaN and Inf counts), for `.npy` / `.npz` / `.mat` /
  `.h5` / `.hdf5` / `.csv`. HDF5 arrays are streamed, so a
  multi-GB PDEBench file is described without loading it. `to_dict()` is the
  JSON view for a program or an agent that will write the `coords=` /
  `fields=` mapping.
- `PDEDataset.source` and `kd.DatasetSource`: the provenance `kd.load`
  records on a file-loaded dataset (resolved path, sha256, container, layout
  or loader name, hints, sample selection). It is not part of the dataset
  fingerprint (content identity) and is carried through `stride_subsample`
  and `ratio_subsample`; datasets built by a layout or a mapping are named
  by the file stem.
- `h5py` joins the core dependencies: the HDF5 reader behind the file ingress
  (`kd.load` / `kd.inspect_file`), which is the container PDEBench ships every
  dataset in and the one behind MATLAB v7.3 `.mat` files.
- `CITATION.cff` at the repository root, so GitHub's "Cite this repository"
  and citation tools pick up the software citation.
- `kd.resolve_checkpoint` / `kd.eligible_checkpoint_iterations` (also on
  `kd.search`), with `ResolvedCheckpoint` and `CheckpointSelectionError`: the
  one rule for which archive a segment restores from a `kd-ckptman-v1`
  checkpoint directory (periodic entries plus a completed final, the latest
  when no iteration is named, the last entry for a repeated iteration, nothing
  for a run that archived no directory). Until now kd shipped the ledger and
  every resuming caller re-derived the rule; example 10 and the controller
  lanes call this instead.
- `Model(algorithm="eqgpt")` works after a plain install: the pretrained GPT
  checkpoint (151.7 MB) and the 23 wave-breaking per-case surrogate
  checkpoints are fetched from the KD Hub mirror (`timeoutHao/KD-data`,
  `eqgpt/`, pinned to one commit, the GPT checkpoint sha256-verified) into the Hugging
  Face cache the first time they are needed. `KD_EQGPT_ASSET_DIR` and
  `KD_V1_WAVE_ASSETS` keep overriding the download with a local tree.
  `kd.data.remote` exports `fetch_eqgpt_weights`, `fetch_wave_surrogate_tree`
  and the directory fetch `fetch_hub_tree` behind them.
- `kd.search.EvaluationBudgetCallback`: the candidate-evaluation sibling of
  `WallClockBudgetCallback`. It charges every result handed across the
  propose/evaluate boundary (the same count `RunCost.boundary_results`
  publishes) and stops the search at the first iteration boundary at or past
  `max_evaluations`, so a run may overshoot the cap by one iteration and
  reports that spend rather than hiding it. It is this-experiment-only by
  design: cross-segment accounting stays with the caller's ledger.
- `kd.core.recovery` (exported from `kd`): the three-axis recovery judge
  `judge_recovery` / `RecoveryVerdict` / `span_floor`, re-homed byte for byte
  from the validation suite's conftest so experiment scripts and the
  controller package import one implementation, plus two partial-credit
  scores beside the binary verdict: `term_set_jaccard` (a set score on labels
  the caller has canonicalized) and `load_bearing_recall` (read off the
  verdict's drop-one margins, no canonicalization needed).
- `kd.core.rates` (exported from `kd`): `wilson_interval`, `rate_summary` /
  `RateSummary`, and `paired_exact_test` (two-sided exact McNemar on the
  discordant pairs), the reporting helpers the benchmark standard fixes for
  every success-rate cell.
- `kd.data.stride_subsample` / `kd.data.ratio_subsample`: the benchmark's
  sampling axis. `stride` decimates a GRID dataset per axis and keeps the grid
  (finite differences accept the result; a periodic axis refuses a stride that
  breaks the seam); `ratio` keeps a seeded uniform fraction of the boundary-
  trimmed interior as a SCATTERED dataset for the surrogate-derivative path.
  Both keep the dataset `name`; the fingerprint is what changes.
- `kd.data.resample_to_grid(dataset, axes=...)`: fit a `FieldModel` to a
  SCATTERED dataset and sample it on the regular `axes` you name, returning a
  GRID dataset the existing algorithms and the finite-difference provider
  accept with no schema or provider change. Training options follow
  `FieldModelTrainer.fit`; the task, LHS, periodicity, noise level and ground
  truth carry over, the name gains `_resampled`, and `source` describes a
  derived dataset. Sampling in a hole or outside the observations is
  unconstrained extrapolation: there is no coverage mask.
- `kd.load` reads a header-bearing CSV as one observation per row. Without
  mappings the field named by `lhs` is the only field and every other column is
  a coordinate, in header order; `coords=` / `fields=` select columns by name
  when there are several fields or unused columns. Coordinate combinations that
  are unique and complete pivot to a GRID dataset (the axes must be uniform),
  any other point set stays SCATTERED. Rows are never dropped: a selected
  column that is non-numeric or holds a non-finite value is an error.
  `kd.inspect_file` lists the named columns, text columns included. A
  header-less numeric matrix keeps the existing route.
- `ExperimentResult.search_trajectory()` and `has_search_trajectory()`, with
  `kd.search.SearchTrajectoryCandidate`: the distinct fitted structures a
  search evaluated, recorded per iteration. Each candidate carries the
  evaluated candidate IR, the instrument's native score, the fitted support
  (the selected non-zero columns) and the LHS it targets. SGA, DLGA, DISCOVER,
  LLM4ED and EqGPT record it; the one-shot backends (PySINDy, PySR) do not.
  `VizRecorder(trajectory_top_k=10)` bounds how many distinct structures an
  iteration keeps and `0` turns the recording off; it is a recording policy,
  so it is not part of the run identity or the checkpoint. A resume records
  from where it resumed and does not reconstruct the archived run's history.
- Two report figures read that record: `term_presence` (which fitted terms are
  active across the recorded structures, with targets kept apart) and
  `search_score_distribution` (median, interquartile range and min-max of the
  native scores). They are rendered beside the convergence curve whenever a
  result carries the record, and describe the recorded selection rather than
  the whole population.
- `kd.viz.plots.plot_equation_card` and the `equation_card` report figure: the
  published equation, its per-term coefficients next to the dataset's ground
  truth when there is one (aligned on structural term identity, so aliases and
  the constant column line up), and the native search-fit NMSE. It is a
  projection of the result and never refits.
- `Model.report(output_dir, animate=False)`: a fitted model renders its own
  figures and HTML report, so a caller no longer wires `VizEngine` by hand. In
  a notebook the returned `ReportResult` displays the report inline.
- `kd.viz.plots.plot_field(dataset, ax=None)`: one field of a dataset as it was
  measured, before any search (a heatmap for a GRID dataset with one spatial
  axis, a scatter for a SCATTERED one). `plot_plugin(algorithm, name, ax=None)`
  draws one named panel from an instrument's own visualization extension, and
  `save_field_animation(dataset, integration_result, path)` writes the measured
  and integrated fields to a GIF and returns the panel's disclosure notes. The
  single-axis `plot_*` functions now default to `ax=None` and create their own
  axes.
- `kd.evaluate.compare_results(dataset, results)` and `ComparisonResult`: refit
  every run's discovered structure on the same finite-difference features and
  the dataset's LHS, so runs are compared on one scale instead of their native
  scores. A run with no evolution equation, a different LHS, or terms the
  evaluator refuses is listed in `exclusions` with the reason.
  `ComparisonResult.render(path)` writes the equation table, the normalized
  progress and the refit-NMSE sheet.
- `kd.data.loaders.load_wave_breaking_datasets(path=None, case_filter="")`: the
  wave-tank cases as SCATTERED datasets in sorted case order, elevation named
  `u` with LHS `u_t`; `case_filter="N"` selects the paper's discovery
  experiments.
- `kd.format_pde(terms, coefficients, lhs=...)` with `kd.FormattedEquation`,
  and `kd.render_lhs_label(lhs_spec)` with `kd.LhsSpec`: the equation
  formatting kd's own reports use (LaTeX, Unicode and the SymPy objects
  behind them), so a script or an agent prints a discovered law the same way
  the report does.
- `kd.SKETCH_EXIT_VERIFY` and the `verify=` keyword on `Model.fit` and
  `kd.harness.run_episode`: the residual gate the sketch exit judges the
  lifted law with. See Changed for what this changes about a sketch run.
- `sail-kd[agent]` and the `kd-agent` command: the LLM controller ships inside the
  package as an experimental extra (`pip install "sail-kd[agent]"`), with the seven
  instrument cards as package data so its tool descriptions match the
  installed algorithms. `kd-agent setup` stores the endpoint, model name and
  API key in a `0600` config file; `kd-agent run` discovers an equation in one
  run and `kd-agent chat` does it as a conversation. Both accept `--dataset`
  for a catalog id or a saved input reference, `--data` with `--load-options`
  for a file of your own (read through `kd.load`), `--workspace` to reuse an
  existing run ledger, `--recursion-limit` and `--time-budget-minutes`. Every
  run and every chat turn writes `<workspace>/report.md` from the sealed
  records and the ledger, without reloading data or refitting. Installed
  without the extra, the command still parses its arguments and `run` exits 2
  naming the dependency it needs.
- `examples/notebooks/kdv_walkthrough.ipynb`: SGA recovers
  `u_t = -u u_x - 0.0025 u_xxx` from the bundled KdV benchmark at its
  validated budget, read term by term, with the convergence, parity,
  equation-tree and coefficient figures and the HTML report. Outputs are
  committed.

### Changed

- **Breaking** for install commands only: `pysindy`, `openai` and
  `huggingface_hub` are core dependencies, so `pip install sail-kd` runs six of the
  seven algorithms (SGA, DLGA, DISCOVER, EqGPT, PySINDy, LLM4ED) with no second
  step. `pysr` stays an extra because its first import downloads a Julia
  runtime, and `agent` stays one because it carries the controller's LangChain
  stack; `sail-kd[all]` selects both. The "install it with `uv sync --extra ...`"
  errors for the three folded-in backends are gone with them; the PySR one now
  names `pip install "sail-kd[pysr]"`. The `huggingface_hub` floor is 1.0 (the httpx
  client), and `socksio` rides along so the Hub downloads work behind a
  SOCKS proxy (`ALL_PROXY=socks5://...`), where that client otherwise
  raises `ImportError` on the first request.
- **Breaking** for sketch runs: the sketch exit judges the lifted law against
  the data, not only against its clauses. `Model.fit(..., verify=None)` and
  `kd.harness.run_episode(..., verify=None)` resolve to `kd.SKETCH_EXIT_VERIFY`
  (`VerifyPolicy(nmse_max=0.05)`), so a run whose clauses all matched but whose
  law does not fit the data now returns no certified solution, and the failure
  names the measured NMSE and the threshold. Pass an explicit `VerifyPolicy()`
  to restore the clause-only exit. A closed sketch is gated the same way, and a
  run where verification cannot be performed withholds the solution instead of
  issuing it. `verify` without a `sketch` raises `ValueError` before the plugin
  is built. The policy is not part of the run identity or the checkpoint.
- **Breaking** for PySR PDE runs: `PySRConfig.unary_operators` defaults to the
  empty tuple, so a PDE search no longer wraps structural terms in `sin`,
  `cos`, `exp` or `log` and fits constants inside them. The tabular mode and
  `Model(algorithm="pysr")` on a tabular task keep those four; an explicit
  `unary_operators=` is used as given either way.
- **Breaking**: `DerivativeProvider.diff` takes a keyword-only
  `is_periodic: bool | None = None`, and the executor passes it on every call
  (`None` keeps the provider's own periodicity). A provider of your own must
  accept the keyword. See Fixed for the artifact it exists to prevent.
- Ten instrument parameters move from `init_only` to `resume_safe`, so a resume
  may change them: SGA `depth`, `width`, `aic_ratio` and `lam`; DLGA `epsilon`;
  DISCOVER `max_length`; EqGPT `top_k` and `sparsity_alpha`; LLM4ED `pool_size`
  and `reward_limit`. The resume applies the live value to the carried state
  rather than the archived one: the pricing parameters re-evaluate the carried
  population, pool or champion instead of converting old scores, and the cap
  parameters re-gate the carried state (SGA eliminates the members that exceed
  the live `depth` / `width` and refills the population, DISCOVER clears a
  champion longer than the live `max_length`, EqGPT and LLM4ED re-gate the
  carried pool). A resume that leaves the value alone is bit-identical to
  before.
- EqGPT, LLM4ED and PySR declare `Segmentation(reseed=True)` and implement
  `reseed()`, so branching off an archived run re-derives the random streams
  from the live seed instead of continuing the archived lineage. PySR passes
  the seed through to `random_state` on a warm-started fit.
- The HTML report prints a caption under each of its standard figures saying
  what to read from it; downstream galleries reuse the same words.
- Progress lines and `repr(Model)` print the published equation the way the
  report renders it, instead of the raw IR of the best expression.

### Removed

- The `hub`, `pysindy` and `llm4ed` extras, whose dependencies are core now,
  and the wheel-level `dev` extra that duplicated the development dependency
  group (pip warns on an unknown extra and installs the package anyway).
  `pysr` and `agent` remain, and `sail-kd[all]` is `sail-kd[pysr,agent]`.

### Fixed

- Reports and plots of a sketch run show the published equation, fixed
  terms included, in the formula, the coefficient bar, the expression tree,
  the comparison table and the time integration; a sketch with no certified
  solution says so instead of presenting the partial fit as the law. Native
  fit metrics and residual plots name their target (the LHS minus the fixed
  terms) rather than being relabelled as full-equation quality.
- The visualization integrator refuses a law whose actual LHS is not the
  dataset's first-order evolution (a DLGA `u_tt` result, for instance) and
  returns the reason instead of integrating it as first order.
- Error heatmaps disclose the p99 colorbar clipping: colorbar arrows and the
  real error range per panel.
- The EqGPT steady surrogate panel computes R² as the plain SSE/SST ratio, so
  it no longer depends on the physical scale of the observations.
- `kd.load` rejects a file whose interior coordinates are not uniform instead
  of silently rebuilding the axis; kd-npz, explicit-mapping and PDEBench
  routes validate every point at the existing 1e-5 tolerance.
- Consensus verification of a TABULAR dataset uses the default derivative
  factory's own provider declaration (`none`), so tabular plans verify.
- A reused `ExperimentRunner` no longer reports the previous resume source
  on a fresh run's manifest.
- DLGA labels its LHS from the dataset's field and axis, and LLM4ED refuses
  a dataset whose field is not `u` or whose axis is not `x`, instead of
  either one hard-coding the names.
- A sketch anchor written as `u_x` matches a candidate written as
  `diff_x(u)`: anchored clauses compare column fingerprints, so a derivative
  alias no longer fails certification.
- A sketch is certified against the LHS the search actually reported. A plugin
  that fits `u_tt` (DLGA picks its target when `lhs_auto_select` is on, which
  a second-order target requires) was certified as covering the sketch's `u_t`,
  and its residual was scored against `u_t` as well, so the quality figure
  belonged to a different equation.
- SGA and PySINDy reject a sketch `operators` clause at compile time, naming
  the hole and the operator, when it contains a name the backend cannot emit or
  when the clause as a whole admits no operator. Until now the clause was
  dropped with a note in the compile report and the search ran in a space that
  could not contain the intended law. An empty `frozenset()` still means "bare
  terms only".
- Differentiating an expression that contains the differentiated axis's own
  coordinate as a leaf uses the non-periodic stencil. The coordinate is a ramp,
  not a periodic function, so the wrap-around template returned a jump rather
  than a derivative at the two boundary rows on each side: on the periodic
  Burgers grid, `diff2_x(sub(u, x))` fitted at NMSE 0.79 and now fits at
  3.2e-4, the same as `diff2_x(u)`. An expression that is genuinely periodic
  and carries that coordinate leaf loses the wrap template, which costs
  accuracy on the boundary rows and changes nothing in the interior.
- A PDE equation whose terms cannot be canonicalized (PySR's `sin(2*u)`, for
  instance) is no longer published as an `Equation`: the builder warns once and
  omits it, while the fitted expression, coefficients, scores, predictions and
  Pareto data are unchanged. Regression results keep their fitted constants.
- A resume that lowers SGA's `num` no longer truncates the population on the
  archived run's scores before the live pricing is applied, which could discard
  the member that was best under the live parameters and never recover it.
- Field heatmaps put the grid nodes at pixel centers: the image extent gains
  half a cell on each side, so the plotted coordinates are the dataset's.
- A field panel of a dataset with more than two spatial axes states which axes
  were held fixed and at what value, in the figure title and in the returned
  warnings.
- `PDEDataset.from_xlsx` logs the raw, kept and dropped row counts at INFO when
  missing values in the selected columns remove rows. A complete input and
  `drop_na=False` stay silent.


## [0.7.5] - 2026-09-08

### Added

- `kd.SearchInterrupted`: the exception an embedding process (a signal
  handler, a wall-clock watchdog) raises inside a running `fit` to stop it.
  Every place in the search stack that totalizes a failing candidate and keeps
  going now re-raises this type first, so the interrupt reaches the caller
  instead of being recorded as one more invalid candidate. The signal is
  scoped to one search: `kd.harness.run_plan` seals the interrupted episode as
  `raised` and continues with the next entry.
- `kd.core.platform.resolve_derivative_requirements(plugin)`: the resolver
  that turns an instrument's declared derivative requirements into the
  executor's, previously a private helper of the platform builder.
- `kd.core.expr.executor.DIFF_OPERATOR_PATTERN`: the one regular expression
  that recognizes a derivative operator token, previously copied into three
  private modules.

### Changed

- **Breaking**: `FiniteDiffProvider(method=)` is gone; the keyword was
  accepted and never read. Drop it from the call.
- **Breaking**: `PivotRegressionForm.rhs_irs` is gone; the field was written
  and never read. The form carries `pivot_ir` only.
- **Breaking**: `kd.search.eqgpt.Vocab.encode_sentence` and `decode_sentence`
  are gone; nothing in kd called them. Encode and decode tokens one at a time
  through the primitives `Vocab` keeps.
- **Breaking**: `kd.core.platform._resolve_derivative_requirements` is no
  longer exported; call `resolve_derivative_requirements` (above) instead.
- A plugin of your own that declares no `derivative_requirements` is now
  gated on the resolver's default (`GRID`, target order 1) before the run
  starts, where 0.7.4 skipped the capability gate for it. A plugin that fits
  a non-grid dataset or a second-order target must declare the property, the
  row `FacadeWiringContract` now names. `ExperimentResult.config` carries
  `provider_kind` for every plugin as a consequence.
- The HTML report prints the algorithm identifier (`"sga"`, what `kd.Model`
  takes) instead of the plugin class name (`"SGAPlugin"`), the resolution
  `Model.summary()` already used; the JSON block inside the report keeps the
  class name.
- The PySR audit figures title the re-score as KD rather than kd.

## [0.7.4] - 2026-09-01

### Added

- This changelog.
- `Model.train_surrogate(dataset)`: the surrogate-network training that sga
  (autograd mode) and dlga run inside `fit` becomes a callable step, returning
  the trained module and its `TrainingResult`; the module is bit-identical to
  the fit path's at the same seed. Impossible calls fail loudly
  (`NotImplementedError` / `TypeError` / `ValueError` by cause).
- `kd.models.save_field_model` / `kd.models.load_field_model`: a trained
  `FieldModel` round-trips through one file (`kd-surrogate-v1`); the loaded
  module keeps the source's `torch_module_artifact` digest, so one file
  injects into every segment of a lineage without tripping the resume gate.
- `kd.harness.run_episode(..., sketch=)`: an episode run with a sketch writes
  a `sketch.json` sidecar (per-clause verdicts plus an `evidence_hash`) at the
  run-directory root; the sidecar stays out of the manifest and the seal hash.
- `kd.search.SurrogateTrainer`: the protocol an instrument with a non-empty
  `config_artifact_keys` must implement; a registry-level test enforces the
  pairing.
- `kd.instrument_schemas()`: rows gain `config_artifact_keys` and
  `surrogate_fields`; each mode gains `sketch` (the six clause names mapped to
  a support level, present even when all six are `unsupported`), `lhs_orders`
  (the time-derivative orders the mode fits, a sorted list, or `null` when the
  mode does not declare them), and `score_frame` (the frame the mode's
  self-reported score is measured in, `"undeclared"` when unstated; the three
  frame words are constants in `kd.search.descriptor`).

### Changed

- **Breaking**: `InstrumentDescriptor(...)` requires a third keyword-only
  field, `surrogate_fields: frozenset[str]`, naming the config fields that
  shape the instrument's surrogate network, and refuses a descriptor where it
  and `config_artifact_keys` disagree about being empty. Bundled instruments
  are unaffected; a hand-built descriptor must pass the new set (the same
  adjustment 0.7.3 asked for the first two required fields).

## [0.7.3] - 2026-08-29

### Added

- PySR resumes across processes: a pysr checkpoint now carries the fitted
  `PySRRegressor` (populations and hall of fame) in the plugin's `state`, and
  `Model.fit(..., resume_from=...)` continues from it in a fresh process with
  one warm fit. `generations` on the resumed call is that segment's iteration
  increment, as for the five iterative engines. The archived regressor is
  about 90 KB at kd's defaults, about 140 KB after a resumed segment. A
  checkpoint written by 0.7.2 or earlier holds a fitted model without search
  state; resuming one raises a `ValueError` naming `search_state` instead of
  silently replaying the old fit.
- `kd.list_datasets_answer_blind()`: the bundled catalog as JSON-safe rows
  with six fields, `id`, `axes`, `lhs`, `fmt`, `tier`, `tags`, and nothing
  else. `kd.list_datasets()` still returns full `DatasetSpec`s whose
  `equation` is the ground truth a discovery run is meant to find, so a
  program that shows the catalog to a model should call the answer-blind one.
  The rows are built by naming the shown fields forward, not by deleting
  `equation` from a copy, so a later answer-bearing field cannot leak through.

### Changed

- **Breaking**: `InstrumentDescriptor(...)` requires `identity_breaking_fields`
  and `config_artifact_keys`, both keyword-only `frozenset[str]`: the config
  fields whose change makes a resume a different experiment, and the
  `.artifacts` entries that stand in for config values. The two module-level
  tables in `kd.search.resume_policy` that used to hold this per algorithm are
  gone; each bundled plugin declares its own. Construction refuses a knob that
  shares a name with either set while declaring a tier below
  `identity_breaking`, and the checkpoint writer raises `TypeError` for an
  algorithm without a descriptor. Hand-built descriptors must pass the two
  sets.
- The pysr row of `kd.instrument_schemas()` declares
  `segmentation["archive"] == "progress"` (it was `conclusion`), and
  `niterations` appears among its knobs as `resume_safe`.

### Fixed

- `generations` together with `config=PySRConfig(...)` is refused with a
  `ValueError`: for pysr the facade's `generations` maps into the config's own
  `niterations`, so giving both stated the same quantity twice and the
  config's value won silently. Every other algorithm keeps accepting
  `generations` alongside `config=`, where it drives the runner loop and
  reaches no config field.
- The Recording recipe in `Model.fit`'s docstring writes a resolvable catalog
  row: the 0.7.2 recipe passed `run_dir=str(paths.root)`, relative to the
  working directory, while every path column in a `kd-runcat-v1` row is stored
  relative to the catalog file, so a reader landed on a non-existent
  `runs/runs/<run_id>`. The recipe now passes the bare `run_id`, and the
  docstring states the rule.

## [0.7.2] - 2026-08-28

### Added

- A resume can branch: `Model.fit(..., reseed=True)` keeps the restored search
  state (population / controller weights / best) but re-derives the random
  streams from THIS model's seed the way a cold start would, so two branches
  off one checkpoint explore differently. With `reseed=True` a changed `seed`
  is the one config difference the resume gate accepts. A branch requires
  `resume_from` and an algorithm that declares it can branch; both refusals
  are a `ValueError` raised before any expensive build. The same keyword
  reaches `kd.harness.run_episode`.
- Instruments declare whether they can branch: `kd.instrument_schemas()` gains
  `segmentation["reseed"]`, `True` for sga, dlga and discover, `False` for
  pysr, eqgpt, llm4ed and pysindy. The run-directory and catalog lineage face
  gains a sixth key, `reseed`; artifacts written by earlier versions carry
  five keys, still load, and normalize to `reseed=False`.
- `Model.fit`'s docstring gained a Recording section naming the two public
  paths that put a run on disk (`kd.harness.run_episode`, and a short
  `kd.search` recipe around the facade); a fit by itself writes no run record
  and no catalog row.

### Fixed

- A checkpoint records which dataset it was searched against: the payload
  carries a `dataset_fingerprint` taken at the same facade stage on the
  writing and the reading side, and a mismatch raises `ValueError` instead of
  continuing a search whose restored scores were priced on other data. The key
  is additive-optional; a checkpoint written without it resumes as before.
- DISCOVER re-gates a restored Pareto front row by row through the live
  magnitude filter before Pareto dominance, which is what makes
  `magnitude_filter` honestly `resume_safe`.
- `from_sympy` emits executable IR for non-integer rationals: a non-integer
  `Rational` used to serialize as `div(int, int)`, which the executor cannot
  run. It now emits a single Float literal, exact when the rational is a
  binary float (1/2, 3/4), approximate otherwise (1/3).

[Unreleased]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.7.5...v0.8.0
[0.7.5]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.7.4...v0.7.5
[0.7.4]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.7.3...v0.7.4
[0.7.3]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.7.2...v0.7.3
[0.7.2]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/releases/tag/v0.7.2
