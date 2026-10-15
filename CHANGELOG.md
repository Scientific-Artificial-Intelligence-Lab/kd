# Changelog

All notable changes to KD are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versioning policy: KD is in 0.x development. A patch release may contain a
breaking change; every breaking entry is marked **Breaking** and names what a
caller must change. The seven bundled instruments are updated in the same
release and are never the party that breaks. From 0.8.0 on, each release is
also published on PyPI and as a GitHub Release carrying longer notes.

## [Unreleased]

## [0.8.2] - 2026-10-15

### Added

- Mode declarations that say which input a mode takes and which parameters
  select it. Every mode in `kd.instrument_schemas()` gains nine keys:
  `selected_by` and `order_params` (the parameters that select the mode, the
  second keyed by the left-hand-side order), `required_params`,
  `default_overrides` (defaults the facade applies in this mode only when the
  caller supplied none of the group), `refused_fields`, `field_names` and
  `spatial_axes` (the names the mode's default vocabulary is written in),
  `names_from` (the parameter that carries a vocabulary built from the input's
  own names) and `selectable_from_data`; each schema gains
  `facade_inert_fields`. `kd.mode_accepts(mode, *, topology, form, lhs_order,
  field_names, spatial_axes)` reads one of those declarations and returns a
  `kd.ModeVerdict(accepts, reason, params)` without loading data; a `form`,
  `topology` or `lhs_order` outside the closed sets raises `ValueError`.
  `kd.search.mode_inputs(report)` projects `kd.preview_report(...).to_dict()`
  onto those arguments. `kd.instrument_schema_digest()` is the sha256 of the
  declarations with the descriptive prose left out: it changes when a default,
  a mode's applicability or a parameter's settability changes, and stays the
  same when only a description is reworded.
- Two self-describing NPZ wire formats. `kd.load_tabular(path, *, name=None)`
  reads a regression table (`format="kd-tabular-npz-v1"` with `X`, `y`,
  `var_names`, `target_name`) into a float64 `TabularDataset`, keeping the
  column order and refusing names the tabular bridge refuses. The layout
  `kd-scattered-npz` (`format="kd-scattered-npz-v1"` with `axis_order`,
  `field_names`, `coord__<axis>`, `field__<name>`) loads per-point vectors
  through `kd.load` as a scattered dataset; it keeps the point order and
  repeated coordinates, never infers a grid from points that happen to form
  one, is recognized without naming the layout, and takes `lhs=""` for a
  steady relation and `periodic=[]`.
- `kd.preview_report` and `kd.preview` accept a `TabularDataset`; the report
  has `topology == "tabular"`, `axes is None`, `lhs_order == 0`, the target
  column as `lhs_field` and an empty diagnostics mapping.
- Every configuration field that a JSON caller can set now has a non-empty
  `description` in `kd.instrument_schemas()`. For fields that are not search
  knobs it is the first sentence of the attribute docstring in the
  configuration class.
- Finite differences on non-uniform grids:
  `kd.data.derivatives.finite_diff.coordinate_diff(f, coords, axis, order,
  is_periodic=False)` differentiates along an axis with respect to its
  coordinate values. Uniformly spaced coordinates use the existing stencils and
  give bit-identical results to `FiniteDiffProvider`; non-uniform coordinates
  use finite-difference weights computed for the actual nodes, with the same
  points per row (five on interior rows, three on the second and second-to-last
  rows, one-sided at the ends). A periodic non-uniform axis is refused.
  `FiniteDiffProvider` still requires uniform axes.
- ODE trajectories in `kd.data`: `Trajectory` and `TrajectorySet` (trajectories
  of one system with their own time arrays, states and per-trajectory parameter
  values), `compute_dataset_fingerprint` for a `TrajectorySet`, and
  `trajectories_to_tabular(trajectories, target_state, trim=0)`, which builds a
  `TabularDataset` whose target `d<state>_dt` is the time derivative of one
  state estimated with `coordinate_diff`.
- Twelve generated ODE systems: `kd.data.ODE_SYSTEMS`, `ODE_CORE_TRACK` and
  `generate_ode_trajectories(system, seed=...)`, a port of the KnowledgeDiscover
  benchmark's ODE generator (DOP853 or Radau, `rtol=1e-10`, `atol=1e-12`); each
  system carries its equations with the parameter values substituted. They are
  not entries of `DATASET_CATALOG`, whose loaders return grid datasets.
- A tabular mode for `Model(algorithm="pysindy")`: on a `TabularDataset` the
  library is the constant term `one` followed by every monomial of the feature
  columns up to `degree` (new config field, default 2), solved with PySINDy's
  STLSQ, and the result is a regression equation with the target column on the
  left. A law with an additive constant is represented with the constant as the
  coefficient of `one`.
- `PySINDyConfig.relative_threshold` (default `False`): when `True`, the
  threshold passed to STLSQ is `threshold` times the L2 norm of the target the
  optimizer receives, so a term is kept in a round when its coefficient times
  its column norm is at least `threshold` times the target norm. The selected
  terms then do not change when a column or the target is rescaled. It requires
  `normalize_columns=True`; the other combination raises `ValueError`. With a
  `kd.Sketch`, the target is the one left after pinned terms are subtracted and
  anchored columns are projected out. The field is also a declared knob
  (`init_only`).
- `SGAConfig.restart_interval` (default `None`, also accepted as
  `kd.Model(algorithm="sga", restart_interval=...)`): with an integer k, the
  population is re-drawn from scratch once k generations have been committed in
  the current segment, at the start of the next generation, so a value at or
  above `generations` changes nothing. Restart j (j = 1, 2, ...) re-seeds with
  `seed + 1000 * j` and empties the history of expressions already seen.
  `generations` stays the total across segments, and the re-drawn populations
  are not counted as evaluations. The reported equation is the best population
  head across segments, ties going to the earlier segment. With finite
  differences each segment equals an independent fit of k generations at its
  seed; with `use_autograd=True` the surrogate is trained once and shared by
  every segment. Checkpoints carry the segment state, so a resumed run ends
  with the same answer as an uninterrupted one; a resume under lower `depth` or
  `width` caps drops a stored best that violates them. The field is
  `init_only` and is not a knob.
- `kd.data.savgol_smooth(dataset, window_length, *, polyorder=3)` smooths
  every field of a grid dataset along every axis, in `axis_order`, with
  `scipy.signal.savgol_filter(..., mode="interp")`, and keeps each field's
  dtype and device. It returns a new dataset with the same name, axes,
  left-hand side, `noise_level`, ground truth and source; other topologies
  raise `ValueError`. `kd.Model(savgol_window=...)` (default `0`, off) applies
  it with order 3 before the search, for every algorithm, and
  `Model.train_surrogate` trains on the same smoothed data. The run record's dataset
  fingerprint is that of the smoothed data, and the window is not written into
  the run config, so in a harness plan each smoothing setting needs its own
  `dataset_ref`.
- `kd.Model(small_term_rtol=...)` (default `0.0`, off), and the same keyword
  on `kd.search.ExperimentRunner`: a value r in (0, 1) drops from the final
  answer every term whose finite coefficient magnitude is below r times the
  largest finite coefficient magnitude, without refitting. The largest term
  always stays, and a non-finite coefficient is never dropped. In a
  homogeneous result the pivot term (coefficient fixed at 1) always stays and
  is not counted in the largest magnitude.
  `result_.equation` and the record's `support` and `coefficients` reflect the
  drop; `best_expr_` stays the algorithm's own string, `catalog_fit` keeps the
  dropped terms as inactive with their original coefficients, and metrics,
  score and complexity describe the algorithm's own fit. It is not applied on
  sketch fits. A positive value is written into the run config as
  `small_term_rtol`; at the default the run config and every run hash are
  unchanged. A value outside [0, 1), or NaN, raises `ValueError` when the
  model is built.
- `kd.instrument_schemas()` lists `savgol_window` and `small_term_rtol` in
  every algorithm's `facade_params`, under two new `effect` values,
  `"preprocessing"` and `"answer_selection"`, with no `config_field`. The
  parameter table of the experimental `kd-agent` controller shows both rows
  with a one-line note.
- DLGA accepts sketches that carry no pinned terms. Its `default` mode now
  declares `anchors` and `hole_count` as `exit_checked` and
  `derivative_order`, `operator_set` and `field_axis_set` as
  `generation_enforced`; `fixed_terms` stays `unsupported`, so a sketch with a
  pinned term is still refused by the capability gate. Under a sketch DLGA
  narrows `config.library` to the tokens a hole admits or an anchored term
  needs, refuses before scoring any candidate containing a term that is
  neither an anchor nor admitted by a hole, and publishes its own compile
  report. It fits only the left-hand side the sketch states and does not
  consult `lhs_auto_select`. A sketch DLGA can never satisfy is rejected when
  the plugin is built, before the surrogate is trained: an operator name other
  than `mul`, an anchored term whose factors are missing from the library or
  outnumber `max_module_length`, or more required terms than `max_modules`.
  Fits without a sketch are unchanged.
- DISCOVER accepts sketches in its finite-difference mode. `mode1` now
  declares `fixed_terms` as `lowered`, `anchors` and `hole_count` as
  `exit_checked`, and `derivative_order`, `operator_set` and
  `field_axis_set` as `generation_enforced`; the `tabular` mode declares no
  sketch support. Under a sketch DISCOVER narrows `config.library` to the
  tokens a hole admits or an anchored term needs (configured order kept;
  `lap` and the axis-less `diff` tokens are removed), caps the cumulative
  derivative order at the smaller of `max_diff_order` and the highest order
  a hole or anchored term allows when every hole bounds its derivative
  order, and refuses before scoring any candidate containing a term the
  sketch vocabulary cannot analyze, one that respells a pinned term (same
  law key or column fingerprint), or one that no hole admits and no anchored
  term matches. The platform subtracts pinned terms from the target and the
  exit restores them. The engine is built from a derived configuration, so
  `config`, and with it the resume identity, keeps the caller's values. A
  sketch DISCOVER can never satisfy is rejected before the controller is
  built: an operator clause naming a feature that no configured token
  fitting inside the clause produces, an anchored term whose fields,
  coordinates, operator features or per-axis derivative orders the
  configured tokens cannot build or whose order exceeds `max_diff_order`,
  a narrowed vocabulary left without an operator token, a required hole
  that admits nothing after narrowing or that no kept operator can root
  when `add` and `sub` are not configured, or a sketch that needs two or
  more terms when neither `add` nor `sub` is configured. Fits without a
  sketch are unchanged. The experimental
  `kd-agent` controller still refuses DISCOVER sketches.
- EqGPT accepts sketches in its `single_wave` mode. `single_wave` now declares
  `fixed_terms` as `lowered`, `anchors` and `hole_count` as `exit_checked`, and
  `derivative_order`, `operator_set` and `field_axis_set` as
  `generation_enforced`; `wave_multicase` and `steady` declare no sketch
  support. Under a sketch EqGPT masks the vocabulary tokens that no hole admits
  and no anchored term needs (each token is judged by its own derivative order,
  fields, axes and operators; `*` and `/` are masked when no hole admits `mul`,
  or `mul` with `recip`), and refuses before scoring any candidate containing a
  term the sketch vocabulary cannot analyze, one that respells a pinned term
  (same law key or column fingerprint), or one that no hole admits and no
  anchored term matches. The platform subtracts pinned terms from the target
  and the exit restores them; the declared-variable reward gate counts the axes
  the pinned terms carry as present. The mask applies to the fit's sampler only,
  so `config`, and with it the resume identity, keeps the caller's values. A
  sketch EqGPT can never satisfy is rejected before the pretrained weights load:
  an operator clause naming a feature that no vocabulary token fitting inside
  the clause produces, an anchored term that no single token or `*`/`/` chain
  of tokens can spell, anchored terms whose single-token spellings form a
  combination the reward gate scores 0, more required term positions than one
  sentence of `max_length` tokens holds (an anchored term takes the atoms of
  its shortest spelling, once per column; a hole filled only by a product
  takes two), a required hole that admits no kept token and no product of two
  kept tokens joined by a kept `*` or `/` off the pinned columns, a sketch
  that leaves no term token, a first operator position within `max_length`
  that the configured mask and the sketch mask together leave without any of
  `E`, `+`, `*` and `/`, or an axis the reward gate requires that neither the
  start prefix, a kept token nor a pinned term carries. Fits without a sketch
  are unchanged. The experimental `kd-agent` controller still refuses EqGPT
  sketches.
- PySR accepts sketches in its `default` (PDE) mode. `default` now declares
  `fixed_terms` as `lowered`, `anchors`, `hole_count` and `operator_set` as
  `exit_checked`, and `derivative_order` and `field_axis_set` as
  `generation_enforced`; the `tabular` mode declares no sketch support. PySR
  compiles the sketch when the plugin is built: it reads each configured term
  as the law terms a PySR fit of that term alone converts to, keeps a term when
  each of those is admitted by a hole or matches an anchored term's column
  (configured order kept), removes pinned terms by law key and by column
  fingerprint, and refuses before scoring
  any converted candidate containing a term the sketch vocabulary cannot
  analyze, one that respells a pinned term (same law key or column
  fingerprint), or one that no hole admits and no anchored term matches. The
  kept terms bound the Julia search; the search itself is not steered per hole.
  When PySR's own selected expression is refused, the fit reports no liftable
  law instead of selecting another entry and logs a warning naming the
  expression. PySR's `plus`, `sub`, `mult` and `div` count as `+`, `-`, `*` and
  `/`. The platform subtracts pinned terms
  from the target and the exit restores them. `config` keeps the caller's
  `terms` and publishes the fingerprint of the narrowed library. A sketch PySR
  can never satisfy is rejected when the plugin is built, before Julia starts:
  an anchored term that no configured term converts to (matched by column
  fingerprint, so any spelling of the column counts), an operator clause naming
  a law feature that no configured term the hole admits converts to and no
  configured PySR operator emits, or admitting no configured term and no PySR
  composition as a whole, an empty library after narrowing, a required hole
  that admits no kept term outside the anchored columns, no kept term that
  converts to a pinned column, and no PySR composition of an anchored term
  with itself, a configured term whose conversion yields no law term, or a
  sketch that needs two or
  more terms when neither `+` nor `-` is configured. Fits without a sketch are
  unchanged. The
  experimental `kd-agent` controller still refuses PySR sketches.
- `kd.core.equation.term_admission(term_ir, sketch, *, pinned_keys,
  pinned_fingerprints, anchor_fingerprints)`: the pre-score admission rule the
  DISCOVER, EqGPT and PySR sketch backends share (outside the sketch
  vocabulary, a pinned law key or column fingerprint, an anchored fingerprint,
  or some hole's constraint). It returns `None` for an admitted term and the
  refusal reason otherwise.
- Optional runner-callback member `on_iteration_updated(iteration, algorithm)`:
  the runner calls it right after `algorithm.update()` returns and before any
  `on_iteration_end`, so a callback that labels the algorithm state learns the
  completed iteration independently of its position among the observers.
  Callbacks that do not define it are unaffected.
- `kd-agent`: a successful `run_discovery` envelope carries
  `primary.equation`, the sealed law on one line with its signed
  coefficients against the left-hand side the record names (`0` for a
  homogeneous law, the target column for a regression); every support term
  appears once, in order, with nothing merged or dropped. The line is absent
  when the left-hand side or any coefficient is unknown, and a rendering
  failure is reported in `failures` while the envelope stays a success.
  `primary.display` and `primary.law` are unchanged; `submit_answer` still
  takes `primary.law`.
- `kd-agent`: the segment ledger row's `cost` records `time_cap_seconds` and
  `stopped_by_time_cap` beside the evaluation-cap pair, so two segments that
  differ only in their wall-clock cap can be told apart after the run.
- `kd.instrument_schemas()` and `kd.search.tool_schema` publish
  `sketch_lower_owner`, the plugin's declaration of which component subtracts
  a sketch's pinned terms (`platform` or `native`). Together with each mode's
  `forms`, `lhs_orders` and `provider_kind`, it tells a caller from the schema
  alone which modes kd's capability gate consults for a given sketch.
- The lab mirror ships a tag-triggered release workflow
  (`.github/workflows/release.yml`): a `v*` tag builds the distributions,
  checks the wheel's contents, publishes to PyPI through Trusted Publishing
  and creates the GitHub Release from the matching CHANGELOG section.
  `export-public.py` copies an asset set's `.github/` directory into the
  staging tree; the personal mirror has none.
- `kd.search.DeadlineAwareSearchAlgorithm`: an optional protocol with one
  method, `set_wall_clock_deadline(deadline)`, where `deadline` is a time on
  the `time.perf_counter()` clock. `WallClockBudgetCallback` announces
  `started + max_seconds` to an algorithm that implements it when the
  experiment starts (after `prepare()`, before the first `propose`); other
  algorithms are untouched. PySR implements it.
- `kd-agent`: the offline judge's `kdagent.analysis.judge.Estimator` takes
  `needs_surrogate=True` for DLGA's derivative footprint, where the raw field
  as well as every derivative order is read from the surrogate network
  (estimator name `autograd+surrogate_field`). It requires the autograd
  provider and refuses `surrogate_max_order`.
- `kd.data.synthetic.generate_variable_diffusion_data(k_profile)`: 1-D
  diffusion `u_t = diff_x(mul(K, u_x))` whose conductivity `K` is a second
  dataset field next to `u` (conservative flux form with harmonic-mean face
  conductivity, explicit Euler within the stable step, zero Dirichlet ends,
  101 x 161 grid). Its docstring states how each algorithm uses `K`. The
  catalog dataset `variable-diffusion-smooth` uses `K = 1 + 0.5 sin(2 pi x)`;
  fitted with kd's central differences on the full grid, the true term
  reaches NMSE 9.8e-3. A piecewise-constant `K`, such as the three values
  0.5 / 1.5 / 0.25 of Chang & Zhang (2019), is passed as node values; there
  the full-grid fit reaches NMSE 0.58 because the error sits at the jumps of
  `K`, and fitted away from the jumps (see `rows` and
  `kd.data.interface_mask` below) the coefficient is 1.0001 at NMSE 3.9e-4.
- `kd.evaluate_terms(..., rows=...)`, and the same keyword on
  `Evaluator.evaluate_terms` and `Evaluator.build_theta_matrix`, select the
  rows a fit uses. `rows` is a boolean mask with the dataset's grid shape or a
  1-D integer index tensor. Terms and derivatives are still computed on the
  full grid; the least-squares fit, residuals and every metric use the
  selected rows only. `rows=None` (the default) leaves results unchanged.
- `kd.data.interface_mask(coefficient, *, axis, width=3, trim=None)` builds such
  a mask: it drops the `width` nodes on each side of every jump of a
  piecewise-constant coefficient field along `axis`, and `trim` drops a given
  number of rows at both ends of chosen dimensions (for example the first and
  last time rows).
- `EqGPTConfig.steady_boundary_grid`: the grid used by steady boundary
  deletion, given as six numbers `(x_start, x_stop, x_step, y_start, y_stop,
  y_step)`; each axis is `np.arange(start, stop + step, step)`. Both steps
  must be positive and each axis must start below its stop. It requires
  `steady=True` and `steady_boundary_delete_num`, and a JSON caller sets it as
  a list of six numbers. `EqGPTConfig.steady_preset("eitech")` and
  `steady_preset("smile")` set it, and their evaluation domains are
  bit-identical to before. With the default `None` the grid is the distinct
  coordinate values in the data, which was already the behavior for every
  dataset other than the two catalog datasets below.
- Per-field data diagnostics in `kd.preview_report`. Each `FieldReport` has
  three dictionaries keyed by axis name: `noise_estimate`,
  `first_derivative_gap` and `second_derivative_gap`. An axis is included
  when its spacing is uniform and it has at least five points, and only for a
  field without NaN or Inf values; scattered and tabular datasets give empty
  dictionaries. `noise_estimate` is the median absolute deviation of the
  second difference along the axis, scaled to a standard deviation
  (`1.4826 * MAD / sqrt(6)`) and divided by the population standard deviation
  of the field, so it is on the same scale as the relative `level` of
  `kd.add_noise`. The two gaps compare second-order central differences with
  step 1 and step 2 on the common interior points, normalized by the L2 norm
  of the step-2 result: `||4A - B|| / ||B||` for the second derivative and
  `||2C - E|| / ||E||` for the first, where `A` and `C` use step 1 and `B` and
  `E` use step 2. They describe the raw field, not the derivative method a fit
  uses. An infinite gap is written as `null` in `to_dict()` and the axis key
  is kept. The diagnostics read only the field values and the axis reports,
  never `ground_truth` or `noise_level`. `kd.preview` prints the same text as
  before.
- The reserved constant term `one` works on tabular data: on a
  `TabularDataset` its column is all ones with the length of the target column
  (grid and scattered datasets take the shape from their axes, as before). The
  reserved name is public as `kd.core.expr.executor.UNITY_TOKEN` (value
  `"one"`, unchanged).
- Two answer-blind checks of a candidate law on grid data.
  `kd.block_dispersion(dataset, terms, *, n_blocks=4, axis=None,
  max_order=2, lhs_order=None)` trims two grid points at both ends of every
  axis, splits the remaining indices along one axis (default the first
  spatial axis) into `n_blocks` contiguous blocks, fits the terms on each
  block with `kd.evaluate_terms(..., rows=block)`, and returns the mean over
  terms of `std(c) / max(|mean(c)|, 1e-12)`, where `c` holds one term's
  coefficients across the blocks. A failed block fit raises
  `EvaluationFailedError`; a dataset that is not a grid raises `ValueError`.
  `kd.forward_error(dataset, terms, coefficients, *, method="Radau",
  max_step=None)` integrates the law from the data's first time slice with
  `kd.core.integrate_pde` and returns a `kd.ForwardError`: either `value` is
  `||pred - u||_2 / ||u||_2` over the full grid in float64, or `value` is
  None and `reason` is `"diverged"`, `"solve-failed"` or `"refused"` (for
  example a dataset that is not a grid, a term with a time derivative, or a
  grid dataset whose `lhs_order` is not 1, which `forward_error` itself
  refuses before calling `integrate_pde`); `warning` says why. Both
  functions resolve unset `lhs_field` / `lhs_axis` the way
  `kd.evaluate_terms` does and leave the caller's dataset unchanged. Neither
  function applies a time limit; a call that needs one goes in a subprocess.
  KD gives no rule for when to use either number, and the default answer
  selection is unchanged.
- `kd.core.assemble_rhs(terms, coefficients, keep)` builds the right-hand
  side string that `kd.core.integrate_pde` takes:
  `"(c0)*(term0) + (c1)*(term1) + ..."` over the indices in `keep`, with
  coefficients written by `repr()`. Exactly-zero coefficients are omitted,
  an empty selection gives `"0"`, and a non-finite coefficient raises
  `ValueError`. It was a private function of the visualization layer;
  `kd.viz.build_integration_result` and `kd.forward_error` now both call it.
- `kd.load_wave_breaking_datasets(path=None, case_filter="")` is exported
  from `kd` and `kd.data` (previously only from `kd.data.loaders`). It
  returns the wave-tank cases as SCATTERED `PDEDataset`s, the data layout
  that `Model(algorithm="eqgpt")` in wave multi-case mode accepts; the
  existing `kd.load_wave_breaking` returns a `TabularDataset`.
- Process-group supervision in `kd.harness`: `ProcessSupervisor` launches a
  command as the leader of its own session and process group and supervises the
  whole group (a hard time limit, SIGTERM, a grace period, SIGKILL, a bounded
  wait after SIGKILL, sampled peak resident memory of the group). It can be
  polled one step at a time, stopped on request through the same escalation, or
  waited on with an optional cancellation event; `run_supervised` launches and
  waits in one call. Both return a `ProcessResult` whose `ended_by` is a
  `ProcessEnding` (`exit`, `sigterm`, `sigkill`). `build_capped_command` wraps a
  command in a `systemd-run --user --scope` with `MemoryMax` and
  `MemorySwapMax=0`, and `check_memory_cap_available` checks that such a scope
  can be started.
- `Llm4edConfig.extra_body` (and `OpenAICompatProvider(extra_body=...)`): an
  optional dict of request options sent unchanged with every LLM request, for
  endpoint-specific switches such as
  `{"chat_template_kwargs": {"enable_thinking": false}}` on NVIDIA's
  OpenAI-compatible endpoint. `None` (the default) sends the same request as
  before.
- `kd.harness.EpisodeOutcome.error_traceback`: the full `traceback.format_exc()`
  text of the exception a raised episode caught. `kd.harness.run_episode` sets it
  on every `raised` outcome; it is `None` for `completed` and `no_record`
  outcomes, and defaults to `None` for an outcome constructed directly. Unlike
  `error_message`, it is not cut at 2000 characters. `EvidenceStore` does not
  persist it.
- `kd-agent`: `kd-agent run --memory <document>` reads a `kd-experience-v2`
  document on a file input (`--data`, or `--dataset` pointing at a saved
  input), retrieves up to two pipelines that judged best on training problems
  whose report features and task kind match the input's, and appends them to
  the system prompt as one block. The document must sit outside the workspace.
  A catalog id, or a run that pins no dataset, is refused with `--memory`
  before a model is built. `trace.jsonl` metadata carries only
  `memory_sha256`; the closing result also carries `memory_path`. When nothing
  is retrieved the system prompt is byte-identical to a run without
  `--memory`. `chat` has no `--memory`. `--advice <document>` reads a `kd-experience-advice-v1` document
  under the same rules and independently of `--memory`: usage notes are shown
  under the retrieved rows they apply to, and cost anchors for the algorithms
  and modes that take this input are added to the task section.
- `kd-agent`: file inputs cover every task kind. Besides a grid, `--data`
  takes scattered points (`kd-scattered-npz`, for a steady relation) and a
  table (`--load-options '{"layout":"kd-tabular-npz"}'`, with `--task-kind ode`
  when the target column is a time derivative). The system prompt carries a
  task section and a glossary for the input's task kind (`pde-evolution`,
  `pde-steady`, `tabular-regression`, `ode`), listing which algorithms and
  modes take this input and the parameters each requires, as `kd.mode_accepts`
  reports them. The seven algorithm cards are rewritten on the same
  declarations. An answer on a steady input has no left-hand side; on a table
  it is the target column.
- `kd-agent`: `submit_answer` refuses a law whose `lhs` is missing or differs
  from the left-hand side the named search sealed; the refusal does not end
  the run. Dropping a negligible term from the law is still accepted.
- `kd-agent`: `kd-agent fingerprint [--json]` prints the sha256 of what the
  model is shown for the configured protocol and model, without calling an
  endpoint. `run` and `chat` take `--no-config` (ignore the saved endpoint
  file) and `--effort` (reasoning effort passed to the provider). The
  close-out lines gain `prompt_sha256`, `system_prompt_sha256`, `task_kind`,
  `target` and, with `--advice`, `advice_path` and `advice_sha256`; a reply
  the provider explicitly refused ends the run with `stop_reason: refused`.
- `kd-agent`: `kd-agent run --result <path>` writes the run's close-out lines
  (`stop_reason`, `answer`, `answer_source`, `answer_provenance`,
  `answer_nudges`, `truncation_nudges`, `endpoint`, `model`, `report_path`,
  and the `prior_*` / `memory_*` lines when a document was given) as one JSON
  object, replacing the file in one step. Standard output is unchanged.
- `kd-agent`: `kd-agent run` answers SIGTERM by stopping the searches it is
  supervising (SIGTERM to each search's process group, then SIGKILL after the
  episode grace period), writing a close-out with `stop_reason: interrupted`
  to `--result` when one was given, and exiting with status 143. The searches
  run in sessions of their own, so before this a SIGTERM to the controller
  left them running. The interrupted close-out has `answer`, `answer_source`,
  `answer_provenance`, the two nudge counts and `report_path` set to null and
  carries no document keys. `chat` is unchanged.
- Cross-batch merge in `kd.harness`: `merge_stores(sources, *, root, name,
  version_note=None)` writes one new evidence store from the sealed stores of
  separate `run_plan` batches. The merged plan is the sources' entries
  concatenated in argument order, so a single `run_plan` over the same entries
  under the same name has the same plan hash and entry indices. Records are
  copied byte for byte, and the new store is reloaded and verified. The merge
  is refused, before the new root is created, when one entry was attempted in
  two sources, when one dataset name refers to different data, or when a source
  is itself a merged store. Sources that differ in `kd_version` or `git_sha`
  require a `version_note`; the store report and the consensus report
  (`render_consensus_markdown(report, store=...)`) then list each source's
  version and commit with the note. A merged store writes index version 2
  (keys `sources` and `version_note`, read through `EvidenceStore.sources` and
  `EvidenceStore.version_note`), and `EvidenceStore.create` takes the matching
  keyword-only `sources` and `version_note`. Stores written by `run_plan` keep
  index version 1 and load as before.
- DISCOVER MODE2 through the facade: on a fresh call,
  `Model(algorithm="discover", config=DiscoverConfig(pinn=PINNConfig(...)))`
  or `Model(algorithm="discover", pinn={...})` pretrains a network on the observations, searches on its
  autograd derivatives, and retrains it on the best equation after `n_iterations` generations and then
  every `cycle_n_iterations` generations; `generations` stays the total search length and `n_cycles` is
  not read. On a fresh call a PINN phase runs only when `n_iterations` is smaller than `generations`:
  with the defaults
  (`n_iterations=2000`, `generations=50`) none runs, and the facade warns, naming both values. The mode
  is declared as `mode2` (autograd, grid, first-order left-hand side, `required_params=["pinn"]`); the
  final equation is fitted on the collocation points. A sketch, `EarlyStoppingCallback` and a
  `TabularDataset` are refused with `pinn`.
- DISCOVER MODE2 checkpoints and resume. A checkpoint written with `pinn` carries a `mode2` block in the
  plugin state: the network's weights, the seed the observation and collocation points were drawn with,
  the generation and cycle counters, the pretraining and per-phase metrics, and the stability candidate
  pool with the set of expressions it has seen. `fit(resume_from=...)` restores the network from it and
  does not pretrain. Generations are counted across resumed segments, so the PINN phases keep their
  places; a checkpoint taken on a boundary whose phase has not run resumes by running that phase first,
  inside `prepare()`, before any generation. A checkpoint written while a PINN phase was interrupted
  holds the network as it was before that phase, and the resumed run repeats the phase. A branch
  (`reseed=True`) keeps the parent's observation and collocation points; its search, the local samples
  and the stability selection follow the new seed. `DISCOVERPlugin` implements
  `DeadlineAwareSearchAlgorithm`: with `pinn`, a PINN phase starts no further training epoch once the
  announced deadline is reached and counts as completed, so `WallClockBudgetCallback` ends the run after
  the next generation; the phase a resumed run executes inside `prepare()` precedes any deadline and is
  not shortened. The `mode2` block goes with the PINN mode: a plugin running that mode refuses a state
  without the block, and any other plugin refuses a state with it, both with `ValueError`. MODE2
  checkpoints written before the block existed are therefore refused on resume; run those searches again
  from the start. `CHECKPOINT_VERSION` is unchanged.
- `PINNConfig` is exported from `kd.search.discover` and `kd.search`; `PINNConfig.obs_ratio` (default
  1.0, the share of grid points the network is trained on) and `PINNConfig.collocation_cut_ratio`
  (default 0.05). A nested config dataclass such as `pinn` can be given as a JSON object
  (`pinn={"pretrain_epoch": 5000}`); `kd.instrument_schemas()` publishes its fields under `object_fields`
  and each mode's `selector` (the config field whose being set or unset selects the mode, or `null`).
  A plugin declares it with `kd.search.ModeSelector(field, present)` on `InstrumentMode.selector`.
  `kd.search.descriptor.assert_sketch_supported` takes `config=` and then asks only the mode the config
  selects.
- `ExperimentResult.config["provider_kind"]` records the derivative source the run actually used
  (an algorithm's `derivative_source`, the optional protocol `kd.search.DerivativeSourceAlgorithm`),
  so an SGA `use_autograd=True` run and a DISCOVER MODE2 run record `autograd`.

### Changed

- **Breaking**: a fit now refuses, before any component is built or a surrogate
  is trained, an input whose field or axis names the algorithm's default
  vocabulary does not cover. These inputs used to run and return a result
  computed on the wrong names or on a subset of the data.
  - `Model(algorithm="discover")` on a grid raises `NotImplementedError` when
    the dataset's fields differ from the effective `library.state_vars`, or a
    spatial axis is neither in `coord_vars` nor reached by a derivative
    operator along it (`lap` reaches every spatial axis). Set `library` through
    a typed `DiscoverConfig` that covers the input.
  - `Model(algorithm="eqgpt")` in the single-wave mode raises `ValueError`
    unless the only field is `u` and the axes are among `x`, `y`, `z` plus `t`.
  - `Model(algorithm="llm4ed")` raises `ValueError` when the dataset has a field
    other than `u`.
  - `Model(algorithm="pysr")` and `Model(algorithm="pysindy")` on a grid raise
    `ValueError` when `terms` is omitted and the fields are not exactly `u` or
    the spatial axes are not exactly `x`: pass `terms` written in the input's
    names. The default on a one-dimensional `u(x, t)` grid is unchanged.
  - `Model(algorithm="pysindy")` raises `ValueError` when `terms` contains the
    left-hand-side column itself (`u_t`, `diff_t(u)` or `diff1_t(u)` for a
    `u_t` target).
  - `Model(algorithm="dlga")` raises `ValueError` naming the term when a
    `library` term uses a field or axis the dataset does not have.
  - `Model(algorithm="dlga")` without a sketch raises `ValueError` when
    `library` contains a time derivative of the left-hand field that the run
    fits: the order `target_lhs_order` names (`u_t`, `diff_t(u)`, `diff1_t(u)`
    for order 1; `u_tt`, `diff2_t(u)` for order 2), or both orders while
    `lhs_auto_select` is on. Such a token fits its own branch exactly, so the
    search used to return the identity `u_t = 1.0*u_t` whatever the data. The
    time derivative of an order the run does not fit stays an ordinary library
    term. The check runs in `fit` only and compares these literal spellings;
    `train_surrogate` does not read the library.
- **Breaking**: the `kd-npz` layout no longer recognizes an NPZ file that holds
  an entry named `format`; that key now marks the other self-describing
  layouts. A grid file with a field array of that name is reported as an
  unrecognized layout; rename the array or load the file with explicit
  `coords` and `fields` mappings.
- **Breaking**: `kd.AxisReport` has a new required field `periodic` (whether
  the dataset declares the axis periodic), also present in
  `DatasetReport.to_dict()`. Code that constructs an `AxisReport` must pass it.
- The message raised for a second-order time target by the algorithms that do
  not fit one now names the route: `Model(algorithm="dlga")` with
  `target_lhs_order=2`.
- **Breaking**: `Model(algorithm="dlga")` fits every candidate against the
  time derivative of the order `target_lhs_order` names and no other;
  `DLGAConfig.lhs_auto_select` now defaults to `False`. Before, the default
  fitted both `u_t` and `u_tt` and kept the branch with the lower NMSE, so a
  fit on a `u_t` dataset could return a `u_tt` law and a fit on a `u_tt`
  dataset a `u_t` law. `target_lhs_order=2` no longer requires
  `lhs_auto_select=True` (the `ValueError` for that combination is gone) and
  fits `u_tt` alone; `DLGAConfig.wave_preset()` and `kg_preset()` do the
  same. Pass `lhs_auto_select=True` for the former two-branch selection, in
  which the order of the time derivative is chosen from the data, as the
  DLGA-PDE paper sets out to do. On a target of either
  order a candidate's fitness is no longer the better of two branches, so a
  search can take a different course from the one an earlier release took
  with the same seed. A checkpoint written under the former default records
  `lhs_auto_select=True`: resuming it is refused as a change of the run's
  identity unless `lhs_auto_select=True` is passed again. In
  `kd.instrument_schemas()` the `dlga` mode's `order_params`
  for order 2 is `{"target_lhs_order": 2}`, and
  `kd.instrument_schema_digest()` changes with it.
- The summaries and mode descriptions of `discover` and `eqgpt`, and the
  default-mode descriptions of `pysr`, `pysindy` and `llm4ed`, state the data
  each mode takes (default vocabulary, fields, axes, and for the EqGPT steady
  mode that every call retrains a surrogate).
- The tabular defaults of `discover` (`batch_size=500`, `epsilon=0.05`) and
  `pysr` (`unary_operators`) are read from the mode declarations
  (`default_overrides`); the values and the rule that an explicit keyword
  switches the group off are unchanged.
- `Model(algorithm="discover")` on a `TabularDataset` now samples 500 candidate
  expressions per iteration with `epsilon=0.05` when the caller does not pass
  `batch_size` or `epsilon`; grid fits keep 16 and 0.10, and so do the
  `DiscoverConfig` defaults and the published schema. The tabular mode removes
  invalid candidates before the risk-seeking quantile, and with the old
  defaults most iterations kept fewer than 11 valid candidates, so the quantile
  was the batch maximum and the policy-gradient term was zero (42 of 50
  iterations on Nguyen-1). With the new defaults the same fit recovers
  `x1**3 + x1**2 + x1`. The default 50-iteration fit on that 20-row table
  took 83 s instead of 8 s. Explicit `batch_size` / `epsilon` keywords and a supplied
  `config=DiscoverConfig(...)` are used as given.
- `Model(algorithm="discover")` on a `TabularDataset` now searches with the
  operators add, sub, mul, div, sin, cos, exp and log instead of add, sub, mul
  and div. This is the same set as the PySR tabular default. The constant
  fitter evaluates exp and log with the protection of kd's expression
  registry: exp clamps its argument to [-50, 50], and log takes the logarithm
  of |x| floored at 1e-10. The tabular mode still owns the library, and a
  user-supplied library is still refused. sqrt is not included. On the five
  kd-bench tabular rows timed with both libraries, a default fit took 85 s on
  average instead of 73 s.
  The fitter's `div` follows `safe_div` as well (1e-10 with the sign of the
  denominator), so a candidate the platform scores finite has a finite fitting
  objective.
- A sketch's pinned terms are reserved algebraically. Before a candidate term
  is scored, its law key is expanded into a polynomial in the fields, their
  derivatives and the coordinates (an open-form derivative by the chain rule;
  division by an identically zero column, and the log of one, read as a
  constant, as the protected operators compute them); a term whose expansion
  contains a monomial of a pinned term is refused as regenerating the pinned
  column, whatever its spelling: beside a pinned `u*u_x`, `diff_x(n2(u))`,
  `u*u_x*u/u`, `(u_x+u_x)*u` and `exp(x/x)*u*u_x` are refused, and so is a
  sum or product that carries `u*u_x` beside another monomial. Feature sets
  are no longer compared for pins, so a different column with the same
  features (`u*u*u_x`, `u_x*u_x`) is now admitted. For an open
  sketch, two pins sharing a monomial and an anchored term carrying a pinned
  monomial are rejected when the sketch is compiled (a closed sketch, all
  pinned with no holes or anchors, compiles as before). Only a sketch with
  pinned terms pays the expansion; results are kept in a bounded cache keyed
  by the term as passed and the vocabulary. The compile reports of SGA, DISCOVER, EqGPT and
  PySR state the rule and its remaining boundary: the expansion cancels no
  common factor between a numerator and a denominator, so a rational
  respelling such as `(u*u*u_x - u*u_x*u_x)/(u - u_x)` is admitted; several
  admitted terms can add up to the pinned column, since terms are judged one
  at a time; and function identities (`sin` or `cos` squared, `log` of
  `exp`) respell it when those operators are configured. **Breaking** for
  code calling `kd.core.equation.pinned_fingerprints` or passing
  `pinned_fingerprints=` to `term_admission`: call `pinned_monomials` and
  pass `pinned_monomials=`; `kd.core.expr` exports `jet_monomials` and
  `Monomial`.
- DISCOVER's default `epsilon` is `0.10` (was `0.05`). With the default batch of
  16 and the `R_e` baseline, `0.05` put the risk-seeking quantile at the batch
  maximum, so the reward term of the policy-gradient loss was identically zero
  and only the entropy term trained the controller. At equal evaluation budgets
  the new default recovers pde-divide in 17/20 and 19/20 seeds against 9/20 and
  13/20; Burgers is unchanged within noise. Configurations that set `epsilon`
  or use a packaged preset are unaffected. Results of default-configuration
  DISCOVER runs change.
- The platform least-squares seam is deterministic. `LeastSquaresSolver`,
  `STRidgeSolver` and `kd.core.recovery` (`judge_recovery`, `span_floor`) now
  solve through `kd.core.linear_solve.deterministic_lstsq` (new export), the
  scoped `torch.use_deterministic_algorithms` mechanism SGA's own solver has
  used since July; SGA imports the shared helper instead of a private copy.
  Repeated evaluation of one library on CPU used to return up to about 20
  distinct coefficient vectors per 100 calls on an ill-conditioned library
  (default gelsy driver, allocator-state dependent), which could flip a
  STRidge support or a recovery verdict between two runs of one seeded
  experiment; it now returns one. The driver is unchanged, so every value
  stays inside the set the bare call already produced. EqGPT's final refit
  still calls `torch.linalg.lstsq` directly.
- `kd.law_signature` signs a PDE law that contains one term twice, such as
  `u` beside `neg(u)` or `diff2_x(u)` written twice: the term is signed once
  with the sum of its coefficients, and a sum of exactly zero stays as a
  zero-coefficient term. It used to raise `duplicate canonical term`, which
  made `kd.verify_equation` fail on such a law and left the run out of
  consensus as unsignable; DISCOVER and SGA both produce such laws. A law that
  repeats its left-hand side or the pivot term of a homogeneous law, and a
  tabular (regression) law with a repeated term, still raise. Every law that
  signed before gets the same signature, and the signature version is
  unchanged.
- SGA recognises a sketch's anchored terms by column fingerprint instead of
  by spelling, as the sketch exit already does when it attributes terms to
  anchors. During generation a term with an anchored term's fingerprint is
  admitted even when no hole admits it: beside an anchored `u*u_x`, the
  respelling `diff_x(u)*u` and the product `u*u*u_x` (a fingerprint does not
  record how often a factor repeats) used to be refused and are now
  admitted. The hole-capacity check after scoring leaves the same terms out
  of the hole count, so a selected support holding `diff_x(u)*u` and one
  other term now fits a hole with `max_count=1`. SGA and DLGA judge each
  candidate term with `kd.core.equation.term_admission`, the predicate
  DISCOVER, EqGPT and PySR already use. DLGA's behaviour is unchanged, and so
  is SGA's under a sketch without anchored terms. Results of SGA runs under a
  sketch with anchored terms can change.

- `kd-agent`: `run_discovery(sketch=...)` checks a sketch against the same
  modes as kd's capability gate: a mode counts when its form includes
  `EVOLUTION`, its left-hand-side orders include the sketch's, and, when kd
  subtracts the pinned terms, it has derivatives. The agent can therefore
  narrow DISCOVER, PySR and EqGPT through their PDE modes; before, their
  tabular, multi-case and steady modes, which declare no sketch support,
  refused every sketch. A sketch that no mode can run, such as one with a
  second-order left-hand side for SGA or PySINDy, is now refused before the
  search starts; before, kd refused it inside the search subprocess.
- DLGA's `default` mode publishes a `description` in `kd.instrument_schemas()`:
  the raw field and every derivative order come from the trained surrogate,
  and 5000 surrogate epochs is the smallest measured budget at which the true
  Burgers law fits the surrogate's own columns within 5% on every coefficient
  (noise 0 and 0.05, three seeds). `DLGAConfig.surrogate_max_epochs` still
  defaults to 50000.
- PySINDy keeps a sketch's anchored terms in every STLSQ round, and
  `kd.instrument_schemas()` declares its `anchors` clause as `fit_enforced`
  (was `exit_checked`). The anchored columns are eliminated from the target and
  from the other columns before STLSQ, and their coefficients are recovered by
  least squares afterwards, so the result equals a joint least-squares fit on
  the anchors and the selected support and an anchor is never thresholded
  away. A library column that equals an anchor numerically under another
  spelling gets coefficient 0, and the anchor columns are scaled to unit norm
  before the rank check, so a change of coordinate units does not report
  independent anchors as linearly dependent. The exit check is unchanged: with
  the default `support_threshold=0` an anchor counts as present whenever its
  fitted coefficient is nonzero, so set a positive `support_threshold` to
  refuse a negligible anchor coefficient.
- **Breaking**: building a `PDEDataset` now rejects a field whose name is a
  reserved kd token (a registered operator name, a `diff*_<axis>` operator name
  or the unity token `one`). Execution resolves these tokens before looking up
  fields, so such a field was silently replaced by the token's value (for `one`,
  a column of ones). Rename the field: the error says what kd reads the name
  as and suggests a replacement such as `sinval`. The same check guards the tabular bridge
  and `SINDyRegressor`, which previously did not reserve `one`.
- `build_consensus` leaves the dataset's shared left-hand-side term out of
  the class adjacency comparison, so two classes with no right-hand-side term
  in common are reported as `disjoint` with Jaccard 0.0. Every class on one
  dataset carries `u_t`, so `disjoint` was unreachable and Jaccard had a
  positive floor. Example 19 and the batch design page now point to
  `kd.evaluate_terms` for telling a notation difference from a real
  disagreement.
- `PlatformBuilder` raises `ValueError` for `provider_kind="autograd"` with
  neither `needs_surrogate=True` nor a `surrogate_model`; it used to build an
  untrained network as the derivative source. No bundled algorithm declares
  that combination.
- `SVDNullSpaceSolver` computes the condition number only when constructed
  with `compute_condition_number=True`, like `LeastSquaresSolver` and
  `STRidgeSolver`; by default `SolveResult.condition_number` is `None`. The
  extra SVD took about a quarter of each DLGA solve, and DLGA never reported
  the value.
- **Breaking**: EqGPT's steady boundary deletion no longer chooses its grid by
  dataset name. A built-in grid used to be looked up for datasets named
  `eqgpt-laplacian-eitech` and `eqgpt-laplacian-smile`; the lookup is removed,
  so a renamed copy of either dataset gets the same evaluation domain as the
  original. A steady configuration that sets `steady_boundary_delete_num` on
  these datasets without using the preset now takes the grid from the
  distinct coordinate values in the data: on eitech the evaluation domain
  cannot be built and a `ValueError` naming `steady_boundary_grid` is raised
  after the surrogate is trained, and on smile the evaluation coordinates
  differ from the preset's in the last bits. Pass
  `steady_boundary_grid=[4.3, 141.0, 0.170875, 19.6, 54.6, 0.175]` for eitech
  or `[-4.0, 4.0, 0.032, -4.0, 4.0, 0.032]` for smile to get the preset's
  grid. Disabling deletion on these presets with
  `steady_boundary_delete_num=None` now also needs
  `steady_boundary_grid=None`. EqGPT checkpoints written before this change
  are refused on resume in every mode: their stored config lacks the new
  field, and a field present on one side only counts as changed.
- EqGPT's multi-case wave mode checks the dataset passed to `fit` against the
  primary case by content. Axis order, field names and every coordinate and
  field tensor must equal those of
  `wave_breaking_case_to_dataset(cases[primary_case])`; the dataset name,
  `source` and `ground_truth` are not compared. A renamed copy of the primary
  case's dataset is now accepted (it was rejected because its name differed
  from `wave-breaking-<case>`), and a dataset carrying that name with
  different arrays, such as a subsample or a noisy copy, is now rejected. The
  error still contains `primary dataset mismatch` and names the item that
  differs.
- **Breaking**: the SGA and PySINDy configs each gain a field (`restart_interval`,
  `relative_threshold`) that is recorded at its default value too, so the
  `run_spec_hash` and `record_hash` of an SGA or PySINDy run record differ from
  the ones an earlier version wrote for the same settings. Both fields are
  `init_only`: an SGA or PySINDy checkpoint written before this change is
  refused on resume, with the field reported as present only in the live
  config. Migration: SGA and PySINDy checkpoints written before `eb15c48c`,
  and EqGPT checkpoints written before `30acd43f` (the `steady_boundary_grid`
  entry above), cannot be resumed; run those searches again from the start.
- **Breaking**: `kd.FieldReport` has three new required keyword-only fields,
  `noise_estimate`, `first_derivative_gap` and `second_derivative_gap`
  (each `dict[str, float]`), and `FieldReport.to_dict()` has the same three
  keys. Code that constructs a `FieldReport` directly must pass the three
  dictionaries; reports built by `kd.preview_report` are unaffected. Because
  of the dictionary fields, `FieldReport` instances are no longer hashable.
- PySR's grid (PDE) mode keeps an additive constant of a selected expression
  as the reserved constant term `one`, whose coefficient the refit estimates;
  it used to drop the constant. A selected expression that is a nonzero
  constant becomes the law `one` instead of raising `RuntimeError`, and a
  hall-of-fame entry that is a nonzero constant is kept as a `one` candidate.
  An expression that reduces to zero still converts to no term. Under a
  sketch, a configured term that converts to a constant (`one`, `div(u,u)`)
  is now kept as `one` when a hole admits it, and `one` is admitted only by
  an anchor or by a hole without a `fields` clause. A sketch with two or more
  holes (each then carries a `fields` clause) or with one `fields`-constrained
  hole therefore refuses every candidate with an additive constant before
  scoring, and a fit whose selected expression has one reports no law; the
  constant used to be dropped and the other terms scored. The tabular mode is
  unchanged: it keeps constants inside its single whole-expression term and
  refuses a constant model with no feature symbols. Results of PySR grid runs
  whose selected expression has an additive constant change.
- The batch dispatcher (`run_dispatch`, `dispatch_plan`) now launches and
  supervises its workers through `kd.harness.ProcessSupervisor`. Four
  behaviours change. A shard with `memory_max_gb` now runs under
  `systemd-run ... -p MemoryMax=<n>G -p MemorySwapMax=0 --`, so the cap holds
  on machines with swap enabled (before, only `MemoryMax` was set and the
  worker could exceed it by swapping); `worker_command` returns that argv.
  Workers now start with stdin closed instead of inheriting the dispatcher's.
  A worker whose own process had already exited when the dispatcher was
  interrupted is recorded with its exit code and no `kill_reason`, instead of
  `dispatcher_interrupted`. When any shard sets `memory_max_gb`,
  `run_dispatch` now checks with `check_memory_cap_available` that a capped
  user scope actually starts, so a machine where `systemd-run` exists but
  cannot apply the cap (for example, no user manager) raises
  `DispatchRunError` before the first worker starts; before, only the
  presence of `systemd-run` on `PATH` was checked.
- `kd-agent`: the agent's episode, surrogate-training and input-preparation
  subprocesses are now supervised by `kd.harness.run_supervised`. One behaviour changes:
  when the worker exits on its own but leaves processes behind in its process
  group, those processes now receive SIGTERM and, after the kill grace
  period, SIGKILL, instead of an immediate SIGKILL; the wait after SIGKILL is
  bounded.
- `OpenAICompatProvider` now retries a reply whose message content is `None`
  within its existing bounded retry (same seed, same backoff, same exhaustion
  error). Reasoning models return such a reply when their reasoning uses the
  whole `max_tokens` budget; the length varies between identical requests, so a
  retry usually returns text. Replies with no choices or no model id still fail
  immediately.
- **Breaking**: `PINNConfig.coef_pde` defaults to `1.0` (the physics residual enters PINN training);
  a caller that relied on `0.0` passes `coef_pde=0.0`.
- **Breaking**: `PINNConfig.local_sample` defaults to `False`; a caller that trains on a subsample of
  the grid passes `local_sample=True`.
- The facade no longer raises on `DiscoverConfig.pinn`; the `n_iterations` warning applies to MODE1 only.
- The report's autograd-domain note reads `provider_kind` alone; a result config that carries
  `use_autograd=True` with `provider_kind="finite_diff"` no longer gets the note.

### Removed

- The log warning DLGA emitted at the end of a fit whose result had an NMSE
  above 1e-2 and more than five tokens, which suggested raising `epsilon`. The
  length term of the fitness is described in the `epsilon` field's
  documentation instead.
- **Breaking**: `kd.core.equation.lower_to_regression`, `RegressionForm` and
  `PivotRegressionForm` are removed (module `kd.core.equation.lowering`). No kd
  code called them; a caller that needs the term list of an equation reads
  `eq.terms` directly.
- **Breaking**: `kd.search.eqgpt.FakeGPTBackend` is removed from the package. It
  was a test double; any object satisfying `kd.search.eqgpt.GPTBackend` can still
  be injected with `EqGPTPlugin(config, backend=...)`.
- **Breaking**: `kd.search.lifecycle` (`SearchLifecycle`, `LifecycleState`,
  `LifecycleError`) and the `ExperimentRunner.lifecycle` property are removed.
  Nothing read the state machine; whether a run resumed a checkpoint is reported
  by `ExperimentResult.manifest.resumed`.
- The module `kd.search.discover.utils.canonicalize` is removed; import
  `canonicalize_expression` from `kd.core.equation`.
- The module `kd.search.discover.search` (`rebuild_batch`) is removed; it had no
  caller outside its own tests.
- The DISCOVER research drivers are no longer installed with the package: the
  command `python -m kd.search.discover`, the package
  `kd.search.discover.runners`, and the modules `kd.search.discover.data` (the
  raw `.mat` and `.npy` readers `load_burgers_mat`, `load_kdv_mat` and their
  siblings, and `load_allen_cahn_2d`), `kd.search.discover.golden` and
  `kd.search.discover.paths`. Nothing else in the package imported them. A
  DISCOVER fit runs through `kd.Model(algorithm="discover")` on a dataset from
  the top-level loaders (`kd.load_burgers`, `kd.load_kdv`,
  `kd.load_chafee_infante` and the rest).
- `kd.search.recorder.__all__` no longer lists the private aliases
  `_detach_tensor`, `_is_json_serializable`, `_make_json_safe` and
  `_sanitize_float`; the functions are `kd.core.jsonsafe.detach_tensor`,
  `is_json_serializable`, `make_json_safe` and `sanitize_float`.
- `kd.search.callbacks` no longer re-exports `CHECKPOINT_VERSION`; import it
  from `kd.search.checkpoint_payload`, which is also the home of
  `atomic_torch_save` and `build_checkpoint_payload`.
- `kd.search.sga.train.compute_aic` is removed; it was an alias of
  `kd.core.metrics.aic_no_n`.
- **Breaking**: the batch dispatch family is removed from `kd.harness`, with no
  deprecation period: `run_dispatch`, `dispatch_plan`, `merge_shards`, the
  dispatch manifest (`DispatchManifest`, `DispatchDatasetSpec`,
  `DispatchRecording`, `build_dispatch_manifest`, `read_dispatch_manifest`,
  `write_dispatch_manifest`, `resolve_dataset`, `DISPATCH_ARTIFACT_TAG`), the
  dispatch log (`DispatchLog`, `WorkerLogRow`, `read_dispatch_log`,
  `write_dispatch_log`, `decode_dispatch_log`, `DISPATCH_LOG_ARTIFACT_TAG`),
  the batch report (`build_batch_report`, `render_dispatch_markdown`,
  `write_dispatch_report`), the result and error types (`DispatchResult`,
  `DispatchRunError`, `DispatchManifestError`, `DispatchAllocationError`,
  `DatasetResolverError`, `DispatchLogError`, `DispatchMergeError`,
  `MergeReplayError`, `ShardMappingError`, `ShardMissingError`) and the worker
  entry point `python -m kd.harness.worker`. The schema tags `kd-dispatch-v1`
  and `kd-dispatch-log-v1` leave `kd.core.schema_registry`. Run each batch with
  `run_plan` and combine the stores with `merge_stores`. The shard stores under
  an earlier dispatch directory are ordinary evidence stores and still load
  with `EvidenceStore.load`. Process-group supervision (`ProcessSupervisor`,
  `run_supervised`, `build_capped_command`, `check_memory_cap_available`) is
  unchanged.
- **Breaking**: `kd.harness.run_plan` no longer accepts `entry_indices`,
  `resume_from` or `plan_hash`; remove them from the call. Only the workers of
  the removed batch dispatch passed them. `run_plan` starts every entry of its
  plan fresh, and its `kd-runcat-v1` catalog rows carry the entry's index in
  that plan and the store's plan hash, with `resume_from` and `parent_run_id`
  left `null`. To run parts of a study in separate processes, give each process
  its own plan and combine the stores with `merge_stores`. Resuming one run
  from a checkpoint is still done with `run_episode(resume_from=...)` or
  `Model.fit(resume_from=...)`.

### Fixed

- `kd-agent`: the report writer no longer crashes on a tool call the
  framework rejected before the tool ran (an argument that failed
  validation). The tracer stores such a call with empty outputs and the
  error text, and `report.md` now lists that error under the call instead
  of raising `KeyError` and leaving the run without a close-out.
- Numeric literals inside a call no longer make an expression fail on the
  executor's fast path. The fast path used to pass a literal to the
  operator as a Python float, and `div`, `exp`, `log`, `sin` and `cos`
  refuse a float. So `div(x1, 2.0)`, `exp(1.0)` and `add(x1, exp(1.0))`
  raised `TypeError`, and DISCOVER's tabular mode scored them as invalid
  candidates: 58.6% of the candidates in a default fit on nguyen-5. The
  full path already evaluated them. Literals now reach the operator as 0-dim
  float64 tensors on the context's device, so the result keeps the column's
  dtype, as it did with a Python float. Expressions that worked before keep
  their values, except that a subexpression of literals only, such as
  `add(0.1, 0.2)`, is now computed in float64 instead of float32.
- `Model(algorithm="pysr", config=PySRConfig(...))` on a tabular dataset
  searched only `+ - * /`. The `config=` route replaced the default `terms`
  with the feature names but kept `unary_operators` at the PDE default `()`,
  while the keyword route and `PySRPlugin(mode="tabular")` use
  `("sin", "cos", "exp", "log")` (a 0.8.0 regression). A default-valued
  `unary_operators` in a tabular `config=` is now treated as unset, like
  `terms`; a user-set value is kept. Because a frozen dataclass cannot tell
  an explicit `unary_operators=()` from the default, that spelling also yields
  the tabular set under `config=` on a table (the keyword route still keeps
  it). `unary_operators` is one of PySR's identity-breaking fields, so a
  tabular `config=` checkpoint written by 0.8.x is refused on resume.
- `OpenAICompatProvider` builds the SDK client with `max_retries=0`. The SDK's
  own default of two internal retries per request was stacked on kd's bounded
  loop, so one call could issue 18 HTTP requests of up to 60 s each instead of
  the documented `1 + max_retries_per_call` = 6. The wait between attempts is
  now only kd's fixed backoff schedule; the SDK's `Retry-After` handling goes
  with its retries.
- `preview_report` (and `kd.preview`) no longer warns "Model.fit() will fall
  back to ('u', 't')" on a steady dataset (`lhs_order == 0`): the fit path
  returns such a dataset unchanged, so the warning described a fallback that
  never happens. The same text reached the dataset summary kdagent hands the
  LLM.
- `kd.preview` prints the order-aware LHS label: a second-order dataset such
  as `kd.load_wave()` now shows `LHS: u_tt` where it printed `LHS: u_t`
  (`preview_report` already reported `u_tt`). First-order datasets are
  unchanged.
- The kdagent offline judge spelled a mixed partial derivative as `u_tx`, a
  name kd cannot execute, so a truth law containing one (eq-6-2-12) was
  recorded as a `KeyError` instead of a verdict. It now writes kd's compound
  form `u_t_x` (one segment per axis, axes sorted); same-axis names such as
  `u_xx` are unchanged.
- The DLGA expression-bloat warning no longer states "Truth structure is still
  recovered"; the plugin has no truth expression on user data and the helper
  it cited exists only in test code. The epsilon retuning advice is unchanged.
- The multi-run summary table's "Δ Terms vs Run 0" column diffs each run's
  published (sparse-fit) law, the same `active_law` projection the Expression
  cell renders, instead of the full candidate catalog. Runs with identical gene
  lists but different STRidge supports showed "=" while their expressions
  differed, and identical laws over different catalogs showed spurious changes.
  A run whose law has no active term now degrades to "n/a" with a warning.
- DISCOVER through `Model.fit` (and the kdagent discover lane) now releases
  the evaluator's term-column cache once per batch, as the standalone
  `DiscoverEngine.run_iteration` loop has since 2026-07-07. The facade path
  drove `propose`/`update` directly and relied only on the 1 GiB byte-budget
  flush, so every DISCOVER process grew its term cache to that ceiling.
  `DISCOVERPlugin.update` invalidates in a `finally`; the sketch-gated
  evaluator forwards the call to the evaluator it wraps.
- The DISCOVER report's "Reward (Mean of Top ε)" caption described the
  `ε=0.05` default: a single-row subset equal to the per-iteration maximum and
  coincident with the Reward Baseline plot. Under the current default
  (`ε=0.10`, batch 16) the subset holds two rows; the caption now states the
  quantile rule and when the curve equals the maximum. Published HTML reports
  on the site still carry the old sentence until they are regenerated.
- Fixed-coefficient verification no longer reports an allocation failure as a
  candidate problem. Every exception raised while executing a verification term
  was wrapped into `ValueError("failed to execute verification term ...")`, so a
  CUDA out-of-memory or a CPU allocation failure during verification was
  recorded as `verify: ...` and withheld a solution whose verdict had passed.
  Both now propagate, matching the arms the platform builder already had. The
  predicate that recognises a CPU allocation failure moved from the private
  `kd.core.linear_solve._helpers` to the new `kd.core.oom`; two of its three
  callers were already outside `linear_solve`.
- `examples/04_noisy_data.py` returned a wrong equation in both derivative
  modes on 5% noise. Its autograd surrogate was undertrained at 500 epochs; the
  example now trains it for 3000 and recovers `-0.9355*u*u_x + 0.09737*u_xx`
  against the Burgers ground truth, while finite differences stay at `-1.102*u`.
  The docstring now states the measured outputs and that recovery at this noise
  level depends on the search seed, replacing an unverified rule of thumb.
  Runtime rises from about 25 seconds to about 2.5 minutes.

- The final checkpoint written when a search crashes carries the iteration
  whose `update()` completed. When an observer ordered before
  `CheckpointCallback` raised inside `on_iteration_end`, the final
  `checkpoint_final.pt` saved the post-iteration state under the previous
  iteration's label, and a resume from it counted one generation short.

- Batch dispatch (`run_dispatch`): each worker now leads its own process
  group, and the timeout and interrupt handling signals the whole group. A
  PySR fit's Julia child, or any other process a worker spawned, stops with
  the worker, and the worker's slot is released only once its group is
  empty. A terminal Ctrl-C now reaches the dispatcher only; its interrupt
  handling terminates the workers, and a Ctrl-C is held while a worker is
  being registered and for the whole shutdown, so a second one hurries the
  SIGKILL instead of cutting the shutdown or the final ledger short.

- `kd-agent`: an autograd search failed at its first step on a machine whose
  GPU driver is older than the CUDA build of the installed torch (for example
  torch 2.14 built for CUDA 13.0 on a driver that supports CUDA 12.8). The
  agent's search subprocess set `PYTORCH_NVML_BASED_CUDA_CHECK=1`, so
  `torch.cuda.is_available()` reported the card as usable although CUDA could
  not start, and the field-model trainer raised `RuntimeError` while saving
  the GPU random state. The subprocess no longer sets the variable: on such a
  machine torch reports CUDA as unavailable and the search trains on the CPU.
- `FieldModelTrainer.fit` isolates the random-number generator of the device
  it trains on, not of every CUDA device torch reports. A fit on the CPU no
  longer reads or seeds any CUDA generator (reading one initialized CUDA and,
  on a driver older than torch's CUDA build, raised `RuntimeError` although
  `torch.cuda.is_available()` was True); a fit on a CUDA device saves and
  restores that device's generator only. The CPU random stream is unchanged,
  so seeded results are the same as before.
- PySR honors a wall-clock budget. A `WallClockBudgetCallback`, passed as
  `Model(..., callbacks=[WallClockBudgetCallback(s)])` or set by `kd-agent`'s
  time cap, becomes PySR's own `timeout_in_seconds` for each fit: the time left
  before the deadline minus 45 s for start-up and 5 s for wind-down, at least
  15 s, and never above a configured `extra_pysr_kwargs["timeout_in_seconds"]`.
  The runner reads the budget only between iterations and a PySR segment is a
  single fit, so before this the fit ran past the budget; under `kd-agent` it
  was killed at the cap plus 120 s and returned nothing. Now the fit returns
  its hall of fame near the deadline and the run seals as usual. The value is
  not written into the config, so it does not reach the resume gate or the run
  identity, and a resumed segment times its fit from its own deadline.
  **Breaking** for a custom `kd.search.pysr.PySRBackend`: `fit` takes a new
  keyword-only `timeout_in_seconds` (`None` for no limit), which the plugin
  passes on every fit.
- `kd-agent`: a submitted law that copies the cited sealed law with a
  coefficient changed only by the text round trip (for example
  `0.1001876458695404` for `0.10018764586954039`) is reported as that sealed
  law, with its native measurements. The report compared the two laws with
  exact equality; it now requires the same fields, left-hand side and terms
  and compares the coefficients with `kd.core.law_agreement` at
  `coeff_atol=1e-12` on kd's normalized law signatures. A law kd refuses to
  sign stays unevaluated.
- `kd-agent`: `kdagent.analysis.sources.tool_calls` parses the return
  envelope only for the six controller verbs. A transcript in which the model
  ran deepagents' `execute` tool, whose output can be JSON followed by an
  exit-code line, raised `JSONDecodeError`; the framework's own tools now keep
  `envelope = None`.
- The unity term `one` now takes the dtype of the left-hand-side field instead
  of always `float32`, so `kd.evaluate_terms(ds, ["one"])` fits on `float64`
  datasets instead of raising a dtype mismatch.
- The PySINDy coefficient plot notes "native invalid" when the native fit is
  missing and only the refit bar is drawn, matching the existing
  "refit invalid" note.
- DLGA refuses library terms on a dataset field other than the left-hand
  field, such as a known coefficient `K` or its derivative `K_x`, before the
  surrogate is trained. DLGA's surrogate models the left-hand field only; such
  a term used to fail as an invalid candidate, and the fit returned an
  equation without it and without a message.
- DLGA keeps its best genome from one generation to the next, as Xu's
  reference and the predecessor do: the best survivor still goes through crossover and
  mutation (so the random stream is unchanged) and is then put back unmodified
  at index 0. The port had dropped this step, so the best genome reached the
  next generation intact only about 4% of the time. The reported equation was
  unaffected (the best genome is recorded separately); multi-generation search
  trajectories change.
- LLM4ED shows the LLM the STRidge-pruned equation, as EDL does: the prompt
  histories and the elite-pool display keep only terms with a nonzero
  coefficient, at most five, without a leading sign. They used to show every
  proposed term, including those STRidge set to zero. Scores and
  `best_expression` are unchanged.
- PySR's selected-model marker (the star on the Pareto, score-agreement and
  audit plots, and the recorded `selected_complexity` / `selected_loss`) sits
  on the entry PySR selected. In PDE mode, entries that differ only in their
  constants convert to the same KD expression, and the marker used to land on
  the first of them.
- Under a sketch, SGA applies the hole-capacity check to the initial
  population and to the refill after a restore, not only to offspring. An
  initial individual whose selected support overflowed a hole could hold a
  population slot, including the elite one, for the whole run.
- The generic convergence plot no longer draws a 0 for iterations in which
  EqGPT, LLM4ED or DISCOVER had no candidate yet. Their `best_score` is a 0.0
  placeholder until the first candidate; the recorded series now holds a gap
  there, so an all-empty run no longer reads "constant at 0" and a run with
  negative EqGPT rewards no longer shows a decline from 0.
- Axis regularization in `kd.load`, the kd-npz and PDEBench readers and
  `kd.data.resample_to_grid` measures float32 jitter against a fraction
  (1e-3) of one grid step plus the rounding of the stored dtype (four machine
  epsilons of the largest coordinate) instead of an absolute 1e-5, and
  rebuilds the axis between its two endpoints. Large-magnitude or long
  float32 uniform axes such as `linspace(0, 100, 1024)` used to be rejected,
  and genuinely non-uniform axes with a small span were silently replaced by
  a uniform one; the first now load and the second raise "not uniformly
  spaced". A float64 or text axis whose stored values deviate from uniform by
  more than 1e-3 of a step is now also rejected.
- `kd.core.integrate_pde` accepts the reserved constant term `one`. Its
  pre-flight check filed `one` under unrecognised symbols, so a grid law
  with a constant term (from a term library that includes `one`, or a PySR
  grid result that keeps a constant term) was refused with "unrecognised
  symbols ['one']" and the report's forward-integration plots fell back to
  their degraded form. Such laws now integrate, with `one` evaluated as a
  column of ones.
- SGA now sparsifies candidates whose least-squares coefficients are all
  small. When every column-normalised least-squares coefficient of a candidate
  fell below the tolerance being tried (the first sweep step uses `d_tol`,
  default 1.0), SGA's internal STRidge returned the dense least-squares
  solution. With the default `lam=0` that solution tied the `tol=0` baseline,
  so the tolerance sweep accepted it, kept raising the tolerance and never
  backed off; the candidate then kept every term. STRidge now returns the zero
  solution in that case, and the sweep rejects the empty support and backs off
  to smaller tolerances, as the predecessor does with the default `normalize=2`. With
  `normalize=0` and `lam=0`, the predecessor keeps the dense solution in this case,
  while kd now backs off. The condition is checked per candidate; when the
  candidate's columns are close to orthogonal it roughly means that the L2
  norm of the target over all samples is below `d_tol`. On such data `kd-sga`
  answers change: for example, on Burgers with the time axis stretched 1000
  times, almost every candidate of a five-generation seed-0 fit was affected,
  and the fit no longer reports a spurious `7.574e-08*u` term. With the
  defaults (`d_tol=1.0`, `maxit=10`) the smallest tolerance the sweep tries is
  about 1.4e-4, so a candidate whose normalised coefficients are all below
  that still keeps every term, as in the predecessor. The same branch is also reached by
  individual poorly fitting candidates on other data, which can change a
  seeded search; the five-generation seed-0 quick-start fit on
  `kd.load_burgers()` gives the same equation and score as before.
- `kd-agent`: the offline judge (`kdagent.analysis.judge.judge_law`) now judges
  a candidate law whose support contains the constant term `one` on kd's
  constant column, with its coefficient, instead of raising `JudgeError`.

### Documentation

- The installation instructions (README, the site's installation page and
  controller guide) state that the CUDA build of PyTorch must be one the
  installed driver supports: `uv pip install sail-kd --torch-backend=auto`
  selects such a build (a specific one can be named, for example `cu128`),
  and with pip torch is installed first from the PyTorch index for the CUDA
  version the driver supports. With the newest CUDA build on an older driver,
  `torch.cuda.is_available()` returns False and KD trains on the CPU.

## [0.8.1] - 2026-09-15

### Added

- `kd-agent skill install`: copies the packaged KD workflow skill
  (`skills/kd/SKILL.md`, a step-by-step checklist for a coding agent writing
  Python against an installed `kd`: inspect and load the data, preview, choose
  an algorithm from the schema, fit, verify, report) into the Claude Code skills
  directory under the current directory; `--to` picks another directory and
  `--force` overwrites. It works on a base install without the `agent` extra. The public
  repository carries the same file at `skills/kd/SKILL.md`.

### Fixed

- **Breaking:** `load_burgers()` and `load_kdv()` now declare the periodic
  spatial boundaries used to generate their bundled data. This corrects
  forward reconstruction, changes boundary derivative estimates and dataset
  fingerprints, and requires new runs when using the corrected defaults
  instead of resuming checkpoints tied to the previous dataset identity.
- Field-reconstruction reports now disclose periodic boundaries or endpoints
  fixed at their initial values. Refreshed published examples retain their
  original search equations and metrics and identify corrected reconstructions.

### Documentation

- Revised the SAIL README with an earlier quick start, a PyPI badge and a
  Burgers discovery animation. Installation and controller workflows have
  dedicated documentation pages.
- Refreshed historical reconstruction reports while preserving their original
  search results, and clarified the limitations of the saved PySR and DISCOVER
  examples at their report links.

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

[Unreleased]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.8.2...HEAD
[0.8.2]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.7.5...v0.8.0
[0.7.5]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.7.4...v0.7.5
[0.7.4]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.7.3...v0.7.4
[0.7.3]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.7.2...v0.7.3
[0.7.2]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/releases/tag/v0.7.2
