# Changelog

All notable changes to KD are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versioning policy: KD is in 0.x development. A patch release may contain a
breaking change; every breaking entry is marked **Breaking** and names what a
caller must change. The seven bundled instruments are updated in the same
release and are never the party that breaks. From 0.8.0 on, each release is
also published on PyPI and as a GitHub Release carrying longer notes.

## [Unreleased]

## [0.9.0] - 2026-10-26

### Added

- `kd.Sketch` takes keyword shorthand: `pinned` as a dict of term to fixed
  coefficient (`{"u*u_x": -1.0}`), `anchored` as term strings whose
  coefficients are fitted (`["u_xx", "u"]`) and `holes` as `kd.Hole`
  objects, any of them omitted. The shorthand returns a `kd.SketchSpec`;
  `Model.fit(sketch=...)` and `kd.harness.run_episode(sketch=...)` resolve it
  against the dataset they fit (`spec.resolve(dataset)` does it by hand),
  taking the LHS, the vocabulary (every field and axis) and an exact match
  policy from the dataset; an explicit `lhs_spec` must equal the dataset's
  LHS, an explicit `vocabulary` may only narrow the dataset's, and an
  explicit `match_policy` replaces the exact one. A term string is a name, a
  product (`u*u_x`), a positive integer
  power (`u**2`, a raw product distinct from `n2(u)`) or funcall IR; a term
  outside that grammar, or one the dataset cannot evaluate, is refused with
  its clause position (`anchored[1]`). The run identity, checkpoint and
  sketch sidecar carry the resolved `kd-sketch-v2` payload, and
  `sketch_to_dict`, `compile_sketch` and `DiscoveryTask.from_sketch` refuse
  an unresolved spec. The typed construction remains supported; complete
  Sketches are written as `kd-sketch-v2`, and `kd-sketch-v1` payloads are
  still read.
- `kd.Hole(max_terms, min_terms=1, max_deriv_order=None, operators=None,
  fields=None)` describes one hole of a shorthand Sketch. The hole at
  position `i` becomes the `TermHole` with id `"hole_i"`. `operators` takes
  law feature names (`"mul"`, `"recip"`, `"n2"`, ...; division is `"mul"`
  with `"recip"`); `"*"`, `"div"` and `"^2"` are refused. Counts and orders
  must be integers (a `bool` is refused), and several holes need disjoint
  `fields`, as in the typed form.
- `kd.vote_terms(supports, *, min_count, coefficients=None,
  small_term_rtol=0.0)` counts each law term once per recorded support, by
  its law-term identity, and returns a `kd.TermVoteResult` with one
  `kd.TermVoteRow` per term (count and supporting indices) and the terms
  that reach `min_count`. With coefficients, exact zeros are not counted and
  the small-term rule of `Model(small_term_rtol=...)` applies.
- `kd.sketch_from_anchors(dataset, anchors, *, min_additional_terms,
  max_additional_terms, max_derivative_order)` builds a `kd-sketch-v2`
  Sketch from selected anchor terms and one bounded hole. It refuses, by
  name, an anchor with a symbol outside the dataset's fields and axes, an
  unknown operator or a wrong argument count. An anchor may be written in
  product notation (`u*u_x`, `u**2`); the Sketch stores its funcall IR.
- `kd.save_sketch(sketch, path)`, also as `Sketch.save(path)` and
  `SketchSpec.save(path)`, writes a Sketch to a JSON file. A complete Sketch
  is written as its `kd-sketch-v2` payload (`kd.sketch_to_dict`). A
  `SketchSpec`, which `kd.Sketch(...)` returns for shorthand clauses, is
  written unbound as `kd-sketch-spec-v1`: terms in the caller's spelling and
  order, pinned values as `{"term", "value"}` entries, every `kd.Hole` field,
  `null` for an LHS, vocabulary or match policy left to the dataset, and
  sorted name sets.
- `kd.load_sketch(source, *, dataset=None)` reads such a file from a path or
  an already decoded mapping. Without a dataset it returns the `SketchSpec`
  or the complete Sketch the file holds; a `kd-sketch-v1` payload is read and
  saves back as `kd-sketch-v2`. With a dataset it always returns the
  complete Sketch on that dataset: a spec is resolved against it, and a
  complete Sketch must have the dataset's LHS and a vocabulary within its
  names. A malformed file raises `ValueError` naming the entry.
- `kd.search.SearchedVocabularyAlgorithm`: an optional protocol through
  which a search plugin reports the vocabulary it searched. DISCOVER
  implements it, and its runs record `operators`, `state_vars`,
  `coord_vars`, `token_names` and `token_arities` as
  `result.config["searched_vocabulary"]`.
- `kd.refit_terms(dataset, terms, coefficients=None, *, small_term_rtol=0.0,
  lhs_order=None, max_order=3)` refits a fixed set of terms and returns a
  `kd.RefitResult`: the kept terms, their refitted coefficients, each pruned
  term with its input coefficient, NMSE, R-squared, the condition number, the
  LHS order used and `derivative_source="finite_diff"`. With
  `small_term_rtol > 0` it first drops every term whose input coefficient is
  smaller than that fraction of the largest finite one, by the rule of
  `Model(small_term_rtol=...)`, and then fits the remaining terms once with
  `kd.evaluate_terms`; with `small_term_rtol=0` the numbers equal
  `kd.evaluate_terms(..., max_order=3)`. A `small_term_rtol` outside `[0, 1)`,
  a positive one without coefficients, or coefficients that are not one per
  term, raises `ValueError`.
  Derivatives are finite differences of the dataset passed in; to refit on a
  smoothed field, pass the output of `kd.data.savgol_smooth`. Terms are
  executed as written, without algebraic simplification, and the function
  gives no verdict on the equation.
- `kd.refit_terms` records how it measured. Keywords: `method`
  (`"finite_diff"`, the default, `"autograd"` or `"weak"`), `trim` (an
  integer, a per-axis mapping or `"by_order"`), `rows`, `surrogate` (a stored
  surrogate whose identity binds it to the dataset, `autograd` only) and
  `weak` (`kd.WeakFormOptions`). Finite differences are taken on the complete
  input and rows are selected afterwards; `"by_order"` removes the stencil
  footprint of the submitted terms and the LHS, nested derivatives included.
  `autograd` takes field values, derivatives and the LHS from the one stored
  model and rejects a term on a field other than the LHS field. `RefitResult`
  also carries `mse`, `weak` and `frame`, a `kd.RefitFrame` (input identity;
  LHS field, axis and order; field source; derivative method and implementation
  version; surrogate identity; trim; hash of explicit rows; sample or window
  count; condition-number basis). With the default arguments the numbers are
  those of the finite-difference refit described in the previous entry. No
  method falls back to another, and a
  combination outside the rules (for example `trim` with `rows`, or
  `surrogate` without `autograd`) raises `ValueError`.
- `refit_terms(method="weak")` fits a fixed support by integration by parts
  against degree-4 bump test functions on the raw field (`kd-weak-v1`). It
  takes one real field, one spatial axis, a uniform grid and a first-order
  evolution, and the terms `one`, the field, its square and cube (`n2` and
  `n3` of it, or the raw products `mul(u,u)` and `mul(mul(u,u),u)` in either
  bracketing, each fitted as the same power column under its own label), pure
  spatial derivatives of order 1 to 3 and `mul` of the field with its first
  derivative; other terms raise `NotImplementedError`. Window half-widths are
  given or chosen per axis from the spectral corner with `spectrum_tau`, and
  an axis with no legal split or width raises `ValueError`.
  `noise_correction="gaussian"` replaces `u^2` and `u^3` by `u^2 - sigma^2`
  and `u^3 - 3 sigma^2 u`, with sigma from the second-difference MAD estimate
  (off by default). `result.weak` is a `kd.WeakFormDiagnostics`; `nmse` is the
  weak residual MSE over the variance of the integrated target.
- `kd.ablate_terms(dataset, terms, *, lhs_order=None, max_order=3,
  method="finite_diff", trim=0, rows=None, surrogate=None, weak=None)` refits
  a support and then refits it once without each term, all on one frame
  (rows or windows, target and correction). It returns a
  `kd.AblationResult` with one `kd.AblationRow` per dropped term (`mse`,
  `nmse`, `delta_mse`, `relative_rise`). Dropping the only term scores the
  zero model; a zero baseline MSE sets `relative_rise_reason="zero_baseline"`;
  a failed reduced fit raises `EvaluationFailedError` naming the dropped term.
- `kd.score_term_sets(dataset, term_sets, *, ...)` (the keywords of
  `ablate_terms`) fits several fixed supports on one frame resolved for their
  union and returns `kd.TermSetScores`: one `kd.TermSetEntry` per set with its
  fit, or with a failure classified `"refused"` or `"fit_failed"` and no fit.
- `kd.forward_error(..., max_seconds=None)` stops at a cooperative deadline
  on the monotonic clock, checked before and after preparing the RHS, at
  every RHS evaluation and before a value is returned, and then returns
  `value=None, reason="timeout"`. `ForwardError` gains `terms`,
  `coefficients`, `input_identity`, `interval` and `boundary`.
  `kd.core.integrate_pde` takes `deadline=` and its result gains `timed_out`.
- `kd.data.lowpass_smooth(dataset, *, modes, axes=None, time_window=0,
  polyorder=3)`, also exported as `kd.lowpass_smooth`, returns a copy of a
  GRID dataset with every field low-passed per axis: along each selected axis
  the coefficients with index `>= modes` are set to zero, with the real FFT on
  a periodic axis and the orthonormal DCT-II on any other (`modes=1` there
  keeps each line's mean). `modes` is one count for every selected axis or a
  mapping that names exactly the selected axes; `axes` defaults to every axis
  but the left-hand-side axis. `time_window > 0` then applies a Savitzky-Golay
  filter along the left-hand-side axis (odd window, larger than `polyorder`).
  The input is not modified; a field holding NaN raises `ValueError`.
- `kd.data.lowpass_spectrum(dataset, field, axis)`, also exported as
  `kd.lowpass_spectrum`, returns the fraction of a field's energy that
  `lowpass_smooth(modes=m)` keeps along `axis`, for `m = 1..n`, under the
  same transform (one-sided FFT bins weighted as in Parseval's identity). The
  array is non-decreasing and clipped to `[0, 1]`: on a clean field the
  cumulative sum could round to `1 + 2e-16`.
- `kd.refine_pinn(dataset, terms, coefficients, *, epochs, pretrain_epochs,
  train_coefficients=True, device=None, seed=0, max_seconds=None, ...)` fits
  DISCOVER's PINN network to a GRID dataset and a candidate law together.
  The data-only pretraining runs up to `pretrain_epochs` epochs under
  DISCOVER's own validation patience (500 epochs, checked after a warmup of
  `min(pretrain_epochs, 5000)` epochs), whatever `epochs` is. After it, the
  joint training minimises the data loss plus
  `residual_weight` times the law's mean squared residual at Latin hypercube
  collocation points, derivatives by autograd; with `train_coefficients` the
  coefficients are trained in the same optimisation. It returns a
  `kd.PinnRefinement` with the initial and final coefficients and losses,
  the joint epochs run, whether `max_seconds` stopped it, the pretraining
  epochs requested (`pretrain_epochs`) and run (`pretrain_epochs_run`),
  whether the patience ended the pretraining (`pretrain_stopped_early`;
  it can be true with every requested epoch run), and the network's
  prediction of every field as a dataset whose axes are all non-periodic.
  The left-hand-side order must be 1. The caller's global random state is
  left as it was.
- `kd.data.select_lowpass_modes(dataset, field, *, terms, refit, axes=None,
  time_window=0, tolerance=0.02, max_counts=8)`, also exported as
  `kd.select_lowpass_modes`, chooses the mode count of
  `kd.data.lowpass_smooth` per axis without a reference equation. Per axis it
  starts at the knee of `kd.data.lowpass_spectrum` (the first count whose
  missed energy is within 1.5 times the energy missed at `n // 8` modes),
  smooths the dataset and refits `terms` with the caller's `refit` (for
  example `refit=kd.refit_terms`) at the knee and at each next count, and
  stops at the first step whose largest per-term relative coefficient change
  is at most `tolerance`; the chosen count is the earlier count of that pair.
  Several axes are scanned in turn, each already scanned axis held at its
  chosen count. It returns a `kd.data.lowpass.LowpassSelection` with the
  chosen counts, the knees and the per-axis trace (count, energy fraction,
  coefficients, change). It never reads `dataset.ground_truth`, and the
  copies passed to `refit` carry none. A scan that does not settle within
  `max_counts` counts raises `ValueError` listing the counts and changes.
- `kd.data.network_smooth(dataset, *, recipe, seed=0, device=None)`, also
  exported as `kd.network_smooth`, trains one `FieldModel` on the single
  field of a GRID dataset with `FieldModelTrainer` under a
  `kd.models.FieldRecipe` and a seed, and returns a `kd.data.NetworkSmoothing`:
  the network's field on the dataset's grid as a `PDEDataset`
  (`kd.surrogate_field` of the trained model), the trained model on CPU, the
  `TrainingResult`, the recipe, a JSON-safe preparation identity and a
  quality block (`kd.data.NetworkSmoothingQuality`). Both types are also
  exported from `kd`. The identity covers every field and coordinate byte,
  the axis and field order, the dtype, the source periodicity, the recipe,
  the seed and the trained weights; the file provenance
  (`PDEDataset.source`), the device and the ground truth are not part of it.
  The quality block reports the requested and executed epochs, the stopping
  reason, the final training loss and the best validation loss with their
  scale (the variance of the observed field), a per-field noise estimate
  `sigma_hat` in field units (the second-difference MAD estimate, the
  smallest over axes with at least 5 points) with
  `rms(observed - predicted) / sigma_hat`, the training device and its CUDA
  peak (read without resetting the counter), and a reason for each quantity
  it could not compute. Training runs on `device`, by default the field's
  device, in the recipe's `training_dtype` when it names one and in the
  field's dtype otherwise; the model is returned in the field's dtype. A
  dataset that is not GRID, holds more than
  one field or has NaN in its field raises `ValueError`. The diagnostics do
  not measure derivative accuracy or recovery.
- `kd.surrogate_field(dataset, model, *, batch_size=8192)`, also
  `kd.data.surrogate_field`, samples a trained `FieldModel` on the grid of a
  single-field GRID dataset in batches of at most `batch_size` points on the
  model's device and dtype, in eval mode without gradients, and restores each
  module's train flag. The result keeps the axes, axis order, left-hand side
  and metadata, holds the predictions in the source field's dtype and on its
  device, has no ground truth, and declares every axis non-periodic, because
  an unconstrained network does not agree at the two ends of a periodic axis.
  The input is not modified and no random number is drawn. A model saved
  with `kd.models.save_field_model(model, path, identity=smoothing.identity)`
  and read back as `kd.models.load_stored_surrogate(path).module` samples to
  the same bytes. Mismatched coordinate or field names, a non-positive
  `batch_size` or a non-finite prediction raise `ValueError`.
- `kd.models.FieldRecipe`, the architecture and training settings of one
  `FieldModel` fit (name, version, hidden sizes, activation, learning rate,
  epoch budget, patience, validation ratio, best-state restore, training
  precision), with `FieldRecipe.to_dict()` as its JSON form, and
  `kd.models.DLGA_NN_1`, DLGA's NN_1 network: five 50-unit sine layers,
  50,000 epochs at learning rate 1e-3, validation ratio 0.2, no early
  stopping, the best validation state restored. `DLGAConfig` reads its seven
  `surrogate_*` network defaults from `DLGA_NN_1`; every value is unchanged.
- `training_dtype`, the floating precision a field network is trained in:
  `None`, the default, trains in the data's dtype as before; `"float32"` or
  `"float64"` names one. `FieldModelTrainer.fit`, `kd.models.FieldRecipe`
  (and so `kd.network_smooth`), `kd.data.resample_to_grid`, `DLGAConfig` and
  `SGAConfig` take it; `kd.Model(algorithm="dlga", training_dtype=...)` and
  `kd.Model(algorithm="sga", derivatives="autograd", training_dtype=...)`
  pass it through the config, and `Model.train_surrogate` and the surrogate
  a fit trains read the same value. The trainer casts the model and copies
  of the detached inputs, so normalization, forward pass, loss, gradients,
  optimizer state and the best-epoch snapshot use that precision, then
  returns the network in the data's dtype on its entry device; the caller's
  tensors are not modified, and derivatives and the search are computed in
  the data's dtype. Any other value (another string, a `torch.dtype`,
  float16, bfloat16, a number or a boolean) raises `ValueError` (through
  `kd.Model`, a number or a boolean is refused at construction with
  `TypeError`, and a `torch.dtype` with `ValueError` when `fit` builds the
  config); SGA's finite-difference and tabular modes refuse a value other
  than `None`.
  An explicit value is recorded in the DLGA surrogate identity, the
  `network_smooth` identity's recipe, the SGA and DLGA run config and the
  `resample_to_grid` training hints, and is left out when it is `None`, so
  default identities, stored DLGA surrogates and resume comparisons are
  unchanged. A stored DLGA surrogate trained under another precision is
  refused at load and injection (`surrogate.training_dtype`), and a resume
  that adds, drops or changes the value is refused as `identity_breaking`.
  `training_dtype` is an identity-breaking and surrogate field of `sga` and
  `dlga`, so `kd.instrument_schema_digest()` changes; kd-agent experience
  and action-experience documents carry the digest, so documents built
  before this change are refused and must be rebuilt. The kd-agent SGA and
  DLGA cards render the new field, so their text and the pdeagent
  `prompt_sha256` move too.
- `sqrt` in kd expression IR. The default function registry binds it to raw
  `torch.sqrt` (not a protected operator): a negative argument gives NaN, and
  the existing non-finite checks reject that column. `from_sympy` accepts
  powers with exponent denominator 2: `x**(1/2)` becomes `sqrt(x)`,
  `x**(-1/2)` becomes `recip(sqrt(x))`, and other odd numerators become that
  integer power of `sqrt(x)`. Exponents with any other denominator, and
  Float or symbolic exponents, raise `ValueError`. PySR output containing
  `sqrt` therefore converts to kd terms. The LLM4ED column registry binds raw
  `sqrt`. `integrate_pde` returns
  `success=False` for a right-hand side that is non-finite at the initial
  state, such as `sqrt` of a negative initial field, under every solver
  method. **Breaking**: `sqrt` is a reserved field and variable name.
  `PDEDataset`, the tabular bridge, `SINDyRegressor` and
  `kd.data.prepare_trajectory_regression` reject a field, feature, target,
  `var_names` entry, state variable or parameter named `sqrt`, and the error
  suggests a replacement such as `sqrtval`; rename that column before building
  the dataset or fitting.
- SGA's `tabular` mode searches formulas with unary functions and a constant
  table. `SGAConfig.functions` names distinct functions from `sin`, `cos`,
  `exp`, `log`, `sqrt`; `SGAConfig.constants` lists finite, distinct values
  other than 0 and 1, which become leaves. Both default to `()`, which adds
  no function or constant to the tabular grammar; grid modes refuse non-default
  values. In tabular mode `exp`, `log` and `sqrt` are raw torch functions and a
  non-finite column is dropped; `div`, `n2` and `n3` keep their protected
  definitions. The same function of arguments that reduce to the same
  expression is one fitted column, a constant inside a function argument is
  part of that column, and a tree
  with a function whose argument is zero or contains no input feature is
  rejected. Tabular scoring accepts exact fits on noiseless data: the AIC uses
  `max(MSE, F)` with a floor `F` scaled to the target, the `-100` lower bound
  applies only to grid modes, and a support with at least as many columns as
  rows, or with two columns that differ only in inner constants, is invalid.
- LLM4ED has a `tabular` mode for `TabularDataset` targets. On tabular data
  `Model("llm4ed")` runs this mode; `functions=[...]` allows `sin`, `cos`,
  `exp`, `log` and `sqrt` with raw semantics. The model writes inner constants as
  placeholders `c1`, `c2`, ...; each candidate's placeholders are fitted to
  the table before sparse regression, and published terms carry the fitted
  values. A numeric constant term, or a term made only of placeholders, is the
  intercept column `one`. Terms are deduplicated by comparing columns
  elementwise, and a candidate whose active terms differ only in inner
  constants is invalid. Feature names that SymPy defines (`I`, `E`, `beta`,
  ...) are read as data columns; names of the form `c<k>` are refused. Default
  (EDL) mode results are unchanged. In both modes, candidates that used to
  raise (an overflowing literal, a column too small to normalize, a line that
  parses to a comparison) are invalid candidates.
- `kd.core.expr.constant_fit.fit_placeholders` fits the placeholders `c1`,
  `c2`, ... in a list of term IRs by variable projection: outer coefficients
  by least squares over all terms, BFGS on the residual sum of squares divided
  by the target's sum of squares, from the fixed starts `(1.0, -1.0, 0.1)`,
  evaluated in float64. A term that evaluates to a scalar raises `ValueError`:
  every term must reference a column or `one`.
- `kd.refit_constants(dataset, terms)` refits the numeric literals inside the
  terms of an answer on a `TabularDataset` and returns a `kd.ConstantRefit`:
  the refitted terms, the fitted constants and the literals they replaced, the
  outer least-squares coefficients and the NMSE. The NMSE follows
  `kd.core.metrics.nmse`: the mean squared residual over the population
  variance of the target, or the raw mean squared residual when that variance
  is at most 1e-15. A literal stays when the path from the root of its term
  reaches it only through `mul` (either argument), `neg` and the numerator of
  `div`, because the outer coefficient absorbs it; every other literal is
  refitted. The literals are the first candidate, and another candidate
  replaces them only when it lowers the fit objective below
  `incumbent * (1 - 1e-9) - 1e-20`, so the terms come back unchanged when no
  fit improves on them, also for a literal the data do not identify, such as
  the 2.0 in `n2(mul(2.0, x1))`. A table
  without rows, a non-finite target, a refit with no finite candidate and a
  term that evaluates to a scalar raise `ValueError`. Terms are evaluated with
  plain division, unclamped powers and raw functions, the new
  `FunctionRegistry.create_plain()`, which LLM4ED's columns now use too.
- **Breaking**: `SGAConfig` gains `functions` and `constants`, and
  `Llm4edConfig` gains `functions`; all three are identity-breaking fields.
  Resuming an SGA or LLM4ED checkpoint written by an earlier release, in any
  mode, is refused as a new lineage; start a new run. Tabular SGA checkpoints
  also carry the score policy `canonical-basis-aic-floor-v2` and a vocabulary
  record. The new fields and the
  LLM4ED mode change `kd.instrument_schema_digest()` and every SGA and LLM4ED
  `run_spec_hash` and `record_hash`; kd-agent experience and action-experience
  documents and the exports they are built from carry the digest, so documents
  built before this change are refused and must be rebuilt.
- Experimental agent action memory via `kd-agent run --action-memory
  <document>` (file input only; the document must sit outside the
  workspace): retrieves evidence for continue, restart, retune and
  algorithm-switch choices from the current search state, legal operations
  and remaining budget. Two document formats are read. In
  `kd-action-experience-v1` the action outcomes describe OLS-refitted
  input-grid error, not equation recovery. `kd-action-experience-v2` adds
  selected-answer recovery effects judged offline and keeps them apart from
  candidate-pool gains and proxy fit improvements; its retrieval keeps
  unknown, failure and cost denominators, binds the selector identity and
  requires judged support from several training families. A document whose
  schema digest or implementation identity differs from the installed kd is
  refused. The tools that build these documents are not part of this
  release.
- SGA accepts `TabularDataset` through its `tabular` mode, including scalar
  ODE regression tables. Candidate terms use feature columns and `one`, exclude
  the target column and spatial derivatives, and reject algebraically zero
  subexpressions. Tabular candidates are algebraically normalized before fitting:
  polynomial roots become independent monomial columns, valid rational roots
  remain whole columns, and exact linear dependencies are removed. The existing
  width bounds the independent fitted basis as well as the genetic trees. Native
  STRidge and AIC use the selected canonical columns returned to callers.
  Checkpoints bind that scoring policy and feature order. Grid-mode search
  results are unchanged by this mode; run identities and resume change as
  the **Breaking** `SGAConfig` entry states.
- Reusable ODE preprocessing in `kd.data`:
  `prepare_trajectory_regression` estimates derivatives separately for each
  trajectory using finite differences or one multi-output neural model per
  trajectory. `TrajectoryRegression.table` selects a target and observed or
  neural RHS states without retraining. `TrajectoryNNConfig` configures the
  neural fit; `save_trajectory_regression` and `load_trajectory_regression`
  preserve numeric arrays, training reports and content identities in JSON/NPZ.
- PySINDy enforces a sketch's hole term count after the STLSQ fit when the
  holes admit at most `HOLE_SUBSET_MAX = 3` terms in total. If STLSQ's support
  is empty, or its number of non-anchor terms lies outside the summed
  `min_count..max_count` of the holes, the fit is replaced by a subset
  selection over the same columns and pin-deducted target. Every subset of the
  non-anchor columns outside the anchor span (a column inside that span, such
  as an alias spelling of an anchor, is not a hole term) within the
  summed count is fitted together with the anchor columns by ordinary least
  squares on unit-norm columns and scored by `aic_no_n(mse, k)` at ratio 1,
  with `k` counting the subset and the anchors. The lowest score wins, and
  scores equal to within `1e-12` relative go to the smaller subset, then to the
  smaller column indices. With fewer columns outside the anchor span than the
  holes' summed `min_count`, STLSQ's fit stands and the exit reports the hole
  as short. The recorded `stlsq_history` stays STLSQ's
  trajectory, and the compile report carries a note with the summed bounds. A
  selection that leaves no term, possible only with `min_count=0` and no
  anchors, raises `RuntimeError`. Sketches whose holes admit more than three
  terms in total keep the STLSQ fit, and the declared `hole_count` level stays
  `exit_checked`.
- `kd-agent`: a successful `run_discovery` envelope carries
  `primary.now_selected`, the `run_id` the run publishes when no answer is
  submitted: the `select_node` choice, otherwise the default selection over
  the policy tree, or null when that rule has no winner; the run never falls
  back to the search it just ran, also when the tree could not record that
  search. It is
  the provenance the answer channel is written with, the value `prune_node`
  and `select_node` already reported.
- `kd-agent`: `python -m kdagent.tree surrogate train` takes `--device`, with
  the choices of `kd-agent run --device` (`cpu`, the default, or `cuda`). The
  surrogate id does not depend on the device, so a surrogate trained on one
  device is reused on the other.
- `kd-agent`: `kd-agent run`, `chat` and `fingerprint` take
  `--training-dtype float32|float64`, and `kdagent.run`, `prepare_run` and
  `chat` take `training_dtype=None`: the precision every field network the
  run trains is trained in (the surrogates `train_surrogate` trains, the
  surrogate a dlga or sga-autograd search trains for itself, and
  `smooth_input(method="network")`). Omitted, the run makes no choice and
  every tool value stands as before. With a choice, a tool value that is
  missing or null inherits it and a different explicit value is refused as
  `invalid_argument` before any worker starts; only a mode that declares
  `training_dtype` and trains a network receives it, so sga's
  finite-difference and tabular modes and the other instruments run
  unchanged. The value enters `params.training_dtype` of a surrogate recipe,
  `network_recipe.training_dtype` of a network recipe and the search's
  `model_kwargs`, so it is part of reuse, plan and resume identity: a
  surrogate or network trained under another precision is not reused,
  injected, sampled by id or differentiated by an autograd measurement
  (`refine_candidate`, `ablate_candidate` and `score_term_sets` with
  `method="autograd"` refuse it as `surrogate_training_dtype_mismatch`
  before any worker starts), and kd's resume gate refuses a lineage
  continued under another one. The trace metadata, the run record and the
  close-out carry `training_dtype` after `device` when it is set. The prompt
  is unchanged. `Session(training_dtype=...)` validates the value with kd's
  `FieldRecipe`. `fingerprint` accepts the flag and ignores it.
- `kd-agent`: `python -m kdagent.tree surrogate train` takes
  `--training-dtype float32|float64`, resolved by the same rule; the
  precision is part of the surrogate recipe, so a later training run reuses
  the file only under the same choice.
- `kd-agent`: a `run_discovery` search whose run directory sealed `completed`
  also leaves `result.json` there, the fit's `kd.ExperimentResult` written
  with `ExperimentResult.save` and read back with
  `kd.ExperimentResult.load`, so a report or figure drawn later does not
  refit. On burgers the file is about 4 MB (`record.json` about 3 KB).
- `kd-agent`: `kdagent.analysis.trajectory_figure(workspace)` returns a
  matplotlib `Figure` of one run read from its `tree.jsonl` and
  `trace.jsonl`: each search segment at the minutes since the run started,
  its NMSE on a log axis (the neutral full-grid frame where the row has it,
  the instrument's own otherwise, as separate series per instrument), pruned
  segments and the subtrees below them crossed out, and the chosen candidate
  starred with its law written by `kd.format_pde`.
- `kd-agent`: `run_discovery`, `train_surrogate` and
  `refine_candidate(pinn_epochs=...)` replies carry `diagnostics.resources`,
  measured in the worker process that ran the search, the training or the
  PINN fit: `device` (`"cuda"` when that process allocated on the card,
  `"cpu"` otherwise) and `cuda_max_memory_allocated_bytes`
  (`torch.cuda.max_memory_allocated()` there, `null` when it never
  initialized CUDA). A failure reply carries it whenever the worker wrote a
  result, a training stopped at its cap included; a reused surrogate or
  PINN field ran no worker and has none.
  `provenance.execution_conditions.device` stays the run's policy.
- `kd.harness.environment_fingerprint()` returns a sixth field, `git_status`:
  `"dirty"` when a tracked file of kd's repo differed from `git_sha` (staged
  or unstaged) as the fingerprint was taken, `"clean"` otherwise, and
  `"unknown"` under the same failures as `git_sha`. Untracked files do not
  count. The fingerprint is the `env` of a fresh evidence store, so a row run
  on a tree with uncommitted changes now records that its SHA does not
  describe the loaded code. `merge_stores` keeps `git_status` in the merged
  top-level `env` only when all sources agree on it (each source row keeps
  its own value), and sources that differ in it need no `version_note`,
  unlike sources that differ in `kd_version` or `git_sha`.
- DISCOVER's grid modes (`mode1`, `mode2`) take a grid whose spatial axes are
  drawn from `x` and `y` under the default config. The default vocabulary is
  derived from the dataset's axes: the coordinate leaves `[*axes, t]` and,
  per axis `a` in dataset order, the operators `diff_a` and `diff2_a` after
  the six base operators
  (`kd.search.discover.config.default_library_config(spatial_axes, fields)`,
  which also makes every field a leaf, sorted by name);
  a `library` other than the dataclass default is never widened, and an axis
  outside `x`, `y` keeps the refusal. On a one-axis `x` grid the derived
  vocabulary equals the previous default in the same token order, so 1-D
  runs, their recorder
  keys and their resume identity are unchanged. The two modes declare
  `spatial_axes=("x", "y")` in `kd.instrument_schemas()` (previously
  `("x",)`); no mixed (`u_xy`) or third-order terms are added. The run
  record's `config.library` keeps the user's (1-D default) declaration while
  the engine searches the derived vocabulary, as on the sketch path.
- `kd.Model(algorithm="pysindy")`, `"pysr"` and `"dlga"` take a grid whose
  spatial axes are drawn from `x` and `y` under the default config. A catalog
  the caller did not set is built over the dataset's spatial axes in dataset
  order: `u`, then per axis `a` the group of the 1-D default with `x` replaced
  by `a` (pysindy and pysr `terms`: `u_a, u_aa, mul(u, u_a)`; dlga `library`:
  `u_a, u_aa, u_aaa`). A `terms=` / `library=` keyword is never rebuilt, nor is
  a `config=` whose catalog differs from the dataclass default; a `config=`
  with the default catalog is rebuilt, as on tabular data. On a one-axis `x`
  grid the default is kept as is, so 1-D configs and library fingerprints are
  unchanged. Under the default catalog all three
  refuse a grid with a spatial axis the default does not cover (for example
  `x, y, z`); for dlga this refusal is new and also applies to
  `Model.train_surrogate` (dlga previously searched such a grid over the `x`
  library without a message). A hand-built `PDEDataset` whose `lhs_axis` is
  unset keeps the 1-D default, so on a 2-D grid it is refused until
  `lhs_axis` is set. No mixed (`u_xy`) terms are added. On 2-D input the
  recorded `config.terms` / `config.library` is the catalog the search was
  built over. The three default modes declare `spatial_axes=("x", "y")` in
  `kd.instrument_schemas()` (previously `("x",)`), and the published field
  docstrings of `PySINDyConfig.terms`, `PySRConfig.terms` and
  `DLGAConfig.library` name the 2-D default, so `kd.mode_accepts` admits a
  2-D grid for these modes without `terms` / `library`. The schema digest
  and the pdeagent `prompt_sha256` move with it.
- LLM4ED's default mode takes a grid whose spatial axes are a non-empty
  subset of `x`, `y`, with the one field `u`. On two axes the operands are
  `u, u_x, u_xx, u_xxx, u_y, u_yy, u_yyy, x, y`: EDL's finite-difference
  templates applied along each axis. The prompt's spatial sentence names
  the axes present, and within one equation the columns are deduplicated by
  elementwise comparison, as in the tabular mode (EDL's comparison of
  absolute sums dropped `u_xx` from `u_xx + u_yy` on a field symmetric in `x`
  and `y`). A lone `y` axis is accepted; any other axis name is refused as
  before. On a one-axis `x` grid the prompts, the operand columns, the
  target and the recorder keys are unchanged. The mode declares
  `spatial_axes=("x", "y")` (`spatial_dims` `[1, 2]`), so the schema digest
  moves.
- `kd.instrument_schemas()` publishes `spatial_dims` on every mode: the
  axis counts `kd.mode_accepts` admits from the mode's declaration,
  `[1, ..., n]` for a fixed set of `n` axis names without a `names_from`
  field, `null` for any count (no axis set, or a `names_from` field built
  from the input's axes). The schema digest moves with the new key; the
  controller's cards render discover's `spatial_axes`, so the discover card
  text and the pdeagent `prompt_sha256` move too.
- DISCOVER's grid modes (`mode1` and the PINN mode `mode2`) take a grid
  dataset with more than one field. The target is the dataset's left-hand side;
  every field is a leaf of the default vocabulary, sorted by name (the
  dataset fingerprint identifies the fields as a sorted set, so a
  checkpoint resumes on the same fields in any dict order); a library other
  than the dataclass default is never widened and still refuses a field it
  does not name. On one field `u` the vocabulary, the recorder keys and the
  resume identity are unchanged. The two modes declare no field set in
  `kd.instrument_schemas()` (`field_names` was `["u"]`), so a single-field
  grid whose field is not named `u` is accepted too; the discover card text
  and the pdeagent `prompt_sha256` move with it.
- `kd.instrument_schemas()` publishes `field_counts` on every mode: the
  field counts `kd.mode_accepts` admits from the mode's declaration, `[n]`
  for a fixed set of `n` field names without a `names_from` field (the
  input must carry exactly those names), `null` for any count. EqGPT's
  `single_wave` and `steady` modes and LLM4ED's default mode publish `[1]`;
  every other mode publishes `null`. The
  schema digest moves with the new key.
- `PDEDataset.retarget(target)`: a copy of a multi-field dataset whose
  left-hand side is `target`, a field derivative such as `v_t` or `v_tt`
  (on a tabular dataset, the target column), every tensor shared and
  nothing else changed. The copy's fingerprint differs from the input's
  through the left-hand side alone, so the two targets never share a cache
  or checkpoint key. An unknown field or axis raises `ValueError`.
- `PDEDataset.with_field(name, values)`: a copy of a GRID dataset whose
  field `name` holds `values` (a tensor or a numpy array), converted to that
  field's dtype and device. Axes with their periodicity and `allow_nan`, the
  other fields, the left-hand side, name, task type, noise level, ground
  truth and source are carried unchanged, and the replaced field keeps its
  `allow_nan` flag. A values array of another shape, Inf, or NaN in a field
  that does not permit it raises `ValueError`; an unknown field raises
  `KeyError`.
- `PDEDataset.window(axis, *, start=None, stop=None)`: a copy of a GRID
  dataset restricted to `start <= c <= stop` on one axis, with both bounds
  inclusive and in coordinate units (`None` leaves that side open), and
  every field sliced along that axis. A periodic axis stays periodic only
  when the window keeps all of its points. A range that keeps fewer than
  two points raises `ValueError`; an unknown axis raises `KeyError`.
  `with_field` and `window` both refuse SCATTERED and TABULAR datasets with
  `ValueError`.
- `kd.load(path)` recognizes a plain grid without load options, through the
  registered layout `plain-grid`: an `.npz` or MATLAB `.mat` file (v5 or
  v7.3) holding coordinate arrays `t` and at least one of `x`, `y`, `z`,
  each 1-D or a row or column vector, and one or more field arrays of one
  shape with one dimension per coordinate. 0-d arrays (a stored viscosity, a
  label) are ignored. Fields keep their stored names (the bundled
  `Burgers_equation.mat` loads with field `usol` and left-hand side
  `usol_t`). The axis order is read from the dimension lengths, and the
  dataset equals the one `kd.load(path, coords=..., fields=...)` builds with
  the coordinates listed in storage order; `periodic` is never inferred.
  When two coordinates share a length, `kd.load` raises `ValueError` naming
  the arrays and the `coords=` / `fields=` / `field_axes=` call. A file
  holding `axis_order` or `format` keeps its own layout, and a file with any
  other array (a parameter vector) or with fields stored in different axis
  orders still needs the explicit mapping.
- LLM4ED's default mode accepts sketches. Pinned terms are deducted from
  LLM4ED's own finite-difference target (`sketch_lower_owner` is
  `"native"`), and a candidate term that regenerates a pinned column is
  refused before scoring. Every scored candidate must contain each anchored
  term, and anchored columns are kept out of STRidge thresholding. Holes
  narrow the operands the prompt offers and the parser admits, and narrow
  the prompt's operator list as a hint; a term no hole admits is refused
  before scoring. With a sketch the prompt gains one paragraph naming the
  required terms, the terms already accounted for and the term budget;
  without one the prompt is unchanged. The compile refuses, naming the term
  or hole: a pin or anchor not spelled over the operand vocabulary with
  EDL's operators, an anchor that expands to a sum, an operator clause that
  names an operator outside `add`, `sub`, `mul`, `recip`, `n2`, `n3`, an
  empty kept vocabulary, and a required hole that no kept operand fills.
  The default mode declares `fixed_terms` as `lowered`, `anchors` as
  `fit_enforced`, `hole_count` and `operator_set` as `exit_checked`, and
  `derivative_order` and `field_axis_set` as `generation_enforced` (before:
  all `unsupported`), so `kd.instrument_schema_digest()` changes. The
  `tabular` mode still refuses sketches.
- A trained DLGA surrogate can be written and read back as a kd file.
  `Model.train_surrogate(dataset, save_to=path)` writes the network as a
  `kd-surrogate-v1` file with one new optional key, `identity`: the
  fingerprint of the LHS-resolved dataset, a SHA-256 of every byte of the
  training target, the config's surrogate fields and the training seed
  (`kd.search.dlga_surrogate_identity(dataset, config)`). `save_to` on an
  instrument other than dlga raises `ValueError`. `Model(algorithm="dlga",
  surrogate_model=...)` takes a module as before, a stored entry or a file
  path. An entry or a path is checked in `prepare()` against the search's
  dataset and config on every identity key but the training seed, and a
  mismatch raises `ValueError` naming the differing keys; an entry or a
  file without an identity is refused; a bare module is injected unchecked,
  as before. `kd.models.load_stored_surrogate(path, *, device=None)` returns
  a `kd.models.StoredSurrogate` (`module`, `identity`, `path`);
  `kd.models.save_field_model` takes `identity=`, and `load_field_model`
  takes `device=`. `kd.search.save_dlga_surrogate` and
  `kd.search.load_dlga_surrogate` write and read such a file, the loader
  comparing every identity key, the seed included. The 0.8.2 loader refuses
  a file that carries an identity (unexpected keys).
- `kd-agent`: eight controller verbs. `refine_candidate` refits a search's
  law, or a given law or term list, with its terms held fixed on the data of
  the search it comes from: on a grid with `kd.refit_terms`, optionally after
  dropping small terms (`small_term_rtol`), after Savitzky-Golay
  (`savgol_window`) or low-pass (`lowpass_modes`, `time_window`) smoothing,
  or by a joint network fit (`pinn_epochs`, `kd.refine_pinn`); on a table
  with `kd.refit_constants`. It returns a law `submit_answer` accepts. With
  `pinn_epochs`, the coefficient frame is
  `kd-refine-pinn-e<epochs>-p<pretrain>-w<weight>-v2` and the derived
  input's `pinn_result` records `pretrain_epochs_run` and
  `pretrain_stopped_early`; a `pinn_pretrain_epochs` budget up to 5000
  epochs (the default) is pretrained in full, and a larger one until the
  patience or the budget ends it. `smooth_input` writes a low-passed copy of a
  grid file input (`lowpass_modes`, `time_window`) as a new input
  `<dataset_id>+lowpass<modes>-t<time_window>`, accepted wherever its parent
  is and listed by `list_datasets` with its `derived_from`.
  `ablate_candidate` refits a search's
  law without each term in one measurement frame (`kd.ablate_terms`);
  `score_term_sets` fits several term sets in one frame resolved for their
  union (`kd.score_term_sets`), each fit with its own weak diagnostics;
  `verify_candidate` reports term validation, block dispersion and the
  forward error of a law as three separate measurements, the forward
  integration under a cooperative deadline and the worker under a hard
  limit; `vote_terms` counts the terms of several searches' sealed laws
  (`kd.vote_terms`), refusing repeated searches, two segments of one
  lineage and two input roots; `make_sketch` builds and registers a Sketch
  from cited evidence (`kd.sketch_from_anchors`) and returns its
  `sketch_id`; `finish_without_answer` ends a route with an explicit
  unresolved result (`source="abstained"`) and its reason. Votes,
  ablations, batches and verifications return an `evidence_id`, recorded in
  the session and in `<workspace>/evidence.jsonl`.
- `kd-agent`: `smooth_input(method="network")` trains a network field once
  per input, recipe (kd's `FieldRecipe`, default `DLGA_NN_1`) and seed,
  registers the model and writes the sampled field as
  `<input>+field-<surrogate_id>`; a repeated request trains nothing, and
  `surrogate_id=` samples a registered model. `lowpass_modes="auto"` lets
  `kd.select_lowpass_modes` choose the counts from refits of `terms` and
  reports its whole trace. `max_seconds` caps the whole call.
- `kd-agent`: `refine_candidate` takes `evaluation_input_id`, `method`
  (`"finite_diff"`, `"autograd"`, `"weak"`), `trim`, `surrogate_id` and
  `weak` and forwards them to `kd.refit_terms`; the reply carries kd's
  `RefitFrame`. Every successful refinement returns a `refinement_id` and
  stores an immutable receipt (session and `<workspace>/refinements.jsonl`):
  parent, source search, support origin, exact law and digest, method and
  parameters, input records, frame and metrics.
- `kd-agent`: `run_discovery` takes `sketch_id` (a Sketch `make_sketch`
  registered) and `verify={"nmse_max": x}` (the Sketch exit's residual
  threshold, finite and nonnegative); a resumed segment keeps its lineage's
  threshold. `diagnostics.sketch` adds the certificate: the sealed
  candidate, its residual NMSE, the threshold, whether it passed and where
  it was measured.
- `kd-agent`: `submit_answer` takes `refinement_id`. A submitted
  provenance carries `producer_ref` (the search, the refinement receipt or
  `{"kind": "edited"}`) and `cited_run_id`.
- `kd-agent`: `run`, `prepare_run` and `chat` take `initial_sketch`, and
  `kd-agent run` / `chat` take `--sketch PATH`: a user Sketch payload, the
  content of a file `kd.save_sketch` writes (`kd-sketch-v2`, unbound
  `kd-sketch-spec-v1` shorthand, or a released `kd-sketch-v1` payload, which
  kd reads under the v2 rule), bound to the pinned input in a worker
  (`kd.load_sketch(payload, dataset=)`) and registered before the endpoint
  is built (a run that pins no input, or a payload kd does not decode or
  cannot bind to that input, raises `ValueError`); every search that names
  no Sketch runs under it. Its `sketch_sha256` is in the
  trace metadata, the close-out and the report, apart from `prompt_sha256`.
- `kd-agent`: `finish_without_answer` ends the run like a submission: no
  further model turn and no nudge; the close-out has `answer` null,
  `answer_source` `"abstained"`, `stop_reason` `"not_converged"`, and the
  model's `reason` and `evidence_run_ids`, and no selection is published in
  its place.
- `kd-agent`: a submission a refinement produced adds `answer_receipt`, that
  refinement's session receipt (method, parameters, input records, resolved
  frame, metrics), to the close-out.
- `kd.search.last_completed_iteration(run_dir)` returns the last completed
  iteration a run wrote to its `events.jsonl`, as a `kd.search.IterationEvent`,
  for a run killed before it sealed its record. It reads the last two lines:
  a final line that the kill tore is skipped and the line before it is used.
  A missing or empty file returns `"no_events"`, a file holding only a torn
  line returns `"no_completed_iteration"`, and a line it reads that is not a
  valid event raises `ValueError`.
- `kd.harness.environment_fingerprint(strict=True)` raises `RuntimeError`
  naming each field that would be `"unknown"` (`git_sha`, `git_status`,
  `torch_version`) and why it could not be read. Without `strict`, the call
  never raises and returns the same mapping, with `"unknown"` in a field
  that cannot be read.
- `kd.judge_law(terms, coefficients, truth_terms, truth_coefficients, *,
  axes, fields=("u",), small_term_rtol, coeff_rtol, scale_free=False)`
  judges a discovered law against a truth equation term by term, without
  data, and returns a `kd.TermVerdict`. Both sides are expanded with
  `kd.jet_polynomial`, so every spelling of one monomial is one key
  (`u_xx`, `diff2_x(u)`, `diff_x(diff_x(u))`; `u*u_x` and half of
  `diff_x(n2(u))`). The verdict reports missing and extra monomials after
  the candidate's small terms are trimmed, the relative error of each shared
  coefficient, the Jaccard score of the two supports, and the two
  thresholds, which have no defaults; `scale_free=True` judges a homogeneous
  law up to a common factor, dividing both sides by their coefficients of
  the first monomial they share (`TermVerdict.gauge`); with no shared
  monomial the coefficients are not ok and `TermVerdict.coefficients_reason`
  is `"no_shared_monomial"`. `kd.law_from_equation(equation, *, axes,
  fields=("u",))` reads truth strings such as `u_t = -u * u_x + 0.1 * u_xx`
  and `u_xx + u_yy + 1 = 0` into a `kd.ParsedLaw` of terms and
  coefficients. The left side may also be the ODE spelling
  `d<field>_d<axis>` (`dx_dt`), read as the derivative `<field>_<axis>` and
  recorded in that spelling. `kd.judge_recovery` keeps its numerical job.
  No module a search runs may import the judge.
- `kd.forward_error_trajectories(trajectories, laws, *, method="Radau",
  max_step=None, max_seconds=None, rtol=1e-6, atol=1e-9)` integrates one law
  per state of a `kd.TrajectorySet` (keyed `d<state>_dt`, each a pair of
  terms and coefficients) from each trajectory's first sample on its own
  time points, and returns a `kd.TrajectoryForwardError`: the relative L2
  error over all trajectories, states and times, the same error per
  trajectory and per target (None for an entry whose observations are all
  zero, where the relative error is undefined), the predicted trajectories
  when the integration succeeded, and the reasons of `kd.forward_error`
  (`refused`, `diverged`, `solve-failed`, `timeout`). `kd.Trajectory` and
  `kd.TrajectorySet` are exported from `kd`. `kd.core.integrate_trajectory`
  integrates one trajectory with the solve loop, deadline and failure
  classification of `kd.core.integrate_pde`.
- `kd.harness.run_plan(..., parallel=kd.harness.ParallelOptions(...))` runs
  each plan entry in its own supervised child process under
  `kd.harness.ResourceLimits`: slots, an estimated-memory budget, GPU slots
  and a card pool (`gpus`, or the cards `CUDA_VISIBLE_DEVICES` lists), with
  `fifo`, `backfill` or `resource_queues` admission. Records, run-dir
  pointers, catalog rows and the plan hash are those of the serial run; the
  index lists attempts in completion order. `device_by_instrument` sends
  chosen instruments to a card; `hard_seconds` stops an entry as a `raised`
  episode, and a child that ends without an outcome is recorded as a
  `raised` `EpisodeProcessEnded` attempt. Process facts per entry are in
  `jobs/entry-NNNN/process.json`. `parallel=None` (the default) is
  unchanged.
- `kd.harness.run_jobs`, `Job`, `JobResult`, `SchedulerState`,
  `resolve_gpu_pool` and `child_environment`: the scheduler behind it.
- `kd.viz.plots.plot_trajectory_overlay`, `plot_state_panels` and
  `save_trajectory_animation` draw observed and discovered-law ODE
  trajectories (two `kd.data.TrajectorySet`s) in state space (2D or 3D),
  per state over a time window, or as a movie. `plot_state_panels` draws
  each state against time with the observed curve solid and the discovered
  law's curve dashed over it, as the profile movie and waterfall do; a
  figure shows one trajectory best, so the caller passes a set holding the
  one to show. `save_trajectory_animation` titles every frame with its time,
  with the same number of decimals on every frame, and writes frames of the
  profile movie's size; on 3D axes the box takes the frame's height, and the
  time title and the legend sit in the frame's top-left corner.
  `plot_profile_animation`, `save_profile_animation` and
  `plot_profile_waterfall` draw a 1D field's observed and integrated
  profiles as a movie or as a stacked poster; the movie keeps the observed y
  range, and a frame whose discovered profile leaves it says so with the
  actual range. `plot_profile_waterfall` raises each row by 0.6 of the
  observed range rounded to one significant figure and states that value in
  its title (an explicit `offset` is used as given). Movies are GIF or MP4
  (H.264) by file suffix; MP4 needs ffmpeg.
- `field_cmap="sequential" | "diverging"` on `plot_field_comparison`,
  `plot_field_animation` and `save_field_animation` (the diverging map for
  two-phase fields), and `difference` on the two animation functions, `True`
  by default (a Predicted - True panel with its own symmetric scale;
  `difference=False` leaves it out).
- `Model.fit(max_seconds=)`: a wall-clock budget for the whole fit. The clock
  starts when `fit` is called, so smoothing, the checkpoint fingerprint, the
  component build and `prepare()` come out of it. The search stops at the
  first iteration boundary after it; DLGA's surrogate training, SGA's
  autograd surrogate and EqGPT's steady surrogate stop at the first epoch or
  iteration boundary after it, and each cut is logged (DLGA's also shows as
  `TrainingResult.stopped_at_deadline` on the result `save_surrogate`
  returns). A budget spent before the search still runs one iteration and
  seals a record. The budget is not part of the run identity.
  `kd.harness.run_episode` forwards `max_seconds`; `WallClockBudgetCallback`
  takes an optional fixed origin (`started_at`) and a `deadline` property.
  `kd.search.PlatformComponents` gains the keyword-only field `deadline`
  (the fit's deadline on the `time.perf_counter()` clock, `None` without a
  budget), which a plugin that trains a surrogate in `prepare()` reads to
  stop at the budget.
- `RunCost` records what ended the search (`stop_reason`: one of
  `iterations_exhausted`, `algorithm_done`, `wall_clock_budget`,
  `wall_clock_budget_before_search`, `evaluation_budget`, `early_stopping`,
  `callback`), the budget (`max_seconds`), the process's peak RSS
  (`peak_rss_bytes`), the CUDA high-water mark on the run device
  (`cuda_max_memory_allocated_bytes`, None when the run device is not a CUDA
  device) and the run device (`device`). The two peaks are process-wide
  marks. `ExperimentResult.stop_reason` reads it. Records sealed before these
  fields still load and verify.
- `Model.save_surrogate(path)` writes the DLGA network a fit trained or
  restored as a stored surrogate whose identity carries the seed it was
  trained with; a network the fit's deadline cut short is marked so that
  `kd.search.load_dlga_surrogate` refuses it as the training of its
  configuration.
- `kd.refit_terms`, `kd.evaluate_terms` and `kd.evaluate.compare_results`
  take `fixed={term: value}` or `sketch=`: the fixed coefficients are held,
  not refitted. Each fixed column times its value is deducted from the
  left-hand side, the other terms are fitted to the rest, and the result
  reports the full equation: the fixed values as given, and the residual,
  MSE, NMSE and R-squared of the returned coefficients measured on the
  original left-hand side, so a constrained and an unconstrained refit of
  one support use one ruler. A key matches the input term with the same law
  identity (`mul(u,u_x)` fixes `mul(u_x,u)`, `neg(...)` folds into the sign)
  and is appended when no input term has it. A Sketch contributes its
  pinned terms; a shorthand Sketch is resolved on the refit's own
  `lhs_order`. With every term fixed nothing is solved and the condition
  number is None. Pruning by `small_term_rtol` never drops a fixed term, and
  a pruned term is not checked against the constraint. `RefitResult.fixed`
  names the fixed terms and `ComparisonResult.fixed` the constraint;
  `compare_results` refits every run under it, so a run searched without
  the Sketch is compared on the constrained structure instead of being
  excluded. Refused: a zero or non-finite value, a key that is the
  left-hand side in any spelling or sign, two keys naming one column, an
  input term that is an alias spelling of a fixed column, one key matching
  several input terms, and `fixed` together with `sketch`. The weak form
  (`method="weak"`) applies the same rule to its integrated target. Calls
  without these arguments are unchanged.
- `kd.load` reads ODE trajectories and regression tables as well as grids.
  `coords` naming only the time (`coords={"t": "time"}`) with every state in
  `fields` returns a `kd.TrajectorySet`: single vectors, matrices with a
  shared or per-row time grid, or a long table grouped by `trajectory_id`
  (trajectories keep the order their ids first appear in), with `params`
  mapping per-trajectory constants. `lhs` naming a column without `coords`
  returns a `kd.TabularDataset` whose other columns are the inputs; `fields`
  selects and renames columns. NPZ, CSV, XLSX, HDF5 and MATLAB v5/v7.3 files
  work for both (text ids, byte strings included, are kept as written).
  `kind="pde" | "trajectory" | "tabular"` states the kind
  (and narrows the static return type); a target that is also a column's
  derivative (`lhs="u_t"` on columns `u`, `t`, `u_t`) needs it. `kd.load`
  recognizes a `kd-tabular-npz` file and loads it like `kd.load_tabular`.
- `TrajectorySet.file_source` and `TabularDataset.file_source`: the file a
  set or table was read from (`kd.DatasetSource`), kept through the tabular
  bridge and `trajectories_to_tabular`; never part of a fingerprint or a
  saved result.
- `kd.inspect_file` reports XLSX files: the columns of the active worksheet
  under its first-row header, with NaN counts for empty cells and text
  columns without extrema.
- `kd.data.compute_dataset_fingerprint` accepts a `TabularDataset` and
  returns the fingerprint of the dataset `Model.fit` builds from it.
- `kd.register_algorithm(name, plugin_cls)` registers a search plugin class as
  an algorithm. `kd.instrument_schemas()` then lists its schema after the
  seven built-in algorithms (and `kd.instrument_schema_digest()` changes),
  and `kd.Model(algorithm=name)` and `kd.harness.run_plan` run it. A name
  follows `[a-z][a-z0-9_]*` and registers once per process. `kd.Model` builds
  the config as `config_cls(seed=seed, **fields)` from its `seed` and the
  config fields passed by name, or takes a `config=` instance of that class,
  and passes the constructor exactly the keywords the class lists in its
  `constructor_kwargs` class attribute (`mode`, `task`, `surrogate_model`,
  `provider`); `surrogate_model=` and `provider=` are refused for an
  algorithm that does not list them. The plugin's `config` mapping must carry
  `"algorithm": name` and the config's seed, or `fit` refuses it. The plugin
  contract `FacadeWiringContract` and its base `ScoreContract` are imported
  from `kd.search.protocol`; a homogeneous producer's form-specific plots
  use `kd.viz.extension.HomogeneousVizExtension`, and `kd.viz.plugin_panels`
  is the built-in plugins' shared panel skeleton.
- Entry-point group `kd.extensions`: `import kd` runs every installed hook
  (value `module:function`, a function without arguments that calls
  `kd.register_algorithm`) once, sorted by hook name. A hook that fails makes
  `import kd` raise `ImportError` naming the hook, its distribution and its
  version.
- `kd.extension_source(kind, name)` returns `"builtin"`, `"entry_point"` or
  `"process"` for a registered extension (`kind="algorithm"`).
  `kd.harness.run_plan(..., parallel=...)` refuses, before it creates a
  store, an algorithm whose source is `"process"`, since a parallel worker
  imports kd without it; a serial plan and `kd.Model` run it.
- `kd.extension_origin(kind, name)` returns the whole origin of a registered
  extension as a new dict: `source` (as `kd.extension_source`),
  `distribution`, `version` and `hook` (set for a registration made by an
  installed `kd.extensions` hook, else `None`) and `implementation`, the
  registered class as `module:qualname`. An unknown kind or name raises
  `KeyError`.
- The result config of a run of an algorithm registered by a hook or in the
  running process carries its origin under `"extension"` (source,
  distribution, version, hook and plugin class), so `run_spec_hash` changes
  when the extension is upgraded. The result configs of the seven built-in
  algorithms are unchanged.
  A checkpoint of such an algorithm (the periodic and final checkpoints of
  `fit` and `ExperimentRunner.save_checkpoint`) carries the same origin under
  `"extension"` in its config snapshot, and `fit(resume_from=...)` refuses,
  before any state is applied, a checkpoint whose origin differs from the one
  registered now (another release, hook or plugin class), with `reseed=True`
  too, and a checkpoint of such an algorithm that carries no origin. The
  refusal is the identity-breaking `ValueError` whose `changes` names the
  field `extension`. Checkpoints of the seven built-in algorithms are
  unchanged.
- `kd-agent`: every verb but `list_datasets` takes a `reason` (optional,
  except on `prune_node`, `select_node` and `finish_without_answer`, which
  require it), and the discovery tree (`tree.jsonl`) records it: `run_discovery` on its
  segment row, every other verb as an `action` event with the verb, the ids
  the call produced, the cited run, the status and the reason. A call
  refused before it started anything is not recorded; one whose worker
  started and failed is recorded with status `failure`. A verb refuses a
  given but blank reason. `python -m kdagent.tree record` and
  `surrogate train` take `--reason`; `tree status` and the report's
  "Decision log" table show the recorded actions. A tree with `action`
  events does not replay in an older controller.
- `kd-agent`: LLM4ED can run inside the agent on its own OpenAI-compatible
  endpoint. `kd-agent setup --llm4ed` saves an `llm4ed` block (endpoint
  `nvidia` with its preset URL or `openai` with a given URL, the model, the
  key and optional call, token and time limits) beside the controller's
  settings; `run`, `chat` and `fingerprint` take `--llm4ed` / `--no-llm4ed`,
  and LLM4ED is off unless enabled. The key is read from the saved file and
  set as `OPENAI_API_KEY` only in the environment of an LLM4ED search worker;
  no option or environment variable supplies it, and LLM4ED together with
  `--shell` is refused. `run_discovery` writes the run's endpoint and limits
  into the search's parameters, refuses a `base_url`, `model` or
  `extra_body` other than the run's, a limit above the configured one and a
  `tape_record_path`, and caps the search at the smallest of the configured
  seconds, `max_search_seconds` and the remaining run time. The public
  settings, never the key, are in the trace metadata, the close-out, the
  report and the prompt fingerprint.
- `kd-agent`: the derivative network a dlga search trains for itself (no
  `surrogate` injected, no segment resumed) is saved by kd
  (`Model.save_surrogate`) and registered under an id of its own. The reply
  carries the id as `provenance.trained_surrogate` and in
  `diagnostics.surrogate` (`surrogate_id`, `stopped_at_deadline`,
  `savgol_window`, `train_seconds`, `epochs`, `final_loss`, or a `reason`
  when none was registered); the tree row stores it as `trained_surrogate`,
  apart from `surrogate` (an injected network), and the report shows it.
  The registry row names the run that trained it (`source_run`). A repeated
  recipe registers another id and no earlier id changes. A completed
  network trained on the input as loaded counts as the training of its
  recipe, so `train_surrogate` with the same settings and seed returns its
  id. A network the cap cut short (`stopped_at_deadline` true) does not
  count: kd refuses it as a search's surrogate, and the autograd
  measurements take it by id. A network trained after `savgol_window`
  smoothing has the window in its recipe and is taken only by a search with
  the same window. A failure to save or register the network is reported
  in `failures` and the search stays a success.

- `examples/gallery/` holds three scripts that run a discovery and integrate
  the discovered law forward against the observations: `kdv.py` (two KdV
  solitons, SGA, seed 42), `allen_cahn.py` (Allen-Cahn fronts, SGA, a search
  seed on the command line) and `lorenz.py` (the Lorenz system from two
  trajectories with 1 % noise, one SGA law per state from neural-network
  derivatives). Each writes its figures, a movie (needs ffmpeg), the HTML
  report of its run (one per law for Lorenz) and a run record.
- `kd` exports `derivative_target`, `trajectories_to_tabular` and
  `compute_dataset_fingerprint` (from `kd.data`) and `to_sympy` and
  `from_sympy` (from `kd.core.expr`) at the top level; the names in the
  submodules are unchanged.
- `kd-agent`: `--data` reads a file the way `kd.load` does. `--load-options`
  takes every `kd.load` keyword (`kind`, `trajectory_id` and `params`
  included), so `kd-agent run --data` reads ODE trajectories
  (`"kind": "trajectory"`) and tables with a header row (`"kind": "tabular"`
  with `lhs`). On raw trajectories each search fits one state's time
  derivative on kd's regression table (`kd.trajectories_to_tabular`, finite
  differences, two points trimmed at each end); `--target-state` names that
  state when the file has several, and `run_discovery(target_state=...)`
  searches another state of the same trajectories. `--task-kind` and
  `--target-state` are refused inside `--load-options`. Input descriptions
  are written as `kd-agent-input-v2`, which records the kind kd loaded, the
  task kind, the projection and the fingerprint of the raw trajectories. A
  description written before keeps its reference and is read as before;
  `--data` on the same file adds a second input with another reference, and
  the earlier lineage continues with `--dataset <earlier reference>`.
- `kd-agent`: `--sketch` reads a file written by `kd.save_sketch` (a
  complete `kd-sketch-v2` Sketch or unbound `kd-sketch-spec-v1` shorthand; a
  released `kd-sketch-v1` file is read too) and binds it to the run's input
  in a worker (`kd.load_sketch(payload, dataset=)`) before the endpoint is
  built, so a Sketch whose left-hand side or vocabulary does not fit the
  input is refused before the run starts. The new verb
  `set_sketch(anchored, pinned, holes, input_id=None, reason=None)` binds
  Sketch shorthand, such as the terms a researcher names in `kd-agent chat`,
  to an input and makes it the active Sketch of later searches; when the run
  was given a user Sketch, only that Sketch is accepted. A search that names
  a Sketch other than the active one is refused (`sketch_conflict`), and in
  an unpinned chat a search on an input of another root is refused naming
  `set_sketch(input_id=...)`. Every arrival of one Sketch (the file,
  `make_sketch`, `set_sketch`) is appended to its `registrations` with its
  origin, evidence, input, reason and time, each envelope echoes it, and the
  report lists the active Sketch with every registration and names the
  Sketch each answer was found under.
- `kd-agent`: algorithms registered with `kd.register_algorithm` get
  generated cards (capabilities, parameters, narrowing and, when a mode uses
  the `autograd` provider, the surrogate section), and `run_discovery`'s
  description names every registered algorithm and, for each mode
  selectable from data, every required parameter with its type and
  description. The Availability notes list a required parameter that has
  no example value by name, type and description and without a value, so
  the model chooses it. An algorithm registered only in the
  controller's process (`kd.extension_source` is `process`) is refused by
  `run_discovery` and `train_surrogate` before a worker starts
  (`instrument_unavailable`, naming the `kd.extensions` hook), because
  workers import kd afresh. The term arguments of the controller's tools
  take KD's term notation (`u*u_x`), which kd's public functions read.
- `kd-agent`: the new verb `verify_system(members, input_id=None,
  reason=None)` integrates one sealed search law per state of raw ODE
  trajectories together with `kd.forward_error_trajectories` (Radau,
  `rtol=1e-6`, `atol=1e-9`, 10 s of integration in a worker of at most 30 s,
  never more than the run has left) and returns the system's id, the
  verification id, the members, their laws and kd's total, per-state and
  per-trajectory relative errors. Each attempt is a `system_verify` event in
  `tree.jsonl`; a timeout or a failed worker does not replace a system's
  earlier result. Once a system of the run's trajectories has a result, the
  systems decide the default answer (policy
  `kd-agent-ode-system-selection-v1`: among the systems whose error is at
  most twice the smallest, the fewest terms over all states, then the
  smaller error, then the earlier system) and the winning system's law for
  the target is published; pruning a member, or a search a member was
  resumed from, takes its systems out of the running. With no system on
  raw trajectories the single-target rule answers, and when it has a
  winner its decision carries `reason: "system_not_assembled"`.
  `select_node` takes `system_id`
  (exactly one of `run_id` and `system_id`, recorded as `system_select`),
  `submit_answer` takes `system_id` (the run and law must be that system's
  member for the target), and the close-out (`answer_system`), `tree
  status` (`systems`, `system_decision`) and the report (an "ODE systems"
  section) carry the system. A tree with system events does not replay in
  an older controller.

### Changed

- **Breaking**: `kd.SourceReport` has a new required field `layouts`, the
  names of the registered layouts that recognize the file, also present in
  `to_dict()` under the key `"layouts"`. An empty list means `kd.load` needs
  `coords=` / `fields=`; a recognized file can still need a load option
  (`select=` for a PDEBench file with several samples, `field_axes=` for a
  plain grid whose coordinates share a length). A caller that builds a
  `SourceReport` itself passes `layouts`.
- **Breaking**: `kd.Sketch` is a constructor function, no longer the Sketch
  class, so `isinstance(x, kd.Sketch)` raises `TypeError` and `kd.Sketch` is
  not a valid annotation. Import the class from `kd.core.equation` for
  annotations and runtime type checks (`from kd.core.equation import
  Sketch`); `kd.SketchSpec` is the type of an unresolved shorthand.
  `kd.Sketch(...)` with the six typed fields returns that class as before.
- The functions that take a list of terms read the product notation of
  `kd.Sketch` as well as funcall IR: `kd.validate_terms`,
  `kd.evaluate_terms`, `kd.refit_terms` (finite-difference, weak and
  autograd frames), `kd.ablate_terms`, `kd.score_term_sets`,
  `kd.block_dispersion`, `kd.refit_constants`, `kd.predict_terms`,
  `kd.refine_pinn`, `kd.vote_terms`, `kd.sketch_from_anchors` and
  `kd.evaluate.compare_results`, and the `fixed=` keys of `kd.refit_terms`,
  `kd.evaluate_terms` and `kd.evaluate.compare_results`. A term such as
  `u*u_x` gives the numbers of `mul(u,u_x)`. Products keep the caller's
  operand order and grouping, `u**k` is the raw repeated product (`u**2` is
  `mul(u,u)`, not the protected `n2(u)`), and a numeric literal keeps its
  value and sign (`-0.5*u`) where the function takes literals
  (`kd.vote_terms` and `kd.sketch_from_anchors` refuse one). Results name
  each term as the caller wrote it,
  in the caller's order, and keep duplicate spellings; `kd.vote_terms`
  counts spellings of one law term together and shows the first,
  `kd.sketch_from_anchors` stores each anchor as funcall IR and refuses two
  spellings of one term, and
  `kd.refit_constants` returns a term whose literals it refitted as funcall
  IR with the fitted values. `{"u*u_x": -1}` and `{"mul(u,u_x)": -1}` fix
  the same column; giving both raises `ValueError` even with equal values.
  Before, a term outside funcall IR was refused. A sum, `/`,
  `^` or a unary minus on a term is still rejected, with the reason
  (`kd.predict_terms` evaluates a signed bare term such as `-u`).
  `kd.forward_error`, `kd.forward_error_trajectories` and
  `kd.core.integrator.assemble_rhs` keep their own expression path, where
  `u**k` is the executor's power: it equals the repeated product bit for bit
  for `k` = 2 and 3, and not for `k` = 4 and 5 (measured on 10^6 values in
  [-2, 2], float32 and float64).
- `kd.core.equation.TermLibrarySpec.from_terms`, `kd.PySINDyConfig(terms=...)`
  and `kd.PySRConfig(terms=...)` accept the product notation and store each
  such entry as its funcall IR (`u*u_x` as `mul(u,u_x)`, `u**2` as
  `mul(u,u)`); entries already in funcall IR are kept as written, and
  `TermLibrarySpec` canonicalizes them as before. Before, PySINDy and PySR
  refused such a term when the plugin built its library.
- **Breaking**: a Sketch with two anchored terms whose jet expansions share
  a monomial raises `ValueError` when it is built (by the typed `kd.Sketch`,
  `kd.sketch_from_anchors`, `kd.sketch_from_dict` or `kd.load_sketch` of a
  complete Sketch; a shorthand `kd.SketchSpec` raises it when it is resolved
  against a dataset), naming both terms and the shared monomials. This
  covers two spellings of
  one column (`u_xx` and `diff2_x(u)`, `mul(u,u_x)` and `diff_x(mul(u,u))`)
  and overlapping expansions (`u_xx` and `diff_x(add(mul(u,u),u_x))`). Each
  anchored term has its own fitted coefficient, which two anchors on one
  column cannot have. Before, the Sketch was built, and PySR and DISCOVER
  counted two spellings of one column as one required term. A caller keeps
  one anchor per column: drop the second spelling, or anchor the separate
  terms of an expanded anchor.
- DISCOVER run identities change: the searched vocabulary is part of the
  result config, so `run_spec_hash` and the catalog `config_hash` of a
  DISCOVER run differ from those recorded before this change for the same
  request. Checkpoints and resume comparison are unchanged.
- `kd-agent`: the Availability table gives the `algorithm` value
  `run_discovery` takes and the mode in separate cells, how each mode is
  selected, the `params` to pass with a valid value for every required
  parameter (`{}` for DISCOVER's `pinn`), and the parameter that sets the
  budget. Before, it printed `algorithm/mode` labels, config field names
  and kd's `"<set>"` placeholder, which agents passed as written.
  `run_discovery` refuses an `algorithm` containing `/` with
  `unknown_instrument` and one sentence naming the call that runs that
  mode; the combined spelling is not an alias.
- `kd-agent`: LLM4ED is listed unavailable in the Availability table, with
  the reason, when the run configures no LLM4ED endpoint (the default; see
  `kd-agent setup --llm4ed` under Added). Before, it was listed available
  whatever the run's configuration, and an LLM4ED call failed at kd's
  endpoint check unless the agent passed `base_url` in `params` and the
  provider key was in the run's environment.
- **Breaking** (experimental `kd-agent`): the shell is off by default.
  `kd-agent run`, `chat` and `fingerprint` take `--shell` to give the model
  deepagents' `execute`, which every run installed before, and
  `kdagent.run.run`, `prepare_run`, `kdagent.chat.chat`,
  `orchestration.build_agent`, `compose_system_prompt`, `install_skills`
  and `cards.common_card` take `shell`, default `False`. The default prompt
  is the no-shell prompt, so the default `prompt_sha256` changes. Without
  the shell the model reaches kd only through the verbs and cannot read a
  dataset's reference equation with `import kd`. A run that relied on the
  shell passes `--shell` (or `shell=True`).
- `kd-agent`: `run_discovery` refuses with `untrusted_asset`, before a
  worker starts, a search parameter that names a file for kd to load
  (EqGPT's `weights_path`, `asset_dir`, `wave_pkl_path`, `v1_asset_dir`),
  `extra_pysr_kwargs` entries other than `deterministic` and `parallelism`,
  and PySR operator definitions; a null value passes. Without the shell the
  model's file tools cannot write, edit or delete under `runs/`,
  `surrogates/` or `inputs/`. `run`, `chat` and `prepare_run` refuse a
  workspace that holds the saved configuration file or overlaps the
  installed kd package. **Breaking**: this workspace check is new; they
  (and `kd-agent run` and `chat --workspace`) raise `ValueError` before the
  run starts for a workspace containing `~/.config/kd/agent.json` (the home
  directory, even before setup writes the file) or the installed kd package
  (for example a project directory holding the `.venv` kd is installed in).
  Use a dedicated directory outside both.
- `kd-agent`: `train_surrogate`'s description lists, per instrument, the
  `params` names it accepts (sga: the four `autograd_train_*` and
  `training_dtype`; dlga: the seven `surrogate_*`, `target_lhs_order` and
  `training_dtype`), from the same rule as its refusal of an unknown name.
- DLGA accepts sketches with pinned terms and fits anchored terms. Its
  `default` mode declares `fixed_terms` as `lowered` and `anchors` as
  `fit_enforced` (before: `unsupported` and `exit_checked`), so
  `kd.instrument_schema_digest()` changes. For an open sketch with pinned
  terms DLGA regresses against the target the platform has lowered by the
  pins, and the sketch exit adds each pin back at its exact value; a pinned
  sketch whose left-hand order differs from `target_lhs_order` raises
  `ValueError` in `prepare`. Every candidate is fitted with the anchored
  terms appended as extra columns; they stay in the fitted law with free
  coefficients and are not counted in the candidate's `complexity` or
  length penalty. A generated term that shares a jet monomial with a pinned
  or anchored term is refused before scoring. Anchored terms no longer take
  genome modules: they no longer count against `max_modules` or
  `max_module_length`, and their factor tokens are kept for generation only
  when a hole admits them. A sketch with pinned terms only runs the
  configured library, and its law is the pins. Refused when the plugin is
  built, before the surrogate is trained: an anchored term on a dataset
  field other than the left-hand one. Two anchored terms that share a jet
  monomial are refused when the Sketch is built (the **Breaking** entry on
  anchored terms above).
- **Breaking**: a DLGA sketch with anchored terms and no hole raises
  `ValueError` when the plugin is built. Before, the search regenerated the
  anchored terms from their factor tokens. Such a sketch needs a hole.
- EqGPT draws each epoch's candidate batch in lockstep: every sentence of the
  batch gains one token per model call, so a batch costs at most
  `max_length - len(start_words)` forward passes instead of one per token
  (`kd.search.eqgpt.sampling.SamplingConfig.sampler = "batched"`, the
  default; `"sequential"` is the previous one-sentence, one-token-at-a-time
  draw). The per-token law is unchanged: with probability `exploration_rate`
  a uniform legal token, otherwise a draw from the model's softmax over the
  legal tokens. The two samplers consume the random stream in a different
  order, so the same seed gives different candidates under each. The plugin
  always samples batched (`EqGPTConfig` has no sampler field; the sequential
  draw is reached only by building a `SamplingConfig` directly), so a run or
  a checkpoint from before this change does not reproduce its old candidate
  stream under the new version. Measured on the CPU, the sequential draw was
  85-90 % of an epoch.
- EqGPT puts its GPT, and in steady and wave multi-case modes its private
  surrogates, on the device the fit asked for: `PlatformComponents` gained
  `device` (the builder's resolved device, `None` for the CPU), filled on
  every bundle including the light ones, and `EqGPTPlugin.prepare` passes it
  to `RealGPTBackend.from_assets`, `SteadyEvaluator.from_components` and
  `MultiCaseEvaluator.from_config`. Before, the three constructors were
  called without a device and the GPT ran on one CPU thread whatever
  `Model(device=...)` said: on an A100 an epoch of
  400 candidates took 17.5 s with the GPT on the CPU and 2.3 s on the card.
  A CPU fit is unchanged byte for byte. A `cuda` fit's candidates change
  only through the float rounding of the card's logits and fine-tune
  updates: the batched sampler
  draws on the CPU, so the random stream itself does not depend on the
  device. The steady and multi-case paths are wired but were exercised on
  the CPU only. Checkpoints written by a `cuda` fit keep the GPT weights and
  the Adam state on the CPU, so they load on a machine without CUDA.
- **Breaking**: `kd.core.Evaluator.evaluate_terms` now lets a
  `torch.cuda.OutOfMemoryError` raised while executing a term, stacking the
  columns or solving propagate, and likewise a CPU allocation failure (torch's
  `RuntimeError` naming the allocator) raised at any of the three. Before, a
  CUDA OOM became an invalid candidate with the message "autograd OOM" and the
  search went on, a CPU allocation failure became an "Execution error" for the
  term, or a skipped term under `skip_invalid=True`, and one raised by the
  solver became the invalid result "Solver error: ...". A fit with a candidate
  that runs out of memory, such as a deep `diff_*` term in DISCOVER's PINN
  mode or DLGA on a small card, now raises at that candidate instead of
  marking it and every later such candidate invalid. A fit that hits this
  needs less memory per candidate (fewer points or a smaller population) or a
  larger device.
- **Breaking**: `kd.validate_terms` and `kd.evaluate_terms`, and through the
  latter `kd.refit_terms` and `kd.evaluate.compare_results`, let a
  `torch.cuda.OutOfMemoryError` or a CPU allocation failure raised while
  executing a term propagate. Before, `kd.validate_terms` reported the term as
  rejected, with the reason `execution: CUDA out of memory during term
  execution` for a CUDA OOM and `execution error for '<term>': ...` for a CPU
  allocation failure, and `kd.evaluate_terms` raised `InvalidTermsError` for
  it, or dropped the term with a warning under `skip_invalid=True`. A call
  that hits this needs fewer points or a larger device.
- The PySR worker process takes its Julia thread count from the caller's
  environment: the first of `PYTHON_JULIACALL_THREADS`, `JULIA_NUM_THREADS`
  and `OMP_NUM_THREADS` that is set, cut to its first comma-separated field,
  goes into each of `PYTHON_JULIACALL_THREADS` and `JULIA_NUM_THREADS` that
  the caller left unset. Before, PySR set `PYTHON_JULIACALL_THREADS=auto` in
  the worker and Julia ran one thread per core whatever the BLAS pin said,
  and a lone `JULIA_NUM_THREADS` had no effect because juliacall's
  `--threads` overrides it. Under `OMP_NUM_THREADS=1` a PySR fit now keeps
  one core busy: for a 100-iteration fit on a 24-core machine, CPU time over
  wall time went from 2.83-2.96 to 1.00 and wall time from 22.7-24.1 s to
  26.3-30.9 s. A caller who wants
  more Julia threads under a one-thread BLAS pin sets
  `PYTHON_JULIACALL_THREADS`. With none of the three variables set, the
  worker starts as before. A worker with a thread count set ignores PySR's
  `UserWarning` that `PYTHON_JULIACALL_THREADS` is not `auto` (one
  `PYTHONWARNINGS` entry, after any the caller set), and kd logs the count
  and the variable it came from at debug level. When the caller sets
  `PYTHON_JULIACALL_THREADS` and `JULIA_NUM_THREADS` to different counts, kd
  emits one `UserWarning` naming both: juliacall starts Julia with
  `--threads` from the first, so the second is ignored.
- **Breaking** (experimental `kd-agent`): the controller's `train_surrogate`
  task has kd write a DLGA surrogate (`Model.train_surrogate(save_to=)`, a
  `kd-surrogate-v1` file), and the registry row stores kd's identity of that
  file next to its recipe (`null` for an SGA `field_model`). A search
  injecting it hands kd the stored entry, which kd checks against the
  search's dataset and configuration; a surrogate field the search params
  omit takes the trained network's value, and an explicit conflicting one is
  refused by kd. The registry's format tag becomes `kdagent-surrogates-v2`:
  a registry written before this change is refused at its tag on
  construction; retrain those surrogates in a fresh workspace.
- **Breaking** (experimental `kd-agent`): the controller's default
  selection (the law published when the model submits and selects nothing)
  ranks searches of different algorithms on the forward error of their
  sealed laws instead of `platform_nmse` (policy
  `kd-agent-default-selection-v2`): the fewest terms within twice the
  smallest forward error, then the error, then the earlier search. After
  every search with a submittable law the controller measures that law
  once, with its own coefficients and left-hand side, on the input the
  search ran on (Radau, at most 10 s inside a worker capped at 30 s or the
  run's remaining time); the result is `diagnostics.selection_forward` and
  the tree row's `selection_forward` (`tree.jsonl` stays `kd-segtree-v1`).
  The measurement's time counts against the run's budget, and the search's
  reply reports the budget left after it. A search without a forward value
  is left out, laws with different left-hand sides are not ranked together,
  and `platform_nmse` ranks only a dataset whose laws cannot be integrated
  forward at all (a common left-hand side that is homogeneous or not first
  order, or a steady, scattered, tabular or ODE input). When the rule has
  no winner the run publishes no default answer (`answer` and
  `answer_source` null); the close-out carries the decision as
  `answer_selection`. One algorithm's searches keep their own score. A tree
  written before this change reads its rows as not measured, so a
  multi-algorithm first-order tree from before has no automatic winner.
  `python -m kdagent.tree record --forward-eval` measures a run directory
  the same way when it files the row (a row already on the tree is not
  measured again), and `status` prints `selection_decision`.
  `prompt_sha256` changes.
- `kd-agent`: a submitted answer's `provenance.nmse` is the measurement of
  the operation that produced its coefficients (the search, or the
  refinement whose receipt holds that exact law) and null for a law that
  no search or receipt produced; before, it was always the cited search's
  NMSE. Without `refinement_id`, the latest receipt of that exact law is
  chosen, then the search's own sealed law.
- `kd-agent`: when the run has a user Sketch, a search that names none runs
  under it, and a search naming another Sketch, or a resume naming no
  Sketch whose lineage ran under another Sketch or none, is refused with
  `sketch_conflict` (a resume that names the user Sketch while its lineage
  ran under another Sketch or none gets `invalid_argument`).
- `kd-agent`: the close-out gains `answer_input`, the input that produced
  the submitted coefficients (`answer_provenance.producer_ref`): a
  refinement's evaluation input, the search's input, or null for an edited
  law. The report attributes a submitted
  law's measurements by `producer_ref` instead of comparing the law with the
  sealed one, so a copy of a sealed law that differs in one digit now reads
  as unevaluated; it lists the outcome and result of every verb but
  `list_datasets`.
- `kd-agent`: the cards and the installed kd skill document the new
  verbs, the measurement frames (a search's `platform_nmse` is in the frame
  of the input it ran on), the weak form's term class and Gaussian
  assumption, voting, ablation and Sketch revision, verification and ending
  without an answer; the cards render `field_counts` and `spatial_dims`.
- The Sketch column fingerprint (`kd.core.expr.column_fingerprint`) has a
  fifth element, the pure-power degree of the term (the field 1, a raw
  same-field product or `n2`/`n3` of a field its degree, any other term 0);
  `kd.core.expr.TermFeatures` has the new required field `power_degree`. The
  raw square and the raw cube of one field shared one fingerprint before, so
  a raw-cube anchor could be satisfied by a raw-square candidate, a square
  beside a cube anchor never filled a hole, and the anchor observed the sum
  of both coefficients. Law keys and law signatures are unchanged; a stored
  Sketch outcome whose verdict relied on
  that collision can evaluate differently when it is recomputed, and stored
  outcomes are not rewritten. EqGPT's Sketch compile charges a raw power one
  term position per factor, so a Sketch with raw-power anchors can now
  exceed a `max_length` it fitted before.
- **Breaking**: `kd.llm.OpenAICompatProvider` reports an SDK transport
  error by its class, its HTTP status when the SDK gives one, and the
  number of attempts, and nothing else. An error that is not retried, such
  as an authentication failure (HTTP 401) or a bad request (HTTP 400), now
  raises `kd.llm.LLMBackendError` on the first attempt; before, the SDK's
  own exception left `complete()` unchanged. After the retries are spent
  the message reads `OpenAI-compatible LLM provider failed after N
  attempt(s): <Class> (HTTP <status>)` (before: `exhausted N attempt(s)`).
  The error is raised without a cause or context, and each retry warning
  names the same class, status and attempt number. A caller that caught
  the SDK's exception classes must catch `kd.llm.LLMBackendError`; the
  SDK's message, response body and chained cause are no longer available
  from it, because an endpoint can echo request details there and kd
  records exception text and tracebacks. Retry count, backoff and the
  empty-completion error are unchanged.
- Every kd figure draws from one token set in `kd.viz.style`: a validated
  teal / terracotta / violet series palette, viridis for True/Predicted
  fields and one teal-grey-terracotta diverging map for every signed
  difference on a symmetric scale (it replaces `RdBu_r`), warning text in a
  dark red ink instead of `red`, and figure sizes chosen for the report
  width. `DEFAULT_STYLE` changes accordingly (`savefig.dpi` 300 to 200).
- Shared scales read off one colorbar (field comparison, PDE residual field,
  2D time slices); coefficient bars are horizontal; convergence, comparison,
  search-score and plugin series curves have no per-point markers, except a
  dot on a sample with no finite neighbour, which a line cannot show; PySR's
  two Pareto curves keep their point markers.
- The HTML report leads with the discovered law (headline metrics, ground
  truth and the published-vs-true term table), shows equations as plain
  text that MathJax replaces once loaded (an offline page no longer shows
  raw TeX), puts figures on cards, moves the run configuration into a
  collapsed section and follows the light or dark colour scheme. The
  equation card figure is no longer part of `VizEngine.render_all`
  (`plot_equation_card` stays for direct calls).
- **Breaking**: `WallClockBudgetCallback` refuses a budget that is not
  finite (`math.inf`, `nan`) with the `ValueError` it already raised for a
  budget that is not positive. A caller that passed `math.inf` for no limit
  leaves the callback out instead.
- **Breaking**: an anchored term is one independent fitted column, and the
  Sketch payload tag is `kd-sketch-v2`. A term spells an anchor only when it
  has the anchor's column fingerprint and the same jet expansion (`u_xx`,
  `diff2_x(u)` and `diff_x(u_x)` spell one anchor; `u*u*u_x` does not spell
  `u*u_x`). Before scoring, every algorithm refuses a term that carries an
  anchor's monomial without spelling it (`diff_x(u*u + u_x)` is
  `2*u*u_x + u_xx` beside an anchored `u_xx`), and each fit keeps one
  spelling per anchor. The sketch exit reports such a term unassigned and
  fails an anchor that two or more present terms spell, so the certified
  anchor coefficient is the net coefficient of its column. A `kd-sketch-v1`
  payload still decodes as the constraint for a new run (it is read under
  the new rule and written back as `kd-sketch-v2`); a checkpoint stored
  under v1 is refused on resume, and the message says to start a fresh fit.
  An anchored term whose head has no jet expansion (a user-registered
  operator) is refused when the Sketch is built. Callers holding v1 payload
  files need no change for new runs unless a file anchors two terms whose
  jet expansions share a monomial (one column twice included) or a term
  without a jet expansion, which now raises as described above; resumed v1
  checkpoints must be rerun.
- **Breaking**: `kd.load(..., coords={"t": ...}, fields=...)` with only a
  time coordinate returns a `TrajectorySet` instead of a one-axis grid; pass
  `kind="pde"` for the grid. `kd.load`'s `lhs` defaults to `None`: a grid
  with one field, mapped (array mapping, CSV columns, XLSX) or found by a
  registered layout, defaults to that field's `<field>_t`, other grids to
  `u_t` as before. Without `kind=`, `kd.load`'s static return type is
  `PDEDataset | TrajectorySet | TabularDataset` (it was `PDEDataset`), for
  catalog ids as well; a type-checked caller that needs a `PDEDataset`
  passes `kind="pde"`, which catalog ids accept.
- **Breaking**: `kd.load(<catalog id>, lhs=...)` accepts only `None` or the
  left side the catalog entry declares, and the refusal names the declared
  one. It used to accept `lhs="u_t"` for every entry and ignore it, so
  `kd.load("klein-gordon", lhs="u_t")` returned the `u_tt` dataset and
  `kd.load("eqgpt-poisson-disk", lhs="u_t")` the steady one, while
  `kd.load("klein-gordon", lhs="u_tt")` was refused. Now `klein-gordon` and
  `wave` take `lhs="u_tt"`, the steady entries take no `lhs`, and the other
  entries take `lhs="u_t"` as before.
- **Breaking**: `kd.data.resample_to_grid` declares every axis of the sampled
  grid non-periodic, as `kd.surrogate_field` does: the network's two ends of
  a periodic source axis were never constrained to agree, and a periodic
  flag made finite differences wrap across them. The source flags are kept
  in `source.hints["source_periodic"]`. Derivatives taken on a grid
  resampled from a periodic source now use one-sided stencils at the ends,
  and the resampled dataset's fingerprint changes accordingly.
- **Breaking**: `kd.load(path, coords=..., fields=...)` without `field_axes`
  reads each field's storage order from its dimension lengths and transposes
  the field into mapping order; the mapping order sets only the dataset's
  axis order. A non-square field stored in another order now loads (it used
  to fail with a shape mismatch). When two mapped coordinates have the same
  length the lengths cannot give the order, and `kd.load` raises
  `ValueError`: for a two-coordinate field the message gives both calls,
  `field_axes=["x", "t"]` and `field_axes=["t", "x"]`, each with the meaning
  of `u[i, j]`, and for more coordinates one call that names the tied
  coordinates. Such a field used to be read in mapping order with nothing to
  check it, so a square `u(x, t)` stored as `(t, x)` loaded transposed
  without a warning; a square field stored in mapping order now also needs
  `field_axes`. A field whose dimension lengths are not the coordinate
  lengths in any order raises with its shape and the call with `field_axes`
  (and `select` for an extra axis). Calls with `field_axes` load as before.
  The plain-grid layout reads the order with the same rule.
- **Breaking**: `kd.load_tabular` sets the table's `source` to `""` and
  records the path, sha256 and route in `file_source`; it used to write the
  path into `source`.
- **Breaking**: `TrajectorySet` refuses a state, parameter or time name that
  kd reads as a reserved token (`one`, an operator such as `sin`,
  `diff*_<axis>`), with the rename the other datasets suggest. A set built
  with such a name at 0.8.2 must rename it (for example through `fields=`
  when loading).
- `kd.viz.plots.plot_field_comparison` names its panels Observed, Discovered
  law, integrated and Residual (discovered - observed), and writes the
  "clipped, actual [lo, hi]" note as one muted line under the clipped panel
  instead of a second title line. The 2D layout names each row once on a
  heading line and titles every panel with its time.
- `kd.core.expr.sympy_bridge.to_sympy` reads `^` as power, as
  `kd.law_from_equation` does, so ground truths such as `u - u^3` render as
  math in reports.
- Reports typeset a tabular target named `d<state>_dt` (the
  `kd.data.derivative_target` spelling) as the dotted state, `\dot{y}` for
  `dy_dt`, when `<state>` appears in the fitted law's terms (unselected ones
  included); `dose_dt` with terms in `dose` and `k` keeps its name.
  `kd.format_pde` takes the table's
  names as `variables`; without them the name's pattern decides. The
  plain-text equation keeps the target name.
- `kd-agent`: `run_discovery`'s wall-clock cap is kd's fit deadline
  (`run_episode(max_seconds=)`, what is left of the cap when the fit
  starts), so it also cuts short a derivative network trained before the
  search (dlga, sga under autograd). `diagnostics.stopped_by_time_cap`,
  `diagnostics.stopped_by_evaluation_cap` and the tree row's cost verdicts
  are read from the sealed record's `cost.stop_reason`; a worker the parent
  terminated at the cap also counts as stopped by the time cap. DISCOVER's
  pretraining is not cut by the deadline; such a search is still ended by
  the worker's termination at the cap.
- `examples/11_evaluate_terms.py` writes its convection term in product
  notation (`u*u_x`) and shows an infix sum (`u + u_x`) as the syntax
  rejection; it used to present `u * u_x` as invalid. Examples 02 and 07
  name `kd.inspect_file` / `kd.load` for files (with `field_axes=` when two
  coordinates share a length) and `kd.Model` keywords for DISCOVER
  settings. Example 12 and the public README's tabular block run DISCOVER
  at 200 generations x batch 1000 (was 100 x 500): at seed 0 the smaller
  budget no longer puts the published chromatography law on the Pareto
  front, the larger one does.
- The public README has an "ODE trajectories" section (`kd.load(...,
  kind="trajectory")`, `kd.trajectories_to_tabular`, one law per state) and
  the three-line Sketch shorthand. It names SGA-PDE, DISCOVER, LLM4ED, PySR
  and PySINDy as accepting feature tables, and states that every algorithm
  accepts a sketch for PDE search on gridded data except DISCOVER's PINN
  mode and EqGPT's multi-case wave mode.
- The getting-started, your-own-data and KdV notebooks are rerun against
  this release. The KdV fit selects `diff_x(n2(u))` and `diff2_x(u_x)`
  (coefficients -0.500004 and -0.002500) and its prose follows that run.
- **Breaking** (experimental `kd-agent`): a search segment recorded under
  `kd-sketch-v1` is not resumed (`sketch_format_not_resumable`, naming the
  recorded format); start a new search instead. It still reads and reports
  as recorded. A `kd-sketch-v1` payload given to `run_discovery(sketch=)`
  for a new search is checked, run and recorded in kd's `kd-sketch-v2`
  encoding.
- `kd-agent`: `select_node(reason, run_id=None, system_id=None)` takes
  `reason` first; a Python caller that passed `run_id` positionally passes
  it by keyword. The model's tool calls name their arguments and are
  unchanged.
- `kd-agent`: the prompt fingerprint (`kd-agent fingerprint`) is computed on
  representative inputs that include an ODE input both as a
  pre-differentiated table and as raw trajectories, under the current input
  references. With the new verbs and texts, every `prompt_sha256` changes.
- `kd-agent`: an XLSX input's summary lists every column (key, shape,
  dtype, no values), read from kd's file inspection.
- `kd-agent`: a worker that fails outside kd's call reports the last 2000
  characters of its traceback, where the exception and its message are,
  instead of the first 2000.

### Fixed

- EqGPT no longer scores an allocation failure as a bad candidate. A CUDA
  out-of-memory error or a CPU allocator `RuntimeError` raised while a
  candidate's terms are executed (single case, each wave case, the final
  refit of the single-case, multi-case and steady paths) now propagates out
  of the fit instead of becoming an invalid candidate, a skipped case or an
  `evaluation_error` final result. Ordinary execution errors keep those
  verdicts. In steady mode a winner whose pivot column cannot be assembled
  now raises instead of reporting a zero target; a winner with no terms
  still has a zero target.
  The same holds for the steady reproduction-plot data `Model.report()`
  builds: an allocation failure there propagates out of `report()` instead
  of leaving that panel without data, and so does one in any plugin
  plot the report engine renders.
- `kd.llm.OpenAICompatProvider` logs a warning when the endpoint reports a
  reply cut at the token limit (`finish_reason == "length"`); the text is
  still returned as sent. Such a reply used to look like a complete proposal
  to LLM4ED.
- `kd.llm.OpenAICompatProvider` returns `usage=None` for a call that retried
  after discarding an empty (`None`-content) reply. It returned the
  successful reply's counts, which leave out the discarded one's, and
  LLM4ED sealed them as the call's whole cost; LLM4ED now withholds its
  token totals for such a run.
- `kd.Model(small_term_rtol=r)` with `r > 0` on a sketch fit warns
  (`UserWarning`) that the drop is not applied. It was ignored silently.
- The report's Ground truth term table expands both laws into monomials with
  the expansion `kd.judge_law` uses, so every spelling of one term is one row:
  `diff_x(u_x)` and `u_xx` (Allen-Cahn), `diff2_x(u_x)` and `u_xxx`, and
  `diff_x(n2(u))` at -0.5 and `u*u_x` at -1 (KdV) were two rows with a zero
  each. A line under the table says so when a published term was expanded.
- SGA accepts a compound derivative terminal anchor such as `u_xx` or
  `u_xxx`. It compiles to SGA's derivative tree, which renders as
  `diff2_x(u)` for `u_xx` and `diff_x(diff2_x(u))` for `u_xxx`; the compiler
  compared law keys and refused the anchor as
  unreachable. Reachability is now judged by the identity the sketch exit
  uses to match anchors: the column fingerprint and the jet expansion.
- `kd.core.integrate_pde` re-raises `SearchInterrupted`,
  `torch.cuda.OutOfMemoryError` and a CPU allocation failure raised inside an
  RHS evaluation. It reported them as `solve_ivp failed`, so
  `kd.forward_error` returned them as solver failures.
- `kd-agent`: instrument cards that the multi-axis changes made false now
  state what kd does: EqGPT's steady preparation runs on the run's device
  (not on the CPU whatever the device); DISCOVER's default grid vocabulary
  takes every field of the dataset and its `x` and `y` axes; PySINDy's and
  PySR's default terms follow the spatial axes `x` and `y`, and supplied
  terms are used as given.
- `SVDNullSpaceSolver.solve` re-raises `torch.cuda.OutOfMemoryError` and CPU
  allocation failures from `torch.linalg.svd`. Before, both became an
  invalid solve ("SVD failed: ..."), so an allocation failure in a DLGA
  search, whose default solver this is, was counted as an invalid
  candidate; it now propagates out of `kd.evaluate_terms` and out of a
  DLGA search.
  Other SVD failures are still an invalid solve.
- DLGA's final result re-raises `torch.cuda.OutOfMemoryError` and CPU
  allocation failures. Before, `DLGAPlugin.build_final_result` returned an
  invalid final result naming the error.
- Restoring a checkpoint maps every stored tensor to the CPU
  (`ExperimentRunner` loads it with `map_location="cpu"`), and each plugin
  places what it restores. Before, a checkpoint that held CUDA tensors, such
  as a DLGA surrogate written by a `cuda` fit, failed to load on a machine
  without CUDA (`RuntimeError: Attempting to deserialize object on a CUDA
  device`).
- `Model(device="cuda").fit(dataset)` with a dataset on the CPU builds on a
  copy of the dataset whose axis and field tensors are on the card. Before,
  SGA with `derivatives="autograd"` trained its surrogate on the dataset's own
  device while the evaluator ran on the card, and the fit failed with "Init
  population failed ... invalid AIC". With `device=None`, or a dataset already
  on the device, the dataset is used as passed, without a copy.
  `Model.train_surrogate` places the dataset the same way for SGA, so its
  surrogate trains on the card; before, it trained on the dataset's own
  device.
- A `kd.SearchInterrupted` raised while the evaluator stacks the term columns
  now stops the search. It was reported as "Failed to build theta matrix",
  counted as one invalid candidate, and the search continued.
- `kd-agent`: a CSV input prepared without a mapping loads again by its
  reference. Its description recorded kd's load route (`csv-longform`) as
  a layout, which kd's CSV loader refused on the second load. A description
  already written that way keeps its reference, and the refusal now says to
  prepare the file again with `--data`.

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

[Unreleased]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.8.2...v0.9.0
[0.8.2]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.7.5...v0.8.0
[0.7.5]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.7.4...v0.7.5
[0.7.4]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.7.3...v0.7.4
[0.7.3]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.7.2...v0.7.3
[0.7.2]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/releases/tag/v0.7.2
