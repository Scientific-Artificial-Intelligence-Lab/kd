# Changelog

All notable changes to KD are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versioning policy: KD is in 0.x development. A patch release may contain a
breaking change; every breaking entry is marked **Breaking** and names what a
caller must change. The seven bundled instruments are updated in the same
release and are never the party that breaks. Longer release notes with worked
examples accompany each release on its GitHub Release page.

## [Unreleased]

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

[Unreleased]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.7.4...HEAD
[0.7.4]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.7.3...v0.7.4
[0.7.3]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/compare/v0.7.2...v0.7.3
[0.7.2]: https://github.com/Scientific-Artificial-Intelligence-Lab/kd/releases/tag/v0.7.2
