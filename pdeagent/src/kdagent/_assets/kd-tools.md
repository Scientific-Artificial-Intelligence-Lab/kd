---
name: kd-tools
description: Data tools, search records, continuation, measurement frames, refinement receipts, evidence and sketches, surrogate reuse and ending a run.
---

## Tools

`list_datasets` lists a file input as id, task kind, target, data report and
array inventory, and a catalog dataset as id, axes, lhs, fmt, tier and tags
with no report.
`run_discovery` requires `dataset_id` and `algorithm`; JSON parameters go in
`params`. `train_surrogate` trains and registers a reusable derivative model.
`prune_node` abandons a search branch; `select_node` chooses an existing search,
or an ODE system `verify_system` returned, for publication. `submit_answer`
publishes a structured answer with support
and aligned coefficients; `finish_without_answer` ends the run with an
explicit unresolved result instead. `refine_candidate` refits a search's law
with its terms held fixed. `smooth_input` writes a smoothed copy of the
pinned grid input (low-passed, or a network's field) as a new input whose id
every data tool accepts. `ablate_candidate`, `score_term_sets` and
`verify_candidate` measure a law in a declared frame; `verify_system`
integrates one search's law per state of raw ODE trajectories together and
compares the result with the observations; `vote_terms` counts the
terms that recur across searches, and `make_sketch` registers a Sketch built
from that evidence for `run_discovery(sketch_id=...)`. `set_sketch` binds the
terms the researcher states as the Sketch every later search runs under. Use
the pinned input's id, or an id derived from it, for every data tool.

Every tool that takes terms (a law's `support`, `terms`, `term_sets`,
anchors and Sketch terms) reads KD's term notation: a name (`u`, `u_xx`), a
product (`u*u_x`), a positive integer power (`u**2`) or KD's IR
(`mul(u,u_x)`, `n2(u)`), mixed freely; a sum, infix division or `^` is
refused. A result names each term as you wrote it, in your order. kd's
law-term identity decides when two spellings are one term: in a vote, for a
Sketch anchor, and for whether a refit's terms came from the search.

Every tool but `list_datasets` takes a `reason`, why you make the call; it is
required by `prune_node`, `select_node` and `finish_without_answer` and
optional elsewhere, and an empty one is refused. A search stores it on its
segment row in `tree.jsonl`. Every other call that ran adds one `action` line
there: the tool, the id it produced, the search it cites, your reason and
whether it ran to success or failed. A refused call started nothing and adds
no line. The report lists both.

<!-- shell -->
The Python interpreter in `execute` can `import kd`. With `d` read from
`inputs/<id>.json`, `kd.load(d["path"], **d["load_options"])` loads a grid
file input as the searches see it (their records name it by the id: pass
`name=<id>` as well for the same fingerprint); the three functions below take
that dataset. For an input with a projection `p = d["projection"]`, the same
call loads the raw trajectories, and `kd.trajectories_to_tabular(raw,
p["target_state"], trim=p["trim"])`, renamed to the id with
`dataclasses.replace`, is the table its searches fit.
`kd.validate_terms(dataset, terms)` executes each term
without fitting and lists the rejected ones with a reason. On grid data,
`kd.forward_error(dataset, terms, coefficients)` integrates a first-order law
from the first time slice and returns `value`, the relative L2 error against
the data, or a `reason` (`diverged`, `solve-failed`, `refused`, `timeout`); it
has no horizon argument. `forward_error(..., max_seconds=10)` stops at a
cooperative deadline and returns `reason="timeout"`; a single long solver
step is not interrupted, so keep `timeout` as the hard limit.
`verify_candidate` runs the same three checks in a worker under its own hard
limit.
`kd.block_dispersion(dataset, terms)` refits the terms on four blocks along
the first spatial axis and returns one number: the mean over terms of each
coefficient's coefficient of variation across the blocks. In 58 of 105
stored benchmark cases (seven problems, relative noise 0 to 5e-2, three
seeds; each case holds the final answers of the fixed pipelines run on it)
the pipelines' answers disagreed, all 58 at noise 0 to 1e-2; there, with
terms of |coefficient| at most 1e-4 of the largest dropped first, the fewest
terms within twice the smallest forward error picked the right answer 49
times, the smallest block dispersion 38 times and the same fewest-terms rule
on `platform_nmse` 32 times. Forward error was capped at 10 s per candidate
there and was undefined for 173 of 698 candidates.
<!-- /shell -->

When omitted, `generations` is {DEFAULT_GENERATIONS} where consumed and a
fresh run uses seed {DEFAULT_SEED}. Supply `seed` separately from `params`.
A resumed run inherits its seed; `reseed` derives a new random stream from
restored search state. A new segment's generation budget is additional work.

## Results and records

Read `failures` and status first; a `partial` result is not a successful
search. `primary.law` contains support, aligned coefficients and lhs;
`primary.equation` is the equation display. Read structured values instead of
extracting coefficients from display text. A null law is not submittable.
`diagnostics.score_kind`, `score_direction`, `nmse`, `mse` and
`segment_report` describe fit and segment facts. Native scores and derivative
domains do not establish comparisons across algorithms.
`platform_nmse` and `platform_coefficients` are separate neutral refit
measurements, with `unavailable_reason` when that evaluation is unsupported.
`term_count` is the number of terms in the law.
`primary.now_selected` names the `run_id` this run publishes if you submit
nothing: your `select_node` choice, otherwise the default selection over the
tree, or null when that has no answer.
After each search KD measures the sealed law once, in a separate worker:
the forward error of the law with its own coefficients and left-hand side
(not `platform_coefficients`), integrated with Radau from the first time
slice of the input the search ran on, at most 10 s of integration inside a
worker capped at 30 s or the run's remaining time, whichever is smaller.
That time comes out of your budget. `diagnostics.selection_forward` holds the
outcome: `status` `measured` with a `value`, or with kd's `reason` for having
none (a timeout, a divergence, a failed solve, a refusal); `not_measured`
with the reason (`budget_exhausted`, `worker_failed`); or `inapplicable` when
kd cannot integrate that law on that input at all (a left-hand side that is
not first order, a steady, scattered, tabular or ODE input). The default
selection compares one algorithm's searches on that algorithm's own score.
Across algorithms it takes the law with the fewest terms among those whose
forward error is at most twice the smallest, then the smaller error, then
the earlier search; a search without a forward value is left out and not
ranked against the others, and when none has a value there is no default
answer. Only when every search is `inapplicable` are they ranked the same
way on `platform_nmse`. Laws with different left-hand sides are never
ranked against each other; choose one with `select_node` or `submit_answer`.
On raw ODE trajectories (an input with a `projection`) every search's
`selection_forward` is `inapplicable`, so until a system `verify_system`
measured has a result, the searches of this run's target are ranked by the
rules above: one algorithm's on its own score, several algorithms' on
`platform_nmse`. From then on the systems decide the default answer (see
Verification), and `primary.now_selected` names the winning system's member
for this run's target.
`provenance.headline_coefficient_source` says whether coefficients are native
or refitted; `provenance.run_id` names the search for selection and resume.
`provenance.run_dir` points to `record.json` (sealed evidence), `recorder.json`
(iteration series) and `phases.jsonl` (phase boundaries). The fitted target is
in `evidence.catalog_fit.lhs_spec`.

## Continuation

Use `resume_from_run_id` for the latest checkpoint; add `resume_iteration`
to roll back and fork. `resume_safe` parameters can change on resume,
`init_only` parameters cannot affect restored state, and `identity_breaking`
parameters require a new lineage. An archive classified `conclusion` cannot
resume; rerun it with other parameters. `decision_parent` records the parent
of a fresh switch or narrowing decision. A segment report contains
observations, not a decision about which branch deserves more budget.
A supplied catalog prior is an annotated historical row per instrument;
it is not a judgment of the current input or result.

Measured on stored noise-free benchmark records, counted as seeds where one
arm found the equation and the other did not:
- SGA restarting from scratch every five generations: see `restart_interval`
  below. SGA resuming runs from their checkpoints with continue, retune
  or fork decisions, against one continuous run: 3 against 2.
- DISCOVER (batch 128, epsilon 0.05, not the defaults) in eight segments
  against one continuous run: cold-started segments 6 against 22, segments
  that carry the network and optimizer state 11 against 13. Carried against
  cold-started segments: 23 against 9.
- More budget: SGA (population 10) from 20 to 100 generations raised the
  success count on each of 3 problems, from 6/10 to 7/10, 1/10 to 3/10 and
  9/10 to 10/10; on two of them the count stopped changing after 40
  generations. DISCOVER from 8000 to 96000 evaluations left four problems at
  0/3, 0/3, 1/3 and 1/3, from 0/3 each.

## Refining a candidate

`refine_candidate(run_id=...)` refits the coefficients of that search's law
with its terms held fixed and returns a law that `submit_answer` accepts with
the same `run_id`; pass `law` (for example the law with a term removed) or
`terms` to refit a variant instead. On grid data it is one least-squares fit
on the whole grid with finite-difference derivatives; on a table it refits
the numeric constants inside the terms. It is not a search and counts against
no evaluation cap. Without smoothing, `diagnostics.nmse` is in the frame of
`platform_nmse`, so a refined law compares directly with the searches.
`diagnostics` also reports `r2`, `condition_number`, `terms_dropped` and
`original_coefficients` (on a table, `constants` and `initial`).

A refit is measured in a frame you declare. `evaluation_input_id` names the
input it runs on: omitted, the input the search ran on; otherwise that
input's root (the pinned input) or an input derived from that root this run,
never another root. `method` is `"finite_diff"` (default), `"autograd"`
(derivatives of the model `surrogate_id` names, which has to be trained on
that input) or `"weak"` (integrals against test-function windows, set by
`weak`: `half_widths`, `spectrum_tau`, `noise_correction`, `noise_sigma`;
see Weak form below). `trim` drops boundary rows after the derivatives are
taken (finite differences only: an integer, a per-axis mapping or
`"by_order"`). `diagnostics.refit_frame` is kd's resolved frame (input
identity, `derivative_method`, `field_source`, `trim`, sample count),
`diagnostics.weak` the weak windows, rank and column count, and
`diagnostics.frame` a tag that names a method or trim other than the
default. An `nmse` compares only with fits in the same frame.

Every successful refinement returns `primary.refinement_id`, the id of a
receipt this run keeps: the exact law, the method and parameters, the
search's and the measurement's inputs, the frame and the metrics. Submit a
refined law with `submit_answer(law=<its primary.law>, run_id=<the search>,
refinement_id=<its id>)`; the answer then carries that refinement's
measurement and input (`provenance.producer_ref`). A law that differs from
every search law and refined law of this run, even in the last digit of one
coefficient or in the spelling of one term (`mul(u,u_x)` for a returned
`u*u_x`), is recorded as edited: it is accepted, carries no measurement, and
the input its coefficients came from is unknown.

Measured on stored benchmark records:
- On noise-free data the refit corrected 20 of 20 DLGA answers that had a
  second time derivative and the right terms, all on one problem; each had
  one coefficient 12.7% off before the refit, and the deviation comes from
  the trained network. On noise-free data the refit changed no correct
  answer into a wrong one.
- An answer with extra small terms: the measured gain (under
  `small_term_rtol` below) came from `small_term_rtol` in `run_discovery`
  params, which drops terms without a refit. `refine_candidate`'s
  `small_term_rtol` drops the same terms and then refits the rest on finite
  differences, so on noisy data the refit numbers below apply to it.
- On noisy data a finite-difference refit is not better than the algorithm's
  own coefficients: of 108 noisy answers with the right terms, the
  algorithms' coefficients passed on 64 and the refit on 57.
- Do not replace DLGA's coefficients with a finite-difference refit on noisy
  data. DLGA takes derivatives from its trained network; at 1% noise its
  Burgers and KdV coefficients were 0.1% to 3.5% off, and on Burgers a refit
  moved them far off (refit NMSE 0.91).
- A refit on a smoothed field (`savgol_window`) is another derivative source,
  to be judged on each problem: on the same noisy answers it turned 8 failures
  into passes and 16 passes into failures.

Three parameters with measured values:
- `small_term_rtol` drops terms whose coefficient is below that fraction of
  the largest one (in `run_discovery` params on the algorithm's own result,
  and in `refine_candidate` before the refit). Measured as dropping the terms
  from the algorithm's result without a refit: at 1e-4 it turned 73 of 1641
  noise-free answers right and none wrong, and 27 of 1678 later answers
  (noisy ones included) right and none wrong; at 1e-3 it turned 29 answers
  on one problem wrong.
- `restart_interval` (SGA) redraws the population every k generations. With
  k = 5 and population 10, SGA found the right equation on 23 seeds where the
  run without restarts did not, and the reverse happened on 4 (two problems,
  same evaluation count); on five more problems the count was 38 against 19.
- `savgol_window` smooths every field before the search (or the refit). With
  window 11, SGA found the right equation on 40 of 84 noisy runs against 29
  without; on noise-free Burgers it found it on 0 of 3 against 3 of 3,
  because the window spans the shock.

## Noisy data

`smooth_input(dataset_id, lowpass_modes, time_window=0)` writes a copy of the
pinned grid input in which every field keeps only its lowest `lowpass_modes`
spectral modes along each axis other than the target's (real Fourier modes
on a periodic axis, cosine modes otherwise), optionally Savitzky-Golay
filtered along the target's axis with an odd `time_window`. The copy is a
new input, `<id>+lowpass<modes>-t<window>`, listed by `list_datasets` with
its `derived_from`; `run_discovery` and `train_surrogate` take its id where
they take the pinned one, a search on it seals the same left-hand side, and
its `run_id` goes to `refine_candidate` and `submit_answer` as usual. The
same call twice returns the same id (`primary.reused`). A low-passed copy
can be smoothed again: the new id appends to its id and its `derived_from`
is that copy; a PINN field (below) is not taken. A derived input is
accepted only in the run that wrote it: on a workspace an earlier run left,
the same call derives it again before it can be used. One whose descriptor
or file was changed is refused (`input_unverified`), and so is the call that
named it when repeated, which then derives nothing; a search on it can no
longer be submitted or refined, so derive a copy under other parameters.
`refine_candidate`
also takes `lowpass_modes` and `time_window` directly, so a candidate from a
search on the raw input can be refitted on a low-passed field without a
second search; `diagnostics.frame` is then
`kd-evaluate-terms-full-grid-lowpass<modes>-t<window>-v1`, and
`savgol_window` is not combined with it.

`lowpass_modes="auto"` with `terms` lets kd choose the counts: from each
axis's knee it refits `terms` at the knee and the following counts (at most
`max_counts`) and chooses the first count whose refit the next count
reproduces within `tolerance`, the largest relative coefficient change
(`diagnostics.selection` holds the whole trace); a failed selection derives
nothing, and a term with a coefficient near zero can keep it from settling.

`smooth_input(dataset_id, method="network")` instead fits a network to the
data (kd's `FieldRecipe` in `network_recipe`, by default five sine layers of
50 trained for 50,000 epochs, and `seed`) and writes its field as
`<id>+field-<surrogate_id>`; a repeated request trains nothing
(`primary.reused`), and `surrogate_id` with the id of a model trained on
that input (a network or a `train_surrogate` surrogate) samples it without
training. `diagnostics.quality` is the training report; it does not say the
derivatives are accurate.

A network field and a search surrogate are different things. The field is an
ordinary derived input that every search and measurement takes. A
`train_surrogate` model is injected by one instrument into its own search
(`run_discovery(surrogate=<id>)`); sampling it with `smooth_input` neither
retrains it nor makes it a surrogate of another instrument, and a
`smooth_input` network is refused as a search surrogate.

The envelope's `primary.energy_kept` is, per smoothed axis, the fraction of
the target field's spectral energy the kept modes hold;
`diagnostics.spectrum.<axis>` adds the axis length `n`, `modes_at_0.99` and
`modes_at_0.999` (the smallest count whose kept fraction reaches that value)
and `modes_at_knee`, the first count past which the remaining energy is the
noise floor (within 1.5 times the remainder at n/8 modes). All of them are
computed from the whole spectrum, so the first call with any legal count
reports them.

Recipe, by data features:
1. Judge the noise from the input's data report (`noise_estimate` per axis,
   as a fraction of the field's standard deviation; the preview compares the
   field with a smoothed copy of itself). At or above about 0.3 %,
   finite-difference derivatives of second order and above are dominated by
   noise, and the finite-difference searches (pysindy, sga default, discover,
   pysr, eqgpt) return a wrong structure or an empty support on the raw
   input. The data report's `periodic` flag per axis decides which transform
   that axis gets; the route needs a periodic axis to be flagged (below).
2. Call `smooth_input` once with any legal count and read
   `diagnostics.spectrum`. Use `modes_at_knee`, not `modes_at_0.999`: the
   99.9 % count is reached while the field's own modes are still being cut.
   The usable counts are the first few at or above the knee, and the higher
   the derivative order of the law, the more counts above the knee the refit
   needs (a second-order law at the knee, a third-order one one or two above,
   a fourth-order one two or three above). Write one derived input at the
   knee and one or two at higher counts, with `time_window` 7 or 11.
3. Run the cheap finite-difference searches on the derived inputs, not on the
   raw input, and compare their supports as you would on clean data. A
   default library cannot spell a third or fourth derivative or a cubic
   field: pass `terms` where the data report's order or the fit says so.
4. Refit the agreed support with `refine_candidate` on each count (the
   `run_id` of a search on that derived input, or `lowpass_modes` and
   `time_window` on a raw-input search's `run_id`) and keep the count at
   which the coefficients stop moving between neighbouring counts (within
   about 2 %); below the knee they move with every count. Drop terms below
   about 1 % of the largest with `small_term_rtol` and read `nmse`, `r2`
   and `condition_number` on the same frame: a working copy brings the
   library's condition number down by one to four orders of magnitude.
5. Submit the support whose coefficients keep their sign and size across the
   neighbouring counts, the raw frame and two algorithms or two seeds. A
   small `nmse` after smoothing is not evidence of the right coefficients
   when two library columns are collinear (a field that stays one decaying
   mode has `u_xx` proportional to `u`; the library's condition number is in
   the thousands on the raw field and stays high on every copy): the refit
   then lands on a combination of the two, and no count changes that.

Measured on kd's generators at 1 % noise (a fraction of the field's standard
deviation, three seeds) and on one 511-point fourth-order grid from the
benchmark pilot, refitting the true terms with `kd.refit_terms` and
counting a setting as recovering when every coefficient is within 5 %:
- On the raw field every law refit with a 100 % coefficient error, and
  `savgol_window` 7, 11 or 21 left 50 to 100 %: it recovered nothing.
- A second-order law with a shock on a 256-point periodic axis: 6 to 12
  modes recovered 3 of 3 seeds with or without the time window (8 modes
  with window 7 or 11: 0.04 to 0.5 %); 4 modes smeared the shock (9.4 %);
  16 modes recovered only with the window; 24 and more left 13 to 76 %. sga
  at kd defaults on the 8-mode copy returned the right support on 3 of 3
  seeds with coefficients within 0.5 %, in 52 to 132 s; pysindy at its
  default threshold dropped a true term with a small coefficient, so keep
  its threshold at 0.01 or lower on a smoothed copy.
- A third-order law whose third-derivative term has a small coefficient: 7
  or 8 modes with window 7 or 11 recovered 2 of 3 seeds (coefficients
  about 1 to 10 % off, the third seed 11 to 19 %); the first-order term was
  within 1 % at 8 modes on every seed, and no setting recovered all three.
- A fourth-order law on the 511-point periodic axis (one dataset): 11 or 12
  modes with window 11 gave 3.7 to 4.0 %; 9 modes 24 %, 16 modes 39 %;
  window 21 was worse than 11 at every count (a chaotic field with a time
  step of 0.2). The usable window was two counts wide.
- The `modes_at_0.999` count picked 4, 4, 2 and 7 on the four grids and
  failed on each (9.4 %, 250 %, collinear, 67 %); `modes_at_knee` picked 6,
  7, 2 and 9: right for the second-order law, borderline for the
  third-order one, one count short for the fourth-order one. The true-term
  library's condition number fell from 380, 5e3 and 7e4 on the raw fields
  to 12, 28 and 10 on the working copies.
- The time window never hurt a working count, lowered the refit `nmse` 4 to
  10 times on the second-order law, and brought the third- and fourth-order
  laws into range.
- Transform choice: the real Fourier transform on an axis that is not
  periodic (16 time modes) destroyed every fit (`nmse` about 1), and the
  cosine transform on a periodic axis that was not flagged needed far more
  modes than any count tried, so no count worked. The time axis is never
  low-passed; `time_window` is its filter.

### PINN refinement

`refine_candidate(run_id, pinn_epochs=...)` replaces the least-squares refit
with a network fitted to the data and to the candidate law at once, the law's
coefficients trained with the network (`kd.refine_pinn`): first
`pinn_pretrain_epochs` epochs on the data alone (default 5000), then
`pinn_epochs` epochs on the data misfit plus `pinn_residual_weight` times the
law's mean squared residual (default 1.0; 0 fits the data alone and keeps the
coefficients). It needs a grid file input, a first-order time-derivative
left-hand side and a number for every coefficient, and takes none of
`terms`, `small_term_rtol`, `savgol_window` or `lowpass_modes`. It returns
the trained coefficients as a law for `submit_answer` on the same `run_id`
(`diagnostics.frame` `kd-refine-pinn-e<epochs>-p<pretrain>-w<weight>-v2`,
the data and residual losses before and after, `nmse` null), and writes the
network's field as a derived input, `<id>+pinn<epochs>-<digest>`
(`primary.dataset_id`), which `run_discovery` takes like a low-passed copy
and whose `run_id`s go to `refine_candidate` (the least-squares refit, now
on the network's field) and `submit_answer`. That field has no periodic
axis: the network does not wrap an axis, and a periodic stencil across its
ends refits nothing (93 % error measured before the flag was dropped). It
fits the input the search ran on, or the pinned input its chain starts from
when that is a derived input, on the run's device: 2000 joint epochs on a 256 by 101 grid took 2 to
6 minutes on one card running alone, after a 5000-epoch pretraining. On one
CPU thread a joint epoch took 0.85 s and a pretraining epoch 0.024 s on that
grid (second-order law), so the same call is about 30 minutes there and a
third-order law longer: on a CPU run budget `pinn_epochs` to the time left.

Recipe, by data features, when the noise estimate is several per cent or
more:
1. Get a candidate support first (a search on a low-passed copy, a
   surrogate-derivative search, or a search on a data-only network field:
   `pinn_epochs` with `pinn_residual_weight` 0 on any candidate).
2. Refine with `pinn_epochs` 2000 and `pinn_residual_weight` 0.01, not the
   default 1.0: at weight 1.0 the coefficients stay near where they start
   whenever the network can bend to the given law, so the result depends on
   the starting coefficients.
3. Run `sga` (default mode) on the network-field input and refit its support
   there with `refine_candidate`; compare those coefficients with the
   PINN-trained ones. Agreement within a few per cent on every term with a
   first or second derivative is the check.
4. Read a third-derivative coefficient from either route as uncertain: the
   network smooths the field's third derivative, so both routes miss it by
   tens of per cent in either direction.
5. The PINN does not remove a wrong term: a term that is not in the law
   kept about a third of its starting coefficient. Drop it with a `law`
   without it and compare the losses.

Measured on kd's generators with relative Gaussian noise of 1, 5, 10 and 25 %
of the field's standard deviation, true terms started 30 % off (one term
high and the other low, then the reverse), 2000 joint epochs after 5000
pretraining epochs, on one card:
- The finite-difference refit of the true terms on the raw field was
  100 % off at every level.
- A second-order law with a shock, weight 1.0: started high, the nonlinear
  coefficient came within 2 to 3 % on one noise draw and 7 to 8 % on
  another; started low, it stopped 18 % short at every level (10000 epochs
  brought it to 4 %); the second-derivative coefficient was within 2 % in
  every run. Weight 0.01: every coefficient within 0.6 to 2 % from both
  starts at 10 and 25 %.
- A third-order law whose third-derivative term has a small coefficient:
  the first-order nonlinear coefficient came within 6 to 8 % at weight 1.0
  and 2 to 6 % at weight 0.01; the third-derivative coefficient stayed near
  its start at weight 1.0 (27 to 36 % off when started low, 1 to 20 % when
  started high) and was 6 to 53 % off at weight 0.01.
- The network field was within 0.4 to 1.0 % of the clean field (relative L2)
  at 1 to 25 % noise when fitted to the data alone or to the right law, and
  3 % when the law it was fitted to had a coefficient 18 % off.
- On the network field fitted to the data alone, the least-squares refit of
  the true terms (finite differences) was within 2 to 4 % for the
  second-order law at 1 to 25 % noise; `sga` at kd defaults returned the
  right support at 1, 5 and 10 % (2 to 3 % off) and an extra term at 25 %;
  on the field fitted with a wrong extra term it returned the right support
  at 5, 10 and 25 %. `pysindy` at its default threshold dropped the
  small-coefficient term on every field. Neither library at kd defaults can
  spell a third derivative, so the third-order law was found as its
  first-order term alone.

## Measurement frames

A number is measured in one frame, an input and a derivative source:
- raw: the pinned input with finite differences;
- filtered: a low-passed copy (`<id>+lowpass...`) or a refit's inline
  smoothing (`lowpass_modes`, `savgol_window`);
- network: a network's field (`<id>+field-...`) with finite differences, or
  `method="autograd"` with the network's `surrogate_id` on the input it was
  trained on;
- PINN: a field and coefficients trained together with the law
  (`pinn_epochs`);
- weak: `method="weak"`, the law tested against smooth windows on any of
  these inputs, with no pointwise derivative.
A search's `platform_nmse`, and a finite-difference `refine_candidate` refit
without `evaluation_input_id` or inline smoothing, are measured on the input
that search ran on: in the raw frame only for a search on the pinned input,
in that input's frame for a search on a low-passed copy or a network field.
A smaller `nmse` in one frame than in another says nothing about which law
is right: the frames measure different things. Compare candidates in one
frame, and name the frame with every number you report.

## Weak form

`method="weak"` (`refine_candidate`, `ablate_candidate`, `score_term_sets`)
moves every derivative onto a smooth test function, so noise is not
differentiated. It takes one field on one spatial axis, a uniform grid and a
first-order time derivative, and only the terms `one`, the field, its square
and cube (`n2(u)`, `n3(u)`, or the raw products `mul(u,u)` and
`mul(mul(u,u),u)` in either bracketing, for a field `u`), pure spatial
derivatives of order one to three and the field times its first spatial
derivative (`mul` in either order); any other term is refused, never fitted
another way. A raw product keeps its own label in the result. Windows have
to fit inside the grid, so a short
axis leaves few of them; `diagnostics.weak` reports the windows, the rank
and the column count. Without `half_widths`, the widths follow each axis's
spectral corner, and a smaller `spectrum_tau` gives wider windows.
`noise_correction: "gaussian"` removes the bias noise puts into the power
terms, assuming additive, zero-mean Gaussian noise of one standard deviation
everywhere (`noise_sigma`, or kd's estimate from second differences when
omitted); on other noise the correction is not established, and `"none"`
(the default) is the uncorrected fit. Report both when they differ.

## Narrowing by evidence

A narrowed search is a sequence of hypotheses, each step citing the one
before:
1. `vote_terms(run_ids, min_count)` counts how many independent searches
   (one vote per lineage, on one input root) found each term, by kd's
   law-term identity; `primary.selected` are the terms that reached
   `min_count`. A recurring term is a hypothesis, not a finding.
2. `ablate_candidate` refits a law once with all its terms and once without
   each, in one frame; `relative_rise` is how much the fit worsens without
   the term. A term whose rise stays small across frames (weak windows,
   inputs) carries little of the fit there.
3. `make_sketch(evidence_ids, anchor_term_keys, min_additional_terms,
   max_additional_terms, max_derivative_order)` builds a Sketch that
   requires the anchors you keep, each one a term of the cited evidence, and
   leaves the stated number of other terms to the search; pass its
   `sketch_id` to `run_discovery`. A changed choice is a new Sketch made from
   the new evidence; nothing is merged. An anchor matches the evidence by
   kd's law-term identity but keeps the spelling you give it, in KD's IR
   (`u*u_x` is the anchor `mul(u,u_x)`). That identity does not rewrite a
   product of a field with itself as its `n2`/`n3` power, so a search whose
   library spells the term the other way can refuse the anchor (pysindy
   requires every anchor among its `terms`).
Coefficient cutoffs depend on units: `small_term_rtol` is relative to the
largest coefficient of the same law, and an absolute threshold (pysindy's
`threshold`) or a coefficient's size means something different on data in
other units or scales. State the cutoff with every count or rise.

## Verification

`verify_candidate(run_id, law, input_id, max_seconds)` makes three separate
measurements of a law on one input: `validation` (whether kd can evaluate
each term), `dispersion` (how much its coefficients spread when refitted on
separate grid blocks) and `forward` (the relative error of the field
integrated from the first time slice). They measure discrepancy: a small
forward error does not establish a law, and a large one can come from the
integrator or the boundary as well as from the law. A forward check that
reaches `max_seconds` returns `reason="timeout"` and no value: it was not
measured, which is not an infinite error. `diverged`, `solve-failed` and
`refused` are separate outcomes, and a measurement that could not be made
(out of memory, an unsupported input) is listed under `failures` with
status `partial`. The default selection reads only the automatic
measurement of each sealed law (`diagnostics.selection_forward`), never a
`verify_candidate` result, so a lower forward error you measured on an
edited law, another input or a longer budget changes nothing until you
submit that law or select a search.

`verify_system(members, input_id, reason)` checks a whole ODE system on raw
trajectories. Search each state's time derivative first: the input's own
target with `run_discovery`, every other state with
`run_discovery(target_state=...)` on the same input. `members` maps each
state's target (`dx_dt` for state `x`) to the `run_id` of a search of that
target that succeeded this run; every state needs exactly one. KD integrates the
sealed laws together with their sealed coefficients (nothing is refitted),
from each trajectory's first observation to its last (Radau, rtol 1e-6, atol
1e-9, 10 s of integration in a worker of at most 30 s, never more than the
run has left), and returns `primary.system_id` (the same searches are always
the same system), `verification_id`, `members`, `laws` and `forward`: `value`,
the relative L2 error over every trajectory, state and time point,
`per_target`, `per_trajectory`, the integration `intervals`, the `solver`, or
kd's `reason` with null values when it could not integrate (`refused`,
`diverged`, `solve-failed`) or ran out of time (a timeout, not measured).
A timeout or a failed worker leaves the system's current result
(`standing_verification_id`) as it was; the latest other outcome replaces
it. Once a system of the run's raw trajectories has a result, the default
answer is a system's: among the systems whose error is at most twice the
smallest, the fewest terms over all states, then the smaller error, then the
earlier system, with its member for this run's target published; when none
of them has a finite error there is no default answer. Pruning a member
takes its systems out of the running. `select_node(system_id=...)` publishes
a system's member for the target, and `submit_answer(..., system_id=...)`
submits that member's sealed law unchanged with the system beside it. The
error compares the laws with the observations they were found on; it is not
a held-out test.

## Ending without an answer

When your searches and checks do not support a law (no term survives the
evidence, no candidate certifies, refits disagree beyond the tolerance you
declared, a check you rely on was not measured), call
`finish_without_answer(reason, evidence_run_ids)` instead of submitting the
closest candidate. The run then records an unresolved result with your
reason and the searches it rests on, and no selected search is published in
its place. Like a submission, it ends the run.

## Sketches

Pass a `kd-sketch-v2` JSON payload as `sketch` (a released `kd-sketch-v1`
payload is read too and recorded as `kd-sketch-v2`; a segment recorded under
a `kd-sketch-v1` Sketch is not resumed). Its `lhs_spec` specifies
`field`, `axis`, `order`; `vocabulary` names fields and coordinates;
`pinned` fixes `term_ir` and `value`; `anchored` fits mandatory `term_ir` entries.
Each hole names an `id`, `min_count`, `max_count` and a `constraint` over
`max_deriv_order`, `operators`, `fields` and `axes`. `match_policy` specifies
coefficient tolerances, support threshold and identity/assignment schemes.
An example payload is
```json
{{
  "schema": "kd-sketch-v2",
  "lhs_spec": {{
    "field": "u",
    "axis": "t",
    "order": 1
  }},
  "vocabulary": {{
    "fields": [
      "u"
    ],
    "coordinates": [
      "t",
      "x"
    ]
  }},
  "pinned": [],
  "anchored": [],
  "holes": [
    {{
      "id": "rest",
      "min_count": 0,
      "max_count": 3,
      "constraint": {{
        "max_deriv_order": 2,
        "operators": [
          "mul"
        ],
        "fields": [
          "u"
        ],
        "axes": [
          "x"
        ]
      }}
    }}
  ],
  "match_policy": {{
    "coeff_atol": 1e-08,
    "coeff_rtol": 1e-05,
    "support_threshold": 0.0,
    "term_identity": "kd-lawsig-v2",
    "hole_assignment": "disjoint",
    "derivative_order": "total-effective"
  }}
}}
```
A payload that fixes a term with its coefficient (`pinned`) and leaves one
or two more terms to the search:
```json
{{
  "schema": "kd-sketch-v2",
  "lhs_spec": {{
    "field": "u",
    "axis": "t",
    "order": 1
  }},
  "vocabulary": {{
    "fields": [
      "u"
    ],
    "coordinates": [
      "t",
      "x"
    ]
  }},
  "pinned": [
    {{
      "term_ir": "mul(u,u_x)",
      "value": -0.5
    }}
  ],
  "anchored": [],
  "holes": [
    {{
      "id": "diffusion",
      "min_count": 1,
      "max_count": 2,
      "constraint": {{
        "max_deriv_order": 2,
        "operators": null,
        "fields": null,
        "axes": null
      }}
    }}
  ],
  "match_policy": {{
    "coeff_atol": 1e-09,
    "coeff_rtol": 1e-09,
    "support_threshold": 0.0,
    "term_identity": "kd-lawsig-v2",
    "hole_assignment": "disjoint",
    "derivative_order": "total-effective"
  }}
}}
```
The same structure with the term required but its coefficient fitted
(`anchored`):
```json
{{
  "schema": "kd-sketch-v2",
  "lhs_spec": {{
    "field": "u",
    "axis": "t",
    "order": 1
  }},
  "vocabulary": {{
    "fields": [
      "u"
    ],
    "coordinates": [
      "t",
      "x"
    ]
  }},
  "pinned": [],
  "anchored": [
    {{
      "term_ir": "mul(u,u_x)"
    }}
  ],
  "holes": [
    {{
      "id": "diffusion",
      "min_count": 1,
      "max_count": 2,
      "constraint": {{
        "max_deriv_order": 2,
        "operators": null,
        "fields": null,
        "axes": null
      }}
    }}
  ],
  "match_policy": {{
    "coeff_atol": 1e-09,
    "coeff_rtol": 1e-09,
    "support_threshold": 0.0,
    "term_identity": "kd-lawsig-v2",
    "hole_assignment": "disjoint",
    "derivative_order": "total-effective"
  }}
}}
```
The terms and the value in these examples only show the format. Pin only a
term and coefficient that agreed across your own earlier searches in this
run, across seeds or algorithms; the task text is not a source. No field
records which searches justified a pin.
A pin with a wrong coefficient is not repaired by the search: a hole
candidate whose expansion shares a monomial with a pinned term is refused
before scoring, so no hole term can offset the pin, and the law then fails
the exit residual gate. In a rerun on a noise-free 128 by 51 Burgers grid, a
pin with the wrong sign gave no certified law in 8 of 8 DISCOVER runs and 10
of 10 SGA runs.
No hole clause excludes an explicit coordinate factor along an axis the hole
differentiates: `fields` limits which data fields a term contains, and `axes`
limits derivative axes and coordinate factors together, so `axes: ["x"]`
refuses any dependence on t but still admits a factor of x. Leaving x out of
`vocabulary.coordinates` makes its derivatives unresolvable.
`small_term_rtol` is not applied on a sketch run.
Use the input's names. The per-algorithm table declares each clause's level;
tabular modes do not accept sketches. Operator constraints use IR feature
names, such as `mul` and `recip`, not search-backend spellings.

What the levels mean. `lowered`: the clause is compiled into the search arithmetic itself, as a pinned term deducted from the regression target before the fit. `generation_enforced`: the clause filters which candidates the search may generate, which decides how fast a compliant law is found rather than whether it is compliant. `fit_enforced`: the anchored column is in every fit, never thresholded, and its coefficient is not shrunk by the penalty; the exit still requires |coefficient| >= support_threshold, and at support_threshold=0 an anchored clause on a fit_enforced backend fails only on an exactly zero coefficient. `exit_checked`: the search itself is not constrained by that clause at all, and compliance is judged only at the exit.

There are two exit gates. First, every used clause is judged whatever its level; `diagnostics.sketch.satisfied` reports that clause verdict. Second, the full law must pass the exit residual verification policy on the platform footprint (by default NMSE <= 0.05). `diagnostics.sketch.certified` reports whether kd delivered a solution after both gates. A law can satisfy every clause yet fail the residual gate: `satisfied` true, `certified` false and a null `primary.law`; `diagnostics.sketch.failure` explains the refusal. kd never delivers a law that fails either gate, but the search may find no certified law. A false `certified` and a null `primary.law` are expected outcomes, and clause failure is likelier when more used clauses sit at `exit_checked`.

`run_discovery(sketch_id=...)` runs under a Sketch `make_sketch` or
`set_sketch` registered,
and `verify={{"nmse_max": x}}` sets that search's residual threshold (a
finite number, at least 0; default 0.05); a resumed segment keeps its
lineage's threshold. A larger threshold lets exploratory candidates certify;
it does not make them physically right. `diagnostics.sketch` adds the sealed
candidate, its residual NMSE, the threshold, whether it passed and the
input it was measured on. A search on a network field measures the gate on
that field, and a later refit does not inherit the certificate.

When the researcher states a structure ("every law has the advection term",
"no third derivative"), `set_sketch(anchored, pinned, holes, input_id,
reason)` binds it on the input: `anchored` terms are fitted, `pinned` terms
are a term-to-coefficient mapping, and each hole is `{{"max_terms": n}}` with
optional `min_terms`, `max_deriv_order`, `operators` and `fields`; a term is
written `u*u_x`, `u_xx`, `u**2` or in KD's IR. The bound Sketch is then active:
every search that names no Sketch, on that input or one derived from it, runs
under it; a search naming a different Sketch is refused (`sketch_conflict`),
and so is a search on another input (`sketch_input_mismatch`) or a resume of
a lineage that ran under another Sketch. Replace it with the next
`set_sketch`; one that fails leaves it as it was. Searches that ran earlier
and an earlier answer keep the Sketch they were found under. When the run was
given a user Sketch, it is the active one for the whole run and `set_sketch`
cannot replace it.

The residual gate uses platform finite differences of the data the search
sees, after `savgol_window` smoothing when that is set, and `run_discovery`
cannot change it. Least-squares fits of the true terms on finite differences
of unsmoothed noisy benchmark fields had median NMSE above 0.05 on 4 of 7
problems at relative noise 1e-4, 6 of 7 at 1e-3 and 7 of 7 at 1e-2 and 5e-2,
so on such data a sketch cannot certify the true law. After smoothing with
window 11 they were above 0.05 on 1 of 7 problems without noise and at 1e-4,
3 of 7 at 1e-3, 6 of 7 at 1e-2 and 7 of 7 at 5e-2.

An earlier comparison of sketch runs with blind search ran on code of
2026-09-12, before two derivative fixes, on that 128 by 51 Burgers grid with
relative noise added, with SGA (population 20, 100 generations, five seeds)
and coefficients judged within 40% on SGA's own derivatives. Blind search
found the equation on 3 of 5 seeds at noise 1e-4 and 4 of 5 at 2e-4, and a
constrained hole on 5 of 5 at both. Without noise both found it on 5 of 5,
and a pinned term with a constrained hole took a median 6.3 s against 17.2 s
blind. At 3e-4 and above every arm found it on 0 of 5. That comparison covers
one problem and one algorithm. On current code, DISCOVER on KdV with a pinned
term and a constrained hole, in three sketch configurations (pinned, hole
only, wrong pin) over three rounds and two budgets, certified 0 of 24 runs,
and 6 blind runs did not find the equation either; the hole candidates
carried coordinate factors the sketch could not exclude.

## Surrogates

Only algorithms declaring an injection channel accept `train_surrogate` and
`run_discovery(surrogate=<id>)`. Training accepts the declared surrogate
fields plus mode order parameters and a separate seed. Equal recipes reuse
the registered model. A resumed lineage inherits its surrogate and sketch;
changing either requires a fresh lineage. Read each algorithm's Surrogate
section for the mode selection and training behavior.

A dlga search that injects no surrogate and resumes nothing trains its own
network, and that network is registered under an id of its own,
`provenance.trained_surrogate` in the search's reply; `diagnostics.surrogate`
gives the id, `stopped_at_deadline`, `savgol_window` and the training facts,
or the reason none was registered. A repeated recipe registers another id;
no earlier id changes. The id works where a `train_surrogate` id does, and a
completed network on the input as loaded is what `train_surrogate` returns
for the same settings and seed. A network the search's cap cut short
(`stopped_at_deadline` true) is not a complete training: kd refuses it as a
search's surrogate and `train_surrogate` trains anew, while
`method="autograd"` measurements take it by id. A network trained after
`savgol_window` smoothing is that smoothed field's network: only a search
with the same window takes it, and the calls bound to the input as loaded
refuse it.

The operator fixes the device for the whole run (`kd-agent run --device`,
cpu or cuda): every search and every surrogate training runs there, and no
call takes a device. The numbers below were measured on the CPU. In one
pilot of 42 agent runs, ten DLGA searches used their whole 60-minute cap and
two separate DLGA surrogate trainings took 51 and 57 minutes; none of the 42
final answers came from DLGA. With four CPU trainings in parallel, 5000
surrogate epochs took 2638 s; kd's default is 50000.
