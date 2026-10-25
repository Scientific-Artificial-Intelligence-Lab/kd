Input data selects between the grid modes and tabular mode, and `pinn`
selects between the two grid modes; there is no `mode` parameter. The grid
vocabulary is built from the input: every field (the target is the
left-hand side, not a token), the spatial axes and `t` as coordinates, and
`diff_<axis>` and `diff2_<axis>` for each spatial axis. Spatial axes other
than `x` and `y` are refused. The vocabulary a run actually searched is
sealed in its record: `run_spec.config.searched_vocabulary` in `record.json`
holds the operators, the state and coordinate variables, and the engine's
token names and arities (on a sketch run, the library narrowed to the
sketch). Read it there rather than reconstructing it from the parameters
you passed.
`library` requires the separate typed Python route and is unavailable through
`run_discovery`. `savgol_window` is the available preprocessing control for
noisy grids.

On grid data, `pinn` selects PINN mode (`mode2`): pass a JSON object of the
keys listed under Parameters, `{}` for all defaults. On a fresh call set
`n_iterations` (its row is in the Parameters table) to an integer smaller
than `generations`: the first retraining phase follows that generation. With
the default `n_iterations` of 2000, a fresh call has no retraining phase unless
`generations` exceeds 2000. `n_iterations`,
`stability_selection` and `stability_queue_capacity` take effect only with
`pinn`. Plan the time as
`pretrain_epoch` times the number of observations, plus, for each retraining
phase, `pinn_epoch` times the number of collocation points (and of local
samples when `local_sample` is on). The defaults are the research budget of
the reference configuration, so set `pretrain_epoch`, `pinn_epoch` and
`n_collocation` explicitly. A call with `pinn` accepts no sketch.

A PINN-mode run is continued, adjusted, rolled back and forked with
`resume_from_run_id` like a run without `pinn`. The child keeps the parent's
network and does not pretrain. Generations are counted along the lineage, not
per call: the first retraining phase follows generation `n_iterations`
counted from the start of the first call, and the child's `generations` add
to that count. A parent that ended exactly on a retraining boundary (with the
default cycle length, a count that is a multiple of `n_iterations`) has not
run that phase; the child runs it first, before its first generation. That
phase is preparation, as pretraining is in a fresh call. The clean-stop clock
starts after preparation with the remaining budget computed before `Model.fit()`;
preparation neither reduces that budget nor is cut short by the clean stop.
The parent sends SIGTERM at the cap plus 120 seconds from the worker's start,
including preparation; a call still running then is terminated and returns a
failure. A retraining phase reached
later in the call starts no further training epoch once the cap is reached;
the run then ends after the next generation with a sealed record. A fork
(`reseed`) keeps the parent's observation and collocation points; its search
and the local samples follow the new seed.

Plan evaluations as `batch_size` times generations. A small batch and a few
generations are a probe, not a recovery budget. On noise-free benchmark
problems with batch 128 and epsilon 0.05 (not the defaults), eight
cold-started segments found the equation on 6 seeds where one continuous run
did not and the reverse on 22. Eight segments that carry the network and
optimizer state gave 11 against 13 against one continuous run, and 23 against
9 against eight cold-started segments. Raising the evaluation
cap from 8000 to 96000 left four problems at 0/3, 0/3, 1/3 and 1/3, from 0/3
each. Reward combines fit with
complexity and is not a recovery criterion. Read the split law and its aligned
coefficients when the display is a nested genotype. Valid proposals with an
unchanged best score record no improvement; neither a flat curve nor an
improving final iteration establishes representability or recovery.
