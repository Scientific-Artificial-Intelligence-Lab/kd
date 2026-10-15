Input data selects between the grid modes and tabular mode; there is no `mode`
parameter. The grid vocabulary is fixed to `u`, `x`, `t`, `diff_x`, `diff2_x`;
other grid field names, extra fields and extra spatial axes are refused.
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
generations are a probe, not a recovery budget. Reward combines fit with
complexity and is not a recovery criterion. Read the split law and its aligned
coefficients when the display is a nested genotype. Valid proposals with an
unchanged best score record no improvement; neither a flat curve nor an
improving final iteration establishes representability or recovery.
