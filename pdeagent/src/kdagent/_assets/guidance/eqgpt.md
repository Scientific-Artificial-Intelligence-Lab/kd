### EqGPT tool configuration

`weights_path` and `asset_dir` are Python-only and cannot be passed in JSON
`params`. Set `KD_EQGPT_ASSET_DIR` in the controller environment to use local
weights. An uncached offline download can raise `FileNotFoundError`, and a Hub
HTTP failure can raise `HfHubHTTPError`. Omitted `sparsity_alpha` is rejected;
packaged evolution presets use 0.02 and steady presets use 1.0. The steady
mode needs its three settings together, as described in the shared card.

### EqGPT observations

At the then-default samples_per_epoch=400, the tool's ten-generation budget
drew 4000 sentences. Each checkpoint contains transformer and Adam state,
hundreds of megabytes; checkpoint_every=1 and checkpoint_keep_last=5 retained
about 1-2 GB. Increase the interval if that disk cost matters. Keep-last bounds
periodic checkpoints, not the final archive.

A perfect one-term evolution fit at sparsity_alpha=0.02 scores about 0.994
because the column count also includes the pinned target. A zero score can
be a gate-zeroed candidate, ranked above negative rewards, or accompany an
invalid no-candidate outcome; check status. The evolution finite-difference
mask admitted 21 atoms versus the reference's 24, excluding fourth- and
fifth-order x derivatives and the unmapped sinh(u). The reference pool size
was 10. Consider a new seed or larger samples_per_epoch only after checking
the allowed atoms and sentence gates.
