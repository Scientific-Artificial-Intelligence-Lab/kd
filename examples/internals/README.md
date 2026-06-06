# Internal Examples

This directory contains the original component-level examples that show how to
wire kd's internals together by hand:

- `00_components_intro.py` — walks through `PlatformComponents`, executor,
  evaluator, and the search-side configuration objects
- `02_sga_discovery_burgers.py` — assembles `SGAConfig`, `SGAPlugin`, and
  `ExperimentRunner` directly to discover the 1D Burgers equation
- `03_sga_2d_diffusion.py` — same low-level wiring on a 2D diffusion problem
- `04_2d_diffusion.py` — 2D diffusion via the high-level `kd.Model` facade
- `04_sga_with_autograd.py` — swaps the derivative provider to autograd
- `05_save_load_result.py` — round-trips an `ExperimentResult` to disk

These examples are aimed at plugin and extension authors who need full control
over the search loop (e.g. building a new algorithm plugin, customizing the
evaluator, or integrating a different derivative provider).

For the user-facing tutorial that uses the high-level `kd.Model` API, see the
parent `examples/` directory.
