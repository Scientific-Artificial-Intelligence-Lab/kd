# kd Examples

Run any of these end-to-end. In a checkout, `uv run` uses the environment
`uv sync` created (no manual venv activation); after `pip install sail-kd`, run them
with plain `python` instead:

```bash
uv run python examples/01_quickstart.py
uv run python examples/02_your_data.py
uv run python examples/03_visualize.py
uv run python examples/04_noisy_data.py
uv run python examples/05_save_load.py
uv run python examples/06_realworld.py # showcase: 4 real benchmarks, autograd, full reports
uv run python examples/07_discover.py # DISCOVER (LSTM controller + RSPG) on Burgers
uv run python examples/08_dlga.py # DLGA (NN surrogate + GA) on Burgers
uv run python examples/09_compare_algorithms.py # all 7 engines on one dataset, one ruler (optional engines skip gracefully)
uv run python examples/10_checkpoint_resume.py # checkpoint a run, resume it
uv run python examples/11_evaluate_terms.py # score your own terms, no search
uv run python examples/12_symbolic_regression.py # scalar SR: PySR + DISCOVER auto-search y=f(X) (needs the pysr extra: `pip install "sail-kd[pysr]"`, or `uv sync --extra pysr` in a checkout)
uv run python examples/13_sindy_basis_sr.py # scalar SR bypass: you supply the basis, one sparse solve
uv run python examples/14_field_animation_2d.py # 2D Burgers True|Predicted field animation GIF
uv run python examples/15_remote_dataset.py # on-demand HuggingFace dataset (needs network)
uv run python examples/16_eqgpt.py # EqGPT (pretrained GPT proposer) on Burgers (needs pretrained weights)
uv run python examples/17_llm4ed.py # LLM4ED (LLM equation proposer) on diffusion, offline canned provider (zero network)
uv run python examples/19_batch_harness.py # kd.harness: run a plan of fits, seal the evidence, read the consensus
uv run python examples/21_sketch_discovery.py # sketch: pin known terms, constrain the unknown remainder, certified exit
uv run --with jupyter jupyter notebook examples/notebooks/getting_started.ipynb # narrated walkthrough with inline outputs
uv run --with jupyter jupyter notebook examples/notebooks/kdv_walkthrough.ipynb # KdV walkthrough: SGA at its validated budget in four calls, report rendered inline
uv run --with jupyter jupyter notebook examples/notebooks/noisy_burgers_walkthrough.ipynb # Noisy Burgers walkthrough: 15% noise added with kd.add_noise, DLGA at its validated budget, report rendered inline
```

Each file is self-contained.

| File | What you'll learn | Runtime |
|------|-------------------|---------|
| 01_quickstart.py | First SGA fit on synthetic data: Hello World | ~5 s |
| 02_your_data.py | **BYOD: wrap your numpy arrays into a PDEDataset** | ~3 s |
| 03_visualize.py | Render a multi-figure HTML report in one `model.report(...)` call | ~30 s (slow) |
| 04_noisy_data.py | Choose finite_diff vs autograd for noisy data | ~25 s (slow) |
| 05_save_load.py | Persist and reload a fitted result | ~5 s |
| 06_realworld.py | **Showcase: 4 real benchmarks × 2 modes (FD + NN), full viz** through `model.report(...)` for every fit | ~15-25 min |
| 07_discover.py | **DISCOVER**: LSTM-controller symbolic regression (smoke run), with `model.report(...)` for universal and algorithm diagnostics | ~10 s |
| 08_dlga.py | **DLGA**: NN_1 surrogate + GA, incl. the surrogate training curve in `model.report(...)` | ~3 min |
| 09_compare_algorithms.py | **All 7 engines on one dataset, one unified NMSE ruler**: SGA / DLGA / DISCOVER always run; PySINDy also runs with the core install; PySR (needs `pysr` extra) and EqGPT (needs local `.pt` weights) are reported as skipped when absent; llm4ed runs offline via a canned provider (a preset answer, tagged as such, not a live search). `kd.evaluate.compare_results` refits active terms on shared features, then `.render(...)` draws the equation table, normalized progress and NMSE bars | ~6-12 min |
| 10_checkpoint_resume.py | **Checkpoint a long run, resume after a crash** (`checkpoint_dir` / `fit(resume_from=...)`) | ~10 s |
| 11_evaluate_terms.py | **Score candidate terms without a search**: the stateless, fail-loud `kd.evaluate_terms` / `kd.validate_terms` entry (agent-ready) | ~5 s |
| 12_symbolic_regression.py | **Scalar SR through `Model.fit`**: PySR and DISCOVER search `y = f(X)` on the real-world TLC-CC chromatography dataset (`kd.load_tlc_cc`), without PDE derivatives. Both fit constants *inside* functions; DISCOVER exposes the Pareto front with each candidate's fitted outer scale. PySR needs the `pysr` extra | ~minutes (DISCOVER: 100 generations × 500 candidates) |
| 13_sindy_basis_sr.py | **Scalar SR bypass: SINDy**: you supply a candidate-term library, one STRidge sparse solve picks support + coefficients; fail-loud on bad terms | ~3 s |
| 14_field_animation_2d.py | **2D Burgers field animation**: uniformly subsample with `kd.data.stride_subsample`, integrate the ground-truth RHS, and save a True\|Predicted GIF with `save_field_animation` | ~5 s |
| 15_remote_dataset.py | **On-demand remote dataset**: list HuggingFace datasets, fetch one, preview it, and run a tiny SGA fit. Needs network | network dependent |
| 16_eqgpt.py | **EqGPT**: pretrained generative GPT proposes candidate PDEs, kd scores + fine-tunes toward high-reward equations. Uses `EqGPTConfig.burgers_preset()` (per-problem `sparsity_alpha`, D5); needs the pretrained weights (see the script's asset hint) | ~1-2 min |
| 17_llm4ed.py | **LLM4ED**: an LLM proposes candidate PDE right-hand sides as text, kd scores each by an EDL sparse-regression reward and evolves an elite pool. Runs offline with an inline canned `provider=` (zero network, no API key); the script comments show the real-backend path (`base_url` + `OPENAI_API_KEY`, optional `tape_record_path`) | ~10 sec |
| 18 (retired) | Retired (numbering gaps stay retired; no renumbering) | - |
| 19_batch_harness.py | **Batch a plan of fits and aggregate it** (`kd.harness`): an ordered (instrument, dataset, seed) matrix runs into a fresh evidence store, the store re-opens read-only and self-verifies, and `build_consensus` groups the runs into structure classes with a Markdown + JSON rendering. Also shows how to read a class split that is only a notation difference | ~5 s |
| 20 (retired) | Retired (numbering gaps stay retired; no renumbering) | - |
| 21_sketch_discovery.py | **Discover with a sketch** (`Model.fit(dataset, sketch=...)`): pin the advection term of Burgers with its exact coefficient, leave one order-2 hole, and let SGA's native compiler search only the admissible grammar. The exit certifies the solution against every sketch clause and restores the pin exactly | ~6 s |
| notebooks/getting_started.ipynb | **Narrated getting-started notebook**: load Burgers, preview the field, fit SGA, compare against truth, draw convergence with `plot_convergence`, and display `model.report(...)` inline; recorded outputs | ~1-2 min |
| notebooks/your_own_data.ipynb | **Bring your own gridded arrays**: build a two-mode heat field, wrap it with `PDEDataset.from_arrays`, draw it with `plot_field`, fit SGA, and display `model.report(...)` inline | ~1 min |
| notebooks/kdv_walkthrough.ipynb | **KdV walkthrough**: SGA-PDE recovers `u_t = -u*u_x - 0.0025*u_xxx` from the bundled KdV benchmark at the budget its recovery suite validates (200 generations, seed 42), in four calls (`kd.load_kdv`, `plot_field`, `Model.fit`, `Model.report`), the discovered law next to the reference, and the HTML report with every supported figure rendered inline in the notebook; committed outputs | ~10-13 min (the 200-generation fit) |
| notebooks/noisy_burgers_walkthrough.ipynb | **Noisy Burgers walkthrough**: `kd.add_noise` adds 15% relative Gaussian noise to the bundled Burgers field; SGA-PDE differentiating the samples directly returns a derivative-free term, and DLGA at the budget its recovery suite validates (50000-epoch surrogate, population 50, 20 generations, seed 0) recovers `u*u_x` (-0.983) and `u_xx` (0.104) with two small extra terms; the report, with the surrogate training curve, rendered inline; committed outputs | ~3 h (158 min for the DLGA fit in the recorded run) |

### Scalar symbolic regression

Examples 12-13 fit `y = f(X)` on tabular data without PDE derivatives.
Example 12 uses `TabularDataset` and `kd.Model` for automatic search;
example 13 uses the separate `SINDyRegressor` for a user-supplied basis:

- **PySR and DISCOVER (12)**: automatic expression search; fits free-form structure
  *and* constants inside nonlinear functions (e.g. the `-1.85` in `exp(-1.85·x)`).
- **SINDy (13)**: you bring a candidate-term library as domain knowledge; one
  sparse linear solve picks the support + coefficients. It **cannot** fit
  constants inside functions: that limitation is the trade for transparency and
  user control over the hypothesis space.

DISCOVER's Pareto front records accuracy and complexity together with the fitted
outer scale, so a candidate expression must be read with its `entry.scale`.

For component-level wiring (custom Evaluator, plugin authoring, manual schema construction): see `examples/internals/`.
