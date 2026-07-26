# kd Examples

Run any of these end-to-end:

```bash
python examples/01_quickstart.py
python examples/02_your_data.py
python examples/03_visualize.py
python examples/04_noisy_data.py
python examples/05_save_load.py
python examples/06_realworld.py # showcase: 4 real benchmarks, autograd, full reports
python examples/07_discover.py # DISCOVER (LSTM controller + RSPG) on Burgers
python examples/08_dlga.py # DLGA (NN surrogate + GA) on Burgers
python examples/09_compare_algorithms.py # all 7 engines on one dataset, one ruler (optional engines skip gracefully)
python examples/10_checkpoint_resume.py # checkpoint a run, resume it
python examples/11_evaluate_terms.py # score your own terms, no search
python examples/12_symbolic_regression.py # scalar SR bypass: PySR auto-search y=f(X) (needs: uv sync --extra pysr)
python examples/13_sindy_basis_sr.py # scalar SR bypass: you supply the basis, one sparse solve
python examples/14_field_animation_2d.py # 2D Burgers True|Predicted field animation GIF
python examples/15_remote_dataset.py # on-demand HuggingFace dataset (needs network + uv sync --extra hub)
python examples/16_eqgpt.py # EqGPT (pretrained GPT proposer) on Burgers (needs pretrained weights)
python examples/17_llm4ed.py # LLM4ED (LLM equation proposer) on diffusion, offline canned provider (zero network)
python examples/18_eqgpt_wave_breaking.py # EqGPT wave-breaking reproduction on real scattered data (needs .pt weights + KD_V1_WAVE_ASSETS surrogate tree)
python examples/19_batch_harness.py # kd.harness: run a plan of fits, seal the evidence, read the consensus
jupyter notebook examples/notebooks/getting_started.ipynb # narrated walkthrough with inline outputs
```

Each file is self-contained.

| File | What you'll learn | Runtime |
|------|-------------------|---------|
| 01_quickstart.py | First SGA fit on synthetic data: Hello World | ~5 s |
| 02_your_data.py | **BYOD: wrap your numpy arrays into a PDEDataset** | ~3 s |
| 03_visualize.py | Render a multi-figure HTML report | ~30 s (slow) |
| 04_noisy_data.py | Choose finite_diff vs autograd for noisy data | ~25 s (slow) |
| 05_save_load.py | Persist and reload a fitted result | ~5 s |
| 06_realworld.py | **Showcase: 4 real benchmarks × 2 modes (FD + NN), full viz** | ~15-25 min |
| 07_discover.py | **DISCOVER**: LSTM-controller symbolic regression (smoke run) | ~10 s |
| 08_dlga.py | **DLGA**: NN_1 surrogate + GA, incl. surrogate training curve | ~3 min |
| 09_compare_algorithms.py | **All 7 engines on one dataset, one unified NMSE ruler**: SGA / DLGA / DISCOVER always run; PySR (needs `pysr` extra), PySINDy (needs `pysindy` extra), and EqGPT (needs `.pt` weights) skip gracefully when their asset is absent; llm4ed runs offline via a canned provider (a preset answer, tagged as such, not a live search) | ~6-12 min |
| 10_checkpoint_resume.py | **Checkpoint a long run, resume after a crash** (`checkpoint_dir` / `fit(resume_from=...)`) | ~10 s |
| 11_evaluate_terms.py | **Score candidate terms without a search**: the stateless, fail-loud `kd.evaluate_terms` / `kd.validate_terms` entry (agent-ready) | ~5 s |
| 12_symbolic_regression.py | **Scalar SR bypass: PySR**: auto-search `y = f(X)` on the real-world TLC-CC chromatography dataset (`kd.load_tlc_cc`), no PDE/Theta/facade; fits constants *inside* functions. Needs the `pysr` extra | ~10-30 s |
| 13_sindy_basis_sr.py | **Scalar SR bypass: SINDy**: you supply a candidate-term library, one STRidge sparse solve picks support + coefficients; fail-loud on bad terms | ~3 s |
| 14_field_animation_2d.py | **2D Burgers field animation**: integrate the ground-truth RHS and save a True\|Predicted GIF | ~5 s |
| 15_remote_dataset.py | **On-demand remote dataset**: list HuggingFace datasets, fetch one, preview it, and run a tiny SGA fit. Needs network and the `hub` extra | network dependent |
| 16_eqgpt.py | **EqGPT**: pretrained generative GPT proposes candidate PDEs, kd scores + fine-tunes toward high-reward equations. Uses `EqGPTConfig.burgers_preset()` (per-problem `sparsity_alpha`, D5); needs the pretrained weights (see the script's asset hint) | ~1-2 min |
| 17_llm4ed.py | **LLM4ED**: an LLM proposes candidate PDE right-hand sides as text, kd scores each by an EDL sparse-regression reward and evolves an elite pool. Runs offline with an inline canned `provider=` (zero network, no API key); the script comments show the real-backend path (`base_url` + `OPENAI_API_KEY`, optional `tape_record_path`) | ~10 sec |
| 18_eqgpt_wave_breaking.py | **EqGPT wave-breaking reproduction (flagship)**: EqGPT proposes one RHS structure and kd scores it per case across the paper's 12 wave-tank experiments, recovering `eta_t + c1*eta_x + c2*eta_xxx + c3*(eta*eta_x)_xx = 0` on real SCATTERED `(t, x, eta)` data; the report leads with the per-case reward hero panel. Needs the pretrained `.pt` weights, the WaveBreaking pickle, and the per-case v1 surrogate tree (`KD_V1_WAVE_ASSETS`) | ~minutes |
| 19_batch_harness.py | **Batch a plan of fits and aggregate it** (`kd.harness`): an ordered (instrument, dataset, seed) matrix runs into a fresh evidence store, the store re-opens read-only and self-verifies, and `build_consensus` groups the runs into structure classes with a Markdown + JSON rendering. Also shows how to read a class split that is only a notation difference | ~5 s |
| notebooks/getting_started.ipynb | **Narrated getting-started notebook**: load Burgers, preview the field, fit SGA, compare against truth, and keep inline outputs for GitHub | ~1-2 min |

### Scalar symbolic regression (bypass)

Examples 12-13 are a **separate track** from the PDE-discovery platform above:
they fit `y = f(X)` on plain tabular data, with no PDEDataset, derivatives, Theta
matrix, or `kd.Model` facade. They reuse only the numerical kernel and stay
physically isolated from the platform. Two complementary modes:

- **PySR (12)**: fully automatic black-box search; fits free-form structure
  *and* constants inside nonlinear functions (e.g. the `-1.85` in `exp(-1.85·x)`).
- **SINDy (13)**: you bring a candidate-term library as domain knowledge; one
  sparse linear solve picks the support + coefficients. It **cannot** fit
  constants inside functions: that limitation is the trade for transparency and
  user control over the hypothesis space.

(A third tier, vocabulary-driven auto-search à la DISCOVER, is deferred.)

For component-level wiring (custom Evaluator, plugin authoring, manual schema construction): see `examples/internals/`.
