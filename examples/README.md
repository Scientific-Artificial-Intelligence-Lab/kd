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
python examples/09_compare_algorithms.py # SGA/DLGA/DISCOVER + PySR baseline (needs: uv sync --extra pysr)
python examples/10_checkpoint_resume.py # checkpoint a run, resume it
python examples/11_evaluate_terms.py # score your own terms, no search
python examples/12_symbolic_regression.py # scalar SR bypass: PySR auto-search y=f(X) (needs: uv sync --extra pysr)
python examples/13_sindy_basis_sr.py # scalar SR bypass: you supply the basis, one sparse solve
python examples/14_field_animation_2d.py # 2D Burgers True|Predicted field animation GIF
python examples/15_remote_dataset.py # on-demand HuggingFace dataset (needs network + uv sync --extra hub)
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
| 09_compare_algorithms.py | **SGA / DLGA / DISCOVER on one dataset, one unified NMSE ruler**: with the external PySR as a reference baseline (needs the `pysr` extra) | ~10-15 min |
| 10_checkpoint_resume.py | **Checkpoint a long run, resume after a crash** (`checkpoint_dir` / `fit(resume_from=...)`) | ~10 s |
| 11_evaluate_terms.py | **Score candidate terms without a search**: the stateless, fail-loud `kd.evaluate_terms` / `kd.validate_terms` entry (agent-ready) | ~5 s |
| 12_symbolic_regression.py | **Scalar SR bypass: PySR**: auto-search `y = f(X)` on the real-world TLC-CC chromatography dataset (`kd.load_tlc_cc`), no PDE/Theta/facade; fits constants *inside* functions. Needs the `pysr` extra | ~10-30 s |
| 13_sindy_basis_sr.py | **Scalar SR bypass: SINDy**: you supply a candidate-term library, one STRidge sparse solve picks support + coefficients; fail-loud on bad terms | ~3 s |
| 14_field_animation_2d.py | **2D Burgers field animation**: integrate the ground-truth RHS and save a True\|Predicted GIF | ~5 s |
| 15_remote_dataset.py | **On-demand remote dataset**: list HuggingFace datasets, fetch one, preview it, and run a tiny SGA fit. Needs network and the `hub` extra | network dependent |
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
