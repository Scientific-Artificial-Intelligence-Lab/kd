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
python examples/09_compare_algorithms.py # all four engines (needs: uv sync --extra pysr)
```

Each file is self-contained.

| File | What you'll learn | Runtime |
|------|-------------------|---------|
| 01_quickstart.py | First SGA fit on synthetic data — Hello World | ~5 s |
| 02_your_data.py | **BYOD: wrap your numpy arrays into a PDEDataset** | ~3 s |
| 03_visualize.py | Render a multi-figure HTML report | ~30 s (slow) |
| 04_noisy_data.py | Choose finite_diff vs autograd for noisy data | ~25 s (slow) |
| 05_save_load.py | Persist and reload a fitted result | ~5 s |
| 06_realworld.py | **Showcase: 4 real benchmarks × 2 modes (FD + NN), full viz** | ~12-15 min |
| 07_discover.py | **DISCOVER**: LSTM-controller symbolic regression (smoke run) | ~10 s |
| 08_dlga.py | **DLGA**: NN_1 surrogate + GA, incl. surrogate training curve | ~3 min |
| 09_compare_algorithms.py | **All four engines, one dataset, one unified NMSE ruler** (needs the `pysr` extra) | ~10-15 min |

For component-level wiring (custom Evaluator, plugin authoring, manual schema construction): see `examples/internals/`.
