<div align="center">

# Knowledge Discovery (KD)

**Symbolic PDE Discovery from Data**

Discover the governing partial differential equations hidden in your data — automatically.

</div>

---

KD is a modular toolkit for symbolic equation discovery, with a strong focus on
partial differential equations (PDEs). It ships several complementary discovery
engines behind one unified API, a single dataset interface, and built-in
HTML-report visualization.

## Highlights

- **Multiple discovery engines, one API** — symbolic genetic algorithms,
  deep-learning GAs, and RL-based controllers (with more on the way), all driven
  by the same one-line `kd.Model(...).fit(dataset)` call
- **Unified `PDEDataset`** — load a built-in benchmark in one line, or wrap your
  own gridded NumPy/PyTorch arrays with `PDEDataset.from_arrays(...)`
- **Built-in visualization** — `VizEngine` renders a multi-figure HTML report:
  convergence curves, term importance, residual diagnostics, plus each engine's
  own search diagnostics
- **N-D-aware data model** — `PDEDataset` carries axis metadata for
  multi-dimensional spatial grids

## Quick Start

### Installation

Requires **Python >= 3.11** and **PyTorch >= 2.0**.

```bash
git clone -b trunk https://github.com/Scientific-Artificial-Intelligence-Lab/kd.git
cd kd
uv sync
```

### Discover a PDE in a few lines

```python
import kd

# Synthetic Burgers data with a known ground truth.
dataset = kd.generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)

model = kd.Model(algorithm="sga", generations=30, population=15, seed=0)
model.fit(dataset)

print(model.best_expr_)    # u_t = -1.00 u u_x + 0.10 u_xx
print(model.best_score_)   # best AIC
```

<div align="center">
<img src="docs/images/burgers_field_comparison.png" width="760" alt="True vs predicted Burgers field"><br>
<em>True vs predicted solution from the fit above — Burgers equation, residual ~1e-4.</em>
</div>

See [`examples/`](examples/) for runnable scripts covering every engine and
visualization mode.

## Supported Methods

The engines available today — swap the `algorithm=` string to switch:

| Algorithm | `algorithm=` | Approach |
|-----------|--------------|----------|
| **SGA** | `"sga"` | Symbolic genetic algorithm over a finite-difference term library; scored by AIC |
| **DLGA** | `"dlga"` | Neural-network surrogate + gene-expression genetic algorithm (Xu et al. 2020) |
| **DISCOVER** | `"discover"` | LSTM controller + risk-seeking policy gradient (Du et al. 2024) |

Per-engine hyperparameters live in `kd.SGAConfig`, `kd.DLGAConfig`, and
`kd.DiscoverConfig`, passed via `Model(..., config=...)`:

```python
from kd.search.dlga import DLGAConfig

model = kd.Model(
    algorithm="dlga",
    generations=20,
    config=DLGAConfig.burgers_preset(pop_size=80, seed=0),
)
```

## Example Gallery

<div align="center">

<img src="docs/images/burgers_equation.png" width="460" alt="Discovered Burgers equation"><br>
<em>The equation SGA discovers from the Burgers data above.</em>

<br><br>

<img src="docs/images/burgers_parity.png" width="420" alt="Parity plot"><br>
<em>Predicted vs actual — R² = 1.0000.</em>

<br><br>

<img src="docs/images/chafee_field_comparison.png" width="760" alt="Chafee-Infante recovery"><br>
<em>A real benchmark: true vs predicted field for Chafee–Infante.</em>

</div>

See [`examples/`](examples/) for runnable scripts that produce these figures.

## Built-in Datasets

Real PDE benchmarks bundled with kd, each loadable in one line:

```python
import kd

dataset = kd.load_chafee_infante()   # also: load_burgers, load_kdv,
                                     #       load_pde_compound, load_pde_divide
print(dataset.name, "->", dataset.ground_truth)
kd.preview(dataset)                  # sanity-check grid, dtype, NaN/Inf
```

## Bring Your Own Data

Wrap your own arrays — any field on a regular grid — into a `PDEDataset`:

```python
import kd

dataset = kd.PDEDataset.from_arrays(
    coords={"x": x, "t": t},        # 1-D coordinate arrays; order = axis order
    fields={"u": u},                # field tensor shaped (len(x), len(t))
    lhs="u_t",                      # left-hand side of the equation to discover
    periodic={"x"},                 # optional: periodic axes improve fits
    name="my_pde",
    ground_truth="u_t = 0.1 * u_xx",
)
```

## Visualization

After a fit, render a full HTML report (universal figures plus the fitted
engine's own search diagnostics):

```python
import kd

viz = kd.VizEngine(output_dir="out/my_run")
report = viz.render_all(model.result_, algorithm=model.algorithm_, dataset=dataset)
print(report.report)         # path to report.html
print(len(report.figures))   # number of figure files
```

## Package Layout

```
src/kd/
├── api.py        # Model facade — one-line fit() for every algorithm
├── data/         # PDEDataset, synthetic generators, benchmark loaders
├── search/       # sga / dlga / discover engines + their configs
├── viz/          # VizEngine — HTML reports & figures
└── inspect.py    # preview() dataset sanity checks
```

## Acknowledgements

KD draws on ideas and code from several open-source projects:

- [DISCOVER](https://github.com/menggedu/DISCOVER)
- [SGA-PDE](https://github.com/YuntianChen/SGA-PDE) — also the source of the
  bundled benchmark datasets (see [`NOTICE`](NOTICE))
- [SymPy](https://github.com/sympy/sympy)
- [PyTorch](https://github.com/pytorch/pytorch)

## License

[Apache-2.0](LICENSE). Copyright 2026 Mao, Hao and the Scientific Artificial
Intelligence Lab.
