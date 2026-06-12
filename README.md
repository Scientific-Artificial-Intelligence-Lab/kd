<div align="center">

# Knowledge Discovery (KD)

**Symbolic PDE discovery from gridded data**

</div>

---

KD is the lab's unified platform for data-driven discovery of partial
differential equations. It provides refactored re-implementations of
discovery algorithms developed in this lab — **SGA** (Chen et al.), **DLGA**
(Xu et al.), and **DISCOVER** (Du et al.) — on one shared stack: a single
dataset interface, a single term evaluator, and built-in HTML-report
visualization. External engines such as **PySR** (Cranmer) are integrated
unchanged behind the same API. More engines are on the roadmap.

## Install

Requires **Python >= 3.11** and **PyTorch >= 2.0**.

```bash
git clone -b trunk https://github.com/Scientific-Artificial-Intelligence-Lab/kd.git
cd kd
uv sync                      # add --extra pysr for the PySR engine (Julia)
```

## First fit

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

See [`examples/`](examples/) for runnable scripts covering every engine —
including [`09_compare_algorithms.py`](examples/09_compare_algorithms.py),
which runs all four engines on the same dataset and ranks the discovered
equations on one unified NMSE ruler (it needs the `pysr` extra).

## Engines

Swap the `algorithm=` string to switch engines:

| Algorithm | `algorithm=` | Origin | Approach |
|-----------|--------------|--------|----------|
| **SGA** | `"sga"` | Chen et al. 2022 (SGA-PDE) | Genetic algorithm over symbolic trees with finite-difference terms; scored by AIC |
| **DLGA** | `"dlga"` | Xu et al. 2020 | Neural-network surrogate + gene-encoded genetic algorithm |
| **DISCOVER** | `"discover"` | Du et al. 2024 | LSTM controller trained with risk-seeking policy gradient |
| **PySR** | `"pysr"` | Cranmer 2023 | External symbolic-regression engine (Julia); results re-scored by KD's own evaluator |

SGA, DLGA, and DISCOVER are re-implementations of the original algorithms,
refactored onto KD's shared platform; PySR is an external tool integrated
as-is (requires the `pysr` extra). More engines are planned.

Per-engine hyperparameters live in each engine's config dataclass, passed
via `Model(..., config=...)`:

```python
from kd.search.dlga import DLGAConfig

model = kd.Model(
    algorithm="dlga",
    generations=20,
    config=DLGAConfig.burgers_preset(pop_size=80, seed=0),
)
```

## Built-in Datasets

Real PDE benchmarks bundled with KD, each loadable in one line:

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
import torch

import kd

x = torch.linspace(0.0, 1.0, 64)   # spatial grid
t = torch.linspace(0.0, 1.0, 32)   # time grid
u = torch.rand(64, 32)             # your measured field on the (x, t) grid

dataset = kd.PDEDataset.from_arrays(
    coords={"x": x, "t": t},        # 1-D coordinate arrays; order = axis order
    fields={"u": u},                # field tensor shaped (len(x), len(t))
    lhs="u_t",                      # left-hand side of the equation to discover
    periodic={"x"},                 # optional: periodic axes improve fits
    name="my_pde",
    ground_truth="u_t = 0.1 * u_xx",
)
```

## Score Your Own Candidate Terms

You don't have to run a search to use KD's evaluator. `evaluate_terms` fits
a candidate term set directly; `validate_terms` classifies terms without
fitting. Both fail loud with a complete per-term rejection report (reason +
hint), so a caller — human or LLM agent — can repair and resubmit:

```python
import kd

result = kd.evaluate_terms(dataset, ["diff2_x(u)", "mul(u, diff_x(u))"])
print(result.coefficients, result.nmse)

report = kd.validate_terms(dataset, ["u_xx", "u + u_x"])  # no fit performed
for v in report.rejected:
    print(v.term, "->", v.reason)    # "u + u_x" is not canonical funcall IR
```

## Long Runs: Checkpoint & Resume

```python
model = kd.Model(algorithm="discover", generations=500, checkpoint_dir="ckpts")
model.fit(dataset)                   # writes ckpts/checkpoint_*.pt as it goes

# Later — or after a crash — continue from the saved search state:
model = kd.Model(algorithm="discover", generations=200)
model.fit(dataset, resume_from="ckpts/checkpoint_final.pt")
```

The checkpoint restores search state (population / controller weights /
best); generations and other settings come from the new `Model`.

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

Every result also carries a `manifest` (dataset fingerprint, seed, KD
version) so a run can be identified and reproduced later.

## Package Layout

```
src/kd/
├── api.py        # Model facade — one-line fit() for every engine
├── evaluate.py   # evaluate_terms / validate_terms — score terms directly
├── data/         # PDEDataset, synthetic generators, benchmark loaders
├── search/       # sga / dlga / discover / pysr engines + their configs
├── viz/          # VizEngine — HTML reports & figures
└── inspect.py    # preview() dataset sanity checks
```

## Origins & Acknowledgements

The SGA, DLGA, and DISCOVER engines are refactored re-implementations of
algorithms developed in this lab; credit for the methods belongs to the
original works:

- **SGA-PDE** — Chen et al., [SGA-PDE](https://github.com/YuntianChen/SGA-PDE) —
  also the source of the bundled benchmark datasets (see [`NOTICE`](NOTICE))
- **DLGA** — Xu et al. 2020
- **DISCOVER** — Du et al., [DISCOVER](https://github.com/menggedu/DISCOVER)

KD also builds on [PySR](https://github.com/MilesCranmer/PySR) (integrated
as an optional engine), [SymPy](https://github.com/sympy/sympy), and
[PyTorch](https://github.com/pytorch/pytorch).

## License

[Apache-2.0](LICENSE). Copyright 2026 Mao, Hao and the Scientific Artificial
Intelligence Lab.
