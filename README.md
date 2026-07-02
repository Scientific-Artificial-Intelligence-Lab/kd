<div align="center">

# Knowledge Discovery (KD)

**Symbolic PDE discovery from data**

</div>

---

KD discovers the governing partial differential equation from data: give it a
field sampled on a spatiotemporal grid, get back a symbolic PDE. Three
in-house discovery engines (**SGA**, **DLGA**, **DISCOVER**) run behind one
`kd.Model` API, sharing a single dataset interface, term evaluator, and
HTML-report visualization. More engines are on the roadmap.

<div align="center">
<img src="docs/images/burgers2d_animation.gif" width="760" alt="2D Burgers field over time: true evolution vs the ground-truth PDE integrated forward"><br>
<em>2D Burgers (<code>u_t = -u·u_x - u·u_y + 0.01·∇²u</code>): the field's true evolution beside the PDE integrated forward through the platform. Regenerate with <code>examples/14_field_animation_2d.py</code>.</em>
</div>

## Install

Requires **Python >= 3.11** and **PyTorch >= 2.0**.

```bash
git clone -b trunk https://github.com/Scientific-Artificial-Intelligence-Lab/kd.git
cd kd
uv sync
```

## Quick start

```python
import kd

# Generate a synthetic Burgers dataset.
dataset = kd.generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)

model = kd.Model(algorithm="sga", generations=30, population=15, seed=0)
model.fit(dataset)

print(model.best_expr_)    # u_t = -1.00 u u_x + 0.10 u_xx
print(model.best_score_)   # best AIC
```

<div align="center">
<img src="docs/images/burgers_field_comparison.png" width="760" alt="True vs predicted Burgers field"><br>
<em>True vs predicted solution from the fit above (Burgers equation, residual ~1e-4).</em>
</div>

See [`examples/`](examples/) for runnable scripts covering every engine,
including [`09_compare_algorithms.py`](examples/09_compare_algorithms.py),
which runs the engines on the same dataset and ranks the discovered
equations on one unified NMSE ruler.

## Engines

All three engines are re-implementations of algorithms developed in this
lab, refactored onto KD's shared platform. Swap the `algorithm=` string to
switch:

| Algorithm | `algorithm=` | Origin | Approach |
|-----------|--------------|--------|----------|
| **SGA** | `"sga"` | Chen et al. 2022 (SGA-PDE) | Genetic algorithm over symbolic expression trees |
| **DLGA** | `"dlga"` | Xu et al. 2020 | Neural-network surrogate + genetic algorithm |
| **DISCOVER** | `"discover"` | Du et al. 2024 | LSTM controller + policy gradient |

The external PySR regressor can also be driven through the same facade as an
optional fallback for cross-checking (`algorithm="pysr"`, needs
`uv sync --extra pysr`). More engines are planned.

## Datasets

### Simulated PDE benchmarks

KD bundles the simulated benchmark suites used across this lab's
PDE-discovery papers — **SGA-PDE** (Chen et al., *Phys. Rev. Research* **4**,
023174, 2022), **EqGPT** (Xu et al., *Nat Commun* **16**, 10255, 2025) and
**LLM4ED** (Du et al., *Phys. Fluids* **36**, 097121, 2024):

<div align="center">
<img src="docs/images/dataset_gallery.png" width="820" alt="Field snapshots of the bundled PDE benchmark datasets">
</div>

| Dataset | Governing PDE | Grid | What it models |
|---------|---------------|------|----------------|
| `allen-cahn` | `u_t = 0.003·u_xx + u - u³` | `(256, 201)` | phase separation (reaction–diffusion) |
| `burgers` | `u_t = -u·u_x + 0.1·u_xx` | `(256, 201)` | shock waves in fluids |
| `burgers-2d` | `u_t = -u·u_x - u·u_y + 0.01·∇²u` | `(101, 51, 100)` | 2D Burgers flow |
| `chafee-infante` | `u_t = u_xx - u + u³` | `(301, 200)` | reaction–diffusion |
| `convection-diffusion` | `u_t = -u_x + 0.25·u_xx` | `(256, 100)` | advection plus diffusion |
| `eq-6-2-12` | `u_t = -0.1·u_x_t - 0.1·u_x` | `(501, 501)` | handbook equation with a mixed space–time derivative |
| `kdv` | `u_t = -u·u_x - 0.0025·u_xxx` | `(256, 201)` | shallow-water solitons |
| `klein-gordon` | `u_tt = 0.5·u_xx - 5·u` | `(201, 201)` | relativistic wave equation |
| `llm4ed-fisher` | `u_t = 0.02·u_xx + 10·u·(1-u)` | `(x, t)` | population growth with spatial spread |
| `llm4ed-fisher-nonlinear` | `u_t = 0.02·(u·u_xx + u_x²) + 10·u·(1-u)` | `(x, t)` | Fisher growth with nonlinear diffusion |
| `llm4ed-heat` | `u_t = 0.05·u_xx` | `(x, t)` | heat conduction |
| `pde-compound` | `u_t = u·u_xx + u_x²` | `(100, 251)` | constructed compound-structure benchmark |
| `pde-divide` | `u_t = -u_x/x + 0.25·u_xx` | `(100, 251)` | constructed benchmark with a division term |
| `wave` | `u_tt = u_xx` | `(161, 321)` | vibrating string |

Load any bundled dataset with `kd.load_burgers()`, or browse the catalog
programmatically with `kd.list_datasets()` / `kd.get_dataset(id)`; each entry
carries its `.source` and `.license` (see [`NOTICE`](NOTICE)).
`kd.generate_burgers_data()`, `kd.generate_diffusion_data()`, … build
synthetic datasets on demand. Remote (HF) entries
are fetched with `kd.load_from_hub(id)` after installing the hub extra
(`uv sync --extra hub`) and are cached locally, checksum-verified, and
revision-pinned; browse them with `kd.list_remote_datasets()`.

### Real-world experimental data

KD also bundles **real-world experimental data** — measured, not simulated:

| Dataset | Type | Measured quantity | Size | Reference |
|---------|------|-------------------|------|-----------|
| `wave-breaking` | wave-tank experiment (Imperial College London) | surface elevation `η(t, x)` of wave groups approaching breaking | 314,478 points (one of the paper's 12 experiments) | Xu et al., *Nat Commun* **16**, 10255 (2025) |
| `tlc-cc` | automated chromatography experiment | column retention volumes `V_S`, `V_E` vs `(R_F, r)` | 2 tables × 74 conditions | Xu et al., *Nat Commun* **16**, 832 (2025) |

**Wave breaking** — surface elevation of focused wave groups approaching
breaking, reconstructed frame by frame from camera images in the wave-tank
experiments of the EqGPT paper. KD bundles one of the paper's 12 experiments
(case `N_G2Tp12A100_broad`) as scattered `(t, x, η)` points — a table rather
than a gridded `PDEDataset`.

**TLC-CC** — column-chromatography retention volumes measured on an
automated platform (192 compounds, 4 g silica columns), aggregated to mean
start/end retention volumes over 74 `(R_F, r)` conditions — ready for KD's
scalar symbolic-regression entries
([`examples/12`](examples/12_symbolic_regression.py),
[`examples/13`](examples/13_sindy_basis_sr.py)).

```python
wb = kd.load_wave_breaking()          # η(t, x): scattered wave-tank points
cc = kd.load_tlc_cc(target="start")   # X = (R_F, r), y = V_S
```

Experimental background, protocols, and references for both datasets are in
the papers and their Supplementary Information
([wave breaking](https://doi.org/10.1038/s41467-025-65114-2),
[TLC-CC](https://doi.org/10.1038/s41467-025-56136-x)).

## Bring Your Own Data

Wrap your own arrays (any field on a regular grid) into a `PDEDataset`:

```python
import torch

import kd

x = torch.linspace(0.0, 1.0, 64)   # spatial grid
t = torch.linspace(0.0, 1.0, 32)   # time grid
u = torch.rand(64, 32)             # your measured field on the (x, t) grid

dataset = kd.PDEDataset.from_arrays(
    coords={"x": x, "t": t},        # one 1D array per axis; insertion order sets the axis order
    fields={"u": u},                # field shaped (len(x), len(t))
    lhs="u_t",                      # left-hand side of the equation to discover
    periodic={"x"},                 # optional: periodic axes improve fits
    name="my_pde",
    ground_truth="u_t = 0.1 * u_xx",
)
```

The same call handles 2D spatial fields: add a `y` axis and pass an nD field,
e.g. `coords={"x": x, "y": y, "t": t}` with `u` shaped `(len(x), len(y),
len(t))`.

## Score Your Own Candidate Terms

You don't have to run a search to use KD's evaluator. `evaluate_terms` fits
a candidate term set directly; `validate_terms` classifies terms without
fitting. Both fail loud with a complete per-term rejection report (reason +
hint), so a caller (human or LLM agent) can repair and resubmit:

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

# Later (or after a crash), continue from the saved search state:
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

The report renders the discovered equation as a structure-only **expression
tree**, and (for the SGA engine) the raw **genome tree** of the best evolved
individual, so you can see what the search actually produced versus the sparse
equation it was distilled into:

<div align="center">
<img src="docs/images/sga_genome_vs_equation_tree.png" width="820" alt="SGA genome tree vs discovered expression tree"><br>
<em>Example: SGA on the built-in Chafee-Infante benchmark (recovers
<code>u_t = u_xx - u + u^3</code>). Left: the raw GP genome of the best
individual, still carrying evolved bloat (redundant / zeroed terms). Right: the
discovered equation after sparse selection, operators and derivatives only,
coefficients dropped (they stay in the LaTeX equation figure).</em>
</div>

Every result also carries a `manifest` (dataset fingerprint, seed, KD
version) so a run can be identified and reproduced later.

## Package Layout

```
src/kd/
├── api.py        # Model facade: one-line fit() for every engine
├── evaluate.py   # evaluate_terms / validate_terms: score terms directly
├── data/         # PDEDataset, synthetic generators, benchmark loaders
├── search/       # sga / dlga / discover / pysr engines + their configs
├── viz/          # VizEngine: HTML reports & figures
└── inspect.py    # preview() dataset sanity checks
```

## Origins & Acknowledgements

The SGA, DLGA, and DISCOVER engines are refactored re-implementations of
algorithms developed in this lab; credit for the methods belongs to the
original works:

- **SGA-PDE**: Chen et al., [SGA-PDE](https://github.com/YuntianChen/SGA-PDE);
  also the source of the bundled benchmark datasets (see [`NOTICE`](NOTICE))
- **DLGA**: Xu et al. 2020
- **DISCOVER**: Du et al., [DISCOVER](https://github.com/menggedu/DISCOVER)

KD also builds on [PySR](https://github.com/MilesCranmer/PySR) (optional
fallback engine), [SymPy](https://github.com/sympy/sympy), and
[PyTorch](https://github.com/pytorch/pytorch).

## License

[Apache-2.0](LICENSE). Copyright 2026 Mao, Hao and the Scientific Artificial
Intelligence Lab.
