<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/kd-mark-dark.svg">
  <img src="docs/images/kd-mark.svg" alt="" width="92">
</picture>

<h1>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/kd-title-dark.svg">
  <img src="docs/images/kd-title.svg" alt="Knowledge Discovery" height="42">
</picture>
</h1>

Discovering governing equations from data

[![Documentation](https://img.shields.io/badge/documentation-online-21918c?style=flat-square)](https://scientific-artificial-intelligence-lab.github.io/kd/)
[![License](https://img.shields.io/badge/license-Apache--2.0-46256b?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-46256b?style=flat-square)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-46256b?style=flat-square)](pyproject.toml)

[Documentation](https://scientific-artificial-intelligence-lab.github.io/kd/) ·
[Examples](https://scientific-artificial-intelligence-lab.github.io/kd/examples/) ·
[Algorithms](https://scientific-artificial-intelligence-lab.github.io/kd/algorithms/) ·
[API reference](https://scientific-artificial-intelligence-lab.github.io/kd/api/)

<br>

<img src="docs/images/burgers2d_animation.gif" width="740" alt="2D Burgers field over time: true evolution beside the ground-truth PDE integrated forward">

<sub>2D Burgers (<code>u_t = -u·u_x - u·u_y + 0.01·∇²u</code>): the field's true evolution beside the PDE integrated forward through the platform.<br>Regenerate with <a href="examples/14_field_animation_2d.py"><code>examples/14_field_animation_2d.py</code></a>.</sub>

</div>

<br>

KD discovers the governing partial differential equation from data: give it a
field sampled on a spatiotemporal grid, get back a symbolic PDE. The same
`Model.fit` also takes a plain feature table and returns a scalar expression
`y = f(X)`. Five in-house algorithms (SGA, DLGA, DISCOVER, EqGPT, LLM4ED) and
two external baselines (PySR, PySINDy) run behind one `kd.Model` API, sharing a
single dataset interface, term evaluator, and HTML report.

## Install

Requires Python >= 3.11 and PyTorch >= 2.0.

```bash
git clone -b trunk https://github.com/Scientific-Artificial-Intelligence-Lab/kd.git
cd kd
uv sync
```

## Quick start

```python
import kd

# The bundled Burgers benchmark: a 256 × 201 field on an (x, t) grid.
dataset = kd.load_burgers()

model = kd.Model(algorithm="sga", generations=5, seed=0)
model.fit(dataset)

print(model.best_expr_)    # u_t = -1*mul(u_x, u) + 0.1002*diff2_x(u)
print(model.best_score_)   # -28.78 (AIC, lower is better)
```

The printed expression is KD's canonical function-call notation (funcall IR):
`mul(u_x, u)` is $u\,u_x$ and `diff2_x(u)` is $u_{xx}$, so five generations of
search recover

$$u_t = -u\,u_x + 0.1002\,u_{xx}$$

against a ground truth of $u_t = -u\,u_x + 0.1\,u_{xx}$. The same run narrated
step by step, with the figures it produces, is
[Getting started](https://scientific-artificial-intelligence-lab.github.io/kd/examples/getting_started/).

## Tabular data

`Model.fit` also accepts a `TabularDataset`: a plain feature table, searched for
`y = f(X)` with no fields, no derivatives, and no term library. The example below
uses the bundled TLC-CC measurements, 74 chromatography conditions.

```python
import kd

dataset = kd.load_tlc_cc(target="start")     # X = (R_F, r), y = V_S

model = kd.Model("discover", generations=100, seed=0,
                 batch_size=500, reward_alpha=0.005, max_length=15).fit(dataset)

for entry in model.result_.pareto_front():
    print(entry.complexity, entry.loss, entry.scale, entry.expression)
```

Tabular scoring is scale-free: a Pareto entry holds the raw candidate in
`entry.expression` and its fitted outer coefficient in `entry.scale`. At this
budget and seed the complexity-5 entry `div(r, add(0.0737, R_F))` with scale
6.634 normalizes to $r/(0.151\,R_F + 0.0111)$, against the published
$r/(0.147\,R_F + 0.0114)$.

`"discover"` and `"pysr"` run in tabular mode. Both on the same table:
[`examples/12_symbolic_regression.py`](examples/12_symbolic_regression.py); the
worked comparison, with the Pareto fronts side by side, is
[Column chromatography](https://scientific-artificial-intelligence-lab.github.io/kd/examples/two_engines/).

## Algorithms

The five in-house algorithms are refactored re-implementations of methods
developed in this lab. Swap the `algorithm=` string to switch; all seven share
one dataset interface and one result object.

| Algorithm | `algorithm=` | Origin | Approach |
|-----------|--------------|--------|----------|
| [**SGA**](https://scientific-artificial-intelligence-lab.github.io/kd/algorithms/sga/) | `"sga"` | Chen et al. 2022 (SGA-PDE) | Genetic algorithm over symbolic expression trees |
| [**DLGA**](https://scientific-artificial-intelligence-lab.github.io/kd/algorithms/dlga/) | `"dlga"` | Xu et al. 2020 | Neural-network surrogate + genetic algorithm |
| [**DISCOVER**](https://scientific-artificial-intelligence-lab.github.io/kd/algorithms/discover/) | `"discover"` | Du et al. 2024 | LSTM controller + policy gradient |
| [**EqGPT**](https://scientific-artificial-intelligence-lab.github.io/kd/algorithms/eqgpt/) | `"eqgpt"` | Xu et al. 2025 (EqGPT) | Pretrained generative GPT proposes candidate PDEs, then reward-guided fine-tuning |
| [**LLM4ED**](https://scientific-artificial-intelligence-lab.github.io/kd/algorithms/llm4ed/) | `"llm4ed"` | Du et al. 2024 (LLM4ED) | An LLM proposes candidate equations as text, scored by a sparse-regression reward |
| [**PySR**](https://scientific-artificial-intelligence-lab.github.io/kd/algorithms/pysr/) | `"pysr"` | external, Cranmer 2023 | Genetic programming over expression trees (`uv sync --extra pysr`) |
| [**PySINDy**](https://scientific-artificial-intelligence-lab.github.io/kd/algorithms/pysindy/) | `"pysindy"` | external, de Silva et al. 2020 | Native STLSQ sparse regression over the KD term library (`uv sync --extra pysindy`) |

Beyond field data, `"discover"` and `"pysr"` also run on a feature table (above),
and `"sga"` and `"pysindy"` also accept a sketch (below).

EqGPT needs its pretrained GPT weights, which are not vendored; see
[`examples/16_eqgpt.py`](examples/16_eqgpt.py) for where to place them. LLM4ED
runs fully offline with an injected provider, or against any OpenAI-compatible
API (see [`examples/17_llm4ed.py`](examples/17_llm4ed.py)).

Each algorithm's own settings go through the same call: any field of its config
carries as a keyword argument, so `kd.Model(algorithm="pysindy", threshold=0.2,
normalize_columns=True)` and `kd.Model(algorithm="dlga", pop_size=200,
epsilon=1e-4)` need no per-algorithm call form. An unknown name is rejected with
the accepted ones listed. `kd.instrument_schemas()` returns one row per
algorithm, its config fields with types and defaults plus the facade parameters
(`generations`, `population`, `seed`, ...), so a caller holding only JSON can
configure any algorithm without hardcoding names.

## Datasets

### Simulated PDE datasets

The simulated datasets come from this lab's PDE-discovery papers:
**SGA-PDE** (Chen et al., *Phys. Rev. Research* **4**, 023174, 2022),
**EqGPT** (Xu et al., *Nat Commun* **16**, 10255, 2025) and **LLM4ED**
(Du et al., *Phys. Fluids* **36**, 097121, 2024).

<div align="center">
<img src="docs/images/dataset_gallery.png" width="820" alt="Field snapshots of the bundled simulated PDE datasets">
</div>

The catalog runs from Burgers and KdV to 2D Burgers, Klein-Gordon and the
Fisher family. Load any bundled dataset with `kd.load_burgers()`, or browse the
catalog programmatically with `kd.list_datasets()` / `kd.get_dataset(id)`; each
entry carries its `.source` and `.license` (see [`NOTICE`](NOTICE)).
`kd.generate_burgers_data()`, `kd.generate_diffusion_data()`, ... build
synthetic datasets on demand, and remote entries are fetched with
`kd.load_from_hub(id)` after installing the hub extra (`uv sync --extra hub`).

Governing equation, grid and source for every entry:
[Bundled datasets](https://scientific-artificial-intelligence-lab.github.io/kd/data/datasets/).

### Real-world experimental data

KD also bundles measured data, not only simulation:

| Dataset | Type | Measured quantity | Size | Reference |
|---------|------|-------------------|------|-----------|
| `wave-breaking` | wave-tank experiment (Imperial College London) | surface elevation `η(t, x)` of wave groups approaching breaking | 314,478 points (one of the paper's 12 experiments) | Xu et al., *Nat Commun* **16**, 10255 (2025) |
| `tlc-cc` | automated chromatography experiment | column retention volumes `V_S`, `V_E` vs `(R_F, r)` | 2 tables × 74 conditions | Xu et al., *Nat Commun* **16**, 832 (2025) |

```python
wb = kd.load_wave_breaking()          # η(t, x): scattered wave-tank points
cc = kd.load_tlc_cc(target="start")   # X = (R_F, r), y = V_S
```

Wave breaking is surface elevation of focused wave groups approaching breaking,
reconstructed frame by frame from camera images in the wave-tank experiments of
the EqGPT paper, bundled as scattered `(t, x, η)` points. TLC-CC is
column-chromatography retention volumes measured on an automated platform (192
compounds, 4 g silica columns), aggregated to mean start and end retention
volumes over 74 `(R_F, r)` conditions, ready for KD's scalar
symbolic-regression entries. Experimental background, protocols and references
for both are in the papers and their Supplementary Information
([wave breaking](https://doi.org/10.1038/s41467-025-65114-2),
[TLC-CC](https://doi.org/10.1038/s41467-025-56136-x)).

## Examples

<table>
<tr>
<td width="50%">
<a href="https://scientific-artificial-intelligence-lab.github.io/kd/examples/getting_started/"><img src="docs/images/card-getting-started.png" alt=""></a>
<b><a href="https://scientific-artificial-intelligence-lab.github.io/kd/examples/getting_started/">Getting started</a></b><br>
<sub>Load a bundled benchmark, fit in one call, read the result.</sub>
</td>
<td width="50%">
<a href="https://scientific-artificial-intelligence-lab.github.io/kd/examples/your_own_data/"><img src="docs/images/card-your-own-data.png" alt=""></a>
<b><a href="https://scientific-artificial-intelligence-lab.github.io/kd/examples/your_own_data/">Use your own data</a></b><br>
<sub>Two coordinate arrays and one field array, from NumPy to a fitted equation.</sub>
</td>
</tr>
<tr>
<td width="50%">
<a href="https://scientific-artificial-intelligence-lab.github.io/kd/examples/wave_breaking/"><img src="docs/images/card-wave-breaking.png" alt=""></a>
<b><a href="https://scientific-artificial-intelligence-lab.github.io/kd/examples/wave_breaking/">Breaking waves</a></b><br>
<sub>The published EqGPT equation, reproduced on 12 wave-tank experiments.</sub>
</td>
<td width="50%">
<a href="https://scientific-artificial-intelligence-lab.github.io/kd/examples/two_engines/"><img src="docs/images/card-two-engines.png" alt=""></a>
<b><a href="https://scientific-artificial-intelligence-lab.github.io/kd/examples/two_engines/">Column chromatography</a></b><br>
<sub>Two algorithms on one 74-row table, both recovering the published formula.</sub>
</td>
</tr>
</table>

Nineteen runnable scripts covering every algorithm are in
[`examples/`](examples/), including
[`09_compare_algorithms.py`](examples/09_compare_algorithms.py), which runs the
algorithms on one dataset and ranks the discovered equations on a single NMSE
ruler.

## Discovery with a sketch

A blind search starts from "any equation could be here". When part of the law is
already settled physics, `fit(dataset, sketch=...)` states that part and searches
only the rest. A pinned term is subtracted from the regression target before the
search and restored exactly in the solution, so the search cannot spend budget
rediscovering it. A hole declares how many terms may fill it and what shapes they
may take (derivative-order cap, allowed operators, fields, axes).

The exit is certified: `outcome.solution` is published only when the discovered
law satisfies every clause, and otherwise the run reports
`outcome.best_candidate` plus the clause that failed. `"sga"` and `"pysindy"`
accept sketches today; an algorithm that cannot honor a clause refuses the fit
with a `ValueError` naming that clause instead of searching wider than declared.

The full walkthrough, including how the two backends differ, is
[`examples/21_sketch_discovery.py`](examples/21_sketch_discovery.py) (about 30
seconds).

## More in KD

| | | |
|---|---|---|
| **Score candidate terms** | Fit and score a term set directly, or classify one without fitting, with a per-term rejection report | [`examples/11`](examples/11_evaluate_terms.py) · [API](https://scientific-artificial-intelligence-lab.github.io/kd/api/terms/) |
| **Checkpoint and resume** | Atomic search-state checkpoints during `fit`, plus a `manifest.json` ledger to pick a resume point from | [`examples/10`](examples/10_checkpoint_resume.py) · [Guide](https://scientific-artificial-intelligence-lab.github.io/kd/design/resume/) |
| **Batch experiments** | A declarative algorithm × dataset plan, run into a sealed evidence store with environment fingerprints and consensus reports | [`examples/19`](examples/19_batch_harness.py) · [Guide](https://scientific-artificial-intelligence-lab.github.io/kd/design/batch/) |
| **HTML reports** | Convergence, parity, residual maps, field comparisons, the equation in LaTeX, and each algorithm's own search diagnostics | [`examples/03`](examples/03_visualize.py) · [Guide](https://scientific-artificial-intelligence-lab.github.io/kd/viz/) |
| **Dataset preview** | `kd.preview(dataset)` audits axes, spacing, field statistics and the left-hand side before a search | [Data requirements](https://scientific-artificial-intelligence-lab.github.io/kd/data/shape/) |

<div align="center">
<img src="docs/images/sga_genome_vs_equation_tree.png" width="820" alt="SGA genome tree beside the discovered expression tree">
<br>
<sub>From the report, SGA on the bundled Chafee-Infante dataset. Left: the raw genome of the best evolved individual, still carrying redundant branches. Right: the discovered equation after sparse selection, operators and derivatives only.</sub>
</div>

## Origins and acknowledgements

The SGA, DLGA, DISCOVER, EqGPT, and LLM4ED algorithms are refactored
re-implementations of methods developed in this lab; credit for the methods
belongs to the original works:

- **SGA-PDE**: Chen et al., [SGA-PDE](https://github.com/YuntianChen/SGA-PDE);
  also the source of several bundled datasets (see [`NOTICE`](NOTICE))
- **DLGA**: Xu et al. 2020
- **DISCOVER**: Du et al., [DISCOVER](https://github.com/menggedu/DISCOVER)
- **EqGPT**: Xu et al., [EqGPT](https://github.com/woshixuhao/EqGPT),
  *Nat Commun* **16**, 10255 (2025); also the source of several bundled
  datasets, including the wave-breaking experiments (see [`NOTICE`](NOTICE))
- **LLM4ED**: Du et al., [LLM4ED](https://github.com/menggedu/EDL),
  *Phys. Fluids* **36**, 097121 (2024)

KD also builds on [PySR](https://github.com/MilesCranmer/PySR) and
[PySINDy](https://github.com/dynamicslab/pysindy) (optional external
baselines), [SymPy](https://github.com/sympy/sympy), and
[PyTorch](https://github.com/pytorch/pytorch).

## License

[Apache-2.0](LICENSE). Copyright 2026 Mao, Hao and the Scientific Artificial
Intelligence Lab.
