<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/kd-mark-dark.svg">
  <img src="https://raw.githubusercontent.com/Scientific-Artificial-Intelligence-Lab/kd/trunk/docs/images/kd-mark.svg" alt="" width="72">
</picture>

<h1>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/kd-title-dark.svg">
  <img src="https://raw.githubusercontent.com/Scientific-Artificial-Intelligence-Lab/kd/trunk/docs/images/kd-title.svg" alt="Knowledge Discovery" height="42">
</picture>
</h1>

Discovering governing equations from data

[![PyPI](https://img.shields.io/pypi/v/sail-kd?style=flat-square&color=46256b&label=pypi)](https://pypi.org/project/sail-kd/)
[![Documentation](https://img.shields.io/badge/documentation-online-21918c?style=flat-square)](https://scientific-artificial-intelligence-lab.github.io/kd/)
[![Python](https://img.shields.io/badge/python-3.11%2B-46256b?style=flat-square)](https://github.com/Scientific-Artificial-Intelligence-Lab/kd/blob/trunk/pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-46256b?style=flat-square)](https://github.com/Scientific-Artificial-Intelligence-Lab/kd/blob/trunk/LICENSE)

[Documentation](https://scientific-artificial-intelligence-lab.github.io/kd/) ·
[User guide](https://scientific-artificial-intelligence-lab.github.io/kd/guide/) ·
[Examples](https://scientific-artificial-intelligence-lab.github.io/kd/examples/) ·
[API reference](https://scientific-artificial-intelligence-lab.github.io/kd/api/)

</div>

**KD** discovers the governing partial differential equation from a field
sampled on a spatiotemporal grid. The same `Model.fit` also takes a plain
feature table and returns a scalar expression `y = f(X)`.
Five in-house algorithms (SGA, DLGA, DISCOVER, EqGPT, LLM4ED) and two external
baselines (PySR, PySINDy) run behind one `kd.Model` API, with a common dataset
interface, visualization and HTML reports.

## Install

In an activated Python 3.11+ environment, install with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install sail-kd --torch-backend=auto
```

The option selects the PyTorch build for your hardware. See
[Installation](https://scientific-artificial-intelligence-lab.github.io/kd/start/installation/)
for environment setup, pip installation, CPU/CUDA builds and optional components.
For a source checkout, clone this repository's `trunk` branch and run `uv sync`.

## Quick start

Discover Burgers' equation from the bundled 256 × 201 field:

```python
import kd

dataset = kd.load_burgers()
model = kd.Model(algorithm="sga", generations=5, seed=0)
model.fit(dataset)

print(model.result_.equation)  # u_t = -u*u_x + 0.1002*u_xx
```

The five-generation search recovers

$$u_t = -u\,u_x + 0.1002\,u_{xx}$$

against the benchmark equation $u_t = -u\,u_x + 0.1\,u_{xx}$.

<div align="center">
<img src="https://raw.githubusercontent.com/Scientific-Artificial-Intelligence-Lab/kd/trunk/docs/images/burgers-discovery.gif" width="820" alt="Burgers discovery: a moving time marker on the true field and the true profile compared with the predicted profile from the discovered PDE's forward solution">
</div>

The animation follows the field through time. The marker on the left selects
the profile shown on the right, where the observations are compared with the
forward solution of the discovered PDE. The solution uses this fit's coefficients,
starts from the dataset's initial condition and uses its periodic spatial boundary.
The [full field comparison](https://raw.githubusercontent.com/Scientific-Artificial-Intelligence-Lab/kd/trunk/docs/images/burgers-discovery.png) includes the residual.

[First fit](https://scientific-artificial-intelligence-lab.github.io/kd/examples/getting_started/)
explains the result and its diagnostics. `model.report("out/burgers")` writes an
HTML report with convergence, residuals, field comparisons and search diagnostics.

Your own data enters through `kd.load`, which takes a catalog id or a file path.
Inspect a file's arrays first, then name its coordinates and fields:

```python
kd.inspect_file("run.mat")
dataset = kd.load("run.mat", coords={"x": "x", "t": "t"},
                  fields={"u": "usol"}, lhs="u_t")
```

[Use your own data](https://scientific-artificial-intelligence-lab.github.io/kd/data/shape/#load-a-dataset-from-a-file)
also covers registered layouts such as PDEBench HDF5, sample selection and
`PDEDataset.from_arrays` for arrays already in memory.

## Algorithms

The five in-house algorithms are refactored re-implementations of methods
developed in the Scientific Artificial Intelligence Lab. Swap the `algorithm=`
string to switch; all seven share one dataset interface and one result object.
Each linked page covers the method, its parameters and setup.

| Algorithm | `algorithm=` | Origin | Approach |
|-----------|--------------|--------|----------|
| [**SGA-PDE**](https://scientific-artificial-intelligence-lab.github.io/kd/algorithms/sga/) | `"sga"` | Chen et al. 2022 | Genetic algorithm over symbolic expression trees |
| [**DLGA-PDE**](https://scientific-artificial-intelligence-lab.github.io/kd/algorithms/dlga/) | `"dlga"` | Xu et al. 2020 | Neural-network surrogate + genetic algorithm |
| [**DISCOVER**](https://scientific-artificial-intelligence-lab.github.io/kd/algorithms/discover/) | `"discover"` | Du et al. 2024 | LSTM controller + policy gradient |
| [**EqGPT**](https://scientific-artificial-intelligence-lab.github.io/kd/algorithms/eqgpt/) | `"eqgpt"` | Xu et al. 2025 | Pretrained GPT proposes candidate PDEs, followed by reward-guided fine-tuning |
| [**LLM4ED**](https://scientific-artificial-intelligence-lab.github.io/kd/algorithms/llm4ed/) | `"llm4ed"` | Du et al. 2024 | An LLM proposes candidate equations as text, scored by a sparse-regression reward |
| [**PySR**](https://scientific-artificial-intelligence-lab.github.io/kd/algorithms/pysr/) | `"pysr"` | External, Cranmer 2023 | Genetic programming over expression trees |
| [**PySINDy**](https://scientific-artificial-intelligence-lab.github.io/kd/algorithms/pysindy/) | `"pysindy"` | External, de Silva et al. 2020 | Native STLSQ sparse regression over a KD term library |

DISCOVER and PySR also accept feature tables. SGA-PDE and PySINDy support
[sketches](https://github.com/Scientific-Artificial-Intelligence-Lab/kd/blob/trunk/examples/21_sketch_discovery.py) that specify known terms and constrain
the unknown remainder. Algorithm settings can be passed as `Model` keywords;
[the API reference](https://scientific-artificial-intelligence-lab.github.io/kd/api/config/)
documents their types, defaults and programmatic configuration schemas.

## Tabular data

`Model.fit` also accepts a `TabularDataset`: a plain feature table, searched for
`y = f(X)` with no fields, no derivatives and no term library to supply. The
example below uses the bundled TLC-CC measurements, 74 chromatography conditions.

```python
import kd

dataset = kd.load_tlc_cc(target="start")  # X = (R_F, r), y = V_S
model = kd.Model("discover", generations=100, seed=0,
                 batch_size=500, reward_alpha=0.005, max_length=15).fit(dataset)

for entry in model.result_.pareto_front():
    print(entry.complexity, entry.loss, entry.scale, entry.expression)
```

The Pareto front records accuracy and expression size. Each entry holds the raw
candidate in `entry.expression` and its fitted outer coefficient in `entry.scale`;
both are needed to read the fitted expression. DISCOVER and PySR run in tabular
mode. [Column chromatography](https://scientific-artificial-intelligence-lab.github.io/kd/examples/two_engines/)
compares their fronts and the recovered coefficients with the published formula.

## Examples and datasets

<table>
<tr>
<td width="50%">
<a href="https://scientific-artificial-intelligence-lab.github.io/kd/examples/wave_breaking/"><img src="https://raw.githubusercontent.com/Scientific-Artificial-Intelligence-Lab/kd/trunk/docs/images/card-wave-breaking.png" alt="Wave-tank measurements and the published EqGPT equation"></a>
<b><a href="https://scientific-artificial-intelligence-lab.github.io/kd/examples/wave_breaking/">Breaking waves</a></b><br>
<sub>Reproduce the published EqGPT equation on 12 wave-tank experiments.</sub>
</td>
<td width="50%">
<a href="https://scientific-artificial-intelligence-lab.github.io/kd/examples/two_engines/"><img src="https://raw.githubusercontent.com/Scientific-Artificial-Intelligence-Lab/kd/trunk/docs/images/card-two-engines.png" alt="DISCOVER and PySR Pareto fronts for chromatography measurements"></a>
<b><a href="https://scientific-artificial-intelligence-lab.github.io/kd/examples/two_engines/">Column chromatography</a></b><br>
<sub>Compare DISCOVER and PySR on 74 measured conditions and locate the published formula on each Pareto front.</sub>
</td>
</tr>
</table>

For a longer PDE search, the
[KdV walkthrough](https://scientific-artificial-intelligence-lab.github.io/kd/examples/kdv_walkthrough/)
follows 200 generations of SGA-PDE and explains the discovered terms and report.

The simulated datasets come from this lab's PDE-discovery papers, including
SGA-PDE, EqGPT and LLM4ED. The catalog runs from Burgers and KdV to 2D Burgers,
Klein-Gordon and the Fisher family. KD also bundles two experimental datasets:
wave-tank surface elevations and chromatography retention volumes, used with
the permission of the authors. The
[dataset catalog](https://scientific-artificial-intelligence-lab.github.io/kd/data/datasets/)
provides equations, grids, measurement protocols, sources and licenses;
[the dataset API](https://scientific-artificial-intelligence-lab.github.io/kd/api/datasets/)
also covers synthetic generators and downloads.

[All tutorials](https://scientific-artificial-intelligence-lab.github.io/kd/examples/)
and [runnable scripts](https://github.com/Scientific-Artificial-Intelligence-Lab/kd/blob/trunk/examples/) cover every algorithm, including
[tabular regression](https://github.com/Scientific-Artificial-Intelligence-Lab/kd/blob/trunk/examples/12_symbolic_regression.py) and
[algorithm comparison](https://github.com/Scientific-Artificial-Intelligence-Lab/kd/blob/trunk/examples/09_compare_algorithms.py).

## Discovery with a sketch

When part of the law is already known, `fit(dataset, sketch=...)` states that
part and searches the remainder. A pinned term is subtracted from the regression
target before the search and restored with its coefficient in the solution.
An anchored term fixes the structure while leaving its coefficient free.
A hole declares how many terms may fill it and what forms they may take,
including derivative order, operators, fields and axes.

SGA-PDE and PySINDy accept sketches. The
[Burgers sketch example](https://github.com/Scientific-Artificial-Intelligence-Lab/kd/blob/trunk/examples/21_sketch_discovery.py) pins the advection term,
searches for the missing diffusion term and explains how the result is checked
against the sketch and the data.

## Further use

| Capability | Guide or example |
|------------|------------------|
| **Experimental controller**: configure a language-model endpoint, search a dataset and receive a Markdown report with `kd-agent` | [Controller guide](https://scientific-artificial-intelligence-lab.github.io/kd/guide/controller/) |
| **Figures and HTML reports**: inspect convergence, residuals, field comparisons and algorithm-specific diagnostics | [Visualization](https://scientific-artificial-intelligence-lab.github.io/kd/viz/) |
| **Checkpoint and resume**: save search state and continue from a recorded checkpoint | [Resume guide](https://scientific-artificial-intelligence-lab.github.io/kd/design/resume/) |
| **Batch experiments**: run an algorithm × dataset plan with recorded environments and consensus reports | [Batch guide](https://scientific-artificial-intelligence-lab.github.io/kd/design/batch/) |
| **Term evaluation**: fit candidate terms or validate them with per-term rejection reports | [Term API](https://scientific-artificial-intelligence-lab.github.io/kd/api/terms/) |
| **Dataset preview**: inspect axes, spacing, field statistics and the left-hand side before searching | [Data requirements](https://scientific-artificial-intelligence-lab.github.io/kd/data/shape/) |

## Citation and acknowledgements

For research using KD, cite the original works for the methods and datasets you
use. The five lab algorithms are refactored re-implementations of these methods;
their algorithm pages provide the papers and references:

- **SGA-PDE**: Chen et al., *Phys. Rev. Research* **4**, 023174 (2022);
  [original code](https://github.com/YuntianChen/SGA-PDE).
- **DLGA-PDE**: Xu et al. (2020);
  [paper and references](https://scientific-artificial-intelligence-lab.github.io/kd/algorithms/dlga/#references).
- **DISCOVER**: Du et al. (2024);
  [original code](https://github.com/menggedu/DISCOVER).
- **EqGPT**: Xu et al., *Nat Commun* **16**, 10255 (2025);
  [original code](https://github.com/woshixuhao/EqGPT).
- **LLM4ED**: Du et al., *Phys. Fluids* **36**, 097121 (2024);
  [original code](https://github.com/menggedu/EDL).

KD also builds on [PySR](https://github.com/MilesCranmer/PySR),
[PySINDy](https://github.com/dynamicslab/pysindy),
[SymPy](https://github.com/sympy/sympy) and
[PyTorch](https://github.com/pytorch/pytorch).
Dataset attribution and redistribution terms are in [NOTICE](https://github.com/Scientific-Artificial-Intelligence-Lab/kd/blob/trunk/NOTICE).

## License

[Apache-2.0](https://github.com/Scientific-Artificial-Intelligence-Lab/kd/blob/trunk/LICENSE). Copyright 2026 Mao, Hao and the Scientific Artificial
Intelligence Lab.
