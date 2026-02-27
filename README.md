<div align="center">

# Knowledge Discovery (KD)

**Symbolic PDE Discovery from Data**

Discover the governing partial differential equations hidden in your data — automatically.

</div>

---

KD is a modular toolkit for symbolic equation discovery, with a strong focus on
partial differential equations (PDEs). It ships four complementary discovery
engines, a unified dataset interface, and built-in publication-quality
visualisation.

## Highlights

- **4 discovery methods** covering sparse regression, genetic algorithms, deep
  reinforcement learning, and physics-informed neural networks
- **Unified `PDEDataset`** — load built-in benchmarks in one line or bring your
  own gridded data
- **Built-in visualisation** — discovered equations, field comparisons, residual
  diagnostics, expression trees, and more
- **N-D support** — works with 1-D, 2-D, and 3-D spatial data out of the box

## Quick Start

### Installation

Requires **Python >= 3.9** and PyTorch.

```bash
pip install -e .

# (optional) enable PySR-based symbolic regression
pip install 'kd[pysr]'
```

### Discover a PDE in 3 lines

```python
from kd.dataset import load_pde
from kd.model import KD_SGA

dataset = load_pde("burgers")
model = KD_SGA(sga_run=20, num=20, depth=4, width=5)
model.fit_dataset(dataset)
print(model.equation_latex())  # u_t = -1.001 u_x u + 0.100 u_xx
```

### Visualise results

```python
from kd.viz import configure, render_equation, plot_field_comparison
from pathlib import Path

configure(save_dir=Path("artifacts/my_run"))
render_equation(model)               # equation as LaTeX-rendered PNG
plot_field_comparison(model, ...)     # predicted vs true field heatmap
```

<div align="center">
<img src="docs/images/field_comparison_dlga.png" width="700" alt="Field comparison">
<p><em>Exact vs predicted solution — KdV equation via KD_DLGA.</em></p>
</div>

## Supported Methods

| Class | Description |
|-------|-------------|
| **KD_SGA** | PDE discovery via sparse genetic algorithm |
| **KD_DLGA** | PDE discovery via deep-learning genetic algorithm |
| **KD_DSCV** | PDE discovery via reinforcement learning and sparse regression |
| **KD_DSCV_SPR** | DSCV variant with PINN for sparse / noisy data |
| **KD_PySR** | General symbolic regression via PySR (optional) |

## Example Gallery

<table>
<tr>
<td align="center"><img src="docs/images/equation_sga.png" width="420"><br><em>Burgers equation — KD_SGA</em></td>
<td align="center"><img src="docs/images/equation_dscv.png" width="420"><br><em>Burgers equation — KD_DSCV</em></td>
</tr>
<tr>
<td align="center"><img src="docs/images/equation_dlga.png" width="420"><br><em>KdV equation — KD_DLGA</em></td>
<td align="center"><img src="docs/images/expression_tree_dscv.png" width="280"><br><em>Discovered expression tree</em></td>
</tr>
</table>

See [`examples/`](examples/) for 14+ runnable scripts covering every model and
visualisation mode.

## Built-in Datasets

```python
from kd.dataset import load_pde, list_available_datasets

print(list_available_datasets())
# ['chafee-infante', 'burgers', 'kdv', 'fisher', ...]

dataset = load_pde("burgers")   # returns a PDEDataset
```

You can also construct a `PDEDataset` from your own NumPy arrays:

```python
from kd.dataset import PDEDataset

dataset = PDEDataset(
    equation_name="my_pde",
    fields_data={"u": u_array},
    coords_1d={"x": x, "t": t},
    axis_order=["x", "t"],
    target_field="u",
    lhs_axis="t",
)
```

## Package Layout

```
kd/
├── base.py        # Shared estimator base class
├── dataset/       # PDEDataset, registry, loaders
├── model/         # KD_SGA, KD_DLGA, KD_DSCV, KD_PySR
├── viz/           # Unified visualisation facade
└── utils/         # Logging, FD helpers, solver utilities
```

## Acknowledgements

KD draws on ideas and code from several open-source projects:

- [DISCOVER](https://github.com/menggedu/DISCOVER)
- [Deep Symbolic Optimization](https://github.com/dso-org/deep-symbolic-optimization)
- [SymPy](https://github.com/sympy/sympy)
- [DeepXDE](https://github.com/lululxvi/deepxde)
- [PySR](https://github.com/MilesCranmer/PySR)
