# integrator_golden — old-path golden fixtures for VIZ-M2 (IR-native integrator)

Frozen predicted fields from the **OLD** `integrate_pde(rhs_expr: sympy.Expr, ...)`
sympy/lambdify path, captured **before** the -4 IR-native rewrite. They are the
reference side of the equivalence locks in
`tests/unit/core/test_integrator_ir_native.py`: the NEW
`integrate_pde(rhs: str, ...)` path must reproduce these fields.

## Provenance

- Generator: `scripts/scratch/gen_integrator_golden.py` (one-off, TDD red phase of
- Generated at commit: `8fbd7a2a667f` (old sympy signature still in place),
  2026-07-02, torch 2.10.0 / scipy 1.17.1 / numpy 2.4.2, Python 3.11, CPU float64
- `coordinate_2d_periodic.pt` and `product_rule_1d_periodic.pt` were added by
  the RED-review fix pass at commit `5f0240548b93` (old signature still live),
  same environment; the four pre-existing fixtures were regenerated in the same
  run and verified `torch.equal` to the committed files before being restored
- Dataset builders + RHS strings: `tests/unit/core/_integrator_golden_cases.py`
  (single source of truth shared by generator and tests — datasets are fully
  deterministic, no RNG)
- Bitwise reproducibility of the old path on these inputs was verified at
  generation time (re-run produced `torch.equal` fields)

## Cases

| file | RHS (sympy AND IR) | dataset | solver |
|------|--------------------|---------|--------|
| `burgers_1d_periodic.pt` | `u*u_x + u_xx` | 1D periodic, u0=sin(x), 32x10 | Radau |
| `coordinate_1d_periodic.pt` | `u + x` | 1D periodic, u0=sin(x), 32x10 | Radau |
| `heat_1d_dirichlet.pt` | `u_xx` | 1D Dirichlet, u0=sin(pi x), 32x10 | Radau |
| `advection_2d_periodic.pt` | `u_x + u_y` | 2D periodic, u0=sin(x)cos(y), 16x16x5 | Radau |
| `coordinate_2d_periodic.pt` | `u + x` | 2D periodic, u0=sin(x)cos(y), 16x16x5 | Radau |
| `product_rule_1d_periodic.pt` | `u*u_xx + u_x**2` | 1D periodic, u0=1.5+0.5 sin(x), 32x8 | Radau |

`coordinate_2d_periodic` locks the 2D coordinate-broadcast orientation (task
RED checklist "2D 含坐标 RHS（u + x）通过"). `product_rule_1d_periodic` is the
product-rule expansion of the nested cure case `diff_x(mul(u, u_x))`: besides
its equivalence lock it serves as the numeric reference for the nested cure
test at the loose `NESTED_VS_EXPANDED_*` tolerance (the nested and expanded
discrete forms agree only to O(h^2); see `_integrator_golden_cases.py`).

Each `.pt` holds `{"predicted_field": Tensor(float64), "meta": {...}}` and loads
with `torch.load(path, weights_only=True)`.

## Equivalence (rtol) contract — -4 "数值语义"

Old and new paths are **NOT bitwise identical** (three audited causes: executor
materialises `ast.Constant` as float32 tensors; sympy canonically reorders
Add/Mul while the IR string keeps term order under non-associative float
addition; torch vs numpy transcendental ulp differences). The lock is on the
**final field**:

```python
torch.testing.assert_close(
    new_field, golden_field,
    rtol=1e-6, # EQUIVALENCE_RTOL
    atol=1e-6 * golden_field.abs().max().item(), # EQUIVALENCE_ATOL_SCALE * scale
)
```

`atol` is scaled to the field magnitude so near-zero entries (Dirichlet
boundaries) are compared against an absolute floor instead of a vacuous rtol.

## Regeneration

Only meaningful at a commit where the old sympy signature exists (checkout a
pre--4 commit, e.g. `8fbd7a2a667f`):

```bash
uv run python scripts/scratch/gen_integrator_golden.py
```

Do NOT regenerate against the IR-native implementation — that would turn the
equivalence locks into self-comparisons.
