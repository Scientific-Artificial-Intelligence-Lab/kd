
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from kd.core.evaluator import Evaluator
from kd.core.executor.context import (
    ExecutionContext,
)
from kd.core.expr import (
    FunctionRegistry,
    PythonExecutor,
)
from kd.core.linear_solve._helpers import (
    SOLVE_DTYPE,
)
from kd.core.linear_solve.least_squares import (
    LeastSquaresSolver,
)
from kd.data.derivatives.autograd import (
    AutogradProvider,
)
from kd.data.derivatives.finite_diff import (
    DX_ZERO_FLOOR,
    UNIFORM_GRID_RTOL,
    FiniteDiffProvider,
    _check_uniform_grid,
    is_uniform_grid,
)
from kd.search.discover.evaluation.reward import compute_reward

if TYPE_CHECKING:
    from kd.data.schema import PDEDataset






COEF_ABS_TOL: float = 1e-4
REWARD_ABS_TOL: float = 1e-5
METRIC_REL_TOL: float = 1e-4

_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
_DATA_DIR: Path = (
    _PROJECT_ROOT / "refs" / "discover" / "dso" / "dso" / "task" / "pde" / "data_new"
)
_BURGERS_MAT: Path = _DATA_DIR / "burgers.mat"
_CHAFEE_DIR: Path = _DATA_DIR
_MIN_BURGERS_BYTES: int = 100_000

_DATA_READY: bool = (
    _BURGERS_MAT.exists()
    and _BURGERS_MAT.stat().st_size > _MIN_BURGERS_BYTES
    and (_CHAFEE_DIR / "chafee_infante_CI.npy").exists()
)
_SKIP_DATA = pytest.mark.skipif(
    not _DATA_READY,
    reason=f"paper PDE data not found under {_DATA_DIR}",
)







@pytest.mark.unit
def test_d1_solve_dtype_contract_is_float64_internal() -> None:
    assert torch.float64 == SOLVE_DTYPE, (
        f"kd lstsq SOLVE_DTYPE changed: {SOLVE_DTYPE!r}; D1 oracles "
        "below were captured assuming a float64 internal solve."
    )
    g = torch.Generator().manual_seed(0)
    theta = torch.randn(64, 3, generator=g, dtype=torch.float32)
    y = torch.randn(64, generator=g, dtype=torch.float32)
    res = LeastSquaresSolver().solve(theta, y)
    assert res.coefficients.dtype == torch.float32, (
        "coefficients must be cast back to the float32 input dtype"
    )
    assert not res.coefficients.requires_grad, (
        "kd 863059d returns coefficients.detach() — a regression here "
        "would leak the solve graph into discover's reward path"
    )
    assert isinstance(res.residual, float)
    assert isinstance(res.r2, float)


@pytest.mark.unit
def test_d1_ill_conditioned_basin_shift_is_pinned() -> None:
    g = torch.Generator().manual_seed(42)
    theta64 = torch.randn(2000, 4, generator=g, dtype=torch.float64)
    theta64[:, 0] *= 1e3
    theta64[:, 3] *= 1e-3
    true = torch.tensor([0.1, -1.0, 0.5, 2.0], dtype=torch.float64)
    y64 = theta64 @ true + 1e-4 * torch.randn(
        2000,
        generator=g,
        dtype=torch.float64,
    )
    theta32, y32 = theta64.float(), y64.float()

    res = LeastSquaresSolver().solve(theta32, y32)
    coefs = [float(c) for c in res.coefficients]


    expected_coefs = [
        0.10000000149011612,
        -1.0000004768371582,
        0.5000040531158447,
        2.0017266273498535,
    ]
    expected_residual = 1.992519514294943e-05
    expected_r2 = 0.9999999999990461

    for i, (got, exp) in enumerate(zip(coefs, expected_coefs, strict=True)):
        assert abs(got - exp) < COEF_ABS_TOL, (
            f"D1 coef[{i}]: got={got} expected={exp} "
            f"|diff|={abs(got - exp):.3e} (tol={COEF_ABS_TOL:.0e}). "
            "kd lstsq numeric behaviour shifted — FINDING."
        )


    assert coefs[3] > 1.0, (
        f"D1 canary: small-scale coef collapsed to {coefs[3]} — the "
        "float64-upcast path is NOT active (would silently regress SR)."
    )
    assert abs(res.residual - expected_residual) / expected_residual < (
        METRIC_REL_TOL
    ), (
        f"D1 residual: got={res.residual} expected~{expected_residual} "
        "— kd residual computation drifted (feeds nmse -> reward)."
    )
    assert abs(res.r2 - expected_r2) < 1e-9, (
        f"D1 r2: got={res.r2} expected~{expected_r2}"
    )


@pytest.mark.unit
def test_d1_reward_landscape_pinned_for_ill_conditioned_candidate() -> None:

    class _R:
        is_valid = True
        complexity = 4

    g = torch.Generator().manual_seed(42)
    theta64 = torch.randn(2000, 4, generator=g, dtype=torch.float64)
    theta64[:, 0] *= 1e3
    theta64[:, 3] *= 1e-3
    true = torch.tensor([0.1, -1.0, 0.5, 2.0], dtype=torch.float64)
    y64 = theta64 @ true + 1e-4 * torch.randn(
        2000,
        generator=g,
        dtype=torch.float64,
    )
    theta32, y32 = theta64.float(), y64.float()
    res = LeastSquaresSolver().solve(theta32, y32)

    ss_tot = float(((y32 - y32.mean()) ** 2).sum().item())
    nmse = res.residual / ss_tot
    r = _R()
    r.nmse = nmse
    reward = compute_reward(r, alpha=0.01)



    expected_nmse = 9.53947419361684e-13
    expected_reward = 0.9599999990

    assert abs(nmse - expected_nmse) / expected_nmse < 1e-3, (
        f"D1 nmse: got={nmse} expected~{expected_nmse} — the residual "
        "drift changed the reward driver."
    )
    assert abs(reward - expected_reward) < REWARD_ABS_TOL, (
        f"D1 reward: got={reward} expected~{expected_reward} "
        f"(tol={REWARD_ABS_TOL:.0e}). kd lstsq drift moved the SR "
        "reward landscape — FINDING (re-baseline only with justification)."
    )


@pytest.mark.unit
@_SKIP_DATA
def test_d1_burgers_gt_expression_metrics_pinned() -> None:
    from kd.search.discover.data.loader import load_burgers_mat

    ds = load_burgers_mat(_BURGERS_MAT)
    ev = _build_fd_evaluator(ds)
    res = ev.evaluate_expression("add(diff2_x(u), mul(u, diff_x(u)))")

    assert res.is_valid
    expected_mse = 7.631660410276517e-08
    expected_nmse = 1.117832691579564e-05
    expected_coefs = {
        "diff2_x(u)": 0.10018764586954026,
        "mul(u, diff_x(u))": -1.0000022081692412,
    }
    _assert_relative("burgers mse", float(res.mse), expected_mse)
    _assert_relative("burgers nmse", float(res.nmse), expected_nmse)
    got = dict(zip(res.terms, [float(c) for c in res.coefficients], strict=True))
    for term, exp in expected_coefs.items():
        assert term in got, f"missing term {term!r}: got {sorted(got)}"
        assert abs(got[term] - exp) < COEF_ABS_TOL, (
            f"burgers coef[{term!r}]: got={got[term]} expected={exp} "
            f"|diff|={abs(got[term] - exp):.3e} — converged-GT basin "
            "moved under frozen kd (FINDING)."
        )


@pytest.mark.unit
@_SKIP_DATA
def test_d1_chafee_gt_expression_metrics_pinned() -> None:
    from kd.search.discover.data.loader import load_chafee_infante_npy

    ds = load_chafee_infante_npy(_CHAFEE_DIR)
    ev = _build_fd_evaluator(ds)
    res = ev.evaluate_expression("add(diff2_x(u), sub(u, n3(u)))")

    assert res.is_valid
    expected_mse = 0.008009744104737315
    expected_nmse = 0.0002526094924855995
    expected_coefs = {
        "diff2_x(u)": 0.9923814610979442,
        "u": -0.9791696135885406,
        "neg(n3(u))": -0.9925827040281605,
    }
    _assert_relative("chafee mse", float(res.mse), expected_mse)
    _assert_relative("chafee nmse", float(res.nmse), expected_nmse)
    got = dict(zip(res.terms, [float(c) for c in res.coefficients], strict=True))
    for term, exp in expected_coefs.items():
        assert term in got, f"missing term {term!r}: got {sorted(got)}"
        assert abs(got[term] - exp) < COEF_ABS_TOL, (
            f"chafee coef[{term!r}]: got={got[term]} expected={exp} "
            f"|diff|={abs(got[term] - exp):.3e} (FINDING)."
        )







@pytest.mark.unit
def test_d2_uniform_grid_rtol_is_1e_4_public() -> None:
    assert UNIFORM_GRID_RTOL == 1e-4, (
        f"kd UNIFORM_GRID_RTOL changed: {UNIFORM_GRID_RTOL!r} "
        "(expected 1e-4 post-863059d)"
    )
    assert DX_ZERO_FLOOR == 1e-30


@pytest.mark.unit
def test_d2_predicate_band_1e6_to_1e4_now_accepts() -> None:
    base = np.linspace(0.0, 1.0, 1000)
    dx0 = base[1] - base[0]
    jit = base.copy()
    jit[500] += 4e-5 * dx0
    t = torch.tensor(jit)

    assert is_uniform_grid(t) is True, (
        "863059d must ACCEPT a 4e-5 rel-dev grid (rtol 1e-4)."
    )
    assert is_uniform_grid(t, rtol=1e-6) is False, (
        "the same grid must FAIL the OLD 1e-6 predicate — this is the "
        "behavioural drift D2 introduces."
    )

    dx = _check_uniform_grid(t, "x")
    assert math.isfinite(dx) and dx > 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "values", "needle"),
    [
        ("non_finite", [-1.7e308, 1.7e308], "non-finite"),
        ("negative_dx", [0.4, 0.3, 0.2, 0.1, 0.0], "decreasing"),
        ("degenerate_dx", [1.0, 1.0, 1.0, 1.0], "degenerate"),
    ],
)
def test_d2_new_hard_rejects_now_raise(
    name: str,
    values: list[float],
    needle: str,
) -> None:
    arr = torch.tensor(values, dtype=torch.float64)
    with pytest.raises(ValueError, match=needle):
        _check_uniform_grid(arr, "x")
    assert is_uniform_grid(arr) is False


@pytest.mark.unit
@_SKIP_DATA
def test_d2_canonical_pde_grids_uniform_under_both_rtols() -> None:
    from kd.search.discover.data.loader import (
        load_burgers_mat,
        load_chafee_infante_npy,
    )

    for ds in (
        load_burgers_mat(_BURGERS_MAT),
        load_chafee_infante_npy(_CHAFEE_DIR),
    ):
        assert ds.axes is not None
        for ax_name, axis in ds.axes.items():
            v = axis.values
            d = np.diff(v.detach().cpu().numpy())
            rel_dev = float(np.max(np.abs(d - d[0])) / abs(d[0]))
            assert rel_dev <= 1e-12, (
                f"{ds.name} axis {ax_name!r} rel-dev {rel_dev:.3e} "
                "exceeds machine precision — paper grid no longer "
                "uniform under frozen kd (FINDING)."
            )
            assert is_uniform_grid(v) is True
            assert is_uniform_grid(v, rtol=1e-6) is True







@pytest.mark.unit
@_SKIP_DATA
@pytest.mark.parametrize(
    ("code", "needle"),
    [
        ("diff_x(u, u_x)", "exactly 1 argument"),
        ("diff_x()", "exactly 1 argument"),
        ("diff_x(arg=u)", "does not accept keyword arguments"),
        ("diff2_x(u, x)", "exactly 1 argument"),
    ],
)
def test_d3_strict_diff_arity_raises_on_grid(
    code: str,
    needle: str,
) -> None:
    from kd.search.discover.data.loader import load_burgers_mat

    ds = load_burgers_mat(_BURGERS_MAT)
    provider = FiniteDiffProvider(ds, max_order=2)
    ctx = ExecutionContext(dataset=ds, derivative_provider=provider)
    ex = PythonExecutor(FunctionRegistry.create_default())
    with pytest.raises(ValueError, match=needle):
        ex.execute(code, ctx)


@pytest.mark.unit
@_SKIP_DATA
def test_d3_malformed_diff_makes_expression_invalid_not_silent() -> None:
    from kd.search.discover.data.loader import load_burgers_mat

    ds = load_burgers_mat(_BURGERS_MAT)
    ev = _build_fd_evaluator(ds)
    res = ev.evaluate_expression("add(diff_x(u, u_x), u)")
    assert res.is_valid is False, (
        "malformed-diff expression must be INVALID under 863059d "
        "(old kd silently scored it as add(diff_x(u), u))."
    )
    assert compute_reward(res, alpha=0.01) == 0.0


@pytest.mark.unit
@_SKIP_DATA
def test_d3_grid_fd_coord_leaf_is_not_autograd_leaf() -> None:
    from kd.search.discover.data.loader import load_burgers_mat

    ds = load_burgers_mat(_BURGERS_MAT)
    provider = FiniteDiffProvider(ds, max_order=2)
    ctx = ExecutionContext(dataset=ds, derivative_provider=provider)
    ex = PythonExecutor(FunctionRegistry.create_default())

    out = ex.execute("diff_x(x)", ctx).value
    assert out.requires_grad is False, (
        "GRID/FD diff_x(x) must NOT be a requires_grad leaf"
    )
    assert out.dim() == 2, "GRID coord must stay field-shaped for FD"
    flat = out.double().reshape(-1)
    assert abs(float(flat.mean()) - 1.0) < 1e-9
    assert float(flat.std()) < 1e-9


@pytest.mark.unit
def test_d3_autograd_coord_leaf_enables_diff_x_of_x() -> None:
    import torch.nn as nn

    from kd.search.discover.pinn.executor import make_pinn_dataset

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(2, 1)

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            stacked = torch.stack([x.reshape(-1), t.reshape(-1)], dim=1)
            out: torch.Tensor = self.lin(stacked).squeeze(-1)
            return out

    torch.manual_seed(0)
    model = _Net()
    n = 64
    coords = {
        "x": torch.linspace(0.0, 1.0, n, requires_grad=True),
        "t": torch.linspace(0.0, 1.0, n, requires_grad=True),
    }
    meta = make_pinn_dataset(["x", "t"], ["u"], lhs_field="u", lhs_axis="t")
    provider = AutogradProvider(model, coords, meta)
    ctx = ExecutionContext(
        dataset=meta,
        derivative_provider=provider,
        device=torch.device("cpu"),
    )
    ex = PythonExecutor(FunctionRegistry.create_default(), max_depth=1000)

    bare_x = ex.execute("x", ctx).value
    assert bare_x.requires_grad is True, (
        "SCATTERED/autograd bare coord must be a requires_grad leaf "
        "(the D3 coord-leaf change) — old kd returned a detached value"
    )
    d_x = ex.execute("diff_x(x)", ctx).value
    flat = d_x.detach().reshape(-1)
    assert abs(float(flat.mean()) - 1.0) < 1e-9, (
        f"autograd diff_x(x) must be 1.0, got mean={float(flat.mean())} "
        "— the coord-leaf graph is broken (D3 regression)."
    )
    assert float(flat.std()) < 1e-9


    with pytest.raises(ValueError, match="exactly 1 argument"):
        ex.execute("diff_x(x, u_x)", ctx)
    with pytest.raises(ValueError, match="does not accept keyword"):
        ex.execute("diff_x(arg=x)", ctx)







def _build_fd_evaluator(dataset: PDEDataset) -> Evaluator:
    provider = FiniteDiffProvider(dataset, max_order=2)
    context = ExecutionContext(dataset=dataset, derivative_provider=provider)
    registry = FunctionRegistry.create_default()
    u_t = provider.get_derivative(
        dataset.lhs_field,
        dataset.lhs_axis,
        order=1,
    ).flatten()
    return Evaluator(
        executor=PythonExecutor(registry),
        solver=LeastSquaresSolver(),
        context=context,
        lhs=u_t,
    )


def _assert_relative(name: str, actual: float, expected: float) -> None:
    denom = abs(expected) if abs(expected) > 0 else 1.0
    rel = abs(actual - expected) / denom
    assert rel < METRIC_REL_TOL, (
        f"{name}: got={actual} expected={expected} "
        f"|rel|={rel:.3e} (tol={METRIC_REL_TOL:.0e}) — kd drift moved "
        "the converged-expression baseline (FINDING)."
    )
