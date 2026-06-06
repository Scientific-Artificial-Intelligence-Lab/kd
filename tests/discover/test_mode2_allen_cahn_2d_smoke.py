
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor

if TYPE_CHECKING:
    from kd.search.discover.pinn.cycle import PINNCycleResult








_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_PATH: Path = _PROJECT_ROOT / "data" / "allen_cahn_2d_smoke.npz"




_MIN_DATA_BYTES: int = 50_000
_SKIP_REASON = (
    f"Allen-Cahn 2D smoke data not found (or smaller than "
    f"{_MIN_DATA_BYTES} bytes) at {_DATA_PATH}. "
    "Regenerate with: uv run python scripts/generate_allen_cahn_2d.py "
    f"--seed 42 --out {_DATA_PATH}"
)


def _smoke_data_ready() -> bool:
    return _DATA_PATH.exists() and _DATA_PATH.stat().st_size > _MIN_DATA_BYTES


pytestmark = pytest.mark.skipif(not _smoke_data_ready(), reason=_SKIP_REASON)






SEED: int = 42


SMOKE_PRETRAIN_EPOCHS: int = 500
SMOKE_PINN_TRAIN_EPOCHS: int = 100
SMOKE_N_CYCLES: int = 1
SMOKE_N_ITERATIONS: int = 10
SMOKE_BATCH_SIZE: int = 32
SMOKE_MAX_LENGTH: int = 30
SMOKE_N_COLLOCATION: int = 1_000


PRETRAIN_LOSS_CEILING: float = 0.5


PINN_N_LAYERS: int = 4
PINN_N_HIDDEN: int = 20


OBS_SUBSAMPLE_RATIO: float = 0.04
OBS_MIN_POINTS: int = 10



ALLEN_CAHN_OPERATORS: list[str] = [
    "add", "sub", "mul", "div", "n2", "n3", "diff2_x", "diff2_y",
]
STATE_VARS: list[str] = ["u"]
COORD_VARS: list[str] = ["x", "y", "t"]



GROUND_TRUTH_TERMS: list[str] = ["diff2_x(u)", "diff2_y(u)", "u", "n3(u)"]
GROUND_TRUTH_COEFFS: list[float] = [0.0025, 0.0025, 1.0, -1.0]







def _subsample_grid(
    dataset: object, seed: int,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    axes = dataset.axes
    axis_order = dataset.axis_order
    fields = dataset.fields
    axes_tensors = [axes[c].values for c in axis_order]
    grids = torch.meshgrid(*axes_tensors, indexing="ij")
    n_total = grids[0].numel()
    n_obs = max(OBS_MIN_POINTS, int(n_total * OBS_SUBSAMPLE_RATIO))
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n_total, generator=rng)[:n_obs]
    obs_coords = {
        name: grids[i].flatten()[idx].to(torch.float32)
        for i, name in enumerate(axis_order)
    }
    obs_targets = {
        name: fd.values.flatten()[idx].to(torch.float32)
        for name, fd in fields.items()
    }
    return obs_coords, obs_targets


def _axis_bounds(dataset: object) -> dict[str, tuple[float, float]]:
    axes = dataset.axes
    axis_order = dataset.axis_order
    return {
        name: (float(axes[name].values.min()), float(axes[name].values.max()))
        for name in axis_order
    }


@pytest.fixture(scope="module")
def allen_cahn_mode2_result() -> PINNCycleResult:
    from kd.core.evaluator import Evaluator
    from kd.core.executor.context import (
        ExecutionContext,
    )
    from kd.core.expr import (
        FunctionRegistry,
        PythonExecutor,
    )
    from kd.core.linear_solve.least_squares import (
        LeastSquaresSolver,
    )
    from kd.data.derivatives.finite_diff import (
        FiniteDiffProvider,
    )
    from kd.search.discover.builder import build_engine
    from kd.search.discover.config import DiscoverConfig, PINNConfig
    from kd.search.discover.data.allen_cahn_2d import load_allen_cahn_2d
    from kd.search.discover.pinn.collocation import generate_collocation_points
    from kd.search.discover.pinn.cycle import PINNCycleRunner
    from kd.search.discover.pinn.executor import PINNExecutor, make_pinn_dataset
    from kd.search.discover.pinn.model import PINNModel
    from kd.search.discover.tokens.library import LibraryConfig

    dataset = load_allen_cahn_2d(_DATA_PATH)





    provider = FiniteDiffProvider(dataset, max_order=2)
    context = ExecutionContext(
        dataset=dataset, derivative_provider=provider,
    )
    registry = FunctionRegistry.create_default()
    u_t = provider.get_derivative(
        dataset.lhs_field, dataset.lhs_axis, order=1,
    ).flatten()
    evaluator = Evaluator(
        PythonExecutor(registry), LeastSquaresSolver(), context, lhs=u_t,
    )

    pinn_config = PINNConfig(
        number_layer=PINN_N_LAYERS,
        n_hidden=PINN_N_HIDDEN,
        activation="tanh",
        pretrain_epoch=SMOKE_PRETRAIN_EPOCHS,
        pinn_epoch=SMOKE_PINN_TRAIN_EPOCHS,
        n_cycles=SMOKE_N_CYCLES,
        n_collocation=SMOKE_N_COLLOCATION,
        local_sample=False,
        lr=0.001,
        early_stop_patience=200,
    )
    config = DiscoverConfig(
        n_iterations=SMOKE_N_ITERATIONS,
        batch_size=SMOKE_BATCH_SIZE,
        max_length=SMOKE_MAX_LENGTH,
        library=LibraryConfig(
            operators=ALLEN_CAHN_OPERATORS,
            state_vars=STATE_VARS,
            coord_vars=COORD_VARS,
        ),
        num_units=16,
        num_layers=1,
        embedding_dim=4,
        pinn=pinn_config,
    )







    torch.manual_seed(SEED)
    engine = build_engine(config)
    torch.manual_seed(SEED)
    model = PINNModel(COORD_VARS, STATE_VARS, pinn_config)
    pinn_executor = PINNExecutor(registry)
    dataset_meta = make_pinn_dataset(
        COORD_VARS,
        STATE_VARS,
        lhs_field=dataset.lhs_field,
        lhs_axis=dataset.lhs_axis,
    )

    obs_coords, obs_targets = _subsample_grid(dataset, SEED)
    bounds = _axis_bounds(dataset)
    colloc = generate_collocation_points(
        bounds=bounds,
        n_points=pinn_config.n_collocation,
        seed=SEED,
    )

    runner = PINNCycleRunner(
        engine=engine,
        pinn_model=model,
        pinn_executor=pinn_executor,
        initial_evaluator=evaluator,
        observation_coords=obs_coords,
        observation_targets=obs_targets,
        colloc_coords=colloc,
        dataset_metadata=dataset_meta,
        config=config,
        domain_bounds=bounds,
    )
    return runner.run()


@pytest.fixture(scope="module")
def allen_cahn_physics_residual() -> Tensor:
    from kd.core.expr import FunctionRegistry
    from kd.search.discover.config import PINNConfig
    from kd.search.discover.data.allen_cahn_2d import load_allen_cahn_2d
    from kd.search.discover.pinn.executor import PINNExecutor, make_pinn_dataset
    from kd.search.discover.pinn.model import PINNModel

    dataset = load_allen_cahn_2d(_DATA_PATH)
    assert dataset.axes is not None
    assert dataset.axis_order is not None
    assert dataset.fields is not None
    axes = dataset.axes
    axis_order = dataset.axis_order





    axes_tensors = [axes[c].values.to(torch.float32) for c in axis_order]
    grids = torch.meshgrid(*axes_tensors, indexing="ij")
    coords = {
        name: grids[i].reshape(-1).clone().detach().requires_grad_(True)
        for i, name in enumerate(axis_order)
    }

    pinn_config = PINNConfig(
        number_layer=PINN_N_LAYERS,
        n_hidden=PINN_N_HIDDEN,
        activation="tanh",


        pretrain_epoch=0,
        pinn_epoch=0,
        lr=0.001,
        n_cycles=1,
        n_collocation=grids[0].numel(),
        local_sample=False,
    )
    torch.manual_seed(SEED)
    model = PINNModel(COORD_VARS, STATE_VARS, pinn_config)


    coord_stats = {
        name: (axes[name].values.mean(), axes[name].values.std(correction=0))
        for name in COORD_VARS
    }
    field_stats = {
        name: (fd.values.mean(), fd.values.std(correction=0))
        for name, fd in dataset.fields.items()
    }
    model.field_model.set_normalization(coord_stats, field_stats)

    registry = FunctionRegistry.create_default()
    pinn_executor = PINNExecutor(registry)
    dataset_meta = make_pinn_dataset(
        COORD_VARS,
        STATE_VARS,
        lhs_field=dataset.lhs_field,
        lhs_axis=dataset.lhs_axis,
    )




    return pinn_executor.compute_residual(
        model=model,
        terms=GROUND_TRUTH_TERMS,
        coefficients=GROUND_TRUTH_COEFFS,
        coords=coords,
        dataset_metadata=dataset_meta,
        lhs_field="u",
        lhs_axis="t",
    )







class TestMode2AllenCahn2dSmoke:

    @pytest.mark.smoke
    def test_pretrain_final_loss_below_ceiling(
        self,
        allen_cahn_mode2_result: PINNCycleResult,
    ) -> None:
        pretrain_result = allen_cahn_mode2_result.pretrain_result
        assert pretrain_result.epochs_run > 0, (
            f"pretrain reported epochs_run={pretrain_result.epochs_run}; "
            "the trainer never executed a gradient step, so "
            "``train_loss`` below any ceiling is a no-op GREEN"
        )
        assert pretrain_result.train_loss < PRETRAIN_LOSS_CEILING, (
            f"pretrain train_loss={pretrain_result.train_loss:.4f} is not "
            f"below ceiling {PRETRAIN_LOSS_CEILING} "
            f"(epochs_run={pretrain_result.epochs_run})"
        )
        assert pretrain_result.train_loss < pretrain_result.val_loss * 5.0, (
            f"pretrain train_loss={pretrain_result.train_loss:.4f} is more "
            f"than 5x val_loss={pretrain_result.val_loss:.4f}; a "
            "degenerate single-batch output may be leaking through"
        )

    @pytest.mark.smoke
    def test_pinn_cycle_completes_without_raising(
        self,
        allen_cahn_mode2_result: PINNCycleResult,
    ) -> None:
        metrics = allen_cahn_mode2_result.cycle_metrics
        assert len(metrics) == SMOKE_N_CYCLES, (
            f"expected {SMOKE_N_CYCLES} cycle metrics entries, got "
            f"{len(metrics)}"
        )

    @pytest.mark.smoke
    def test_physics_residual_flat_count_matches_u_numel(
        self,
        allen_cahn_physics_residual: Tensor,
    ) -> None:
        from kd.search.discover.data.allen_cahn_2d import load_allen_cahn_2d

        dataset = load_allen_cahn_2d(_DATA_PATH)
        assert dataset.fields is not None
        u_numel = int(dataset.fields["u"].values.numel())
        assert allen_cahn_physics_residual.ndim == 1, (
            f"physics residual must be flat (ndim=1); got shape "
            f"{tuple(allen_cahn_physics_residual.shape)}"
        )
        assert allen_cahn_physics_residual.shape == (u_numel,), (
            f"physics residual flat count {tuple(allen_cahn_physics_residual.shape)} "
            f"does not equal u.numel()={u_numel}; the diff_y autograd "
            f"chain may have altered the scattered-coord count"
        )

    @pytest.mark.smoke
    def test_physics_residual_mean_magnitude_finite(
        self,
        allen_cahn_physics_residual: Tensor,
    ) -> None:
        mean_magnitude = allen_cahn_physics_residual.detach().abs().mean()
        mean_value = float(mean_magnitude.item())
        assert math.isfinite(mean_value), (
            f"physics residual mean magnitude {mean_value} is not finite; "
            f"check diff2_y autograd chain for NaN / Inf propagation"
        )
