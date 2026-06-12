
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

import torch

from kd.search.discover.config import DiscoverConfig, PINNConfig
from kd.search.discover.runners.mode2_helpers import resolve_device
from kd.search.discover.runners.mode2_payload import (
    SUPPORTED_PAYLOAD_PDES,
    assemble_payload,
)
from kd.search.discover.runners.pde_registry import (
    PDE_REGISTRY,
    PROJECT_ROOT,
    PDESpec,
    TierSettings,
)
from kd.search.discover.tokens.library import LibraryConfig

logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "refs" / "baseline" / "results"
OBS_RATIO = 0.04
MIN_OBS_POINTS = 10








PINN_LAYERS = 8
PINN_HIDDEN = 20
PINN_ACTIVATION: Literal["tanh", "sin", "relu"] = "tanh"
NUM_UNITS = 32
NUM_LAYERS = 1
EMBEDDING_DIM = 8
REWARD_ALPHA = 0.01
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
_FD_MAX_ORDER = 2


def _resolve_spec_and_settings(
    pde: str,
    tier: str,
) -> tuple[PDESpec, TierSettings]:
    if pde not in PDE_REGISTRY:
        raise KeyError(f"Unknown PDE {pde!r}; known: {sorted(PDE_REGISTRY)}")
    spec = PDE_REGISTRY[pde]
    if tier not in spec.presets:
        raise KeyError(
            f"Unknown tier {tier!r} for PDE {pde!r}; known: {sorted(spec.presets)}"
        )
    return spec, spec.presets[tier]


def build_configs(
    pde: str,
    tier: str,
    seed: int,
    *,
    scaffold_kwargs: dict[str, Any] | None = None,
) -> tuple[DiscoverConfig, PINNConfig]:
    spec, settings = _resolve_spec_and_settings(pde, tier)
    _ = seed
    pinn_config = PINNConfig(
        number_layer=PINN_LAYERS,
        n_hidden=PINN_HIDDEN,
        activation=PINN_ACTIVATION,
        pretrain_epoch=settings.pretrain_epoch,
        pinn_epoch=settings.pinn_epoch,
        lr=settings.lr,
        coef_pde=settings.coef_pde,
        n_cycles=settings.n_cycles,
        n_collocation=settings.n_collocation,
        local_sample=True,
        early_stop_patience=settings.early_stop_patience,
        cycle_n_iterations=settings.cycle_n_iterations,
    )
    config = DiscoverConfig(
        n_iterations=settings.n_iterations,
        batch_size=settings.batch_size,
        max_length=settings.max_length,
        library=LibraryConfig(
            operators=list(settings.operators),
            state_vars=list(spec.state_vars),
            coord_vars=list(spec.coord_vars),
        ),
        num_units=NUM_UNITS,
        num_layers=NUM_LAYERS,
        embedding_dim=EMBEDDING_DIM,
        attention=settings.attention,
        attn_length=settings.attn_length,
        learning_rate=settings.controller_learning_rate,
        epsilon=settings.epsilon,
        entropy_weight=settings.entropy_weight,
        entropy_gamma=settings.entropy_gamma,
        reward_alpha=REWARD_ALPHA,
        soft_length_loc=settings.soft_length_loc,
        soft_length_scale=settings.soft_length_scale,
        stability_selection=settings.stability_selection,
        pinn=pinn_config,
        **(scaffold_kwargs or {}),
    )
    return config, pinn_config


def make_observation_data(
    noisy_dataset: Any,
    seed: int,
    *,
    device: torch.device,
    obs_ratio: float = OBS_RATIO,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    u_noisy = noisy_dataset.fields["u"].values
    x_vals = noisy_dataset.axes["x"].values
    t_vals = noisy_dataset.axes["t"].values
    big_x, big_t = torch.meshgrid(x_vals, t_vals, indexing="ij")
    n_total = big_x.numel()
    n_obs = max(MIN_OBS_POINTS, int(n_total * obs_ratio))
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n_total, generator=rng)[:n_obs]
    obs_coords = {
        "x": big_x.flatten()[idx].float().to(device=device),
        "t": big_t.flatten()[idx].float().to(device=device),
    }
    obs_targets = {
        "u": u_noisy.flatten()[idx].float().to(device=device),
    }
    pct = 100 * n_obs / n_total
    logger.info(
        "Observations: %d / %d points (%.1f%%)",
        n_obs,
        n_total,
        pct,
    )
    return obs_coords, obs_targets


def _collocation_count(colloc: dict[str, torch.Tensor]) -> int:
    if not colloc:
        return 0
    first_key = next(iter(colloc))
    return int(colloc[first_key].shape[0])


def _format_mode2_start(
    *,
    pde_name: str,
    n_cycles: int,
    total_search_iter: int,
    pretrain_epoch: int,
    pinn_epoch: int,
) -> str:
    return (
        f"Starting MODE2 pde={pde_name} — "
        f"{n_cycles + 1} search passes ({n_cycles} cycles + 1 final), "
        f"{total_search_iter} total search iter, "
        f"pretrain={pretrain_epoch}, pinn={pinn_epoch}/cycle"
    )


def ensure_payload_schema(spec: PDESpec) -> None:
    if spec.pde_name in SUPPORTED_PAYLOAD_PDES:
        return
    raise ValueError(
        f"No MODE2 payload schema registered for PDE {spec.pde_name!r}; "
        f"supported: {sorted(SUPPORTED_PAYLOAD_PDES)}. Failing fast: the "
        "run would complete but results could not be assembled/saved."
    )


def run_pipeline(
    pde: str,
    tier: str,
    seed: int,
    *,
    noise_level: float | None = None,
    noise_scale: str = "std",
    scaffold_kwargs: dict[str, Any] | None = None,
    device: str = "auto",
    diagnostic_scaffold_on: bool = False,
) -> dict[str, Any]:
    spec, settings = _resolve_spec_and_settings(pde, tier)
    ensure_payload_schema(spec)
    effective_noise = (
        spec.default_noise_level if noise_level is None else float(noise_level)
    )
    scaffold_kwargs = scaffold_kwargs or {}
    resolved_device = resolve_device(device)

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger.info(
        "PDE=%s tier=%s seed=%d device=%s noise=%.2f",
        pde,
        tier,
        seed,
        resolved_device,
        effective_noise,
    )
    logger.info("Diagnostic scaffold: %s", diagnostic_scaffold_on)

    if not settings.data_path.exists():
        raise FileNotFoundError(f"{pde} data not found at {settings.data_path}")

    payload, elapsed = _execute_pipeline(
        spec=spec,
        settings=settings,
        tier_name=tier,
        seed=seed,
        noise_level=effective_noise,
        noise_scale=noise_scale,
        scaffold_kwargs=scaffold_kwargs,
        device=resolved_device,
    )
    payload["elapsed_seconds"] = elapsed
    return payload


def _execute_pipeline(
    *,
    spec: PDESpec,
    settings: TierSettings,
    tier_name: str,
    seed: int,
    noise_level: float,
    noise_scale: str,
    scaffold_kwargs: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], float]:
    deps = _import_pipeline_deps()
    clean_dataset = spec.load_data(tier_name)
    noisy_dataset = deps["add_gaussian_noise"](
        clean_dataset,
        noise_level,
        seed,
        scale=noise_scale,
    )
    evaluator, registry = _build_initial_evaluator(noisy_dataset, deps)

    config, pinn_config = build_configs(
        pde=spec.pde_name,
        tier=tier_name,
        seed=seed,
        scaffold_kwargs=scaffold_kwargs,
    )

    torch.manual_seed(seed)
    engine = deps["build_engine"](config)
    model = deps["PINNModel"](
        list(spec.coord_vars),
        list(spec.state_vars),
        pinn_config,
        device=device,
    )
    pinn_executor = deps["PINNExecutor"](registry)
    dataset_meta = deps["make_pinn_dataset"](
        list(spec.coord_vars),
        list(spec.state_vars),
        lhs_field="u",
        lhs_axis="t",
    )
    obs_coords, obs_targets = make_observation_data(
        noisy_dataset,
        seed,
        device=device,
    )
    colloc, x_range, t_range = _build_collocation(
        noisy_dataset,
        settings,
        pinn_config,
        seed,
        device,
        deps,
    )

    runner = deps["PINNCycleRunner"](
        engine=engine,
        pinn_model=model,
        pinn_executor=pinn_executor,
        initial_evaluator=evaluator,
        observation_coords=obs_coords,
        observation_targets=obs_targets,
        colloc_coords=colloc,
        dataset_metadata=dataset_meta,
        config=config,
        stability_seed=seed,
        local_sample_seed=seed,
        domain_bounds={"x": x_range, "t": t_range},
    )
    logger.info(
        "%s",
        _format_mode2_start(
            pde_name=spec.pde_name,
            n_cycles=pinn_config.n_cycles,
            total_search_iter=runner.planned_search_iterations(),
            pretrain_epoch=pinn_config.pretrain_epoch,
            pinn_epoch=pinn_config.pinn_epoch,
        ),
    )
    start = time.time()
    result = runner.run()
    elapsed = time.time() - start
    payload = assemble_payload(
        spec=spec,
        settings=settings,
        config=config,
        pinn_config=pinn_config,
        colloc=colloc,
        seed=seed,
        noise_level=noise_level,
        scaffold_kwargs=scaffold_kwargs,
        result=result,
        engine=engine,
        tier_name=tier_name,
    )
    return payload, elapsed


def _import_pipeline_deps() -> dict[str, Any]:
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
    from kd.search.discover.data.loader import add_gaussian_noise
    from kd.search.discover.pinn.collocation import generate_collocation_points
    from kd.search.discover.pinn.cycle import PINNCycleRunner
    from kd.search.discover.pinn.executor import PINNExecutor, make_pinn_dataset
    from kd.search.discover.pinn.model import PINNModel

    return {
        "Evaluator": Evaluator,
        "ExecutionContext": ExecutionContext,
        "FunctionRegistry": FunctionRegistry,
        "PythonExecutor": PythonExecutor,
        "LeastSquaresSolver": LeastSquaresSolver,
        "FiniteDiffProvider": FiniteDiffProvider,
        "build_engine": build_engine,
        "add_gaussian_noise": add_gaussian_noise,
        "generate_collocation_points": generate_collocation_points,
        "PINNCycleRunner": PINNCycleRunner,
        "PINNExecutor": PINNExecutor,
        "make_pinn_dataset": make_pinn_dataset,
        "PINNModel": PINNModel,
    }


def _build_initial_evaluator(
    noisy_dataset: Any,
    deps: dict[str, Any],
) -> tuple[Any, Any]:
    provider = deps["FiniteDiffProvider"](
        noisy_dataset,
        max_order=_FD_MAX_ORDER,
    )
    context = deps["ExecutionContext"](
        dataset=noisy_dataset,
        derivative_provider=provider,
    )
    registry = deps["FunctionRegistry"].create_default()
    u_t = provider.get_derivative("u", "t", order=1).flatten()
    evaluator = deps["Evaluator"](
        deps["PythonExecutor"](registry),
        deps["LeastSquaresSolver"](),
        context,
        lhs=u_t,
    )
    return evaluator, registry


def _build_collocation(
    noisy_dataset: Any,
    settings: TierSettings,
    pinn_config: PINNConfig,
    seed: int,
    device: torch.device,
    deps: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], tuple[float, float], tuple[float, float]]:
    x_vals = noisy_dataset.axes["x"].values
    t_vals = noisy_dataset.axes["t"].values
    x_range = (float(x_vals.min()), float(x_vals.max()))
    t_range = (float(t_vals.min()), float(t_vals.max()))
    colloc = deps["generate_collocation_points"](
        bounds={"x": x_range, "t": t_range},
        n_points=pinn_config.n_collocation,
        cut_ratio=settings.collocation_cut_ratio,
        seed=seed,
        device=device,
    )
    logger.info(
        "Collocation: %d / %d in x=[%.1f,%.1f] t=[%.1f,%.1f] cut=%.2f",
        _collocation_count(colloc),
        pinn_config.n_collocation,
        *x_range,
        *t_range,
        settings.collocation_cut_ratio,
    )
    return colloc, x_range, t_range


def save_payload(
    spec: PDESpec,
    tier: str,
    seed: int,
    payload: dict[str, Any],
    *,
    output_dir: Path | None = None,
) -> Path:
    out_dir = output_dir if output_dir is not None else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"mode2_{spec.output_prefix}_{tier}_seed{seed}.json"
    out_path = out_dir / filename
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return out_path


__all__ = [
    "OBS_RATIO",
    "OUTPUT_DIR",
    "build_configs",
    "ensure_payload_schema",
    "make_observation_data",
    "run_pipeline",
    "save_payload",
]
