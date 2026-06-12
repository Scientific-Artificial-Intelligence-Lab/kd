
from __future__ import annotations

from typing import Any

import torch

from kd.search.discover.config import DiscoverConfig, PINNConfig
from kd.search.discover.runners.pde_registry import PDESpec, TierSettings


def _serialize_scaffold_kwargs(
    scaffold_kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in scaffold_kwargs.items()
    }


def _collocation_count(colloc: dict[str, torch.Tensor]) -> int:
    if not colloc:
        return 0
    first_key = next(iter(colloc))
    return int(colloc[first_key].shape[0])


def _build_burgers_config_payload(
    *,
    settings: TierSettings,
    config: DiscoverConfig,
    pinn_config: PINNConfig,
    colloc: dict[str, torch.Tensor],
    seed: int,
    scaffold_kwargs: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "data_path": str(settings.data_path),
        "n_iterations": config.n_iterations,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "attention": config.attention,
        "stability_selection": config.stability_selection,
        "learning_rate": config.learning_rate,
        "n_cycles": pinn_config.n_cycles,
        "pretrain_epoch": pinn_config.pretrain_epoch,
        "pinn_epoch": pinn_config.pinn_epoch,
        "n_collocation": pinn_config.n_collocation,
        "n_collocation_requested": pinn_config.n_collocation,
        "n_collocation_actual": _collocation_count(colloc),
        "cycle_n_iterations": pinn_config.cycle_n_iterations,
        "coef_pde": pinn_config.coef_pde,
        "collocation_cut_ratio": settings.collocation_cut_ratio,
        "stability_seed": seed,
        "operators": list(settings.operators),
    }
    payload.update(_serialize_scaffold_kwargs(scaffold_kwargs))
    return payload


def _build_chafee_config_payload(
    *,
    settings: TierSettings,
    config: DiscoverConfig,
    pinn_config: PINNConfig,
    scaffold_kwargs: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "n_iterations": config.n_iterations,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "attention": config.attention,
        "attn_length": config.attn_length,
        "stability_selection": config.stability_selection,
        "learning_rate": config.learning_rate,
        "soft_length_loc": config.soft_length_loc,
        "coef_pde": pinn_config.coef_pde,
        "cycle_n_iterations": pinn_config.cycle_n_iterations,
        "n_cycles": pinn_config.n_cycles,
        "pretrain_epoch": pinn_config.pretrain_epoch,
        "pinn_epoch": pinn_config.pinn_epoch,
        "n_collocation": pinn_config.n_collocation,
        "operators": list(settings.operators),
    }
    payload.update(_serialize_scaffold_kwargs(scaffold_kwargs))
    return payload


def _serialize_candidates(candidates: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "expression": c.expression,
            "reward": c.reward,
            "nmse": c.nmse,
            "n_nodes": c.n_nodes,
            "terms": list(c.terms),
        }
        for c in candidates
    ]


def _build_burgers_diagnostics(
    *,
    final_state: Any,
    candidates: list[Any],
    global_best_expression: str | None,
    global_best_reward: float | None,
) -> dict[str, Any]:
    extras = final_state.extras or {}
    return {
        "global_best": {
            "expression": global_best_expression,
            "reward": global_best_reward,
        },
        "stability_selection": extras.get("stability_selection"),
        "final_cycle_candidates": _serialize_candidates(candidates),
    }


def build_pretrain_block(result: Any) -> dict[str, Any]:
    return {
        "train_loss": result.pretrain_result.train_loss,
        "val_loss": result.pretrain_result.val_loss,
        "epochs_run": result.pretrain_result.epochs_run,
        "stopped_early": result.pretrain_result.stopped_early,
    }


def _build_result_block(result: Any) -> dict[str, Any]:
    final = result.final_state
    return {
        "best_reward": final.best_reward,
        "best_expression": final.best_expression,
        "best_terms": final.best_result_terms,
        "best_coefficients": final.best_result_coefficients,
    }








SUPPORTED_PAYLOAD_PDES: frozenset[str] = frozenset({"burgers", "chafee"})


def assemble_payload(
    *,
    spec: PDESpec,
    settings: TierSettings,
    config: DiscoverConfig,
    pinn_config: PINNConfig,
    colloc: dict[str, torch.Tensor],
    seed: int,
    noise_level: float,
    scaffold_kwargs: dict[str, Any],
    result: Any,
    engine: Any,
    tier_name: str,
) -> dict[str, Any]:
    if spec.pde_name == "burgers":
        return _assemble_burgers(
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
    if spec.pde_name == "chafee":
        return _assemble_chafee(
            spec=spec,
            settings=settings,
            config=config,
            pinn_config=pinn_config,
            seed=seed,
            noise_level=noise_level,
            scaffold_kwargs=scaffold_kwargs,
            result=result,
            tier_name=tier_name,
        )
    raise ValueError(
        f"No payload schema registered for PDE {spec.pde_name!r}; "
        f"supported: {sorted(SUPPORTED_PAYLOAD_PDES)}"
    )


def _assemble_burgers(
    *,
    spec: PDESpec,
    settings: TierSettings,
    config: DiscoverConfig,
    pinn_config: PINNConfig,
    colloc: dict[str, torch.Tensor],
    seed: int,
    noise_level: float,
    scaffold_kwargs: dict[str, Any],
    result: Any,
    engine: Any,
    tier_name: str,
) -> dict[str, Any]:
    _ = spec
    config_payload = _build_burgers_config_payload(
        settings=settings,
        config=config,
        pinn_config=pinn_config,
        colloc=colloc,
        seed=seed,
        scaffold_kwargs=scaffold_kwargs,
    )
    diagnostics = _build_burgers_diagnostics(
        final_state=result.final_state,
        candidates=engine.cycle_top_candidates,
        global_best_expression=engine.best_expression,
        global_best_reward=engine.best_reward,
    )
    return {
        "tier": tier_name,
        "seed": seed,
        "noise_level": noise_level,
        "config": config_payload,
        "result": _build_result_block(result),
        "diagnostics": diagnostics,
        "pretrain": build_pretrain_block(result),
        "cycle_metrics": result.cycle_metrics,
    }


def _assemble_chafee(
    *,
    spec: PDESpec,
    settings: TierSettings,
    config: DiscoverConfig,
    pinn_config: PINNConfig,
    seed: int,
    noise_level: float,
    scaffold_kwargs: dict[str, Any],
    result: Any,
    tier_name: str,
) -> dict[str, Any]:
    config_payload = _build_chafee_config_payload(
        settings=settings,
        config=config,
        pinn_config=pinn_config,
        scaffold_kwargs=scaffold_kwargs,
    )
    return {
        "pde": "chafee_infante",
        "ground_truth": spec.ground_truth,
        "tier": tier_name,
        "seed": seed,
        "noise_level": noise_level,
        "config": config_payload,
        "result": _build_result_block(result),
        "pretrain": build_pretrain_block(result),
        "cycle_metrics": result.cycle_metrics,
    }


__all__ = [
    "SUPPORTED_PAYLOAD_PDES",
    "assemble_payload",
    "build_pretrain_block",
]
