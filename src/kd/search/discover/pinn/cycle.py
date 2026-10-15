
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from kd.core.evaluator import Evaluator
from kd.core.executor import ExecutionContext
from kd.core.expr import PythonExecutor
from kd.core.interrupt import SearchInterrupted
from kd.core.linear_solve import SparseSolver
from kd.core.oom import is_cpu_alloc_failure
from kd.data.derivatives.autograd import (
    AutogradProvider,
)
from kd.data.schema import PDEDataset
from kd.search.discover.builder import _make_magnitude_filter
from kd.search.discover.engine import extract_active_terms
from kd.search.discover.pinn._memory_log import _log_memory
from kd.search.discover.stability import stability_select

if TYPE_CHECKING:
    from kd.search.discover.config import DiscoverConfig, PINNConfig
    from kd.search.discover.engine import (
        CandidateSnapshot,
        DiscoverEngine,
        EngineState,
    )
    from kd.search.discover.engine_types import Evaluator as EvaluatorProtocol
    from kd.search.discover.pinn.executor import PINNExecutor
    from kd.search.discover.pinn.model import PINNModel, PretrainResult, TrainResult

logger = logging.getLogger(__name__)

_DEGENERATE_BOUNDS_REL_EPS = 1e-6











_SEED_MASK_32 = 0xFFFFFFFF
_BLAKE2B_PERSON = b"kd-pinn-cycle"
LOCAL_SAMPLE_DOMAIN: bytes = b"local_sample"


def _derive_cycle_seed(base: int, cycle_idx: int, domain: bytes) -> int:
    payload = f"{base}:{cycle_idx}".encode() + b":" + domain
    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=_BLAKE2B_PERSON,
    ).digest()
    return int.from_bytes(digest, "big") & _SEED_MASK_32


def _honest_best_terms(
    final_state: EngineState,
) -> tuple[list[str] | None, list[float] | None]:
    if final_state.best_result_is_valid:
        return final_state.best_result_terms, final_state.best_result_coefficients
    logger.warning(
        "Champion '%s' is gate-invalid (coefficients out of magnitude bounds); "
        "dropping it from the final equation report",
        final_state.best_expression,
    )
    return None, None


@dataclass(frozen=True, slots=True)
class RegeneratedData:

    lhs_detached: Tensor
    provider: AutogradProvider
    colloc_coords: dict[str, Tensor]
    dataset_metadata: PDEDataset


def _validate_metadata_consistency(
    source_dataset: PDEDataset,
    pinn_metadata: PDEDataset,
) -> None:
    for field_name in ("lhs_field", "lhs_axis"):
        source_val = getattr(source_dataset, field_name)
        pinn_val = getattr(pinn_metadata, field_name)
        if source_val != pinn_val:
            raise ValueError(
                f"{field_name} mismatch: initial_evaluator source uses "
                f"'{source_val}', PINN metadata uses '{pinn_val}'. "
                f"Use make_pinn_dataset_from(source) to propagate."
            )

    source_axes = list(source_dataset.axis_order or [])
    pinn_axes = list(pinn_metadata.axis_order or [])
    if source_axes != pinn_axes:
        raise ValueError(
            f"axis_order mismatch: initial_evaluator source uses "
            f"{source_axes}, PINN metadata uses {pinn_axes}. "
            f"Use make_pinn_dataset_from(source) to propagate."
        )


def regenerate_metadata(
    model: nn.Module,
    colloc_coords: dict[str, Tensor],
    dataset_metadata: PDEDataset,
    *,
    lhs_field: str,
    lhs_axis: str,
) -> RegeneratedData:
    regen_coords = _clone_collocation_coords(model, colloc_coords)
    provider = AutogradProvider(model, regen_coords, dataset_metadata)
    lhs = provider.get_derivative(lhs_field, lhs_axis, 1)
    lhs_detached = lhs.detach().reshape(-1)
    return RegeneratedData(
        lhs_detached=lhs_detached,
        provider=provider,
        colloc_coords=regen_coords,
        dataset_metadata=dataset_metadata,
    )


def rebuild_evaluator(
    regen_data: RegeneratedData,
    executor: PythonExecutor,
    solver: SparseSolver,
) -> Evaluator:
    context = ExecutionContext(
        dataset=regen_data.dataset_metadata,
        derivative_provider=regen_data.provider,
        device=regen_data.lhs_detached.device,
    )
    return Evaluator(executor, solver, context, lhs=regen_data.lhs_detached)


def _clone_collocation_coords(
    model: nn.Module,
    colloc_coords: dict[str, Tensor],
) -> dict[str, Tensor]:
    device, dtype = _model_tensor_spec(model)
    return {
        name: tensor.detach()
        .clone()
        .to(device=device, dtype=dtype)
        .requires_grad_(True)
        for name, tensor in colloc_coords.items()
    }


def _model_tensor_spec(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(iter(model.parameters()), None)
    if parameter is None:
        raise ValueError(
            f"{type(model).__name__} has no parameters; _model_tensor_spec "
            "cannot infer device/dtype.",
        )
    return parameter.device, parameter.dtype


@dataclass(frozen=True, slots=True)
class PINNCycleResult:

    final_state: EngineState
    cycle_metrics: list[dict[str, float]]
    pretrain_result: PretrainResult


def _pinn_config(config: DiscoverConfig) -> PINNConfig:
    if config.pinn is None:
        raise ValueError("config.pinn must be set for the PINN cycle")
    return config.pinn


def _is_allocation_failure(exc: Exception) -> bool:
    if isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError)):
        return True
    return is_cpu_alloc_failure(exc)


def cycle_iterations(config: DiscoverConfig, cycle_idx: int) -> int:
    cycle_n_iterations = _pinn_config(config).cycle_n_iterations
    if cycle_idx > 0 and cycle_n_iterations is not None:
        return cycle_n_iterations
    return config.n_iterations


def pretrain_split(
    model: PINNModel,
    obs_coords: dict[str, Tensor],
    obs_targets: dict[str, Tensor],
    pinn_config: PINNConfig,
    *,
    split_seed: int,
) -> PretrainResult:
    first_key = next(iter(obs_coords))
    n = obs_coords[first_key].shape[0]
    ratio = pinn_config.pretrain_val_ratio
    n_val = int(n * ratio)

    if n_val == 0:
        empty_c = {k: v[:0] for k, v in obs_coords.items()}
        empty_t = {k: v[:0] for k, v in obs_targets.items()}
        return model.pretrain(
            coords=obs_coords,
            targets=obs_targets,
            val_coords=empty_c,
            val_targets=empty_t,
            config=pinn_config,
        )

    gen = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n, generator=gen)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    return model.pretrain(
        coords={k: v[train_idx] for k, v in obs_coords.items()},
        targets={k: v[train_idx] for k, v in obs_targets.items()},
        val_coords={k: v[val_idx] for k, v in obs_coords.items()},
        val_targets={k: v[val_idx] for k, v in obs_targets.items()},
        config=pinn_config,
    )


def _gated_best_expression(engine: DiscoverEngine, cycle_idx: int) -> str | None:
    best_expr = engine.best_expression
    if not best_expr:
        logger.warning(
            "No valid expression at cycle %d; skipping PINN training",
            cycle_idx,
        )
        return None
    gated_best = engine.best_result
    if gated_best is None or not gated_best.is_valid:
        logger.warning(
            "Champion expression '%s' is gate-invalid on current evaluator "
            "at cycle %d; skipping PINN training",
            best_expr,
            cycle_idx,
        )
        return None
    return best_expr


def _trainable_terms(
    engine: DiscoverEngine,
    evaluator: EvaluatorProtocol,
    config: DiscoverConfig,
    cycle_idx: int,
) -> tuple[list[str], list[float]]:
    best_expr = _gated_best_expression(engine, cycle_idx)
    if best_expr is None:
        return [], []
    fresh_result = evaluator.evaluate_expression(best_expr)
    result_filter = _make_magnitude_filter(enabled=config.magnitude_filter)
    if result_filter is not None:
        fresh_result = result_filter(fresh_result)
    if not fresh_result.is_valid:
        logger.warning(
            "Best expression invalid on current evaluator at cycle %d; "
            "skipping PINN training",
            cycle_idx,
        )
        return [], []
    terms, coefficients = extract_active_terms(fresh_result)
    if not terms:
        logger.warning(
            "No active terms at cycle %d; skipping PINN training",
            cycle_idx,
        )
    return terms, coefficients


def _restore_weights(
    model: PINNModel,
    saved_state: dict[str, Tensor],
    expression: str,
    cycle_idx: int,
) -> None:
    model.load_state_dict(saved_state)
    logger.warning(
        "PINN training failed for expression '%s' at cycle %d; "
        "restoring model checkpoint and skipping",
        expression,
        cycle_idx,
        exc_info=True,
    )


def run_pinn_phase(
    engine: DiscoverEngine,
    model: PINNModel,
    pinn_executor: PINNExecutor,
    evaluator: EvaluatorProtocol,
    *,
    obs_coords: dict[str, Tensor],
    obs_targets: dict[str, Tensor],
    colloc_coords: dict[str, Tensor],
    dataset_metadata: PDEDataset,
    config: DiscoverConfig,
    cycle_idx: int,
    local_coords: dict[str, Tensor] | None,
    deadline: float | None = None,
) -> tuple[dict[str, float], bool, TrainResult | None]:
    skip = {"best_reward": engine.best_reward}
    terms, coefficients = _trainable_terms(engine, evaluator, config, cycle_idx)
    if not terms:
        return skip, False, None
    saved_state = {k: v.clone() for k, v in model.state_dict().items()}
    try:
        train_result = model.train_pinn(
            terms=terms,
            coefficients=coefficients,
            pinn_executor=pinn_executor,
            observation_coords=obs_coords,
            observation_targets=obs_targets,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset_metadata,
            config=_pinn_config(config),
            local_coords=local_coords,
            deadline=deadline,
        )
    except (KeyError, RuntimeError) as exc:
        if _is_allocation_failure(exc):
            raise
        _restore_weights(model, saved_state, engine.best_expression, cycle_idx)
        return skip, False, None
    metrics = {
        "best_reward": engine.best_reward,
        "data_loss": train_result.data_loss,
        "physics_loss": train_result.physics_loss,
        "total_loss": train_result.total_loss,
    }
    return metrics, True, train_result


def make_local_coords(
    obs_coords: dict[str, Tensor],
    bounds: dict[str, tuple[float, float]],
    pinn_config: PINNConfig,
    cycle_idx: int,
    *,
    local_sample_seed: int | None,
) -> dict[str, Tensor] | None:
    if not pinn_config.local_sample:
        return None
    from kd.search.discover.pinn.collocation import generate_local_samples

    seed: int | None = None
    if local_sample_seed is not None:
        seed = _derive_cycle_seed(
            local_sample_seed,
            cycle_idx,
            LOCAL_SAMPLE_DOMAIN,
        )

    return generate_local_samples(
        observation_coords=obs_coords,
        bounds=bounds,
        multiplier=pinn_config.local_multiplier,
        seed=seed,
    )


def rebuild_after_pinn(
    model: nn.Module,
    colloc_coords: dict[str, Tensor],
    dataset_metadata: PDEDataset,
    *,
    executor: PythonExecutor,
    solver: SparseSolver,
) -> Evaluator:
    regen = regenerate_metadata(
        model,
        colloc_coords,
        dataset_metadata,
        lhs_field=dataset_metadata.lhs_field,
        lhs_axis=dataset_metadata.lhs_axis,
    )
    return rebuild_evaluator(regen, executor, solver)


@dataclass(frozen=True, slots=True)
class _StabilityOutcome:

    expression: str
    reward: float
    terms: list[str] | None
    coefficients: list[float] | None
    vote_counts: list[int]
    ran: bool
    error: str | None


def _evaluate_selected_candidate(
    evaluator: EvaluatorProtocol,
    expression: str,
    config: DiscoverConfig,
) -> tuple[list[str], list[float]]:
    result = evaluator.evaluate_expression(expression)
    result_filter = _make_magnitude_filter(enabled=config.magnitude_filter)
    if result_filter is not None:
        result = result_filter(result)
    if not result.is_valid:
        raise ValueError(
            "Selected stability candidate became invalid on final evaluator."
        )
    terms, coefficients = extract_active_terms(result)
    if not terms:
        raise ValueError("Selected stability candidate produced no active terms.")
    return terms, coefficients


def _select_stable(
    candidates: list[CandidateSnapshot],
    evaluator: Evaluator,
    config: DiscoverConfig,
    stability_seed: int | None,
    kept: _StabilityOutcome,
) -> _StabilityOutcome:
    try:
        result = stability_select(
            candidates,
            evaluator,
            top_k=config.stability_selection,
            rng=(
                np.random.default_rng(stability_seed)
                if stability_seed is not None
                else None
            ),
        )
        terms, coefficients = _evaluate_selected_candidate(
            evaluator,
            result.selected.expression,
            config,
        )
    except SearchInterrupted:


        raise
    except Exception as exc:
        if _is_allocation_failure(exc):
            raise
        logger.warning(
            "Stability selection failed; keeping pre-filter result: %s",
            exc,
        )
        return replace(kept, error=str(exc))
    return _StabilityOutcome(
        expression=result.selected.expression,
        reward=result.selected.reward,
        terms=terms,
        coefficients=coefficients,
        vote_counts=result.vote_counts,
        ran=True,
        error=None,
    )


def _with_stability_outcome(
    final_state: EngineState,
    outcome: _StabilityOutcome,
    stability_seed: int | None,
) -> EngineState:
    extras = dict(final_state.extras or {})
    extras["stability_selection"] = {
        "ran": outcome.ran,
        "seed": stability_seed,
        "pre_filter": {
            "expression": final_state.best_expression,
            "reward": final_state.best_reward,
        },
        "selected": {
            "expression": outcome.expression,
            "reward": outcome.reward,
        },
        "vote_counts": outcome.vote_counts,
        "error": outcome.error,
    }
    return replace(
        final_state,
        best_expression=outcome.expression,
        best_reward=outcome.reward,
        best_result_terms=outcome.terms,
        best_result_coefficients=outcome.coefficients,



        best_result_is_valid=True,
        extras=extras,
    )


def _apply_stability_selection(
    engine: DiscoverEngine,
    final_state: EngineState,
    evaluator: Evaluator,
    config: DiscoverConfig,
    stability_seed: int | None,
) -> EngineState:

    honest_terms, honest_coefficients = _honest_best_terms(final_state)
    if config.stability_selection <= 0:
        return replace(
            final_state,
            best_result_terms=honest_terms,
            best_result_coefficients=honest_coefficients,
            best_result_is_valid=True,
        )
    outcome = _StabilityOutcome(
        expression=final_state.best_expression,
        reward=final_state.best_reward,
        terms=honest_terms,
        coefficients=honest_coefficients,
        vote_counts=[],
        ran=False,
        error=None,
    )
    candidates = engine.cycle_top_candidates
    if len(candidates) > 1:
        outcome = _select_stable(candidates, evaluator, config, stability_seed, outcome)
    return _with_stability_outcome(final_state, outcome, stability_seed)


def finalize_cycle_state(
    engine: DiscoverEngine,
    final_state: EngineState,
    evaluator: Evaluator,
    config: DiscoverConfig,
    *,
    stability_seed: int | None,
    seed_plan: dict[str, int | None],
) -> EngineState:
    final_state = _apply_stability_selection(
        engine,
        final_state,
        evaluator,
        config,
        stability_seed,
    )
    extras = dict(final_state.extras or {})
    extras["seed_plan"] = dict(seed_plan)
    return replace(final_state, extras=extras)


class PINNCycleRunner:

    def __init__(
        self,
        engine: DiscoverEngine,
        pinn_model: PINNModel,
        pinn_executor: PINNExecutor,
        initial_evaluator: Evaluator,
        observation_coords: dict[str, Tensor],
        observation_targets: dict[str, Tensor],
        colloc_coords: dict[str, Tensor],
        dataset_metadata: PDEDataset,
        config: DiscoverConfig,
        domain_bounds: dict[str, tuple[float, float]] | None = None,
        stability_seed: int | None = None,
        pretrain_split_seed: int = 0,
        local_sample_seed: int | None = None,
    ) -> None:
        self.pinn_config = _pinn_config(config)
        _validate_metadata_consistency(
            initial_evaluator.context.dataset,
            dataset_metadata,
        )
        self.engine = engine
        self.pinn_model = pinn_model
        self.pinn_executor = pinn_executor
        self.initial_evaluator = initial_evaluator
        self.observation_coords = observation_coords
        self.observation_targets = observation_targets
        self.colloc_coords = colloc_coords
        self.dataset_metadata = dataset_metadata
        self.config = config
        self.stability_seed = stability_seed
        self.pretrain_split_seed = pretrain_split_seed
        self.local_sample_seed = local_sample_seed
        self.local_bounds = (
            domain_bounds if domain_bounds is not None else _infer_bounds(colloc_coords)
        )

    def planned_search_iterations(self) -> int:
        return sum(
            cycle_iterations(self.config, cycle_idx)
            for cycle_idx in range(self.pinn_config.n_cycles + 1)
        )

    def run(self) -> PINNCycleResult:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _log_memory("cycle_runner_start", logger)
        _log_memory("cycle_0_pretrain_start", logger)
        pretrain_result = pretrain_split(
            self.pinn_model,
            self.observation_coords,
            self.observation_targets,
            self.pinn_config,
            split_seed=self.pretrain_split_seed,
        )
        _log_memory("cycle_0_pretrain_end", logger)
        cycle_metrics: list[dict[str, float]] = []
        evaluator = rebuild_after_pinn(
            self.pinn_model,
            self.colloc_coords,
            self.dataset_metadata,
            executor=self.initial_evaluator.executor,
            solver=self.initial_evaluator.solver,
        )
        n_cycles = self.pinn_config.n_cycles
        for cycle_idx in range(n_cycles):
            self._search(evaluator, cycle_idx)
            metrics, evaluator = self._train_and_rebuild(evaluator, cycle_idx)
            cycle_metrics.append(metrics)
        final_state = self._search(evaluator, n_cycles)
        final_state = finalize_cycle_state(
            self.engine,
            final_state,
            evaluator,
            self.config,
            stability_seed=self.stability_seed,
            seed_plan={
                "pretrain_split_seed": self.pretrain_split_seed,
                "local_sample_seed": self.local_sample_seed,
                "stability_seed": self.stability_seed,
            },
        )
        _log_memory("cycle_runner_end", logger)
        return PINNCycleResult(final_state, cycle_metrics, pretrain_result)

    def _search(self, evaluator: Evaluator, cycle_idx: int) -> EngineState:
        _log_memory(f"cycle_{cycle_idx}_search_start", logger)


        state = self.engine.run_cycle(
            evaluator,
            cycle_iterations(self.config, cycle_idx),
            cycle_idx=cycle_idx,
        )
        _log_memory(f"cycle_{cycle_idx}_search_end", logger)
        return state

    def _train_and_rebuild(
        self,
        evaluator: Evaluator,
        cycle_idx: int,
    ) -> tuple[dict[str, float], Evaluator]:
        _log_memory(f"cycle_{cycle_idx}_pinn_start", logger)
        metrics, pinn_ok, _ = run_pinn_phase(
            self.engine,
            self.pinn_model,
            self.pinn_executor,
            evaluator,
            obs_coords=self.observation_coords,
            obs_targets=self.observation_targets,
            colloc_coords=self.colloc_coords,
            dataset_metadata=self.dataset_metadata,
            config=self.config,
            cycle_idx=cycle_idx,
            local_coords=make_local_coords(
                self.observation_coords,
                self.local_bounds,
                self.pinn_config,
                cycle_idx,
                local_sample_seed=self.local_sample_seed,
            ),
        )
        _log_memory(f"cycle_{cycle_idx}_pinn_end", logger)
        if pinn_ok:
            evaluator = rebuild_after_pinn(
                self.pinn_model,
                self.colloc_coords,
                self.dataset_metadata,
                executor=self.initial_evaluator.executor,
                solver=self.initial_evaluator.solver,
            )
        return metrics, evaluator


def _infer_bounds(
    coords: dict[str, Tensor],
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for name, tensor in coords.items():
        lower = float(tensor.min())
        upper = float(tensor.max())
        if lower >= upper:
            center = lower
            eps = max(abs(center), 1.0) * _DEGENERATE_BOUNDS_REL_EPS
            lower = center - eps
            upper = center + eps
        bounds[name] = (lower, upper)
    return bounds


__all__ = [
    "PINNCycleResult",
    "PINNCycleRunner",
    "RegeneratedData",
    "cycle_iterations",
    "finalize_cycle_state",
    "make_local_coords",
    "pretrain_split",
    "rebuild_after_pinn",
    "rebuild_evaluator",
    "regenerate_metadata",
    "run_pinn_phase",
]
