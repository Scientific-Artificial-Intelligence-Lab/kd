
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from kd.core.evaluator import Evaluator
from kd.core.executor import ExecutionContext
from kd.core.expr import PythonExecutor
from kd.core.linear_solve import SparseSolver
from kd.data.derivatives.autograd import (
    AutogradProvider,
)
from kd.data.schema import PDEDataset
from kd.search.discover.builder import _make_magnitude_filter
from kd.search.discover.engine import extract_active_terms
from kd.search.discover.pinn._memory_log import _log_memory
from kd.search.discover.stability import stability_select

if TYPE_CHECKING:
    from kd.search.discover.config import DiscoverConfig
    from kd.search.discover.engine import DiscoverEngine, EngineState
    from kd.search.discover.engine_types import Evaluator as EvaluatorProtocol
    from kd.search.discover.pinn.executor import PINNExecutor
    from kd.search.discover.pinn.model import PINNModel, PretrainResult

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


def _split_obs_data(
    data: dict[str, Tensor],
    val_ratio: float,
    seed: int = 0,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    first_key = next(iter(data))
    n = data[first_key].shape[0]
    n_val = int(n * val_ratio)
    if n_val == 0:
        empty: dict[str, Tensor] = {k: v[:0] for k, v in data.items()}
        return dict(data), empty
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train = {k: v[train_idx] for k, v in data.items()}
    val = {k: v[val_idx] for k, v in data.items()}
    return train, val


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


class PINNCycleRunner:

    def __init__(
        self,
        engine: DiscoverEngine,
        pinn_model: PINNModel,
        pinn_executor: PINNExecutor,
        initial_evaluator: EvaluatorProtocol,
        observation_coords: dict[str, Tensor],
        observation_targets: dict[str, Tensor],
        colloc_coords: dict[str, Tensor],
        dataset_metadata: PDEDataset,
        config: DiscoverConfig,
        domain_bounds: dict[str, tuple[float, float]] | None = None,
        stability_seed: int | None = None,
        source_dataset: PDEDataset | None = None,
        pretrain_split_seed: int = 0,
        local_sample_seed: int | None = None,
    ) -> None:
        if config.pinn is None:
            raise ValueError("config.pinn must be set for PINNCycleRunner")







        effective_source = source_dataset
        if effective_source is None:
            ctx = getattr(initial_evaluator, "context", None)
            effective_source = (
                getattr(ctx, "dataset", None) if ctx is not None else None
            )
        if effective_source is not None:
            _validate_metadata_consistency(effective_source, dataset_metadata)
        self._engine = engine
        self._pinn_model = pinn_model
        self._pinn_executor = pinn_executor
        self._evaluator = initial_evaluator
        self._obs_coords = observation_coords
        self._obs_targets = observation_targets
        self._colloc_coords = colloc_coords
        self._dataset_metadata = dataset_metadata
        self._config = config
        self._pinn_config = config.pinn
        self._stability_seed = stability_seed




        self._pretrain_split_seed = pretrain_split_seed




        self._local_sample_seed = local_sample_seed










        self._local_bounds = (
            domain_bounds if domain_bounds is not None else _infer_bounds(colloc_coords)
        )

    def _cycle_iterations(self, cycle_idx: int) -> int:
        if cycle_idx > 0 and self._pinn_config.cycle_n_iterations is not None:
            return self._pinn_config.cycle_n_iterations
        return self._config.n_iterations

    def planned_search_iterations(self) -> int:
        return sum(
            self._cycle_iterations(cycle_idx)
            for cycle_idx in range(self._pinn_config.n_cycles + 1)
        )

    def run(self) -> PINNCycleResult:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _log_memory("cycle_runner_start", logger)
        _log_memory("cycle_0_pretrain_start", logger)
        pretrain_result = self._pretrain()
        _log_memory("cycle_0_pretrain_end", logger)
        cycle_metrics: list[dict[str, float]] = []
        evaluator = self._rebuild_after_pinn()

        for cycle_idx in range(self._pinn_config.n_cycles):
            _log_memory(f"cycle_{cycle_idx}_search_start", logger)


            self._engine.run_cycle(
                evaluator,
                self._cycle_iterations(cycle_idx),
                cycle_idx=cycle_idx,
            )
            _log_memory(f"cycle_{cycle_idx}_search_end", logger)
            _log_memory(f"cycle_{cycle_idx}_pinn_start", logger)
            metrics, pinn_ok = self._run_pinn_phase(cycle_idx, evaluator)
            _log_memory(f"cycle_{cycle_idx}_pinn_end", logger)
            cycle_metrics.append(metrics)
            if pinn_ok:
                evaluator = self._rebuild_after_pinn()

        n_cycles = self._pinn_config.n_cycles
        _log_memory(f"cycle_{n_cycles}_search_start", logger)
        final_state = self._engine.run_cycle(
            evaluator,
            self._cycle_iterations(n_cycles),
            cycle_idx=n_cycles,
        )
        _log_memory(f"cycle_{n_cycles}_search_end", logger)
        final_state = self._finalize_run(final_state, evaluator)
        _log_memory("cycle_runner_end", logger)
        return PINNCycleResult(
            final_state=final_state,
            cycle_metrics=cycle_metrics,
            pretrain_result=pretrain_result,
        )

    def _pretrain(self) -> PretrainResult:
        first_key = next(iter(self._obs_coords))
        n = self._obs_coords[first_key].shape[0]
        ratio = self._pinn_config.pretrain_val_ratio
        n_val = int(n * ratio)

        if n_val == 0:
            empty_c = {k: v[:0] for k, v in self._obs_coords.items()}
            empty_t = {k: v[:0] for k, v in self._obs_targets.items()}
            return self._pinn_model.pretrain(
                coords=self._obs_coords,
                targets=self._obs_targets,
                val_coords=empty_c,
                val_targets=empty_t,
                config=self._pinn_config,
            )

        gen = torch.Generator().manual_seed(self._pretrain_split_seed)
        perm = torch.randperm(n, generator=gen)
        train_idx, val_idx = perm[n_val:], perm[:n_val]

        return self._pinn_model.pretrain(
            coords={k: v[train_idx] for k, v in self._obs_coords.items()},
            targets={k: v[train_idx] for k, v in self._obs_targets.items()},
            val_coords={k: v[val_idx] for k, v in self._obs_coords.items()},
            val_targets={k: v[val_idx] for k, v in self._obs_targets.items()},
            config=self._pinn_config,
        )

    def _run_pinn_phase(
        self,
        cycle_idx: int,
        evaluator: EvaluatorProtocol,
    ) -> tuple[dict[str, float], bool]:
        skip = {"best_reward": self._engine.best_reward}
        best_expr = self._engine.best_expression
        if not best_expr:
            logger.warning(
                "No valid expression at cycle %d; skipping PINN training",
                cycle_idx,
            )
            return skip, False











        gated_best = self._engine.best_result
        if gated_best is None or not gated_best.is_valid:
            logger.warning(
                "Champion expression '%s' is gate-invalid on current evaluator "
                "at cycle %d; skipping PINN training",
                best_expr,
                cycle_idx,
            )
            return skip, False











        fresh_result = evaluator.evaluate_expression(best_expr)
        result_filter = _make_magnitude_filter(enabled=self._config.magnitude_filter)
        if result_filter is not None:
            fresh_result = result_filter(fresh_result)
        if not fresh_result.is_valid:
            logger.warning(
                "Best expression invalid on current evaluator at cycle %d; "
                "skipping PINN training",
                cycle_idx,
            )
            return skip, False

        terms, coefficients = extract_active_terms(fresh_result)
        if not terms:
            logger.warning(
                "No active terms at cycle %d; skipping PINN training",
                cycle_idx,
            )
            return skip, False

        saved_state = {k: v.clone() for k, v in self._pinn_model.state_dict().items()}
        try:
            train_result = self._pinn_model.train_pinn(
                terms=terms,
                coefficients=coefficients,
                pinn_executor=self._pinn_executor,
                observation_coords=self._obs_coords,
                observation_targets=self._obs_targets,
                colloc_coords=self._colloc_coords,
                dataset_metadata=self._dataset_metadata,
                config=self._pinn_config,
                local_coords=self._make_local_coords(cycle_idx),
            )
        except (KeyError, RuntimeError) as exc:

            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise



            self._pinn_model.load_state_dict(saved_state)
            logger.warning(
                "PINN training failed for expression '%s' at cycle %d; "
                "restoring model checkpoint and skipping",
                self._engine.best_expression,
                cycle_idx,
                exc_info=True,
            )
            return skip, False

        return {
            "best_reward": self._engine.best_reward,
            "data_loss": train_result.data_loss,
            "physics_loss": train_result.physics_loss,
            "total_loss": train_result.total_loss,
        }, True

    def _finalize_with_stability_selection(
        self,
        final_state: EngineState,
        evaluator: Evaluator,
    ) -> EngineState:

        honest_terms, honest_coefficients = _honest_best_terms(final_state)
        if self._config.stability_selection <= 0:
            return replace(
                final_state,
                best_result_terms=honest_terms,
                best_result_coefficients=honest_coefficients,
                best_result_is_valid=True,
            )
        extras = dict(final_state.extras or {})
        selected_expression = final_state.best_expression
        selected_reward = final_state.best_reward
        vote_counts: list[int] = []
        ran = False
        error: str | None = None
        selected_terms: list[str] | None = honest_terms
        selected_coefficients: list[float] | None = honest_coefficients
        candidates = self._engine.cycle_top_candidates
        if len(candidates) > 1:
            try:
                result = stability_select(
                    candidates,
                    evaluator,
                    top_k=self._config.stability_selection,
                    rng=(
                        np.random.default_rng(self._stability_seed)
                        if self._stability_seed is not None
                        else None
                    ),
                )
                selected_expression = result.selected.expression
                selected_reward = result.selected.reward
                vote_counts = result.vote_counts
                selected_terms, selected_coefficients = (
                    self._evaluate_selected_candidate(
                        evaluator,
                        selected_expression,
                    )
                )
                ran = True
            except Exception as exc:
                logger.warning(
                    "Stability selection failed; keeping pre-filter result: %s",
                    exc,
                )
                selected_expression = final_state.best_expression
                selected_reward = final_state.best_reward
                selected_terms = honest_terms
                selected_coefficients = honest_coefficients
                vote_counts = []
                error = str(exc)
        extras["stability_selection"] = {
            "ran": ran,
            "seed": self._stability_seed,
            "pre_filter": {
                "expression": final_state.best_expression,
                "reward": final_state.best_reward,
            },
            "selected": {
                "expression": selected_expression,
                "reward": selected_reward,
            },
            "vote_counts": vote_counts,
            "error": error,
        }
        return replace(
            final_state,
            best_expression=selected_expression,
            best_reward=selected_reward,
            best_result_terms=selected_terms,
            best_result_coefficients=selected_coefficients,



            best_result_is_valid=True,
            extras=extras,
        )

    def _attach_seed_plan(self, final_state: EngineState) -> EngineState:
        extras = dict(final_state.extras or {})
        extras["seed_plan"] = {
            "pretrain_split_seed": self._pretrain_split_seed,
            "local_sample_seed": self._local_sample_seed,
            "stability_seed": self._stability_seed,
        }
        return replace(final_state, extras=extras)

    def _finalize_run(
        self, final_state: EngineState, evaluator: Evaluator
    ) -> EngineState:
        final_state = self._finalize_with_stability_selection(final_state, evaluator)
        final_state = self._attach_seed_plan(final_state)
        return final_state

    def _evaluate_selected_candidate(
        self,
        evaluator: Evaluator,
        expression: str,
    ) -> tuple[list[str], list[float]]:
        result = evaluator.evaluate_expression(expression)
        result_filter = _make_magnitude_filter(enabled=self._config.magnitude_filter)
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

    def _make_local_coords(
        self,
        cycle_idx: int,
    ) -> dict[str, Tensor] | None:
        if not self._pinn_config.local_sample:
            return None
        from kd.search.discover.pinn.collocation import generate_local_samples

        seed: int | None = None
        if self._local_sample_seed is not None:
            seed = _derive_cycle_seed(
                self._local_sample_seed,
                cycle_idx,
                LOCAL_SAMPLE_DOMAIN,
            )

        return generate_local_samples(
            observation_coords=self._obs_coords,
            bounds=self._local_bounds,
            multiplier=self._pinn_config.local_multiplier,
            seed=seed,
        )

    def _rebuild_after_pinn(self) -> Evaluator:
        regen = regenerate_metadata(
            self._pinn_model,
            self._colloc_coords,
            self._dataset_metadata,
            lhs_field=self._dataset_metadata.lhs_field,
            lhs_axis=self._dataset_metadata.lhs_axis,
        )
        if not hasattr(self._evaluator, "executor") or not hasattr(
            self._evaluator,
            "solver",
        ):
            return cast(Evaluator, self._evaluator)
        source = cast(Evaluator, self._evaluator)
        return rebuild_evaluator(regen, source.executor, source.solver)


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
    "rebuild_evaluator",
    "regenerate_metadata",
]
