
from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import Tensor

from kd.core.evaluator import EvaluationResult
from kd.search.discover.candidates import (
    CandidateSnapshot,
    CycleCandidateTracker,
    extract_active_terms,
)
from kd.search.discover.core.tree import trim_to_natural
from kd.search.discover.engine_types import (
    BoolArray,
    CycleCallback,
    EngineState,
    Evaluator,
    FloatArray,
    Generator,
    Int64Array,
    PendingState,
    ResultFilter,
    RewardAdapter,
    SearchProgressCallback,
)
from kd.search.discover.evaluation.dedup import Deduplicator
from kd.search.discover.ir.conversion import tokens_to_ir
from kd.search.discover.tokens.validator import CandidateValidator
from kd.search.discover.training.strategy import BaselineState, RSPGStrategy

DEFAULT_BATCH_SIZE = 256
INITIAL_BEST_REWARD = 0.0
INITIAL_BEST_EXPRESSION = ""
ZERO_REWARD = 0.0
_RESTORED_PLACEHOLDER_ERROR_MESSAGE = "restored from checkpoint without metric replay"








_RESTORED_GATE_INVALID_ERROR_MESSAGE = (
    "restored from checkpoint; magnitude-gate rejected (coefficients out of bounds)"
)
_NONFINITE_REWARD_ERROR_MESSAGE = "reward_adapter must return finite rewards."


class DiscoverEngine:

    def __init__(
        self,
        generator: Generator,
        strategy: RSPGStrategy | None = None,
        reward_adapter: RewardAdapter | None = None,
        validator: CandidateValidator | None = None,
        deduplicator: Deduplicator | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cycle_candidate_capacity: int = 0,
        result_filter: ResultFilter | None = None,
    ) -> None:
        self._generator = generator
        if strategy is None:
            raise TypeError("strategy is required.")
        if reward_adapter is None:
            raise TypeError("reward_adapter is required.")
        if validator is None:
            raise TypeError("validator is required.")
        self._strategy = strategy
        self._reward_adapter = reward_adapter
        self._result_filter = result_filter
        self._validator = validator
        self._deduplicator = deduplicator
        self._batch_size = DEFAULT_BATCH_SIZE
        self.batch_size = batch_size
        if cycle_candidate_capacity < 0:
            raise ValueError("cycle_candidate_capacity must be non-negative.")
        self._cycle_candidate_capacity = cycle_candidate_capacity
        self._cycle_candidates = CycleCandidateTracker(
            self._generator.library,
            cycle_candidate_capacity,
        )
        self._baseline_state = BaselineState()
        self._best_reward = INITIAL_BEST_REWARD
        self._best_expression = INITIAL_BEST_EXPRESSION
        self._best_result: EvaluationResult | None = None
        self._extras: dict[str, Any] | None = None
        self._pending: PendingState | None = None
        self._last_metrics: dict[str, float] = {}

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        if value <= 0:
            raise ValueError("batch_size must be positive.")
        self._batch_size = value

    def propose(self) -> list[str]:
        self._require_idle()
        batch = self._generator.sample(self._batch_size)
        valid_mask = self._validator.validate(batch.actions)
        unique_irs, scatter_map = self._make_unique_irs(batch.actions[valid_mask])
        self._pending = PendingState(
            batch=batch,
            valid_mask=valid_mask,
            scatter_map=scatter_map,
            unique_irs=unique_irs,
        )
        return list(unique_irs)

    def receive_results(self, results: list[EvaluationResult]) -> None:
        pending = self._require_proposed()
        self._validate_result_count(results, pending.unique_irs)
        results = self._apply_result_filter(results)
        pending.unique_results = list(results)
        unique_rewards = self._to_unique_rewards(results)
        unique_eval_valid_mask = self._to_unique_eval_valid_mask(results)
        valid_rewards = Deduplicator.scatter_rewards(
            unique_rewards,
            pending.scatter_map,
        )
        pending.unique_rewards = unique_rewards
        pending.unique_eval_valid_mask = unique_eval_valid_mask
        pending.train_valid_mask = self._expand_train_valid_mask(
            pending.valid_mask,
            pending.scatter_map,
            unique_eval_valid_mask,
        )
        pending.full_rewards = self._expand_rewards(valid_rewards, pending.valid_mask)

    def update(self) -> None:
        pending = self._require_received()
        rewards = pending.full_rewards
        assert rewards is not None
        loss_info, baseline_state = self._strategy.train_step(
            self._generator,
            pending.batch,
            rewards,
            self._baseline_state,
            valid_mask=pending.train_valid_mask,
        )
        self._baseline_state = baseline_state
        self._update_best(
            pending.unique_irs,
            pending.unique_rewards,
            pending.unique_eval_valid_mask,
            pending.unique_results,
        )
        self._last_metrics = self._make_metrics(loss_info, rewards, pending)
        self._pending = None

    def run_iteration(self, evaluator: Evaluator) -> dict[str, float]:
        try:
            ir_strings = self.propose()
            results = [evaluator.evaluate_expression(ir) for ir in ir_strings]
            self.receive_results(results)
            self.update()
        except Exception:
            self._pending = None
            raise
        finally:


            invalidate = getattr(evaluator, "invalidate_term_cache", None)
            if invalidate is not None:
                invalidate()
        return dict(self._last_metrics)

    def run_cycle(
        self,
        evaluator: Evaluator,
        n_iterations: int = 1000,
        progress_callback: SearchProgressCallback | None = None,
        cycle_idx: int | None = None,
    ) -> EngineState:
        if n_iterations < 0:
            raise ValueError("n_iterations must be non-negative.")
        if cycle_idx is not None:
            if not isinstance(cycle_idx, (int, np.integer)) or isinstance(
                cycle_idx,
                bool,
            ):
                raise TypeError(
                    "cycle_idx must be an int or numpy integer, "
                    f"got {type(cycle_idx).__name__}",
                )
            if int(cycle_idx) < 0:
                raise ValueError(
                    f"cycle_idx must be non-negative, got {int(cycle_idx)}",
                )
            self._notify_priors_cycle_start(int(cycle_idx))
        self.rebase_best(evaluator)
        self._reset_cycle_candidates()
        for iteration_idx in range(n_iterations):
            metrics = self.run_iteration(evaluator)
            if progress_callback is not None:
                progress_callback(iteration_idx + 1, metrics, self)
        return self.state

    def run(
        self,
        evaluator: Evaluator,
        n_iterations: int = 1000,
        n_cycles: int = 1,
        between_cycles: CycleCallback | None = None,
    ) -> EngineState:
        if n_iterations < 0:
            raise ValueError("n_iterations must be non-negative.")
        if n_cycles < 0:
            raise ValueError("n_cycles must be non-negative.")
        for cycle_idx in range(n_cycles):
            self.run_cycle(
                evaluator,
                n_iterations=n_iterations,
                cycle_idx=cycle_idx,
            )
            if between_cycles is not None and cycle_idx < n_cycles - 1:
                between_cycles(cycle_idx, self)
        return self.state

    def _notify_priors_cycle_start(self, cycle_idx: int) -> None:
        prior_system = getattr(self._generator, "prior_system", None)
        if prior_system is None:
            return
        for prior in prior_system.priors:
            hook = getattr(prior, "on_cycle_start", None)
            if callable(hook):
                hook(cycle_idx)

    @property
    def state(self) -> EngineState:
        terms: list[str] | None = None
        coefficients: list[float] | None = None
        best_result_is_valid = True
        if self._best_result is not None:
            terms, coefficients = extract_active_terms(self._best_result)
            if not terms:
                terms = None
                coefficients = None
            best_result_is_valid = self._best_result_is_trustworthy()
        return EngineState(
            controller_state_dict=deepcopy(self._generator.state_dict()),
            baseline_state=self._baseline_state,
            best_reward=self._best_reward,
            best_expression=self._best_expression,
            optimizer_state=self._strategy.optimizer_state,
            extras=deepcopy(self._extras) if self._extras is not None else None,
            best_result_terms=terms,
            best_result_coefficients=coefficients,
            best_result_is_valid=best_result_is_valid,
        )

    @state.setter
    def state(self, state: EngineState) -> None:
        self._generator.load_state_dict(deepcopy(state.controller_state_dict))
        self._baseline_state = state.baseline_state
        self._best_reward = state.best_reward
        self._best_expression = state.best_expression
        if state.optimizer_state is not None:
            self._strategy.optimizer_state = deepcopy(state.optimizer_state)
        else:
            self._strategy.reset_optimizer()
        self._extras = deepcopy(state.extras) if state.extras is not None else None
        self._best_result = self._rebuild_best_result(state)
        self._reset_cycle_candidates()
        self._pending = None
        self._last_metrics = {}

    def _best_result_is_trustworthy(self) -> bool:
        best = self._best_result
        if best is None or best.is_valid:
            return True
        return best.error_message == _RESTORED_PLACEHOLDER_ERROR_MESSAGE

    @property
    def best_reward(self) -> float:
        return self._best_reward

    @property
    def best_expression(self) -> str:
        return self._best_expression

    @property
    def best_result(self) -> EvaluationResult | None:
        return self._best_result

    @property
    def last_metrics(self) -> dict[str, float]:
        return dict(self._last_metrics)

    @property
    def cycle_top_candidates(self) -> list[CandidateSnapshot]:
        return self._cycle_candidates.snapshots

    @staticmethod
    def _strip_result(result: EvaluationResult) -> EvaluationResult:
        coefficients = None
        if result.coefficients is not None:
            coefficients = result.coefficients.detach().clone()
        return EvaluationResult(
            mse=result.mse,
            nmse=result.nmse,
            r2=result.r2,
            score=result.score,
            complexity=result.complexity,
            is_valid=result.is_valid,
            error_message=result.error_message,
            expression=result.expression,
            terms=list(result.terms) if result.terms is not None else None,
            coefficients=coefficients,
            selected_indices=(
                list(result.selected_indices)
                if result.selected_indices is not None
                else None
            ),
            residuals=None,
            lhs_name=result.lhs_name,
        )

    @staticmethod
    def _rebuild_best_result(state: EngineState) -> EvaluationResult | None:
        if not state.best_result_terms or not state.best_result_coefficients:
            return None
        error_message = (
            _RESTORED_PLACEHOLDER_ERROR_MESSAGE
            if state.best_result_is_valid
            else _RESTORED_GATE_INVALID_ERROR_MESSAGE
        )
        return EvaluationResult(
            mse=math.inf,
            nmse=math.inf,
            r2=-math.inf,
            score=math.inf,
            complexity=0,
            is_valid=False,
            error_message=error_message,
            expression=state.best_expression,
            terms=list(state.best_result_terms),
            coefficients=torch.tensor(
                state.best_result_coefficients,
                dtype=torch.float32,
            ),
        )

    def _require_idle(self) -> None:
        if self._pending is not None:
            raise RuntimeError("Cannot propose() while a batch is still pending.")

    def _require_proposed(self) -> PendingState:
        if self._pending is None:
            raise RuntimeError("receive_results() requires a prior propose().")
        if self._pending.full_rewards is not None:
            raise RuntimeError("receive_results() already called for this batch.")
        return self._pending

    def _require_received(self) -> PendingState:
        if self._pending is None:
            raise RuntimeError("update() requires a prior propose().")
        if self._pending.full_rewards is None:
            raise RuntimeError("update() requires receive_results() first.")
        return self._pending

    def _make_unique_irs(
        self,
        valid_actions: np.ndarray[Any, np.dtype[np.int32]],
    ) -> tuple[list[str], Int64Array]:
        if valid_actions.shape[0] == 0:
            return [], np.array([], dtype=np.int64)
        if self._deduplicator is not None:
            return self._deduplicator.deduplicate(valid_actions)
        return self._convert_without_dedup(valid_actions)

    def _convert_without_dedup(
        self,
        valid_actions: np.ndarray[Any, np.dtype[np.int32]],
    ) -> tuple[list[str], Int64Array]:
        irs = [
            tokens_to_ir(
                trim_to_natural(action_row, self._generator.library).tolist(),
                self._generator.library,
            )
            for action_row in valid_actions
        ]
        scatter_map = np.arange(valid_actions.shape[0], dtype=np.int64)
        return irs, scatter_map

    def _validate_result_count(
        self,
        results: Sequence[EvaluationResult],
        unique_irs: Sequence[str],
    ) -> None:
        if len(results) != len(unique_irs):
            raise ValueError(
                "results length must match the number of proposed IR strings."
            )

    def _apply_result_filter(
        self,
        results: Sequence[EvaluationResult],
    ) -> list[EvaluationResult]:
        if self._result_filter is None:
            return list(results)
        return [self._result_filter(result) for result in results]

    def _to_unique_rewards(self, results: Sequence[EvaluationResult]) -> FloatArray:
        rewards = np.asarray(
            [self._reward_adapter(result) for result in results],
            dtype=np.float32,
        )
        if not np.isfinite(rewards).all():
            raise ValueError(_NONFINITE_REWARD_ERROR_MESSAGE)
        return rewards

    @staticmethod
    def _to_unique_eval_valid_mask(results: Sequence[EvaluationResult]) -> BoolArray:
        return np.asarray([result.is_valid for result in results], dtype=np.bool_)

    def _expand_rewards(
        self,
        valid_rewards: FloatArray,
        valid_mask: BoolArray,
    ) -> Tensor:
        full_rewards = np.full(valid_mask.shape[0], ZERO_REWARD, dtype=np.float32)
        full_rewards[valid_mask] = valid_rewards
        return torch.tensor(full_rewards, dtype=torch.float32)

    @staticmethod
    def _expand_train_valid_mask(
        syntactic_valid_mask: BoolArray,
        scatter_map: Int64Array,
        unique_eval_valid_mask: BoolArray,
    ) -> BoolArray:
        train_valid_mask = np.zeros(
            syntactic_valid_mask.shape[0],
            dtype=np.bool_,
        )
        train_valid_mask[syntactic_valid_mask] = unique_eval_valid_mask[scatter_map]
        return train_valid_mask

    def _update_best(
        self,
        unique_irs: Sequence[str],
        unique_rewards: FloatArray | None,
        unique_eval_valid_mask: BoolArray | None,
        unique_results: list[EvaluationResult] | None = None,
    ) -> None:
        if unique_rewards is None or unique_eval_valid_mask is None:
            return
        self._cycle_candidates.record(
            unique_irs,
            unique_rewards,
            unique_eval_valid_mask,
            unique_results,
        )
        valid_indices = np.flatnonzero(unique_eval_valid_mask)
        if unique_rewards.size == 0 or valid_indices.size == 0:
            return
        best_offset = int(np.argmax(unique_rewards[valid_indices]))
        best_idx = int(valid_indices[best_offset])
        candidate_reward = float(unique_rewards[best_idx])
        if candidate_reward > self._best_reward:
            self._best_reward = candidate_reward
            self._best_expression = unique_irs[best_idx]
            if unique_results is not None:
                self._best_result = self._strip_result(
                    unique_results[best_idx],
                )

    def rebase_best(self, evaluator: Evaluator) -> None:
        if not self._best_expression:
            return
        result = evaluator.evaluate_expression(self._best_expression)
        if self._result_filter is not None:
            result = self._result_filter(result)
        if result.is_valid:

            reward = float(np.float32(self._reward_adapter(result)))
            if not math.isfinite(reward):
                raise ValueError(_NONFINITE_REWARD_ERROR_MESSAGE)
            self._best_reward = reward
        else:
            self._best_reward = INITIAL_BEST_REWARD
        self._best_result = self._strip_result(result)

    def _reset_cycle_candidates(self) -> None:
        self._cycle_candidates.reset()

    def _count_invalid_in_topk(
        self,
        rewards: Tensor,
        train_valid_mask: BoolArray | None,
    ) -> float:
        if train_valid_mask is None:
            return 0.0
        reward_values = rewards.detach().to(dtype=torch.float32, device="cpu").numpy()
        quantile = float(
            np.quantile(reward_values, 1.0 - self._strategy.epsilon, method="higher")
        )
        topk_mask = reward_values >= quantile
        invalid_topk = np.logical_and(topk_mask, np.logical_not(train_valid_mask))
        return float(np.count_nonzero(invalid_topk))

    def _make_metrics(
        self,
        loss_info: dict[str, float],
        rewards: Tensor,
        pending: PendingState,
    ) -> dict[str, float]:
        reward_max = float(torch.max(rewards).detach().cpu().item())
        unique_eval_valid_mask = pending.unique_eval_valid_mask
        n_eval_valid = 0.0
        if unique_eval_valid_mask is not None:
            n_eval_valid = float(np.count_nonzero(unique_eval_valid_mask))
        metrics = dict(loss_info)
        metrics["reward_max"] = reward_max



        metrics["reward_full"] = float(torch.mean(rewards).detach().cpu().item())
        metrics["best_reward"] = self._best_reward
        metrics["n_valid"] = float(np.count_nonzero(pending.valid_mask))
        metrics["n_eval_valid"] = n_eval_valid
        metrics["n_invalid_in_topk"] = self._count_invalid_in_topk(
            rewards,
            pending.train_valid_mask,
        )
        metrics["n_unique"] = float(len(pending.unique_irs))
        return metrics


__all__ = [
    "CandidateSnapshot",
    "CycleCallback",
    "DiscoverEngine",
    "EngineState",
    "Generator",
    "SearchProgressCallback",
    "extract_active_terms",
]
