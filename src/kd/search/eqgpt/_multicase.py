
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import torch
from torch import Tensor

from kd.core.evaluator import EvaluationResult
from kd.core.term_cache import TermColumnCache
from kd.data.loaders.wave_breaking_eval import WaveBreakingFit
from kd.search.eqgpt._scoring import PENALTY, invalid_final_result, invalid_result
from kd.search.eqgpt.reward import compute_reward
from kd.viz.gap_notes import NO_MEASUREMENT

logger = logging.getLogger(__name__)







WAVE_MAX_ATOMIC_ORDER: Final[int] = 3


WAVE_LHS_ORDER: Final[int] = 1



_COEFF_GENERATION_BASE: Final[int] = 1000


_CASE_ERRORS: Final = (ValueError, KeyError, RuntimeError, IndexError)

if TYPE_CHECKING:
    from kd.core.executor.context import ExecutionContext
    from kd.core.expr.executor import PythonExecutor
    from kd.data.loaders.wave_breaking import WaveBreakingCase
    from kd.models.field_model import FieldModel
    from kd.search.eqgpt.config import EqGPTConfig
    from kd.search.eqgpt.reward import RewardResult







@dataclass(frozen=True)
class WaveCaseBundle:

    case_name: str
    executor: PythonExecutor
    context: ExecutionContext
    pinned_lhs: Tensor
    n_points: int
    cache_generation: int







def assemble_pinned_matrix(
    terms: list[str],
    *,
    executor: PythonExecutor,
    context: ExecutionContext,
    pinned_lhs: Tensor | None,
    cache: TermColumnCache,
    cache_generation: int,
) -> np.ndarray:
    if pinned_lhs is None and not terms:
        raise ValueError("free-pivot matrix requires at least one term")

    columns: list[Tensor] = []
    for term in terms:
        column = cache.get(term, generation=cache_generation)
        if column is None:
            column = executor.execute(term, context).value.reshape(-1).detach()
            cache.put(term, column, generation=cache_generation)
        columns.append(column)
    reference = columns[0] if pinned_lhs is None else pinned_lhs
    reference_np = reference.detach().cpu().numpy().astype(np.float64)
    n_rows = reference_np.shape[0]
    col_nps: list[np.ndarray] = []
    for term, column in zip(terms, columns, strict=True):
        col_np = column.detach().cpu().numpy().astype(np.float64)
        if col_np.shape[0] != n_rows:




            pivot_name = "first term pivot" if pinned_lhs is None else "pinned pivot"
            raise ValueError(
                f"grid-provenance mismatch: term {term!r} produced "
                f"{col_np.shape[0]} rows, {pivot_name} has {n_rows} rows "
                f"(cache generation {cache_generation})."
            )
        col_nps.append(col_np)

    if pinned_lhs is None:
        return np.column_stack(col_nps)
    return np.column_stack([reference_np, *col_nps])







class MultiCaseEvaluator:

    def __init__(
        self,
        *,
        reward_bundles: list[WaveCaseBundle],
        coeff_bundles: list[WaveCaseBundle],
        primary_case: str,
        sparsity_alpha: float,
        cache: TermColumnCache | None = None,
    ) -> None:
        self._reward_bundles = list(reward_bundles)
        self._coeff_bundles = list(coeff_bundles)
        self._primary_case = primary_case
        self._sparsity_alpha = sparsity_alpha
        self._cache = cache if cache is not None else TermColumnCache()

    @property
    def primary_case(self) -> str:
        return self._primary_case

    @classmethod
    def from_config(
        cls,
        config: EqGPTConfig,
        *,
        device: torch.device | None = None,
    ) -> MultiCaseEvaluator:
        from kd.data.loaders.wave_breaking import (
            load_wave_breaking_cases,
            wave_surrogate_checkpoint_path,
        )
        from kd.data.loaders.wave_breaking_eval import _validate_surrogate_layout
        from kd.models.v1_checkpoint import load_v1_field_model

        if config.case_filter is None:
            raise ValueError("from_config requires case_filter (wave mode).")
        cases = load_wave_breaking_cases(config.wave_pkl_path)
        selected = sorted(name for name in cases if config.case_filter in name)
        if not selected:
            raise ValueError(
                f"case_filter {config.case_filter!r} matched no wave cases."
            )
        primary = config.primary_case or selected[0]
        if primary not in selected:
            raise ValueError(
                f"primary_case {primary!r} not among selected cases {selected}."
            )
        reward_bundles: list[WaveCaseBundle] = []
        coeff_bundles: list[WaveCaseBundle] = []
        for index, name in enumerate(selected):
            checkpoint = wave_surrogate_checkpoint_path(name, config.v1_asset_dir)
            surrogate = load_v1_field_model(checkpoint)
            if device is not None:
                surrogate = surrogate.to(device)
            _validate_surrogate_layout(surrogate)
            case = cases[name]
            coeff_gen = _COEFF_GENERATION_BASE + index
            for bundles, ppw, gen in (
                (reward_bundles, config.reward_points_per_window, index),
                (coeff_bundles, config.coeff_points_per_window, coeff_gen),
            ):
                bundles.append(
                    _build_case_bundle(
                        case, surrogate, points_per_window=ppw, cache_generation=gen
                    )
                )
        return cls(
            reward_bundles=reward_bundles,
            coeff_bundles=coeff_bundles,
            primary_case=primary,
            sparsity_alpha=config.sparsity_alpha,
            cache=TermColumnCache(),
        )

    @property
    def case_names(self) -> list[str]:
        return [bundle.case_name for bundle in self._reward_bundles]

    def _assemble(self, terms: list[str], bundle: WaveCaseBundle) -> np.ndarray:
        return assemble_pinned_matrix(
            terms,
            executor=bundle.executor,
            context=bundle.context,
            pinned_lhs=bundle.pinned_lhs,
            cache=self._cache,
            cache_generation=bundle.cache_generation,
        )

    def _score_bundle(
        self, terms: list[str], bundle: WaveCaseBundle
    ) -> RewardResult | None:
        try:
            matrix = self._assemble(terms, bundle)
        except _CASE_ERRORS as exc:
            logger.debug("case %s skipped: %s", bundle.case_name, exc)
            return None
        return compute_reward(matrix, sparsity_alpha=self._sparsity_alpha)

    def _primary_coeff_bundle(self) -> WaveCaseBundle | None:
        for bundle in self._coeff_bundles:
            if bundle.case_name == self._primary_case:
                return bundle
        return None

    def score_candidate(
        self,
        *,
        candidate: str,
        terms: list[str],
    ) -> EvaluationResult:

        survivors = [
            rr
            for rr in (self._score_bundle(terms, b) for b in self._reward_bundles)
            if rr is not None and math.isfinite(rr.r2)
        ]
        if not survivors:
            return invalid_result(
                candidate, terms, "all cases degenerate", is_error=True
            )
        r2 = sum(rr.r2 for rr in survivors) / len(survivors)
        return EvaluationResult(
            mse=PENALTY,
            nmse=(1.0 - r2) if math.isfinite(r2) else PENALTY,
            r2=r2,
            score=sum(rr.reward for rr in survivors) / len(survivors),
            complexity=len(terms),
            coefficients=None,
            is_valid=True,
            error_message="",
            residuals=None,
            terms=list(terms),
            expression=candidate,
        )

    def per_case_rewards(self, terms: list[str]) -> dict[str, float]:
        rewards: dict[str, float] = {}
        for bundle in self._reward_bundles:
            rr = self._score_bundle(terms, bundle)
            if rr is None or not math.isfinite(rr.r2):
                rewards[bundle.case_name] = NO_MEASUREMENT
            else:
                rewards[bundle.case_name] = rr.reward
        return rewards

    def build_final_result(
        self, terms: list[str], *, best_reward: float
    ) -> EvaluationResult:
        bundle = self._primary_coeff_bundle()
        if bundle is None:
            return invalid_final_result(
                "no primary coeff bundle",
                best_reward,
                terms=terms,
                reason="evaluation_error",
            )
        if not terms:
            return invalid_final_result(
                "no terms to fit",
                best_reward,
                terms=terms,
                reason="structural_reject",
            )
        try:
            matrix = self._assemble(terms, bundle)
            theta, target = matrix[:, 1:], matrix[:, 0]
            coefficients = np.linalg.lstsq(theta, target, rcond=None)[0]
        except _CASE_ERRORS as exc:
            return invalid_final_result(
                f"execution error: {exc}",
                best_reward,
                terms=terms,
                reason="evaluation_error",
            )
        residuals = theta @ coefficients - target
        mse = float(np.mean(residuals**2))
        if not math.isfinite(mse):

            return invalid_final_result(
                "non-finite residuals from primary coeff-grid refit",
                best_reward,
                terms=terms,
                reason="non_finite",
            )
        lhs_var = float(target.var()) if target.size > 1 else 0.0
        r2 = 1.0 - mse / lhs_var if lhs_var > 0 else -float("inf")
        return EvaluationResult(
            mse=mse,
            nmse=1.0 - r2 if math.isfinite(r2) else PENALTY,
            r2=r2,
            score=best_reward,
            complexity=len(terms),
            coefficients=torch.as_tensor(coefficients, dtype=torch.float32),
            is_valid=True,
            error_message="",
            residuals=torch.from_numpy(residuals).detach().clone(),
            terms=list(terms),
            expression=" + ".join(terms),
        )

    def result_target(self) -> Tensor:
        bundle = self._primary_coeff_bundle()
        if bundle is None:
            raise ValueError(f"no coeff bundle for primary {self._primary_case!r}.")
        return bundle.pinned_lhs.detach().clone()

    def per_case_fits(self, terms: list[str]) -> dict[str, WaveBreakingFit]:
        fits: dict[str, WaveBreakingFit] = {}
        for bundle in self._coeff_bundles:
            try:
                matrix = self._assemble(terms, bundle)
                theta, u_t = matrix[:, 1:], matrix[:, 0]
                coefficients = np.linalg.lstsq(theta, u_t, rcond=None)[0]
            except _CASE_ERRORS as exc:
                logger.warning("per_case_fits: %s failed: %s", bundle.case_name, exc)
                continue
            resid = u_t - theta @ coefficients
            denom = float(np.sum((u_t - u_t.mean()) ** 2))
            r2 = 1.0 - float(resid @ resid) / denom if denom > 0 else float("nan")

            c = [
                -float(coefficients[i]) if i < coefficients.size else float("nan")
                for i in range(3)
            ]
            fits[bundle.case_name] = WaveBreakingFit(c[0], c[1], c[2], r2)
        return fits


def _build_case_bundle(
    case: WaveBreakingCase,
    surrogate: FieldModel,
    *,
    points_per_window: int,
    cache_generation: int,
) -> WaveCaseBundle:
    from kd.core.platform.builder import PlatformBuilder
    from kd.core.platform.requirements import DerivativeReqs
    from kd.data.loaders.wave_breaking_eval import (
        _build_eval_dataset,
        _surrogate_device_dtype,
        wave_breaking_star_grid,
    )

    device, dtype = _surrogate_device_dtype(surrogate)
    x_star, t_star = wave_breaking_star_grid(case, points_per_window=points_per_window)
    dataset = _build_eval_dataset(
        t_star=t_star.to(device=device, dtype=torch.float64),
        x_star=x_star.to(device=device, dtype=torch.float64),
        query_dtype=dtype,
    )
    reqs = DerivativeReqs(
        provider_kind="autograd",
        surrogate_model=surrogate,
        needs_surrogate=True,
        max_atomic_order=WAVE_MAX_ATOMIC_ORDER,
        lhs_order=WAVE_LHS_ORDER,
    )
    components = PlatformBuilder(dataset, reqs).build()
    if components.evaluator is None or components.context is None:
        raise ValueError(
            f"PlatformBuilder returned no evaluator/context for {case.name!r}."
        )
    pinned_lhs = components.evaluator.lhs.detach().reshape(-1)
    return WaveCaseBundle(
        case_name=case.name,
        executor=components.executor,
        context=components.context,
        pinned_lhs=pinned_lhs,
        n_points=pinned_lhs.numel(),
        cache_generation=cache_generation,
    )


__all__ = [
    "WAVE_LHS_ORDER",
    "WAVE_MAX_ATOMIC_ORDER",
    "MultiCaseEvaluator",
    "WaveCaseBundle",
    "assemble_pinned_matrix",
]
