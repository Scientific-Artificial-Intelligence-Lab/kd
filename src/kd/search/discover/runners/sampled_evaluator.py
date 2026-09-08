
from __future__ import annotations

import math

import torch
from torch import Tensor

from kd.core.evaluator import (
    EvaluationResult,
    Evaluator,
)
from kd.core.interrupt import SearchInterrupted
from kd.core.metrics import make_aic_scorer
from kd.core.metrics import nmse as kd_nmse
from kd.search.discover.evaluation.magnitude import (
    magnitude_reject_reason as magnitude_reject_reason,
)
from kd.search.discover.evaluation.magnitude import (
    magnitude_rejection,
)

_RANK_CHECK_REL_TOL = 1e-8


class _SampledStructuralError(ValueError):
    pass


class _SampledNonFiniteError(ValueError):
    pass


class SampledEvaluator:

    def __init__(
        self,
        base: Evaluator,
        sample_indices: Tensor,
        *,
        rank_check: bool = False,
        magnitude_filter: bool = False,
    ) -> None:
        self._base = base
        self._sample_indices = sample_indices
        self._lhs = base.lhs_target.index_select(0, sample_indices).detach()


        self._lhs_var = float(self._lhs.var(correction=0).item())
        self._scorer = make_aic_scorer(self._lhs.shape[0])
        self._term_cache: dict[str, Tensor] = {}
        self._rank_check = rank_check
        self._magnitude_filter = magnitude_filter

    def build_theta_matrix(self, terms: list[str]) -> tuple[Tensor, list[str]]:
        theta, valid_terms = self._build_theta(terms, skip_invalid=False)
        return theta.detach(), valid_terms

    def evaluate_terms(
        self,
        terms: list[str],
        *,
        skip_invalid: bool = False,
    ) -> EvaluationResult:
        if not terms:
            return self._make_invalid_result(
                "Empty term list", reason="structural_reject"
            )
        with torch.no_grad():
            return self._evaluate_terms_impl(terms, skip_invalid=skip_invalid)

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        from kd.core.expr.terms import split_terms

        try:
            terms = split_terms(expr, self._base.executor.registry)
        except SearchInterrupted:


            raise
        except Exception as exc:
            result = self._make_invalid_result(
                f"split_terms error: {exc}", reason="structural_reject"
            )
            result.expression = expr
            return result

        result = self.evaluate_terms(terms)
        result.expression = expr
        return result

    def _evaluate_terms_impl(
        self,
        terms: list[str],
        *,
        skip_invalid: bool,
    ) -> EvaluationResult:
        try:
            theta, valid_terms = self._build_theta(terms, skip_invalid=skip_invalid)
            solve_result = self._base.solver.solve(theta, self._lhs)
        except _SampledStructuralError as exc:
            return self._make_invalid_result(str(exc), reason="structural_reject")
        except SearchInterrupted:


            raise
        except _SampledNonFiniteError as exc:
            return self._make_invalid_result(str(exc), reason="non_finite")
        except Exception as exc:
            return self._make_invalid_result(str(exc))

        coefficients = solve_result.coefficients
        if self._magnitude_filter:
            rejection = magnitude_rejection(
                coefficients, solve_result.selected_indices
            )
            if rejection is not None:
                detail, invalid_reason = rejection
                return self._make_invalid_result(detail, reason=invalid_reason)
        y_pred = theta @ coefficients
        mse = float(((self._lhs - y_pred) ** 2).mean().item())
        if not math.isfinite(mse):
            return self._make_invalid_result(
                "MSE is NaN or Inf", reason="non_finite"
            )

        complexity = _complexity(solve_result.selected_indices, len(valid_terms))
        return EvaluationResult(
            mse=mse,
            nmse=kd_nmse(mse, self._lhs_var),
            r2=solve_result.r2,
            score=self._scorer(mse, complexity),
            complexity=complexity,
            coefficients=coefficients,
            is_valid=True,
            error_message="",
            selected_indices=solve_result.selected_indices,
            residuals=(y_pred - self._lhs).detach(),
            terms=list(valid_terms),
        )

    def _build_theta(
        self,
        terms: list[str],
        *,
        skip_invalid: bool,
    ) -> tuple[Tensor, list[str]]:
        columns: list[Tensor] = []
        valid_terms: list[str] = []
        for term in terms:
            try:
                column = self._term_column(term)
            except SearchInterrupted:


                raise
            except Exception:
                if not skip_invalid:
                    raise
                continue
            is_bad_column = not torch.isfinite(column).all() or (column == 0).all()
            if skip_invalid and is_bad_column:
                continue
            columns.append(column)
            valid_terms.append(term)
        if not columns:
            raise _SampledStructuralError("No valid terms")
        theta = torch.stack(columns, dim=1)
        if not skip_invalid and not torch.isfinite(theta).all():
            raise _SampledNonFiniteError("Theta contains NaN or Inf")
        if self._rank_check and theta.shape[1] > 1:
            self._assert_full_rank(theta)
        return theta, valid_terms

    @staticmethod
    def _assert_full_rank(theta: Tensor) -> None:


        try:
            singular_values = torch.linalg.svdvals(theta)
        except RuntimeError as exc:
            raise ValueError(f"Could not compute SVD for rank check: {exc}") from exc
        s_max = float(singular_values[0].item())
        if s_max == 0.0:
            raise _SampledStructuralError(
                "Theta has zero largest singular value (all-zero matrix); "
                "likely degenerate basis."
            )
        threshold = _RANK_CHECK_REL_TOL * s_max
        eff_rank = int((singular_values > threshold).sum().item())
        n_cols = int(theta.shape[1])
        if eff_rank < n_cols:
            s_min = float(singular_values[-1].item())
            raise _SampledStructuralError(
                f"Theta is rank-deficient: effective rank {eff_rank} < "
                f"{n_cols} columns (smallest singular value {s_min:.3e}, "
                f"largest {s_max:.3e}, ratio {s_min / s_max:.3e}). "
                f"Likely degenerate basis (e.g. x↔y symmetric data — "
                f"). Set rank_check=False to bypass this check."
            )

    def _term_column(self, term: str) -> Tensor:
        cached = self._term_cache.get(term)
        if cached is not None:
            return cached
        result = self._base.executor.execute(term, self._base.context)
        full_column = result.value.flatten()
        column = full_column.index_select(0, self._sample_indices).detach()
        self._term_cache[term] = column
        return column

    @staticmethod
    def _make_invalid_result(
        error_message: str,
        *,
        reason: str = "evaluation_error",
    ) -> EvaluationResult:
        return EvaluationResult(
            mse=1e10,
            nmse=1e10,
            r2=-float("inf"),
            score=float("inf"),
            complexity=0,
            coefficients=None,
            is_valid=False,
            error_message=error_message,
            invalid_reason=reason,
            selected_indices=None,
            residuals=None,
            terms=None,
            expression="",
        )


def _complexity(selected_indices: list[int] | None, n_terms: int) -> int:
    if selected_indices is None:
        return n_terms
    return len(selected_indices)
