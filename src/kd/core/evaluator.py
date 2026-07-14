
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from kd.core.equation.types import Form
from kd.core.jsonsafe import make_json_safe, sanitize_float
from kd.core.metrics import ScorerFn, make_aic_scorer
from kd.core.metrics import nmse as _metrics_nmse
from kd.core.term_cache import TermColumnCache

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from kd.core.executor import ExecutionContext
    from kd.core.expr import PythonExecutor
    from kd.core.linear_solve import SparseSolver


@dataclass
class EvaluationResult:

    mse: float
    nmse: float
    r2: float
    score: float | None = None
    complexity: int = 0
    coefficients: Tensor | None = None
    is_valid: bool = True
    error_message: str = ""
    selected_indices: list[int] | None = None
    residuals: Tensor | None = None
    terms: list[str] | None = None
    expression: str = ""
    lhs_name: str | None = None
    form: Form = Form.EVOLUTION

    def to_dict(self, *, include_residuals: bool = True) -> dict[str, Any]:
        residuals: Any = None
        if include_residuals:
            residuals = make_json_safe(
                self.residuals,
                key=_RESIDUALS_JSON_KEY,
            )
        return {
            "mse": sanitize_float(self.mse),
            "nmse": sanitize_float(self.nmse),
            "r2": sanitize_float(self.r2),
            "score": sanitize_float(self.score) if self.score is not None else None,
            "complexity": self.complexity,
            "coefficients": make_json_safe(
                self.coefficients,
                key=_COEFFICIENTS_JSON_KEY,
            ),
            "is_valid": self.is_valid,
            "error_message": self.error_message,
            "selected_indices": self.selected_indices,
            "residuals": residuals,
            "terms": self.terms,
            "expression": self.expression,
            "lhs_name": self.lhs_name,
        }








_COEFFICIENTS_JSON_KEY = "final_eval.coefficients"
_RESIDUALS_JSON_KEY = "final_eval.residuals"


class Evaluator:

    def __init__(
        self,
        executor: PythonExecutor,
        solver: SparseSolver,
        context: ExecutionContext,
        lhs: Tensor,
        penalty_value: float = 1e10,
        scorer: ScorerFn | None = None,
        enable_term_cache: bool = True,
    ) -> None:
        self._executor = executor
        self._solver = solver
        self._context = context
        self._lhs = lhs.detach()
        self._penalty_value = penalty_value
        self._term_cache: TermColumnCache | None = (
            TermColumnCache() if enable_term_cache else None
        )





        self._term_cache_provider: object | None = None


        self._lhs_flat = self._lhs.flatten()
        self._n_samples = self._lhs_flat.shape[0]





        self._lhs_var = self._lhs_flat.var(correction=0).item()


        self._scorer = scorer or make_aic_scorer(self._n_samples)

    @property
    def lhs_target(self) -> Tensor:
        return self._lhs_flat.detach()

    @property
    def lhs(self) -> Tensor:
        return self._lhs_flat.detach()

    @property
    def executor(self) -> PythonExecutor:
        return self._executor

    @property
    def solver(self) -> SparseSolver:
        return self._solver

    @property
    def context(self) -> ExecutionContext:
        return self._context

    def invalidate_term_cache(self) -> None:
        if self._term_cache is not None:
            self._term_cache.clear()

    def build_theta_matrix(
        self,
        terms: list[str],
        *,
        skip_invalid: bool = False,
    ) -> tuple[Tensor, list[str]]:
        if not terms:
            raise ValueError("Empty term list")
        with torch.no_grad():
            theta, valid_terms = self._build_theta(
                terms,
                skip_invalid=skip_invalid,
            )
        return theta.detach(), list(valid_terms)

    def evaluate_terms(
        self, terms: list[str], *, skip_invalid: bool = False
    ) -> EvaluationResult:

        if not terms:
            return self._make_invalid_result("Empty term list")






        try:
            with torch.no_grad():
                return self._evaluate_terms_impl(terms, skip_invalid=skip_invalid)
        except torch.cuda.OutOfMemoryError as exc:
            logger.warning(
                "Autograd OOM during evaluate_terms; returning invalid "
                "result. terms=%s error=%s",
                terms[:3],
                exc,
            )
            _release_cuda_memory()
            return self._make_invalid_result("autograd OOM")

    def _build_theta(
        self, terms: list[str], *, skip_invalid: bool = False
    ) -> tuple[Tensor, list[str]]:
        theta_columns: list[Tensor] = []
        valid_terms: list[str] = []
        cache = self._term_cache









        provider = self._context.derivative_provider
        if cache is not None and provider is not self._term_cache_provider:
            if self._term_cache_provider is not None:
                cache.clear()
            self._term_cache_provider = provider




        generation = provider.generation

        for term in terms:




            col = cache.get(term, generation) if cache is not None else None
            if col is None:

                try:
                    result = self._executor.execute(term, self._context)
                    col = result.value.flatten()
                except torch.cuda.OutOfMemoryError:




                    raise
                except Exception as e:
                    if not skip_invalid:
                        raise ValueError(f"Execution error for '{term}': {e}") from e
                    logger.debug(
                        "skip_invalid: skipping '%s' (execution error)", term
                    )
                    continue




                if cache is not None:
                    cache.put(term, col, generation)


            if skip_invalid and not torch.isfinite(col).all():
                logger.debug("skip_invalid: skipping '%s' (NaN/Inf)", term)
                continue


            if skip_invalid and (col == 0).all():
                logger.debug("skip_invalid: skipping '%s' (all zeros)", term)
                continue

            theta_columns.append(col)
            valid_terms.append(term)


        if not theta_columns:
            if skip_invalid:
                raise ValueError("All terms filtered by skip_invalid")
            raise ValueError("No valid terms")

        try:
            theta = torch.stack(theta_columns, dim=1)
        except torch.cuda.OutOfMemoryError:

            raise
        except Exception as e:
            raise ValueError(f"Failed to build theta matrix: {e}") from e


        if not skip_invalid and (torch.isnan(theta).any() or torch.isinf(theta).any()):
            raise ValueError("Theta contains NaN or Inf")

        return theta, valid_terms

    def _get_complexity(self, selected_indices: list[int] | None, n_terms: int) -> int:
        if selected_indices is not None:
            return len(selected_indices)
        return n_terms

    def _evaluate_terms_impl(
        self, terms: list[str], *, skip_invalid: bool = False
    ) -> EvaluationResult:
        try:
            theta, valid_terms = self._build_theta(terms, skip_invalid=skip_invalid)
        except ValueError as e:
            return self._make_invalid_result(str(e))

        try:
            solve_result = self._solver.solve(theta, self._lhs_flat)
        except torch.cuda.OutOfMemoryError:

            raise
        except Exception as e:
            return self._make_invalid_result(f"Solver error: {e}")

        if not solve_result.is_valid:
            return self._make_invalid_result(
                solve_result.error_message or "Solver returned invalid result"
            )

        coefficients = solve_result.coefficients
        y_pred = theta @ coefficients
        mse = ((self._lhs_flat - y_pred) ** 2).mean().item()

        if not math.isfinite(mse):
            return self._make_invalid_result("MSE is NaN or Inf")

        nmse_val = _metrics_nmse(mse, self._lhs_var)
        r2 = solve_result.r2
        complexity = self._get_complexity(
            solve_result.selected_indices, len(valid_terms)
        )
        score_val = self._scorer(mse, complexity)

        return EvaluationResult(
            mse=mse,
            nmse=nmse_val,
            r2=r2,
            score=score_val,
            complexity=complexity,
            coefficients=coefficients,
            is_valid=True,
            error_message="",
            selected_indices=solve_result.selected_indices,
            residuals=(y_pred - self._lhs_flat).detach(),
            terms=list(valid_terms),
        )

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        from kd.core.expr.terms import split_terms

        try:
            terms = split_terms(expr, self._executor.registry)
        except Exception as e:
            result = self._make_invalid_result(f"split_terms error: {e}")
            result.expression = expr
            return result

        result = self.evaluate_terms(terms)
        result.expression = expr
        return result

    def _make_invalid_result(self, error_message: str) -> EvaluationResult:
        return EvaluationResult(
            mse=self._penalty_value,
            nmse=self._penalty_value,
            r2=-float("inf"),
            score=float("inf"),
            complexity=0,
            coefficients=None,
            is_valid=False,
            error_message=error_message,
            selected_indices=None,
            residuals=None,
            terms=None,
            expression="",
        )


def release_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()





_release_cuda_memory = release_cuda_memory
