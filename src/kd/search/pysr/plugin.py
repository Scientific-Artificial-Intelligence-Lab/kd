
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from torch import Tensor

from kd.core.evaluator import EvaluationResult, Evaluator
from kd.core.platform.requirements import DerivativeReqs
from kd.search.protocol import PlatformComponents
from kd.search.pysr import assembly
from kd.search.pysr import viz as _viz_helpers
from kd.search.pysr.backend import PySRBackend, default_backend_factory
from kd.search.pysr.config import PySRConfig
from kd.search.pysr.convert import build_feature_names
from kd.search.recorder import VizRecorder
from kd.viz.extension import PlotInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

ALGORITHM_NAME = "pysr"
INITIAL_BEST_EXPRESSION = ""
INITIAL_BEST_SCORE = float("inf")



_MIN_ATOMIC_ORDER = 1
_LHS_ORDER = 1



_PARETO_COMPLEXITY_KEY = "pareto_complexity"
_PARETO_LOSS_KEY = "pareto_loss"
_PARETO_NMSE_KEY = "pareto_nmse"
_SELECTED_COMPLEXITY_KEY = "selected_complexity"
_SELECTED_LOSS_KEY = "selected_loss"
_SELECTED_NMSE_KEY = "selected_nmse"


_STATE_ALGORITHM = "algorithm"
_STATE_BEST_EXPRESSION = "best_expression"
_STATE_BEST_SCORE = "best_score"
_STATE_FITTED = "fitted"
_STATE_TERMS = "terms"
_STATE_HOF_CANDIDATES = "hof_candidates"
_STATE_HOF_META = "hof_meta"


class PySRPlugin:




    score_kind: ClassVar[str] = "NMSE"
    score_direction: ClassVar[Literal["min", "max"]] = "min"

    def __init__(
        self,
        config: PySRConfig | None = None,
        *,
        backend_factory: Callable[[PySRConfig], PySRBackend] | None = None,
    ) -> None:
        self._config = config or PySRConfig()
        self._backend_factory = backend_factory or default_backend_factory
        self._evaluator: Evaluator | None = None
        self._recorder: VizRecorder | None = None
        self._dataset: Any = None
        self._fitted: bool = False
        self._best_expression: str = INITIAL_BEST_EXPRESSION
        self._best_score: float = INITIAL_BEST_SCORE
        self._terms: list[str] | None = None
        self._hof_candidates: list[str] | None = None
        self._hof_meta: list[tuple[int, float]] | None = None
        self._best_eval: EvaluationResult | None = None
        self._selected_complexity: int | None = None
        self._selected_loss: float | None = None
        self._selected_nmse: float | None = None
        self._pareto_logged: bool = False
        self._restore_pending: bool = False



    def prepare(self, components: PlatformComponents) -> None:
        if not self._restore_pending:
            self._reset_fit_state()
        self._restore_pending = False
        self._evaluator = components.evaluator
        self._recorder = components.recorder
        self._dataset = components.dataset

    def propose(self, n: int) -> list[str]:
        evaluator = self._require_evaluator()
        if self._fitted:
            return []
        self._run_fit(evaluator)
        self._fitted = True
        return list(self._hof_candidates or [])

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        evaluator = self._require_evaluator()
        return [evaluator.evaluate_expression(candidate) for candidate in candidates]

    def update(self, results: list[EvaluationResult]) -> None:
        if not results:
            return
        if self._recorder is None:
            return
        if self._pareto_logged:
            return
        meta = self._hof_meta or []
        self._recorder.log(
            _PARETO_COMPLEXITY_KEY, [complexity for complexity, _ in meta]
        )
        self._recorder.log(_PARETO_LOSS_KEY, [loss for _, loss in meta])
        self._recorder.log(
            _PARETO_NMSE_KEY,
            [result.nmse if result.is_valid else None for result in results],
        )
        self._recorder.log(_SELECTED_COMPLEXITY_KEY, self._selected_complexity)
        self._recorder.log(_SELECTED_LOSS_KEY, self._selected_loss)
        self._recorder.log(_SELECTED_NMSE_KEY, self._selected_nmse)
        self._pareto_logged = True



    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_expression(self) -> str:
        return self._best_expression

    @property
    def config(self) -> dict[str, Any]:
        return {
            _STATE_ALGORITHM: ALGORITHM_NAME,
            **asdict(self._config),
        }

    @property
    def terms(self) -> list[str] | None:
        if self._terms is None:
            return None
        return list(self._terms)

    @property
    def derivative_requirements(self) -> DerivativeReqs:
        max_order = assembly.infer_max_atomic_order(list(self._config.terms))
        return DerivativeReqs(
            provider_kind="finite_diff",
            max_atomic_order=max(_MIN_ATOMIC_ORDER, max_order),
            lhs_order=_LHS_ORDER,
            needs_surrogate=False,
        )

    @property
    def state(self) -> dict[str, Any]:
        return {
            _STATE_ALGORITHM: ALGORITHM_NAME,
            _STATE_BEST_EXPRESSION: self._best_expression,
            _STATE_BEST_SCORE: self._best_score,
            _STATE_FITTED: self._fitted,
            _STATE_TERMS: list(self._terms) if self._terms is not None else None,
            _STATE_HOF_CANDIDATES: (
                list(self._hof_candidates) if self._hof_candidates is not None else None
            ),
            _STATE_HOF_META: (
                [list(entry) for entry in self._hof_meta]
                if self._hof_meta is not None
                else None
            ),
        }

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        if not value:
            self._reset_fit_state()
            self._restore_pending = False
            return
        self._reset_fit_state()
        self._best_expression = str(value.get(_STATE_BEST_EXPRESSION, ""))
        self._best_score = float(value.get(_STATE_BEST_SCORE, INITIAL_BEST_SCORE))
        self._fitted = bool(value.get(_STATE_FITTED, False))
        terms = value.get(_STATE_TERMS)
        self._terms = list(terms) if terms is not None else None
        candidates = value.get(_STATE_HOF_CANDIDATES)
        self._hof_candidates = list(candidates) if candidates is not None else None
        meta = value.get(_STATE_HOF_META)
        self._hof_meta = (
            [(int(entry[0]), float(entry[1])) for entry in meta]
            if meta is not None
            else None
        )
        self._restore_pending = True



    def build_final_result(self) -> EvaluationResult:
        if self._best_eval is not None:
            return self._best_eval
        return self._require_evaluator().evaluate_expression(self._best_expression)

    def build_result_target(self) -> Tensor:
        target = self._require_evaluator().lhs_target
        if not isinstance(target, Tensor):
            raise TypeError(
                "components.evaluator.lhs_target must be a Tensor, "
                f"got {type(target).__name__}."
            )
        return target.detach().clone()



    def list_plots(self) -> list[PlotInfo]:
        return _viz_helpers.list_plot_infos()

    def render_plot(self, name: str, ax: Axes) -> None:
        _viz_helpers.render(name, ax, self._recorder)

    def get_plot_data(self, name: str) -> dict[str, Any]:
        return _viz_helpers.get_data(name, self._recorder)



    def _run_fit(self, evaluator: Evaluator) -> None:
        theta, valid_terms = evaluator.build_theta_matrix(list(self._config.terms))
        self._terms = valid_terms
        x_matrix = theta.detach().cpu().numpy()
        y_vector = evaluator.lhs_target.detach().cpu().numpy().reshape(-1)
        feature_names = build_feature_names(
            valid_terms, reserved=assembly.reserved_names(self._dataset)
        )

        backend = self._backend_factory(self._config)
        backend.fit(x_matrix, y_vector, feature_names)

        self._best_expression = assembly.convert_best(
            backend, valid_terms, feature_names
        )
        candidates, meta = assembly.convert_hall_of_fame(
            backend, valid_terms, feature_names
        )
        self._hof_candidates = candidates
        self._hof_meta = meta
        self._selected_complexity, self._selected_loss = assembly.match_selected_entry(
            self._best_expression, candidates, meta
        )
        self._best_eval = evaluator.evaluate_expression(self._best_expression)
        self._best_score = (
            self._best_eval.nmse if self._best_eval.is_valid else INITIAL_BEST_SCORE
        )
        self._selected_nmse = self._resolve_selected_nmse()

    def _resolve_selected_nmse(self) -> float | None:
        if self._selected_complexity is None:
            return None
        if self._best_eval is None or not self._best_eval.is_valid:
            return None
        return self._best_eval.nmse

    def _reset_fit_state(self) -> None:
        self._fitted = False
        self._best_expression = INITIAL_BEST_EXPRESSION
        self._best_score = INITIAL_BEST_SCORE
        self._terms = None
        self._hof_candidates = None
        self._hof_meta = None
        self._best_eval = None
        self._selected_complexity = None
        self._selected_loss = None
        self._selected_nmse = None
        self._pareto_logged = False

    def _require_evaluator(self) -> Evaluator:
        if self._evaluator is None:
            raise RuntimeError("prepare() must be called before using the plugin.")
        return self._evaluator


__all__ = ["PySRConfig", "PySRPlugin"]
