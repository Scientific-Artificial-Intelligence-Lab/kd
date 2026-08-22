
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from torch import Tensor

from kd.core.equation import Form
from kd.core.equation.library import TermLibrarySpec
from kd.core.evaluator import EvaluationResult, Evaluator
from kd.core.platform.requirements import DerivativeReqs
from kd.data.schema import DataTopology
from kd.search.descriptor import InstrumentDescriptor, InstrumentMode, Knob
from kd.search.protocol import PlatformComponents
from kd.search.pysr import assembly
from kd.search.pysr import viz as _viz_helpers
from kd.search.pysr.backend import PySRBackend, default_backend_factory
from kd.search.pysr.config import PySRConfig
from kd.search.pysr.convert import build_feature_names
from kd.search.recorder import VizRecorder, log_whitelisted_metrics
from kd.viz.extension import PlotInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

ALGORITHM_NAME = "pysr"
INITIAL_BEST_EXPRESSION = ""
INITIAL_BEST_SCORE = float("inf")



_MIN_ATOMIC_ORDER = 1
_LHS_ORDER = 1





_PARETO_COMPLEXITY_KEY = _viz_helpers.PARETO_COMPLEXITY_KEY
_PARETO_LOSS_KEY = _viz_helpers.PARETO_LOSS_KEY
_PARETO_NMSE_KEY = _viz_helpers.PARETO_NMSE_KEY
_PARETO_EXPRESSIONS_KEY = _viz_helpers.PARETO_EXPRESSIONS_KEY
_PARETO_SCALE_KEY = _viz_helpers.PARETO_SCALE_KEY
_SELECTED_COMPLEXITY_KEY = _viz_helpers.SELECTED_COMPLEXITY_KEY
_SELECTED_LOSS_KEY = _viz_helpers.SELECTED_LOSS_KEY
_SELECTED_NMSE_KEY = _viz_helpers.SELECTED_NMSE_KEY
_LOGGED_METRICS = _viz_helpers.LOGGED_METRICS


_CONFIG_LIBRARY_FINGERPRINT = "library_fingerprint"
_STATE_ALGORITHM = "algorithm"
_STATE_BEST_EXPRESSION = "best_expression"
_STATE_BEST_SCORE = "best_score"
_STATE_FITTED = "fitted"
_STATE_TERMS = "terms"
_STATE_HOF_CANDIDATES = "hof_candidates"
_STATE_HOF_META = "hof_meta"
_STATE_MODE = "mode"


class PySRPlugin:




    score_kind: ClassVar[str] = "NMSE"
    score_direction: ClassVar[Literal["min", "max"]] = "min"
    headline_coefficient_source: ClassVar[Literal["native", "platform_refit"]] = (
        "platform_refit"
    )

    config_cls: ClassVar[type[PySRConfig]] = PySRConfig



    one_shot: ClassVar[bool] = True
    sketch_lower_owner: ClassVar[Literal["platform", "native"]] = "platform"

    descriptor: ClassVar[InstrumentDescriptor] = InstrumentDescriptor(
        algorithm="pysr",
        summary=(
            "One-shot PySR search over a configured kd term library (PDE "
            "mode) or dataset-derived feature columns (tabular mode)."
        ),
        cost_class="medium",
        modes=(
            InstrumentMode(
                name="default",
                forms=frozenset({Form.EVOLUTION}),
                topologies=frozenset({DataTopology.GRID}),
                provider_kind="finite_diff",
            ),
            InstrumentMode(
                name="tabular",
                forms=frozenset({Form.REGRESSION}),
                topologies=frozenset({DataTopology.TABULAR}),
                provider_kind="none",
                description="Scalar symbolic regression on tabular X -> y data.",
            ),
        ),
        knobs=(
            Knob(
                "population_size",
                "int",
                "Members per PySR population.",
                resume_tier="init_only",
            ),
            Knob(
                "populations",
                "int",
                "Number of PySR populations.",
                resume_tier="init_only",
            ),
            Knob(
                "maxsize",
                "int",
                "Maximum generated expression size.",
                resume_tier="init_only",
            ),
        ),
    )

    def __init__(
        self,
        config: PySRConfig | None = None,
        *,
        backend_factory: Callable[[PySRConfig], PySRBackend] | None = None,
        mode: Literal["default", "tabular"] = "default",
    ) -> None:
        if mode not in ("default", "tabular"):
            raise ValueError(f"mode must be 'default' or 'tabular', got {mode!r}")
        self._mode = mode
        self._config = config or PySRConfig()
        self._library: TermLibrarySpec = TermLibrarySpec.from_terms(self._config.terms)
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
        if components.evaluator is None:
            raise TypeError(
                "PySRPlugin requires components.evaluator (Theta build + "
                "re-scoring); got None. Assemble the platform with an "
                "evaluator (PlatformBuilder default) to run PySR."
            )
        if self._mode == "tabular":





            target = getattr(components.dataset, "lhs_field", None)
            if target is not None and target in self._library.terms:
                raise ValueError(
                    f"tabular term catalog contains the target column "
                    f"{target!r}; the target must not appear among the "
                    "feature terms (self-regression tautology)"
                )
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
        return [
            self._evaluate_candidate(evaluator, candidate)
            for candidate in candidates
        ]

    def update(self, results: list[EvaluationResult]) -> None:
        if not results:
            return
        if self._recorder is None:
            return
        if self._pareto_logged:
            return
        meta = self._hof_meta or []
        metrics: dict[str, Any] = {
            _PARETO_COMPLEXITY_KEY: [complexity for complexity, _ in meta],
            _PARETO_LOSS_KEY: [loss for _, loss in meta],
            _PARETO_NMSE_KEY: [
                result.nmse if result.is_valid else None for result in results
            ],
            _PARETO_EXPRESSIONS_KEY: list(self._hof_candidates or []),



            **(
                {
                    _PARETO_SCALE_KEY: [
                        float(result.coefficients.flatten()[0])
                        if result.is_valid
                        and result.coefficients is not None
                        and result.coefficients.numel()
                        else None
                        for result in results
                    ]
                }
                if self._mode == "tabular"
                else {}
            ),
            _SELECTED_COMPLEXITY_KEY: self._selected_complexity,
            _SELECTED_LOSS_KEY: self._selected_loss,
            _SELECTED_NMSE_KEY: self._selected_nmse,
        }



        log_whitelisted_metrics(
            self._recorder, _LOGGED_METRICS, metrics, skip_missing=True
        )
        self._pareto_logged = True



    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_expression(self) -> str:
        return self._best_expression

    @property
    def mode(self) -> Literal["default", "tabular"]:
        return self._mode

    @property
    def config(self) -> dict[str, Any]:
        return {
            _STATE_ALGORITHM: ALGORITHM_NAME,
            _CONFIG_LIBRARY_FINGERPRINT: self._library.fingerprint,
            **asdict(self._config),
        }

    @property
    def runner_batch_size(self) -> int:
        return 1

    @property
    def terms(self) -> list[str] | None:
        if self._terms is None:
            return None
        return list(self._terms)

    @property
    def derivative_requirements(self) -> DerivativeReqs:
        if self._mode == "tabular":
            return DerivativeReqs(
                provider_kind="none",
                lhs_order=0,
                lhs_source="field",
                supported_topologies=frozenset({DataTopology.TABULAR}),
            )
        max_order = assembly.infer_max_atomic_order(list(self._library.terms))
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
            _STATE_MODE: self._mode,




            _CONFIG_LIBRARY_FINGERPRINT: self._library.fingerprint,
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







        stored_mode = value.get(_STATE_MODE, "default")
        if stored_mode != self._mode:
            raise ValueError(
                f"PySR checkpoint mode {stored_mode!r} does not match "
                f"plugin mode {self._mode!r}"
            )




        if _CONFIG_LIBRARY_FINGERPRINT in value:
            stored_fingerprint = value[_CONFIG_LIBRARY_FINGERPRINT]
            if stored_fingerprint != self._library.fingerprint:



                raise ValueError(
                    f"PySR checkpoint library_fingerprint {stored_fingerprint!r} "
                    f"does not match this plugin's catalog "
                    f"{self._library.fingerprint!r}; restore requires the same "
                    "term catalog"
                )



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
        evaluator = self._require_evaluator()
        return self._evaluate_candidate(evaluator, self._best_expression)

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

    def render_plot(self, name: str, ax: Axes) -> list[str]:
        return _viz_helpers.render(name, ax, self._recorder)

    def get_plot_data(self, name: str) -> dict[str, Any]:
        return _viz_helpers.get_data(name, self._recorder)



    def _run_fit(self, evaluator: Evaluator) -> None:
        theta, valid_terms = evaluator.build_theta_matrix(list(self._library.terms))
        self._terms = valid_terms
        x_matrix = theta.detach().cpu().numpy()
        y_vector = evaluator.lhs_target.detach().cpu().numpy().reshape(-1)
        feature_names = build_feature_names(
            valid_terms, reserved=assembly.reserved_names(self._dataset)
        )

        backend = self._backend_factory(self._config)
        backend.fit(x_matrix, y_vector, feature_names)

        if self._mode == "tabular":
            self._best_expression = assembly.convert_best_tabular(
                backend, valid_terms, feature_names
            )
            candidates, meta = assembly.convert_hall_of_fame_tabular(
                backend, valid_terms, feature_names
            )
        else:
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
        self._best_eval = self._evaluate_candidate(evaluator, self._best_expression)
        self._best_score = (
            self._best_eval.nmse if self._best_eval.is_valid else INITIAL_BEST_SCORE
        )
        self._selected_nmse = self._resolve_selected_nmse()

    def _evaluate_candidate(
        self,
        evaluator: Evaluator,
        expression: str,
    ) -> EvaluationResult:
        if self._mode == "tabular":





            result = evaluator.evaluate_terms([expression], skip_invalid=False)
            return replace(result, lhs_name=self._dataset.lhs_field)
        return evaluator.evaluate_expression(expression)

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
