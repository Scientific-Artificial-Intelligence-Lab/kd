
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from torch import Tensor

from kd.core.equation import Form
from kd.core.equation.library import TermLibrarySpec
from kd.core.equation.sketch import Sketch
from kd.core.evaluator import EvaluationResult, Evaluator
from kd.core.platform.requirements import DerivativeReqs
from kd.core.platform.sketch_compile import CompileReport, SketchClauseLevels
from kd.data.schema import DataTopology
from kd.search.descriptor import (
    InstrumentDescriptor,
    InstrumentMode,
    Knob,
    Segmentation,
)
from kd.search.protocol import PlatformComponents
from kd.search.pysr import assembly
from kd.search.pysr import viz as _viz_helpers
from kd.search.pysr.backend import PySRBackend, default_backend_factory
from kd.search.pysr.config import PySRConfig
from kd.search.pysr.convert import build_feature_names
from kd.search.pysr.sketch_backend import (
    PySRCompiled,
    candidate_refusal,
    compile_for_pysr,
    fit_expression,
)
from kd.search.recorder import VizRecorder, log_whitelisted_metrics
from kd.search.result import invalid_evaluation_result
from kd.viz.extension import PlotInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.protocol import DiscoveryTask

logger = logging.getLogger(__name__)

ALGORITHM_NAME = "pysr"
INITIAL_BEST_EXPRESSION = ""
INITIAL_BEST_SCORE = float("inf")



_MIN_ATOMIC_ORDER = 1
_LHS_ORDER = 1












PRE_CLOCK_ALLOWANCE_SECONDS = 45.0

POST_ALLOWANCE_SECONDS = 5.0




TIMEOUT_FLOOR_SECONDS = 15.0









_SKETCH_LEVELS = SketchClauseLevels(
    fixed_terms="lowered",
    anchors="exit_checked",
    hole_count="exit_checked",
    derivative_order="generation_enforced",
    operator_set="exit_checked",
    field_axis_set="generation_enforced",
)





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
_STATE_SEARCH_STATE = "search_state"


class PySRPlugin:




    score_kind: ClassVar[str] = "NMSE"
    score_direction: ClassVar[Literal["min", "max"]] = "min"
    headline_coefficient_source: ClassVar[Literal["native", "platform_refit"]] = (
        "platform_refit"
    )

    config_cls: ClassVar[type[PySRConfig]] = PySRConfig



    one_shot: ClassVar[bool] = True
    sketch_lower_owner: ClassVar[Literal["platform", "native"]] = "platform"
    constructor_kwargs: ClassVar[frozenset[str]] = frozenset({"mode", "task"})

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
                field_names=("u",),
                spatial_axes=("x", "y"),
                names_from="terms",
                description="Grid evolution search over explicit kd terms.",
                forms=frozenset({Form.EVOLUTION}),
                topologies=frozenset({DataTopology.GRID}),
                provider_kind="finite_diff",
                sketch=_SKETCH_LEVELS,
                lhs_orders=frozenset({1}),
            ),
            InstrumentMode(
                name="tabular",
                default_overrides={"unary_operators": ["sin", "cos", "exp", "log"]},
                refused_fields={"terms": "any"},
                forms=frozenset({Form.REGRESSION}),
                topologies=frozenset({DataTopology.TABULAR}),
                provider_kind="none",
                description="Scalar symbolic regression on tabular X -> y data.",
                lhs_orders=frozenset({0}),
            ),
        ),
        knobs=(
            Knob(
                "niterations",
                "int",
                "PySR's internal GP iterations per fit; the facade generations "
                "parameter maps here, and on a resume it is the number of "
                "further iterations the new segment runs from the archived "
                "populations.",
                resume_tier="resume_safe",
            ),
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










        segmentation=Segmentation(archive="progress", unit="iterations", reseed=True),
        identity_breaking_fields=frozenset(
            {"terms", "binary_operators", "unary_operators"}
        ),
        config_artifact_keys=frozenset(),
        surrogate_fields=frozenset(),
    )

    def __init__(
        self,
        config: PySRConfig | None = None,
        *,
        backend_factory: Callable[[PySRConfig], PySRBackend] | None = None,
        mode: Literal["default", "tabular"] = "default",
        task: DiscoveryTask | None = None,
    ) -> None:
        if mode not in ("default", "tabular"):
            raise ValueError(f"mode must be 'default' or 'tabular', got {mode!r}")
        if task is not None and mode == "tabular":
            raise ValueError(
                "PySRPlugin(mode='tabular') does not take a task: the "
                "descriptor declares every sketch clause unsupported on the "
                "tabular mode"
            )
        self._mode = mode
        self._config = config or (
            PySRConfig(
                unary_operators=tuple(
                    next(
                        mode.default_overrides["unary_operators"]
                        for mode in self.descriptor.modes
                        if mode.name == "tabular"
                    )
                )
            )
            if mode == "tabular"
            else PySRConfig()
        )
        self._library: TermLibrarySpec = TermLibrarySpec.from_terms(
            self._compile_task(task)
        )


        self._gate: Callable[[str], str | None] | None = None
        self._fit_expression: Callable[[str], str] | None = None
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


        self._search_state: bytes | None = None

        self._segment_fitted: bool = False


        self._deadline: float | None = None



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
        self._deadline = None
        self._evaluator = components.evaluator
        self._recorder = components.recorder
        self._dataset = components.dataset
        if self._open_sketch is not None:
            sketch, compiled = self._open_sketch
            self._gate = partial(
                candidate_refusal,
                sketch=sketch,
                compiled=compiled,
                registry=components.registry,
            )
            self._fit_expression = partial(
                fit_expression, sketch=sketch, registry=components.registry
            )

    def propose(self, n: int) -> list[str]:
        evaluator = self._require_evaluator()
        if self._segment_fitted:
            return []
        self._run_fit(evaluator)
        self._fitted = True
        self._segment_fitted = True
        return list(self._hof_candidates or [])

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        evaluator = self._require_evaluator()
        return [
            self._evaluate_candidate(evaluator, candidate) for candidate in candidates
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

    def set_wall_clock_deadline(self, deadline: float) -> None:
        if self._deadline is None or deadline < self._deadline:
            self._deadline = deadline



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
    def sketch_compile_report(self) -> CompileReport | None:
        return self._sketch_compile_report

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

        max_order = assembly.infer_max_atomic_order(
            [*self._library.terms, *self._pinned_terms]
        )
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

            _STATE_SEARCH_STATE: self._search_state,
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








        if value.get(_STATE_FITTED, False) and value.get(_STATE_SEARCH_STATE) is None:
            raise ValueError(
                "PySR checkpoint carries a fitted conclusion but no "
                "'search_state' (written before PySR archived its search "
                "state); it cannot be resumed. Run a fresh fit in a new "
                "checkpoint directory."
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
        self._search_state = value.get(_STATE_SEARCH_STATE)
        self._restore_pending = True

    def reseed(self) -> None:
        pass



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



    def _compile_task(self, task: DiscoveryTask | None) -> tuple[str, ...]:
        self._sketch_compile_report: CompileReport | None = None
        self._pinned_terms: tuple[str, ...] = ()
        self._open_sketch: tuple[Sketch, PySRCompiled] | None = None
        if task is None:
            return self._config.terms
        compiled = compile_for_pysr(task.sketch, self._config)
        self._sketch_compile_report = compiled.report
        self._pinned_terms = tuple(pin.term_ir for pin in task.sketch.pinned)
        if not task.compiled.closed:
            self._open_sketch = (task.sketch, compiled)
        return compiled.effective_terms

    def _run_fit(self, evaluator: Evaluator) -> None:
        theta, valid_terms = evaluator.build_theta_matrix(list(self._library.terms))
        if self._search_state is not None and list(valid_terms) != list(
            self._terms or []
        ):





            raise ValueError(
                f"PySR search state was archived over terms {self._terms!r}, "
                f"but this segment's Theta yields {valid_terms!r}; the "
                "archived populations index columns by position, so the "
                "segment cannot continue from it"
            )
        self._terms = valid_terms
        x_matrix = theta.detach().cpu().numpy()
        y_vector = evaluator.lhs_target.detach().cpu().numpy().reshape(-1)
        feature_names = build_feature_names(
            valid_terms, reserved=assembly.reserved_names(self._dataset)
        )

        backend = self._backend_factory(self._config)
        backend.fit(
            x_matrix,
            y_vector,
            feature_names,
            search_state=self._search_state,
            timeout_in_seconds=self._fit_timeout(),
        )
        self._search_state = backend.search_state()

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
            self._best_expression, backend.best_sympy(), backend.hall_of_fame()
        )
        self._best_eval = self._evaluate_candidate(evaluator, self._best_expression)
        if (
            self._gate is not None
            and self._best_eval.invalid_reason == "structural_reject"
        ):


            logger.warning(
                "PySR's selected expression %r scored structural_reject before "
                "kd's refit (%s); the selection is not replaced, so this fit "
                "reports no liftable law",
                self._best_expression,
                self._best_eval.error_message,
            )
        self._best_score = (
            self._best_eval.nmse if self._best_eval.is_valid else INITIAL_BEST_SCORE
        )
        self._selected_nmse = self._resolve_selected_nmse()

    def _fit_timeout(self) -> float | None:
        configured = (self._config.extra_pysr_kwargs or {}).get("timeout_in_seconds")
        timeout: float | None
        if self._deadline is None:
            timeout = configured
            source = "config" if configured is not None else "none"
        else:
            timeout = max(
                TIMEOUT_FLOOR_SECONDS,
                self._deadline
                - time.perf_counter()
                - PRE_CLOCK_ALLOWANCE_SECONDS
                - POST_ALLOWANCE_SECONDS,
            )
            source = "wall-clock deadline"
            if configured is not None and configured < timeout:
                timeout = configured
                source = "config, below the wall-clock deadline"
        logger.info("PySR fit timeout_in_seconds=%s (source: %s)", timeout, source)
        return timeout

    def _evaluate_candidate(
        self,
        evaluator: Evaluator,
        expression: str,
    ) -> EvaluationResult:
        if self._mode == "tabular":





            result = evaluator.evaluate_terms([expression], skip_invalid=False)
            return replace(result, lhs_name=self._dataset.lhs_field)
        if self._gate is not None:
            refusal = self._gate(expression)
            if refusal is not None:
                return invalid_evaluation_result(
                    refusal,
                    score=None,
                    expression=expression,
                    reason="structural_reject",
                )
            assert self._fit_expression is not None
            fitted = self._fit_expression(expression)
            if fitted != expression:
                return replace(
                    evaluator.evaluate_expression(fitted), expression=expression
                )
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
        self._search_state = None
        self._segment_fitted = False

    def _require_evaluator(self) -> Evaluator:
        if self._evaluator is None:
            raise RuntimeError("prepare() must be called before using the plugin.")
        return self._evaluator


__all__ = ["PySRConfig", "PySRPlugin"]
