
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from torch import Tensor

from kd.core.equation import Form, LhsSpec
from kd.core.equation.library import TermLibrarySpec
from kd.core.equation.rendering import render_lhs_label
from kd.core.equation.signature import law_term_key
from kd.core.evaluator import EvaluationResult, Evaluator
from kd.core.platform.requirements import DerivativeReqs
from kd.core.platform.sketch_compile import CompileReport, SketchClauseLevels
from kd.data.schema import DataTopology, PDEDataset
from kd.search.descriptor import (
    InstrumentDescriptor,
    InstrumentMode,
    Knob,
    Segmentation,
)
from kd.search.protocol import PlatformComponents
from kd.search.pysindy import viz as _viz_helpers
from kd.search.pysindy.assembly import (
    build_native_result,
    render_best_expression,
    support_from_coefficients,
)
from kd.search.pysindy.backend import (
    PySINDyOptimizerBackend,
    default_backend_factory,
)
from kd.search.pysindy.config import PySINDyConfig
from kd.search.pysindy.hole_subset import HOLE_SUBSET_MAX, select_hole_subset
from kd.search.pysindy.sketch_backend import compile_for_pysindy
from kd.search.recorder import VizRecorder, log_whitelisted_metrics
from kd.search.term_utils import infer_max_atomic_order
from kd.viz.extension import PlotInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.protocol import DiscoveryTask

ALGORITHM_NAME = "pysindy"
INITIAL_BEST_EXPRESSION = ""
INITIAL_BEST_SCORE = float("inf")

_MIN_ATOMIC_ORDER = 1
_LHS_ORDER = 1



_NATIVE_NMSE_KEY = _viz_helpers.NATIVE_NMSE_KEY
_REFIT_NMSE_KEY = _viz_helpers.REFIT_NMSE_KEY
_SUPPORT_SIZE_KEY = _viz_helpers.SUPPORT_SIZE_KEY
_STLSQ_HISTORY_KEY = _viz_helpers.STLSQ_HISTORY_KEY
_LOGGED_METRICS = _viz_helpers.LOGGED_METRICS

_STATE_ALGORITHM = "algorithm"
_STATE_BEST_EXPRESSION = "best_expression"
_STATE_BEST_SCORE = "best_score"
_STATE_FITTED = "fitted"
_STATE_TERMS = "terms"
_STATE_SUPPORT = "support"
_STATE_COEFFICIENT_VALUES = "coefficient_values"
_CONFIG_LIBRARY_FINGERPRINT = "library_fingerprint"
_EMPTY_LAW_MESSAGE = (
    "PySINDy hole-count selection chose the empty law: every admitted subset "
    "scored worse than no term"
)


class PySINDyPlugin:

    score_kind: ClassVar[str] = "NMSE"
    score_direction: ClassVar[Literal["min", "max"]] = "min"
    headline_coefficient_source: ClassVar[Literal["native", "platform_refit"]] = (
        "native"
    )
    config_cls: ClassVar[type[PySINDyConfig]] = PySINDyConfig
    one_shot: ClassVar[bool] = True
    sketch_lower_owner: ClassVar[Literal["platform", "native"]] = "platform"
    constructor_kwargs: ClassVar[frozenset[str]] = frozenset({"task", "mode"})



    descriptor: ClassVar[InstrumentDescriptor] = InstrumentDescriptor(
        algorithm="pysindy",
        summary=(
            "One-shot native PySINDy STLSQ over a configured kd term library "
            "(grid mode) or a feature-derived polynomial library (tabular mode)."
        ),
        cost_class="light",
        modes=(
            InstrumentMode(
                name="default",
                field_names=("u",),
                spatial_axes=("x", "y"),
                names_from="terms",
                refused_fields={"degree": "non_default"},
                description="Grid evolution regression over explicit kd terms.",
                forms=frozenset({Form.EVOLUTION}),
                topologies=frozenset({DataTopology.GRID}),
                provider_kind="finite_diff",
                sketch=SketchClauseLevels(
                    fixed_terms="lowered",
                    anchors="fit_enforced",
                    hole_count="exit_checked",
                    derivative_order="generation_enforced",
                    operator_set="generation_enforced",
                    field_axis_set="generation_enforced",
                ),
                lhs_orders=frozenset({1}),
            ),
            InstrumentMode(
                name="tabular",
                refused_fields={"terms": "any"},
                forms=frozenset({Form.REGRESSION}),
                topologies=frozenset({DataTopology.TABULAR}),
                provider_kind="none",
                description="Polynomial regression on tabular X -> y data.",
                lhs_orders=frozenset({0}),
            ),
        ),
        knobs=(
            Knob(
                "threshold",
                "float",
                "STLSQ sparsity threshold.",
                resume_tier="init_only",
            ),
            Knob(
                "max_iter",
                "int",
                "Maximum STLSQ iterations.",
                resume_tier="init_only",
            ),
            Knob(
                "normalize_columns",
                "bool",
                "Normalize library columns before STLSQ.",
                resume_tier="init_only",
            ),
            Knob(
                "relative_threshold",
                "bool",
                "Read threshold as a fraction of the target L2 norm "
                "(requires normalize_columns).",
                resume_tier="init_only",
            ),
            Knob(
                "degree",
                "int",
                "Maximum total polynomial degree for tabular data.",
                resume_tier="init_only",
            ),
        ),



        segmentation=Segmentation(archive="conclusion", unit="fits", reseed=False),
        identity_breaking_fields=frozenset({"terms"}),
        config_artifact_keys=frozenset(),
        surrogate_fields=frozenset(),
    )

    def __init__(
        self,
        config: PySINDyConfig | None = None,
        *,
        task: DiscoveryTask | None = None,
        backend_factory: Callable[[PySINDyConfig], PySINDyOptimizerBackend]
        | None = None,
        mode: Literal["default", "tabular"] = "default",
    ) -> None:
        if mode not in ("default", "tabular"):
            raise ValueError(f"mode must be 'default' or 'tabular', got {mode!r}")
        if task is not None and mode == "tabular":
            raise ValueError(
                "PySINDyPlugin(mode='tabular') does not take a task: the "
                "descriptor declares every sketch clause unsupported on the "
                "tabular mode"
            )
        self._mode = mode
        self._lhs_name: str | None = None
        self._config = config or PySINDyConfig()
        self._anchor_keys = frozenset(
            law_term_key(anchor.term_ir)
            for anchor in (() if task is None else task.sketch.anchored)
        )
        if task is None:
            effective_terms = self._config.terms
            self._sketch_compile_report: CompileReport | None = None
            self._pinned_terms: tuple[str, ...] = ()
            self._hole_count_bounds: tuple[int, int] | None = None
        else:
            compiled = compile_for_pysindy(task.sketch, self._config.terms)
            effective_terms = compiled.effective_terms
            self._sketch_compile_report = compiled.report
            self._pinned_terms = tuple(pin.term_ir for pin in task.sketch.pinned)
            self._hole_count_bounds = compiled.hole_count_bounds
        self._library = TermLibrarySpec.from_terms(effective_terms)
        self._backend_factory = backend_factory or default_backend_factory
        self._evaluator: Evaluator | None = None
        self._recorder: VizRecorder | None = None
        self._fitted = False
        self._best_expression = INITIAL_BEST_EXPRESSION
        self._best_score = INITIAL_BEST_SCORE
        self._terms: list[str] | None = None
        self._support: list[int] | None = None
        self._coefficient_values: list[float] | None = None



        self._stlsq_history: list[list[float]] | None = None
        self._final_eval: EvaluationResult | None = None
        self._metrics_logged = False
        self._restore_pending = False

    def refuse_lhs_terms(self, dataset: PDEDataset) -> None:
        if self._mode == "tabular":
            return
        lhs = render_lhs_label(
            LhsSpec(dataset.lhs_field, dataset.lhs_axis, dataset.lhs_order)
        )


        target_terms = {
            lhs,
            f"diff_{dataset.lhs_axis}({dataset.lhs_field})",
            f"diff1_{dataset.lhs_axis}({dataset.lhs_field})",
        }
        for term in self._library.terms:
            if term in target_terms:
                raise ValueError(
                    f"terms contains LHS {lhs!r} as {term!r}: tautology; "
                    f"remove that term"
                )

    def prepare(self, components: PlatformComponents) -> None:
        self.refuse_lhs_terms(components.dataset)
        if components.evaluator is None:
            raise TypeError(
                "PySINDyPlugin requires components.evaluator for Theta build "
                "and kd re-scoring; got None"
            )
        if self._mode == "tabular":




            target = components.dataset.lhs_field
            if target in self._library.terms:
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
        self._lhs_name = (
            components.dataset.lhs_field if self._mode == "tabular" else None
        )

    def propose(self, n: int) -> list[str]:
        evaluator = self._require_evaluator()
        if self._fitted:
            return []
        self._run_fit(evaluator)
        self._fitted = True
        return [self._best_expression]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        evaluator = self._require_evaluator()
        return [evaluator.evaluate_expression(candidate) for candidate in candidates]

    def update(self, results: list[EvaluationResult]) -> None:
        if not results or self._recorder is None or self._metrics_logged:
            return
        refit = results[0]
        metrics: dict[str, Any] = {
            _NATIVE_NMSE_KEY: self._best_score,
            _REFIT_NMSE_KEY: refit.nmse if refit.is_valid else None,
            _SUPPORT_SIZE_KEY: len(self._support or []),
            _STLSQ_HISTORY_KEY: self._stlsq_history,
        }
        log_whitelisted_metrics(self._recorder, _LOGGED_METRICS, metrics)
        self._metrics_logged = True



    def list_plots(self) -> list[PlotInfo]:
        return _viz_helpers.list_plot_infos()

    def render_plot(self, name: str, ax: Axes) -> list[str]:
        return _viz_helpers.render(name, ax, self._recorder)

    def get_plot_data(self, name: str) -> dict[str, Any]:
        return _viz_helpers.get_data(name, self._recorder)

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
            _CONFIG_LIBRARY_FINGERPRINT: self._library.fingerprint,
            **asdict(self._config),
        }

    @property
    def runner_batch_size(self) -> int:
        return 1

    @property
    def terms(self) -> list[str] | None:
        return list(self._terms) if self._terms is not None else None

    @property
    def sketch_compile_report(self) -> CompileReport | None:
        return self._sketch_compile_report

    @property
    def derivative_requirements(self) -> DerivativeReqs:
        if self._mode == "tabular":
            return DerivativeReqs(
                provider_kind="none",
                lhs_order=0,
                lhs_source="field",
                supported_topologies=frozenset({DataTopology.TABULAR}),
            )
        max_order = infer_max_atomic_order([*self._library.terms, *self._pinned_terms])
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
            _CONFIG_LIBRARY_FINGERPRINT: self._library.fingerprint,
            _STATE_BEST_EXPRESSION: self._best_expression,
            _STATE_BEST_SCORE: self._best_score,
            _STATE_FITTED: self._fitted,
            _STATE_TERMS: list(self._terms) if self._terms is not None else None,
            _STATE_SUPPORT: (
                list(self._support) if self._support is not None else None
            ),
            _STATE_COEFFICIENT_VALUES: (
                list(self._coefficient_values)
                if self._coefficient_values is not None
                else None
            ),
        }

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        if not value:
            self._reset_fit_state()
            self._restore_pending = False
            return





        if _CONFIG_LIBRARY_FINGERPRINT in value:
            stored_fingerprint = value[_CONFIG_LIBRARY_FINGERPRINT]
            if stored_fingerprint != self._library.fingerprint:



                raise ValueError(
                    f"PySINDy checkpoint library_fingerprint "
                    f"{stored_fingerprint!r} does not match this plugin's "
                    f"catalog {self._library.fingerprint!r}; restore requires "
                    "the same term catalog"
                )



        self._reset_fit_state()
        self._best_expression = str(value.get(_STATE_BEST_EXPRESSION, ""))
        self._best_score = float(value.get(_STATE_BEST_SCORE, INITIAL_BEST_SCORE))
        self._fitted = bool(value.get(_STATE_FITTED, False))
        terms = value.get(_STATE_TERMS)
        self._terms = list(terms) if terms is not None else None
        support = value.get(_STATE_SUPPORT)
        self._support = (
            [int(index) for index in support] if support is not None else None
        )
        coefficients = value.get(_STATE_COEFFICIENT_VALUES)
        self._coefficient_values = (
            [float(coefficient) for coefficient in coefficients]
            if coefficients is not None
            else None
        )
        self._restore_pending = True

    def build_final_result(self) -> EvaluationResult:
        if self._final_eval is not None:
            return self._final_eval
        evaluator = self._require_evaluator()
        if not self._fitted:
            result = evaluator.evaluate_expression(self._best_expression)
            result.lhs_name = self._lhs_name
            return result
        if self._terms is None or self._coefficient_values is None:
            raise RuntimeError(
                "Restored PySINDy fit lacks terms or native coefficient_values"
            )
        theta, valid_terms = evaluator.build_theta_matrix(list(self._terms))
        if valid_terms != self._terms:
            raise RuntimeError(
                "Restored PySINDy term catalog no longer aligns with Theta columns"
            )
        xi = np.asarray(self._coefficient_values, dtype=np.float64)
        self._validate_xi(xi, len(valid_terms))
        rebuilt_support = support_from_coefficients(xi)
        if self._support is not None and rebuilt_support != self._support:
            raise RuntimeError("Restored PySINDy support disagrees with coefficients")
        self._final_eval = build_native_result(
            theta=theta,
            lhs_flat=evaluator.lhs_target,
            xi=xi,
            terms=valid_terms,
            expression=self._best_expression,
        )
        self._final_eval.lhs_name = self._lhs_name




        self._best_score = self._final_eval.nmse
        return self._final_eval

    def build_result_target(self) -> Tensor:
        target = self._require_evaluator().lhs_target
        if not isinstance(target, Tensor):
            raise TypeError(
                "components.evaluator.lhs_target must be a Tensor, "
                f"got {type(target).__name__}"
            )
        return target.detach().clone()

    def _run_fit(self, evaluator: Evaluator) -> None:
        theta, valid_terms = evaluator.build_theta_matrix(list(self._library.terms))
        x_matrix = theta.detach().cpu().numpy().astype(np.float64)
        y_vector = (
            evaluator.lhs_target.detach().cpu().numpy().astype(np.float64).reshape(-1)
        )
        backend = self._backend_factory(self._config)
        if self._anchor_keys:
            xi, history = self._fit_with_anchors(
                backend, x_matrix, y_vector, valid_terms
            )
        else:
            backend.fit(x_matrix, y_vector)
            xi = np.asarray(backend.coefficients(), dtype=np.float64)
        self._validate_xi(xi, len(valid_terms))
        xi, support = self._enforce_hole_count(
            xi, support_from_coefficients(xi), x_matrix, y_vector, valid_terms
        )
        if not support:
            raise RuntimeError(
                "PySINDy STLSQ selected empty support; lower threshold "
                f"(current threshold={self._config.threshold})"
                + (
                    "; threshold is a fraction of the target norm"
                    if self._config.relative_threshold
                    else ""
                )
            )

        expression = render_best_expression(valid_terms, support)
        final_eval = build_native_result(
            theta=theta,
            lhs_flat=evaluator.lhs_target,
            xi=xi,
            terms=valid_terms,
            expression=expression,
        )
        final_eval.lhs_name = self._lhs_name
        self._terms = list(valid_terms)
        self._support = support
        self._coefficient_values = [float(value) for value in xi]
        self._stlsq_history = history if self._anchor_keys else backend.history()
        self._best_expression = expression
        self._final_eval = final_eval
        self._best_score = final_eval.nmse

    def _enforce_hole_count(
        self,
        xi: np.ndarray,
        support: list[int],
        theta: np.ndarray,
        target: np.ndarray,
        terms: list[str],
    ) -> tuple[np.ndarray, list[int]]:
        if self._hole_count_bounds is None:
            return xi, support
        min_terms, max_terms = self._hole_count_bounds
        anchor_indices = [
            index
            for index, term in enumerate(terms)
            if law_term_key(term) in self._anchor_keys
        ]
        hole_terms = len(set(support) - set(anchor_indices))
        if support and min_terms <= hole_terms <= max_terms:
            return xi, support
        candidate_indices = self._identifiable_columns(theta, terms, anchor_indices)
        if len(candidate_indices) < min_terms:
            return xi, support
        choice = select_hole_subset(
            theta,
            target,
            anchor_indices=anchor_indices,
            min_terms=min_terms,
            max_terms=max_terms,
            candidate_indices=candidate_indices,
        )
        chosen = support_from_coefficients(choice.coefficients)
        if not chosen:
            raise RuntimeError(_EMPTY_LAW_MESSAGE)
        return choice.coefficients, chosen

    def _identifiable_columns(
        self, theta: np.ndarray, terms: list[str], anchor_indices: list[int]
    ) -> list[int]:
        rest_indices = [i for i in range(len(terms)) if i not in anchor_indices]
        anchors, rest = theta[:, anchor_indices], theta[:, rest_indices]
        deflated = rest
        if anchor_indices:
            _unit, _scales, q = self._unit_anchor_basis(
                anchors, [terms[index] for index in anchor_indices]
            )
            deflated = rest - q @ (q.T @ rest)
        kept = self._outside_anchor_span(rest, deflated, anchors.shape)
        return [index for index, keep in zip(rest_indices, kept, strict=True) if keep]

    def _fit_with_anchors(
        self,
        backend: PySINDyOptimizerBackend,
        theta: np.ndarray,
        target: np.ndarray,
        terms: list[str],
    ) -> tuple[np.ndarray, list[list[float]]]:
        anchor_indices = [
            index
            for index, term in enumerate(terms)
            if law_term_key(term) in self._anchor_keys
        ]
        rest_indices = [i for i in range(len(terms)) if i not in anchor_indices]
        anchors, rest = theta[:, anchor_indices], theta[:, rest_indices]
        unit_anchors, scales, q = self._unit_anchor_basis(
            anchors, [terms[index] for index in anchor_indices]
        )
        deflated = rest - q @ (q.T @ rest)
        kept = self._outside_anchor_span(rest, deflated, anchors.shape)
        fitted = [index for index, keep in zip(rest_indices, kept, strict=True) if keep]

        def restore(coefficients: np.ndarray) -> np.ndarray:
            self._validate_xi(coefficients, len(fitted))
            xi = np.zeros(len(terms), dtype=np.float64)
            xi[fitted] = coefficients


            residual = target - theta[:, fitted] @ coefficients
            unit = np.linalg.lstsq(unit_anchors, residual, rcond=None)[0]
            xi[anchor_indices] = unit / scales
            return xi


        if not fitted:
            xi = restore(np.empty(0, dtype=np.float64))
            return xi, [xi.tolist()]

        backend.fit(deflated[:, kept], target - q @ (q.T @ target))
        xi = restore(np.asarray(backend.coefficients(), dtype=np.float64))
        history = [
            restore(np.asarray(row, dtype=np.float64)).tolist()
            for row in backend.history()
        ]
        return xi, history

    @staticmethod
    def _outside_anchor_span(
        rest: np.ndarray, deflated: np.ndarray, anchor_shape: tuple[int, ...]
    ) -> np.ndarray:
        span_tol = max(anchor_shape) * np.finfo(np.float64).eps
        return np.asarray(
            np.linalg.norm(deflated, axis=0) > span_tol * np.linalg.norm(rest, axis=0)
        )

    @staticmethod
    def _unit_anchor_basis(
        anchors: np.ndarray, names: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scales = np.linalg.norm(anchors, axis=0)
        if not np.all(scales > 0) or np.linalg.matrix_rank(anchors / scales) < len(
            names
        ):
            raise ValueError(f"PySINDy linearly dependent anchors: {names!r}")
        unit_anchors = anchors / scales
        q, _r = np.linalg.qr(unit_anchors, mode="reduced")
        return unit_anchors, scales, q

    @staticmethod
    def _validate_xi(xi: np.ndarray, n_terms: int) -> None:
        if xi.shape != (n_terms,):
            raise RuntimeError(
                "PySINDy backend coefficients have incompatible shape "
                f"{xi.shape}; expected ({n_terms},)"
            )
        if not np.isfinite(xi).all():
            raise RuntimeError("PySINDy backend coefficients must be finite")

    def _reset_fit_state(self) -> None:
        self._fitted = False
        self._best_expression = INITIAL_BEST_EXPRESSION
        self._best_score = INITIAL_BEST_SCORE
        self._terms = None
        self._support = None
        self._coefficient_values = None
        self._stlsq_history = None
        self._final_eval = None
        self._metrics_logged = False

    def _require_evaluator(self) -> Evaluator:
        if self._evaluator is None:
            raise RuntimeError("prepare() must be called before using the plugin")
        return self._evaluator


__all__ = ["HOLE_SUBSET_MAX", "PySINDyConfig", "PySINDyPlugin"]
