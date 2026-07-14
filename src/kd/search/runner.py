
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from kd.core.equation import (
    HOMOGENEOUS_LHS_LABEL,
    Equation,
    Evolution,
    Form,
    LhsSpec,
    build_equation,
    build_homogeneous,
    render_lhs_label,
)
from kd.core.evaluator import EvaluationResult
from kd.core.expr.naming import build_derivative_name, parse_derivative_name
from kd.core.platform.requirements import DerivativeReqs, assert_dataset_supported
from kd.data.schema import PDEDataset, compute_dataset_fingerprint
from kd.search.callbacks import (
    CHECKPOINT_VERSION,
    RunnerCallback,
    VizDataCollector,
    _algorithm_name,
    atomic_torch_save,
    build_checkpoint_payload,
)
from kd.search.lifecycle import SearchLifecycle
from kd.search.protocol import (
    IterativeSearchAlgorithm,
    PlatformComponents,
    ScoreContract,
    SearchAlgorithm,
    TerminatingSearchAlgorithm,
)
from kd.search.recorder import VizRecorder
from kd.search.result import (
    DEFAULT_SCORE_DIRECTION,
    DEFAULT_SCORE_KIND,
    ExperimentResult,
    RunManifest,
    RunResult,
    invalid_evaluation_result,
)

logger = logging.getLogger(__name__)





_DEFAULT_LHS_LABEL = "u_t"



_NON_PDE_DATASET_FINGERPRINT = "<non-pde-dataset>"



_CHECKPOINT_REQUIRED_KEYS = ("version", "iteration", "algorithm_state")









_SEARCH_ALGORITHM_MEMBERS: tuple[str, ...] = (
    "prepare",
    "propose",
    "evaluate",
    "update",
    "best_score",
    "best_expression",
    "config",
    "state",
    "build_final_result",
    "build_result_target",
)




_MISSING_ATTR: object = object()


def _implements_member(algorithm: object, member: str) -> bool:
    attr = getattr(type(algorithm), member, _MISSING_ATTR)
    if attr is not _MISSING_ATTR and attr is not getattr(
        SearchAlgorithm, member, None
    ):
        return True
    return member in getattr(algorithm, "__dict__", {})


class ExperimentRunner:

    def __init__(
        self,
        algorithm: SearchAlgorithm,
        max_iterations: int = 100,
        batch_size: int = 20,
        callbacks: list[RunnerCallback] | None = None,
    ) -> None:
        self._algorithm = algorithm
        self._max_iterations = max_iterations
        self._batch_size = batch_size
        self._callbacks: list[RunnerCallback] = (
            callbacks if callbacks is not None else []
        )
        self._current_iteration: int = 0



        self._lifecycle: SearchLifecycle | None = None




        self._pending_restore_state: dict[str, Any] | None = None

    @property
    def lifecycle(self) -> SearchLifecycle | None:
        return self._lifecycle

    def run(self, components: PlatformComponents) -> ExperimentResult:
        self._assert_algorithm_protocol()
        self._assert_dataset_supported(components)
        self._current_iteration = 0



        lifecycle = SearchLifecycle()
        self._lifecycle = lifecycle
        pending_restore = self._pending_restore_state
        if pending_restore is not None:
            lifecycle.restore(pending_restore)
        recorder = self._ensure_recorder(components)
        callbacks = self._callbacks_for_run(recorder)
        self._algorithm.prepare(components)
        lifecycle.prepare()



        self._pending_restore_state = None

        for cb in callbacks:
            cb.on_experiment_start(self._algorithm)

        early_stopped = False
        iterative_alg = (
            self._algorithm
            if isinstance(self._algorithm, IterativeSearchAlgorithm)
            else None
        )


















        terminating_alg: TerminatingSearchAlgorithm | None = (
            cast(TerminatingSearchAlgorithm, self._algorithm)
            if _implements_member(self._algorithm, "is_done")
            else None
        )
        try:
            for iteration in range(self._max_iterations):
                self._run_iteration(iteration, callbacks)
                lifecycle.iterate()
                self._current_iteration = iteration + 1
                if any(cb.should_stop for cb in callbacks) or (
                    terminating_alg is not None and terminating_alg.is_done
                ):
                    early_stopped = True
                    break
                if iterative_alg is not None and iteration < self._max_iterations - 1:
                    iterative_alg.between_iterations()
        finally:
            self._finalize_callbacks(callbacks)



        lifecycle.finish()
        return self._build_experiment_result(components, recorder, early_stopped)

    def _assert_algorithm_protocol(self) -> None:
        cls = type(self._algorithm)
        missing = [
            member
            for member in _SEARCH_ALGORITHM_MEMBERS
            if not _implements_member(self._algorithm, member)
        ]
        if missing:
            raise TypeError(
                f"{cls.__name__} does not satisfy the SearchAlgorithm "
                f"protocol; missing or unimplemented members: "
                f"{', '.join(missing)}. Members inherited from the "
                "SearchAlgorithm/IterativeSearchAlgorithm Protocol base "
                "itself are no-op stubs and count as unimplemented. "
                "( merged result building into the protocol; "
                "simple algorithms can delegate via "
                "kd.search.result.default_final_result.)"
            )

    def _assert_dataset_supported(self, components: PlatformComponents) -> None:
        reqs = getattr(self._algorithm, "derivative_requirements", None)
        dataset = getattr(components, "dataset", None)
        if isinstance(reqs, DerivativeReqs) and isinstance(dataset, PDEDataset):
            algorithm = (
                _algorithm_name(self._algorithm) or type(self._algorithm).__name__
            )
            assert_dataset_supported(
                dataset.lhs_order, dataset.topology, reqs, algorithm
            )

    def _run_iteration(
        self,
        iteration: int,
        callbacks: list[RunnerCallback],
    ) -> None:
        for cb in callbacks:
            cb.on_iteration_start(iteration, self._algorithm)

        candidates = self._algorithm.propose(self._batch_size)
        results = self._algorithm.evaluate(candidates)
        self._validate_evaluation_results(candidates, results)
        self._algorithm.update(results)

        for cb in callbacks:
            cb.on_iteration_end(iteration, self._algorithm, candidates, results)

    @staticmethod
    def _validate_evaluation_results(
        candidates: list[str],
        results: list[EvaluationResult],
    ) -> None:
        if len(results) != len(candidates):
            raise RuntimeError(
                f"Plugin contract violation: evaluate returned "
                f"{len(results)} results for {len(candidates)} candidates "
                f"(propose/evaluate must maintain 1:1 correspondence)."
            )
        for i, r in enumerate(results):
            if not isinstance(r, EvaluationResult):
                raise TypeError(
                    f"Plugin contract violation: evaluate result[{i}] is "
                    f"{type(r).__name__}, expected EvaluationResult."
                )

    def _ensure_recorder(self, components: PlatformComponents) -> VizRecorder:
        recorder = getattr(components, "recorder", None)
        if recorder is None:
            recorder = self._callback_recorder()
        if recorder is None:
            recorder = VizRecorder()







        if getattr(components, "recorder", None) is None:
            components.recorder = recorder
        return recorder

    def _callback_recorder(self) -> VizRecorder | None:
        for cb in self._callbacks:
            if isinstance(cb, VizDataCollector):
                return cb.recorder
        return None

    def _callbacks_for_run(self, recorder: VizRecorder) -> list[RunnerCallback]:
        callbacks = list(self._callbacks)
        for cb in callbacks:
            if isinstance(cb, VizDataCollector) and cb.recorder is recorder:
                return callbacks
        callbacks.append(VizDataCollector(recorder))
        return callbacks

    def _finalize_callbacks(self, callbacks: list[RunnerCallback]) -> None:
        for cb in callbacks:
            try:
                cb.on_experiment_end(self._algorithm)
            except Exception:
                logger.exception(
                    "Callback %r.on_experiment_end raised",
                    type(cb).__name__,
                )

    def _build_experiment_result(
        self,
        components: PlatformComponents,
        recorder: VizRecorder,
        early_stopped: bool,
    ) -> ExperimentResult:
        final_eval = self._final_eval()
        actual = self._actual()
        predicted = self._predicted(actual, final_eval)
        score_kind = DEFAULT_SCORE_KIND
        score_direction: str = DEFAULT_SCORE_DIRECTION
        if isinstance(self._algorithm, ScoreContract):
            score_kind = self._algorithm.score_kind
            score_direction = self._algorithm.score_direction
        equation = self._build_equation(components, final_eval)
        if final_eval.form is Form.HOMOGENEOUS:
            lhs_label = HOMOGENEOUS_LHS_LABEL
        elif isinstance(equation, Evolution):
            lhs_label = render_lhs_label(equation.lhs_spec)
        else:
            lhs_label = self._lhs_label(components, final_eval)
        return ExperimentResult(
            best_expression=self._algorithm.best_expression,
            best_score=self._algorithm.best_score,
            iterations=self._current_iteration,
            early_stopped=early_stopped,
            final_eval=final_eval,
            actual=actual,
            predicted=predicted,
            dataset_name=self._dataset_name(components),
            algorithm_name=type(self._algorithm).__name__,
            config=self._result_config(),
            recorder=recorder,
            lhs_label=lhs_label,
            equation=equation,
            manifest=self._build_manifest(components),
            score_kind=score_kind,
            score_direction=score_direction,
        )

    def _build_equation(
        self,
        components: PlatformComponents,
        final_eval: EvaluationResult,
    ) -> Equation | None:
        if final_eval.form is Form.HOMOGENEOUS:
            return build_homogeneous(
                final_eval.terms,
                final_eval.coefficients,
                is_valid=final_eval.is_valid,
            )
        lhs_spec = self._equation_lhs_spec(components, final_eval)
        return build_equation(
            final_eval.terms,
            final_eval.coefficients,
            lhs_spec,
            is_valid=final_eval.is_valid,
        )

    def _equation_lhs_spec(
        self,
        components: PlatformComponents,
        final_eval: EvaluationResult,
    ) -> LhsSpec | None:
        lhs_name = final_eval.lhs_name
        if isinstance(lhs_name, str) and lhs_name:
            return self._lhs_spec_from_name(components, lhs_name)

        dataset = components.dataset
        lhs_field = getattr(dataset, "lhs_field", None)
        lhs_axis = getattr(dataset, "lhs_axis", None)
        if (
            isinstance(lhs_field, str)
            and lhs_field
            and isinstance(lhs_axis, str)
            and lhs_axis
        ):
            return LhsSpec(lhs_field, lhs_axis, getattr(dataset, "lhs_order", 1))
        return None

    def _lhs_spec_from_name(
        self,
        components: PlatformComponents,
        lhs_name: str,
    ) -> LhsSpec | None:
        dataset = components.dataset
        known_fields = self._known_dataset_fields(dataset)
        known_axes = self._known_dataset_axes(dataset)
        parsed = parse_derivative_name(
            lhs_name,
            known_fields=known_fields,
            known_axes=known_axes,
        )
        if parsed is None and known_fields is None and known_axes is None:
            parsed = parse_derivative_name(lhs_name)
        if parsed is None:
            return None
        field, axis, order = parsed
        return LhsSpec(field, axis, order)

    @staticmethod
    def _known_dataset_fields(dataset: object) -> set[str] | None:
        fields = getattr(dataset, "fields", None)
        if not isinstance(fields, dict):
            return None
        return {field for field in fields if isinstance(field, str)}

    @staticmethod
    def _known_dataset_axes(dataset: object) -> set[str] | None:
        axis_order = getattr(dataset, "axis_order", None)
        if not isinstance(axis_order, (list, tuple)):
            return None
        return {axis for axis in axis_order if isinstance(axis, str)}

    def _result_config(self) -> dict[str, Any]:
        config = dict(self._algorithm.config)
        reqs = getattr(self._algorithm, "derivative_requirements", None)
        if isinstance(reqs, DerivativeReqs):
            config["provider_kind"] = reqs.provider_kind
        return config

    def _build_manifest(self, components: PlatformComponents) -> RunManifest:
        from kd import __version__ as kd_version

        dataset = components.dataset
        if isinstance(dataset, PDEDataset):
            fingerprint = compute_dataset_fingerprint(dataset)
        else:
            fingerprint = _NON_PDE_DATASET_FINGERPRINT
        return RunManifest(
            dataset_fingerprint=fingerprint,
            kd_version=kd_version,
            seed=self._algorithm.config.get("seed"),



            terms=getattr(self._algorithm, "terms", None),
            artifacts=getattr(self._algorithm, "artifacts", None),
        )

    def _final_eval(self) -> EvaluationResult:
        result: object = self._algorithm.build_final_result()
        if isinstance(result, EvaluationResult):
            return result
        return self._invalid_final_eval(
            "Final evaluation did not return EvaluationResult"
        )

    def _actual(self) -> Tensor:
        actual = self._algorithm.build_result_target()
        if not isinstance(actual, Tensor):
            raise TypeError(
                f"{type(self._algorithm).__name__}.build_result_target() "
                f"must return Tensor, got {type(actual).__name__}"
            )
        return actual.detach()

    def _predicted(self, actual: Tensor, final_eval: EvaluationResult) -> Tensor:
        if final_eval.residuals is None:
            return torch.zeros_like(actual)



        if not isinstance(final_eval.residuals, Tensor):
            raise TypeError(
                f"final_eval.residuals must be a Tensor, "
                f"got {type(final_eval.residuals).__name__}"
            )
        if final_eval.residuals.shape != actual.shape:
            raise ValueError(
                f"residuals.shape={tuple(final_eval.residuals.shape)} does not "
                f"match actual.shape={tuple(actual.shape)}"
            )
        return actual + final_eval.residuals

    def _dataset_name(self, components: PlatformComponents) -> str:
        name = components.dataset.name
        if isinstance(name, str):
            return name
        return str(name)

    def _lhs_label(
        self,
        components: PlatformComponents,
        final_eval: EvaluationResult,
    ) -> str:
        if isinstance(final_eval.lhs_name, str) and final_eval.lhs_name:
            return final_eval.lhs_name
        dataset = components.dataset
        lhs_field = getattr(dataset, "lhs_field", None)
        lhs_axis = getattr(dataset, "lhs_axis", None)


        if (
            isinstance(lhs_field, str)
            and lhs_field
            and isinstance(lhs_axis, str)
            and lhs_axis
        ):







            lhs_order: int = getattr(dataset, "lhs_order", 1)
            if lhs_order == 1:
                return f"{lhs_field}_{lhs_axis}"
            return build_derivative_name(lhs_field, lhs_axis, lhs_order)
        return _DEFAULT_LHS_LABEL

    def _invalid_final_eval(self, error_message: str) -> EvaluationResult:
        return invalid_evaluation_result(
            error_message,
            score=float("inf"),
            expression=self._algorithm.best_expression,
        )

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(
            build_checkpoint_payload(self._current_iteration, self._algorithm),
            path,
        )
        logger.debug("Saved checkpoint to %s", path)

    def load_checkpoint(self, path: Path) -> None:
        raw = self._torch_load_checkpoint(Path(path))
        data = self._validate_checkpoint_payload(raw)
        self._algorithm.state = data["algorithm_state"]




        self._pending_restore_state = data["algorithm_state"]
        self._current_iteration = data["iteration"]
        logger.debug(
            "Loaded checkpoint from %s (iteration=%d)", path, self._current_iteration
        )

    @staticmethod
    def _torch_load_checkpoint(path: Path) -> object:
        try:
            return torch.load(path, weights_only=False)
        except FileNotFoundError:
            raise
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            raise ValueError(
                f"not a kd checkpoint payload (corrupt or truncated file): {path}"
            ) from exc

    def _validate_checkpoint_payload(self, data: object) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError(
                f"not a kd checkpoint payload: expected a dict, "
                f"got {type(data).__name__}"
            )
        missing = [key for key in _CHECKPOINT_REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(
                f"not a kd checkpoint payload: missing required keys {missing}"
            )



        version = data["version"]
        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"checkpoint version mismatch: payload has version "
                f"{version!r}, this kd build expects {CHECKPOINT_VERSION}"
            )
        algorithm_state = data["algorithm_state"]










        if not isinstance(algorithm_state, dict):
            raise ValueError(
                "not a kd checkpoint payload: 'algorithm_state' must be a "
                f"dict, got {type(algorithm_state).__name__}"
            )
        payload_algorithm = data.get("algorithm")



        plugin_algorithm = _algorithm_name(self._algorithm)
        if (
            payload_algorithm is not None
            and plugin_algorithm is not None
            and payload_algorithm != plugin_algorithm
        ):
            raise ValueError(
                f"checkpoint algorithm mismatch: payload was written by "
                f"algorithm {payload_algorithm!r} but this runner drives "
                f"{plugin_algorithm!r}"
            )
        return data


__all__ = [
    "ExperimentRunner",
    "RunResult",
]
