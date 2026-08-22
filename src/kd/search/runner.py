
from __future__ import annotations

import copy
import logging
import pickle
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, cast

import torch
from torch import Tensor

from kd.core import safety_counters
from kd.core.equation import (
    DEFAULT_LHS_LABEL,
    HOMOGENEOUS_LHS_LABEL,
    Equation,
    Evolution,
    Form,
    LhsSpec,
    build_equation,
    build_homogeneous,
    build_regression,
    render_lhs_label,
)
from kd.core.evaluator import EvaluationResult
from kd.core.expr.naming import parse_derivative_name
from kd.core.platform.requirements import DerivativeReqs, assert_dataset_supported
from kd.core.platform.sketch_compile import SKETCH_CONFIG_KEY, CompileReport
from kd.core.verify import verify_equation
from kd.data.schema import DataTopology, PDEDataset, compute_dataset_fingerprint
from kd.search.callbacks import RunnerCallback, VizDataCollector
from kd.search.checkpoint_payload import (
    CHECKPOINT_VERSION,
    _algorithm_name,
    atomic_torch_save,
    build_checkpoint_payload,
)
from kd.search.descriptor import (
    InstrumentDescriptor,
    assert_sketch_supported,
    mode_for_topology,
)
from kd.search.lifecycle import SearchLifecycle
from kd.search.protocol import (
    DiscoveryTask,
    IterativeSearchAlgorithm,
    PlatformComponents,
    ScoreContract,
    SearchAlgorithm,
    TerminatingSearchAlgorithm,
)
from kd.search.record_assembly import assemble_run_record, cpu_seconds_now
from kd.search.recorder import VizRecorder
from kd.search.records import RecordSchemaError, RunCost
from kd.search.result import (
    DEFAULT_SCORE_DIRECTION,
    DEFAULT_SCORE_KIND,
    ExperimentResult,
    RunManifest,
    RunResult,
    invalid_evaluation_result,
)
from kd.search.run_spec import RunSpec, canonicalize_config
from kd.search.sketch_outcome import SketchOutcome

logger = logging.getLogger(__name__)








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
    if attr is not _MISSING_ATTR and attr is not getattr(SearchAlgorithm, member, None):
        return True
    return member in getattr(algorithm, "__dict__", {})





_UNSET: Final = object()


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
        self._boundary_results = 0
        self._boundary_invalid_results = 0
        self._run_started = 0.0
        self._cpu_started: float | None = None



        self._lifecycle: SearchLifecycle | None = None




        self._pending_restore_state: dict[str, Any] | None = None
        self._resumed = False



        self._resume_source: dict[str, Any] | None = None


        self._task: DiscoveryTask | None = None

    @property
    def lifecycle(self) -> SearchLifecycle | None:
        return self._lifecycle

    def run(
        self,
        components: PlatformComponents,
        *,
        preprocessing_seconds: float | None = None,
    ) -> ExperimentResult:
        raw_task = getattr(components, "task", None)
        self._task = raw_task if isinstance(raw_task, DiscoveryTask) else None
        self._assert_algorithm_protocol()
        self._assert_sketch_capability()
        self._assert_dataset_supported(components)
        self._assert_run_identity_serializable()
        self._current_iteration = 0
        self._boundary_results = 0
        self._boundary_invalid_results = 0




        if safety_counters.counters_enabled():
            safety_counters.reset()
        self._run_started = time.perf_counter()
        self._cpu_started = cpu_seconds_now()



        lifecycle = SearchLifecycle()
        self._lifecycle = lifecycle
        pending_restore = self._pending_restore_state



        self._resumed = bool(pending_restore)
        if pending_restore is not None:
            lifecycle.restore(pending_restore)









        recorder_before = getattr(components, "recorder", None)
        try:
            return self._run_search(components, lifecycle, preprocessing_seconds)
        finally:
            components.recorder = recorder_before

    def _run_search(
        self,
        components: PlatformComponents,
        lifecycle: SearchLifecycle,
        preprocessing_seconds: float | None,
    ) -> ExperimentResult:
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








        crashed = False
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
        except BaseException:
            crashed = True
            raise
        finally:
            finalize_failures = self._finalize_callbacks(callbacks, crashed=crashed)



        lifecycle.finish()




        result = self._build_experiment_result(
            components,
            recorder,
            early_stopped,
            preprocessing_seconds,
            finalize_failures,
        )
        self._emit_safety_counter_summary()
        return result

    def _emit_safety_counter_summary(self) -> None:
        safety_counters.emit_run_summary(
            f"runner:{type(self._algorithm).__name__}", logger
        )

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

    def _assert_sketch_capability(self) -> None:
        if self._task is None:
            return
        descriptor = getattr(self._algorithm, "descriptor", None)
        if not isinstance(descriptor, InstrumentDescriptor):
            raise TypeError(
                f"{type(self._algorithm).__name__} must declare a descriptor "
                "of type InstrumentDescriptor to run a sketch task"
            )
        assert_sketch_supported(
            descriptor,
            self._task.sketch,
            algorithm=descriptor.algorithm,
        )

    def _assert_run_identity_serializable(self) -> None:

        canonicalize_config(self._result_config())
        try:



            getattr(self._algorithm, "artifacts", None)
        except TypeError as exc:
            raise RecordSchemaError(
                f"Injected artifact cannot be serialized into a run identity: {exc}"
            ) from exc

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
        self._boundary_results += len(results)
        self._boundary_invalid_results += sum(
            1 for result in results if not result.is_valid
        )
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

    def _finalize_callbacks(
        self, callbacks: list[RunnerCallback], *, crashed: bool
    ) -> list[str]:
        failures: list[str] = []
        for cb in callbacks:
            member = (
                "on_experiment_end_status"
                if _implements_member(cb, "on_experiment_end_status")
                else "on_experiment_end"
            )
            try:
                if member == "on_experiment_end_status":


                    handler = getattr(cb, member)
                    handler(self._algorithm, crashed=crashed)
                else:
                    cb.on_experiment_end(self._algorithm)
            except Exception as exc:
                logger.exception(
                    "Callback %r.%s raised",
                    type(cb).__name__,
                    member,
                )
                failures.append(
                    f"{type(cb).__name__}.{member}: {type(exc).__name__}: {exc}"
                )
        return failures

    def _build_experiment_result(
        self,
        components: PlatformComponents,
        recorder: VizRecorder,
        early_stopped: bool,
        preprocessing_seconds: float | None,
        finalize_failures: list[str],
    ) -> ExperimentResult:
        final_eval = self._final_eval()
        actual = self._actual()
        predicted = self._predicted(actual, final_eval)
        score_kind = DEFAULT_SCORE_KIND
        score_direction: str = DEFAULT_SCORE_DIRECTION
        if isinstance(self._algorithm, ScoreContract):
            score_kind = self._algorithm.score_kind
            score_direction = self._algorithm.score_direction
        sketch_outcome: SketchOutcome | None = None
        if self._task is not None:
            if final_eval.form is Form.HOMOGENEOUS:
                raise TypeError("sketch tasks cannot produce a HOMOGENEOUS final_eval")
            sketch_outcome = self._build_sketch_outcome(components, final_eval)
            equation = sketch_outcome.solution
        else:
            equation = self._build_equation(components, final_eval)
        if final_eval.form is Form.HOMOGENEOUS:
            lhs_label = HOMOGENEOUS_LHS_LABEL
        elif isinstance(equation, Evolution):
            lhs_label = render_lhs_label(equation.lhs_spec)
        else:
            lhs_label = self._lhs_label(components, final_eval)
        manifest = self._build_manifest(components)
        config = self._result_config()
        library_fingerprint = config.get("library_fingerprint")
        run_spec = RunSpec(
            kd_version=manifest.kd_version,
            config=config,
            library_fingerprint=library_fingerprint,
            dataset_cache_fingerprint=manifest.dataset_cache_fingerprint,
            artifacts=manifest.artifacts,
        )
        search_seconds = time.perf_counter() - self._run_started
        cpu_now = cpu_seconds_now()
        cpu_seconds = (
            cpu_now - self._cpu_started
            if cpu_now is not None and self._cpu_started is not None
            else None
        )







        raw_surrogate_seconds = getattr(
            self._algorithm, "surrogate_train_seconds", _UNSET
        )
        if raw_surrogate_seconds is _UNSET:
            raw_surrogate_seconds = getattr(
                getattr(components.context, "training_result", None),
                "elapsed_seconds",
                None,
            )
        surrogate_seconds = cast("float | None", raw_surrogate_seconds)






        token_totals = cast(
            "dict[str, int] | None",
            getattr(self._algorithm, "llm_token_totals", None),
        )
        tokens_in = None if token_totals is None else token_totals["tokens_in"]
        tokens_out = None if token_totals is None else token_totals["tokens_out"]
        cost = RunCost(
            wallclock_seconds=search_seconds + (preprocessing_seconds or 0.0),
            search_seconds=search_seconds,
            preprocessing_seconds=preprocessing_seconds,
            surrogate_train_seconds=surrogate_seconds,
            cpu_seconds=cpu_seconds,
            boundary_results=self._boundary_results,
            boundary_invalid_results=self._boundary_invalid_results,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        dataset_name = self._dataset_name(components)
        run_record = assemble_run_record(
            instrument=(
                _algorithm_name(self._algorithm) or type(self._algorithm).__name__
            ),
            dataset_name=dataset_name,
            dataset_cache_fingerprint=manifest.dataset_cache_fingerprint,
            seed=manifest.seed,
            final_eval=final_eval,
            equation=equation,
            best_expression=self._algorithm.best_expression,
            best_score=self._algorithm.best_score,
            score_kind=score_kind,
            score_direction=score_direction,


            headline_coefficient_source=getattr(
                self._algorithm,
                "headline_coefficient_source",
                "undeclared",
            ),
            run_spec=run_spec,
            manifest_terms=manifest.terms,
            cost=cost,
            support_from_equation=self._task is not None,
        )
        return ExperimentResult(
            best_expression=self._algorithm.best_expression,
            best_score=self._algorithm.best_score,
            iterations=self._current_iteration,
            early_stopped=early_stopped,
            final_eval=final_eval,
            actual=actual,
            predicted=predicted,
            dataset_name=dataset_name,
            algorithm_name=type(self._algorithm).__name__,
            config=config,
            recorder=recorder,
            lhs_label=lhs_label,
            equation=equation,
            manifest=manifest,
            run_record=run_record,
            score_kind=score_kind,
            score_direction=score_direction,
            finalize_failures=tuple(finalize_failures),
            sketch_outcome=sketch_outcome,
        )

    def _build_sketch_outcome(
        self,
        components: PlatformComponents,
        final_eval: EvaluationResult,
    ) -> SketchOutcome:
        task = self._task
        assert task is not None
        compile_report = self._sketch_compile_report(task)
        lifted = task.compiled.lift(None if task.compiled.closed else final_eval)
        if lifted is None:
            return SketchOutcome(
                solution=None,
                best_candidate=None,
                verdict=None,
                compile_report=compile_report,
                full_verify=None,
                failure="lift: no liftable law from final_eval",
            )

        try:
            verdict = task.sketch.matches(lifted)
        except ValueError as exc:
            logger.warning("Sketch match failed: %s", exc)
            return SketchOutcome(
                solution=None,
                best_candidate=lifted,
                verdict=None,
                compile_report=compile_report,
                full_verify=None,
                failure=f"matches: {exc}",
            )

        full_verify = None
        failure = None
        verify_failed = False
        if components.context is None:
            failure = "verify: platform context is unavailable"
        else:
            try:
                full_verify = verify_equation(
                    lifted,
                    executor=components.executor,
                    context=components.context,
                )
            except ValueError as exc:
                logger.warning("Sketch verification failed: %s", exc)
                failure = f"verify: {exc}"
                verify_failed = True
        solution = lifted if verdict.overall and not verify_failed else None
        return SketchOutcome(
            solution=solution,
            best_candidate=lifted,
            verdict=verdict,
            compile_report=compile_report,
            full_verify=full_verify,
            failure=failure,
        )

    def _sketch_compile_report(self, task: DiscoveryTask) -> CompileReport:
        report = getattr(self._algorithm, "sketch_compile_report", None)
        return report if isinstance(report, CompileReport) else task.compiled.report

    def _build_equation(
        self,
        components: PlatformComponents,
        final_eval: EvaluationResult,
    ) -> Equation | None:
        if components.dataset.topology is DataTopology.TABULAR:
            descriptor = getattr(self._algorithm, "descriptor", None)
            if not isinstance(descriptor, InstrumentDescriptor):
                return None
            mode = mode_for_topology(descriptor, DataTopology.TABULAR)
            if mode is None or Form.REGRESSION not in mode.forms:
                return None
            target = final_eval.lhs_name or components.dataset.lhs_field
            return build_regression(
                final_eval.terms,
                final_eval.coefficients,
                LhsSpec(field=target, axis="", order=0),
                active_indices=final_eval.selected_indices,
                is_valid=final_eval.is_valid,
            )
        if final_eval.form is Form.HOMOGENEOUS:
            return build_homogeneous(
                final_eval.terms,
                final_eval.coefficients,
                active_indices=final_eval.selected_indices,
                is_valid=final_eval.is_valid,
            )
        lhs_spec = self._equation_lhs_spec(components, final_eval)
        return build_equation(
            final_eval.terms,
            final_eval.coefficients,
            lhs_spec,
            active_indices=final_eval.selected_indices,
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
        if self._task is not None:






            config[SKETCH_CONFIG_KEY] = copy.deepcopy(self._task.payload)
        return config

    def _build_manifest(self, components: PlatformComponents) -> RunManifest:
        from kd import __version__ as kd_version

        dataset = components.dataset
        if isinstance(dataset, PDEDataset):
            fingerprint = compute_dataset_fingerprint(dataset)
        else:
            fingerprint = _NON_PDE_DATASET_FINGERPRINT
        return RunManifest(
            dataset_cache_fingerprint=fingerprint,
            kd_version=kd_version,
            seed=self._algorithm.config.get("seed"),



            terms=getattr(self._algorithm, "terms", None),
            artifacts=getattr(self._algorithm, "artifacts", None),
            resumed=self._resumed,
            resume_source=self._resume_source,
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
        if getattr(dataset, "topology", None) is DataTopology.TABULAR:
            return dataset.lhs_field
        lhs_field = getattr(dataset, "lhs_field", None)
        lhs_axis = getattr(dataset, "lhs_axis", None)


        if (
            isinstance(lhs_field, str)
            and lhs_field
            and isinstance(lhs_axis, str)
            and lhs_axis
        ):






            lhs_order: int = getattr(dataset, "lhs_order", 1)
            return render_lhs_label(
                LhsSpec(field=lhs_field, axis=lhs_axis, order=lhs_order)
            )
        logger.warning(
            "Neither the algorithm (final_eval.lhs_name) nor the dataset "
            "(lhs_field/lhs_axis) declared the regression target; defaulting "
            "lhs_label to %r. Downstream plots and reports label the LHS with "
            "this assumption.",
            DEFAULT_LHS_LABEL,
        )
        return DEFAULT_LHS_LABEL

    def _invalid_final_eval(self, error_message: str) -> EvaluationResult:




        return invalid_evaluation_result(
            error_message,
            score=float("inf"),
            expression=self._algorithm.best_expression,
            reason="unclassified",
        )

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            build_checkpoint_payload(self._current_iteration, self._algorithm)
            if self._task is None
            else build_checkpoint_payload(
                self._current_iteration,
                self._algorithm,
                task=self._task,
            )
        )
        atomic_torch_save(
            payload,
            path,
        )
        logger.debug("Saved checkpoint to %s", path)

    def load_checkpoint(
        self,
        path: Path,
        *,
        config_guard: Callable[[object, object], None] | None = None,
        resume_source: dict[str, Any] | None = None,
    ) -> None:
        raw = self._torch_load_checkpoint(Path(path))
        data = self._validate_checkpoint_payload(raw)
        if config_guard is not None:
            config_guard(data.get("config"), data.get("config_canon_scheme"))
        self._resume_source = resume_source
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
