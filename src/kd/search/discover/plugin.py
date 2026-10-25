
from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, cast

import torch
from torch import Tensor

from kd.core.equation import Form
from kd.core.equation.rendering import render_lhs_label
from kd.core.equation.types import LhsSpec
from kd.core.evaluator import EvaluationResult
from kd.core.platform.requirements import DerivativeReqs
from kd.core.platform.sketch_compile import CompileReport, SketchClauseLevels
from kd.data.schema import DataTopology, PDEDataset
from kd.search.descriptor import (
    SCORE_FRAME_PLATFORM_FULL_GRID,
    SCORE_FRAME_SURROGATE_FIELD,
    InstrumentDescriptor,
    InstrumentMode,
    Knob,
    ModeSelector,
    Segmentation,
)
from kd.search.discover import viz as _viz_helpers
from kd.search.discover.builder import _make_magnitude_filter, build_engine
from kd.search.discover.config import (
    DEFAULT_SPATIAL_AXES,
    DiscoverConfig,
    PINNConfig,
    default_library_config,
)
from kd.search.discover.engine import DiscoverEngine, EngineState
from kd.search.discover.evaluation.magnitude import magnitude_rejection
from kd.search.discover.sketch_backend import (
    DiscoverCompiled,
    SketchGatedEvaluator,
    _ExpressionEvaluator,
    compile_for_discover,
    narrowed_config,
)
from kd.search.discover.tabular import (
    ParetoTracker,
    TabularCandidateScorer,
    TabularDeduplicator,
)
from kd.search.discover.tokens.library import (
    _DIFF_OPERATORS,
    _SPECIAL_DIFF_OPERATORS,
)
from kd.search.discover.training.strategy import BaselineState
from kd.search.protocol import (
    DiscoveryTask,
    IterativeSearchAlgorithm,
    PlatformComponents,
)
from kd.search.recorder import VizRecorder, log_whitelisted_metrics
from kd.search.series_keys import (
    PARETO_COMPLEXITY_KEY,
    PARETO_EXPRESSIONS_KEY,
    PARETO_LOSS_KEY,
    PARETO_SCALE_KEY,
)
from kd.search.trajectory import log_search_trajectory
from kd.viz.extension import PlotInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.discover.pinn.plugin_run import PinnRun

ALGORITHM_NAME = "discover"
MIN_PROPOSAL_COUNT = 1
INITIAL_BEST_SCORE = 0.0
INITIAL_BEST_EXPRESSION = ""
ENGINE_STATE_KEY = "engine_state"
CONTROLLER_STATE_KEY = "controller_state_dict"
BASELINE_STATE_KEY = "baseline_state"
BEST_REWARD_KEY = "best_reward"
BEST_EXPRESSION_KEY = "best_expression"
OPTIMIZER_STATE_KEY = "optimizer_state"
EXTRAS_KEY = "extras"
BEST_RESULT_TERMS_KEY = "best_result_terms"
BEST_RESULT_COEFFICIENTS_KEY = "best_result_coefficients"
BEST_RESULT_IS_VALID_KEY = "best_result_is_valid"
EWMA_REWARD_KEY = "ewma_reward"
N_UPDATES_KEY = "n_updates"
_STATE_MODE = "mode"
_TABULAR_FRONT = "tabular_front"


_MODE2_STATE_KEY = "mode2"




_SKETCH_LEVELS = SketchClauseLevels(
    fixed_terms="lowered",
    anchors="exit_checked",
    hole_count="exit_checked",
    derivative_order="generation_enforced",
    operator_set="generation_enforced",
    field_axis_set="generation_enforced",
)





















_LOGGED_METRICS: tuple[str, ...] = (



    "pg_loss",
    "entropy_loss",
    "total_loss",
    "baseline",
    "reward",
    "grad_norm",

    "reward_full",
    "reward_max",
    "best_reward",
    "n_valid",
    "n_eval_valid",
    "n_invalid_in_topk",
    "n_unique",
)


class _TabularEvaluator(Protocol):
    @property
    def lhs_target(self) -> Tensor:
        pass

    def build_theta_matrix(
        self, terms: list[str], *, skip_invalid: bool = False
    ) -> tuple[Tensor, list[str]]:
        pass

    def evaluate_terms(
        self, terms: list[str], *, skip_invalid: bool = False
    ) -> EvaluationResult:
        pass

    def invalidate_term_cache(self) -> None:
        pass


def _serialize_controller_state(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def _deserialize_controller_state(raw: object) -> dict[str, Tensor]:
    if not isinstance(raw, Mapping):
        raise TypeError("controller_state_dict must be a mapping.")
    state_dict: dict[str, Tensor] = {}
    for name, tensor in raw.items():
        if not isinstance(name, str):
            raise TypeError("controller_state_dict keys must be strings.")
        if not isinstance(tensor, Tensor):
            raise TypeError("controller_state_dict values must be tensors.")
        state_dict[name] = tensor.detach().cpu().clone()
    return state_dict


def _serialize_baseline_state(state: BaselineState) -> dict[str, Any]:
    return {
        EWMA_REWARD_KEY: float(state.ewma_reward),
        N_UPDATES_KEY: int(state.n_updates),
    }


def _deserialize_baseline_state(raw: object) -> BaselineState:
    if isinstance(raw, BaselineState):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError("baseline_state must be a mapping.")
    ewma_reward = float(raw.get(EWMA_REWARD_KEY, 0.0))
    n_updates = int(raw.get(N_UPDATES_KEY, 0))
    return BaselineState(ewma_reward=ewma_reward, n_updates=n_updates)


def _parse_state_payload(value: Mapping[str, Any]) -> EngineState:
    raw_engine_state = value.get(ENGINE_STATE_KEY)
    if not isinstance(raw_engine_state, Mapping):
        raise TypeError("state must contain an engine_state mapping.")
    return EngineState(
        controller_state_dict=_deserialize_controller_state(
            raw_engine_state.get(CONTROLLER_STATE_KEY)
        ),
        baseline_state=_deserialize_baseline_state(
            raw_engine_state.get(BASELINE_STATE_KEY)
        ),
        best_reward=float(raw_engine_state.get(BEST_REWARD_KEY, 0.0)),
        best_expression=str(raw_engine_state.get(BEST_EXPRESSION_KEY, "")),
        optimizer_state=raw_engine_state.get(OPTIMIZER_STATE_KEY),
        extras=raw_engine_state.get(EXTRAS_KEY),
        best_result_terms=raw_engine_state.get(BEST_RESULT_TERMS_KEY),
        best_result_coefficients=raw_engine_state.get(BEST_RESULT_COEFFICIENTS_KEY),


        best_result_is_valid=bool(raw_engine_state.get(BEST_RESULT_IS_VALID_KEY, True)),
    )


class DISCOVERPlugin(IterativeSearchAlgorithm):




    score_kind: ClassVar[str] = "reward"
    score_direction: ClassVar[Literal["min", "max"]] = "max"
    headline_coefficient_source: ClassVar[Literal["native", "platform_refit"]] = (
        "platform_refit"
    )

    config_cls: ClassVar[type[DiscoverConfig]] = DiscoverConfig
    one_shot: ClassVar[bool] = False
    sketch_lower_owner: ClassVar[Literal["platform", "native"]] = "platform"
    constructor_kwargs: ClassVar[frozenset[str]] = frozenset({"mode"})

    descriptor: ClassVar[InstrumentDescriptor] = InstrumentDescriptor(
        algorithm="discover",
        summary=(
            "Risk-seeking reinforcement-learning search over PDE "
            "terms (grid) or tabular expressions"
        ),
        cost_class="heavy",
        modes=(
            InstrumentMode(
                name="mode1",
                spatial_axes=DEFAULT_SPATIAL_AXES,
                forms=frozenset({Form.EVOLUTION}),
                topologies=frozenset({DataTopology.GRID}),
                provider_kind="finite_diff",
                description=(
                    "Grid search whose default vocabulary is every field of "
                    "the dataset (the target is its left-hand side), the "
                    "grid's spatial axes (x, and y on a 2-D grid), t, and "
                    "diff and diff2 along each spatial axis."
                ),
                lhs_orders=frozenset({1}),



                score_frame=SCORE_FRAME_PLATFORM_FULL_GRID,
                sketch=_SKETCH_LEVELS,
                selector=ModeSelector("pinn", present=False),
            ),


            InstrumentMode(
                name="mode2",



                required_params=frozenset({"pinn"}),
                spatial_axes=DEFAULT_SPATIAL_AXES,
                forms=frozenset({Form.EVOLUTION}),
                topologies=frozenset({DataTopology.GRID}),
                provider_kind="autograd",
                description=(
                    "PINN surrogate mode that pretrains one network on every "
                    "field of the observations, searches mode1's vocabulary on "
                    "its autograd derivatives, and retrains the network on the "
                    "best equation at every cycle boundary."
                ),
                lhs_orders=frozenset({1}),
                score_frame=SCORE_FRAME_SURROGATE_FIELD,



                selector=ModeSelector("pinn", present=True),
            ),
            InstrumentMode(
                name="tabular",
                default_overrides={"batch_size": 500, "epsilon": 0.05},
                forms=frozenset({Form.REGRESSION}),
                topologies=frozenset({DataTopology.TABULAR}),
                provider_kind="none",
                description="Facade-reachable tabular regression mode.",
                lhs_orders=frozenset({0}),
            ),
        ),


        facade_inert_fields=frozenset(
            {
                "diagnostic_scaffold",
                "diagnostic_scaffold_diffusion_tokens",
                "diagnostic_scaffold_reaction_tokens",
                "diagnostic_scaffold_root_tokens",
                "diagnostic_scaffold_neutral_tokens",
            }
        ),
        knobs=(
            Knob(
                "batch_size",
                "int",
                "Controller samples per Runner iteration.",
                resume_tier="resume_safe",
            ),
            Knob(
                "learning_rate",
                "float",
                "Controller learning rate.",



                resume_tier="resume_safe",
            ),
            Knob(
                "entropy_weight",
                "float",
                "Exploration-entropy weight.",
                resume_tier="resume_safe",
            ),
            Knob(
                "reward_alpha",
                "float",
                "Reward sparsity trade-off.",
                resume_tier="resume_safe",
            ),
            Knob(
                "max_length",
                "int",
                "Maximum generated expression length.",
















                resume_tier="resume_safe",
            ),
            Knob(
                "epsilon",
                "float",
                "Risk-seeking reward quantile: the top-epsilon fraction of a "
                "batch's rewards feeds the policy-gradient update.",















                resume_tier="resume_safe",
            ),
            Knob(
                "magnitude_filter",
                "bool",
                "Reject fitted coefficients outside the reference magnitude bounds.",



                resume_tier="resume_safe",
            ),
        ),
        segmentation=Segmentation(archive="progress", unit="iterations", reseed=True),
        identity_breaking_fields=frozenset({"library", "max_diff_order", "pinn"}),
        config_artifact_keys=frozenset(),
        surrogate_fields=frozenset(),
    )

    def __init__(
        self,
        config: DiscoverConfig | None = None,
        *,
        mode: Literal["default", "tabular"] = "default",
    ) -> None:
        if mode not in {"default", "tabular"}:
            raise ValueError(f"Unknown DISCOVER mode: {mode!r}.")
        self._config = config or DiscoverConfig()


        self._engine_config = self._config
        self._sketch_compiled: DiscoverCompiled | None = None
        self._mode = mode
        self._evaluator: _ExpressionEvaluator | None = None
        self._engine: DiscoverEngine | None = None
        self._pinn_run: PinnRun | None = None

        self._deadline: float | None = None
        self._recorder: VizRecorder | None = None
        self._trajectory_lhs = ""
        self._restore_pending: bool = False
        self._pending_state: EngineState | None = None


        self._pending_mode2: dict[str, Any] | None = None
        self._front = ParetoTracker()
        self._tabular_deduplicator: TabularDeduplicator | None = None
        torch.manual_seed(self._config.seed)

    @property
    def _active_pinn(self) -> PINNConfig | None:
        return self._config.pinn if self._mode == "default" else None

    def refuse_unsupported_names(self, dataset: PDEDataset) -> None:
        if self._mode == "tabular":
            return
        library = self._dataset_config(dataset).library
        fields = set(dataset.fields or {})
        axes = set(dataset.spatial_axes)
        operators = set(library.operators)
        derivative_axes = (
            axes
            if operators & _SPECIAL_DIFF_OPERATORS
            else {
                operator.split("_", 1)[1]
                for operator in operators & _DIFF_OPERATORS
                if "_" in operator
            }
        )
        if (
            fields != set(library.state_vars)
            or not axes <= set(library.coord_vars) | derivative_axes
        ):
            raise NotImplementedError(
                f"DISCOVER dataset fields {sorted(fields)} and spatial "
                f"axes {sorted(axes)} "
                f"are not covered by library state_vars={library.state_vars}, "
                f"coord_vars={library.coord_vars}, operators={library.operators}. "
                "Set library through a typed DiscoverConfig to cover this input."
            )

    def prepare(self, components: PlatformComponents) -> None:
        self.refuse_unsupported_names(components.dataset)
        self._deadline = None
        restore_state = self._pending_state
        if (
            self._restore_pending
            and self._pending_mode2 is None
            and self._engine is not None
        ):
            restore_state = self._engine.state
        task = components.task
        self._compile_task(task, self._dataset_config(components.dataset))
        torch.manual_seed(self._config.seed)
        platform_evaluator = self._coerce_evaluator(components.evaluator)
        self._engine = build_engine(self._engine_config, mode=self._mode)
        self._evaluator = platform_evaluator
        self._pinn_run = None
        pinn = self._active_pinn
        mode2_block = self._pending_mode2
        if pinn is not None:


            from kd.search.discover.pinn.plugin_run import PinnRun





            self._pinn_run = (
                PinnRun.start(self._config, pinn, self._engine, components)
                if mode2_block is None
                else PinnRun.restore(self._config, pinn, components, mode2_block)
            )
            self._evaluator = self._pinn_run.evaluator
        if not self._restore_pending:





            self._front = ParetoTracker()
        if self._mode == "tabular":
            tabular_evaluator = self._coerce_tabular_evaluator(components.evaluator)
            feature_names = list(self._config.library.state_vars)
            theta, _ = tabular_evaluator.build_theta_matrix(
                feature_names,
                skip_invalid=False,
            )
            columns = {
                name: theta[:, index].detach().cpu().numpy().copy()
                for index, name in enumerate(feature_names)
            }
            target = tabular_evaluator.lhs_target.detach().cpu().numpy().copy()
            self._tabular_deduplicator = TabularDeduplicator(
                self._engine.library,
                columns,
                target,
            )
            self._engine.set_deduplicator(self._tabular_deduplicator)
            self._evaluator = TabularCandidateScorer(
                tabular_evaluator,
                lhs_name=components.dataset.lhs_field,
            )
        if (
            task is not None
            and self._sketch_compiled is not None
            and not task.compiled.closed
        ):
            self._evaluator = SketchGatedEvaluator(
                self._evaluator,
                sketch=task.sketch,
                compiled=self._sketch_compiled,
                registry=components.registry,
            )
        self._recorder = components.recorder
        self._trajectory_lhs = ""
        if self._enabled_trajectory_recorder() is not None:
            dataset = components.dataset
            self._trajectory_lhs = (
                dataset.lhs_field
                if dataset.lhs_order == 0
                else render_lhs_label(
                    LhsSpec(dataset.lhs_field, dataset.lhs_axis, dataset.lhs_order)
                )
            )
        if self._restore_pending and restore_state is not None:
            self._engine.state = restore_state
            if self._mode == "tabular":



                self._log_front()


            self._engine.gate_best_by_length()


            self._engine.rebase_best(self._evaluator)
            if self._pinn_run is not None and mode2_block is not None:


                self._pinn_run.resume_cycle(self._engine, mode2_block)
                self._evaluator = self._pinn_run.evaluator
        self._restore_pending = False
        self._pending_state = None
        self._pending_mode2 = None

    def _dataset_config(self, dataset: PDEDataset) -> DiscoverConfig:
        axes = list(dataset.spatial_axes)
        if (
            self._mode == "tabular"
            or self._config.library != DiscoverConfig().library
            or not set(axes) <= set(DEFAULT_SPATIAL_AXES)
        ):
            return self._config
        library = default_library_config(axes, list(dataset.fields or {}))
        if library == self._config.library:
            return self._config
        return replace(self._config, library=library)

    def _compile_task(self, task: DiscoveryTask | None, config: DiscoverConfig) -> None:
        self._sketch_compiled = None
        self._engine_config = config
        if task is not None:
            self._sketch_compiled = compile_for_discover(task.sketch, config)
            self._engine_config = narrowed_config(config, self._sketch_compiled)

    def propose(self, n: int) -> list[str]:
        if n < MIN_PROPOSAL_COUNT:
            raise ValueError(f"n must be >= {MIN_PROPOSAL_COUNT}.")
        engine = self._require_engine()
        engine.batch_size = n
        return engine.propose()

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        evaluator = self._require_evaluator()
        return [evaluator.evaluate_expression(candidate) for candidate in candidates]

    def update(self, results: list[EvaluationResult]) -> None:
        engine = self._require_engine()
        evaluator = self._require_evaluator()
        try:
            engine.receive_results(results)
            trajectory_recorder = self._enabled_trajectory_recorder()
            trajectory = (
                engine.snapshot_pending_candidates(
                    top_k=trajectory_recorder.trajectory_top_k,
                    lhs=self._trajectory_lhs,
                )
                if trajectory_recorder is not None
                else []
            )
            engine.update()
            if self._pinn_run is not None:
                self._pinn_run.completed_generations += 1
            if self._mode == "tabular":
                result_filter = _make_magnitude_filter(
                    enabled=self._config.magnitude_filter
                )
                front_results = (
                    [result_filter(result) for result in results]
                    if result_filter is not None
                    else results
                )
                self._update_front(front_results)
            log_whitelisted_metrics(
                self._recorder,
                _LOGGED_METRICS,
                engine.last_metrics,
                skip_missing=True,
            )
            if self._mode == "tabular":
                self._log_front()
            if trajectory_recorder is not None:
                log_search_trajectory(trajectory_recorder, trajectory)
        finally:





            evaluator.invalidate_term_cache()

    def _enabled_trajectory_recorder(self) -> VizRecorder | None:
        recorder = self._recorder
        if recorder is None or not recorder.enabled or recorder.trajectory_top_k == 0:
            return None
        return recorder

    def between_iterations(self) -> None:
        if self._pinn_run is None:
            return
        self._pinn_run.between_generations(
            self._require_engine(), deadline=self._deadline
        )
        self._evaluator = self._pinn_run.evaluator

    def set_wall_clock_deadline(self, deadline: float) -> None:
        if self._deadline is None or deadline < self._deadline:
            self._deadline = deadline

    @property
    def best_score(self) -> float:
        if self._engine is None:
            return INITIAL_BEST_SCORE
        return self._engine.best_reward

    @property
    def best_expression(self) -> str:
        if self._engine is None:
            return INITIAL_BEST_EXPRESSION
        return self._engine.best_expression

    @property
    def config(self) -> dict[str, Any]:
        return {
            "algorithm": ALGORITHM_NAME,
            **asdict(self._config),
        }

    @property
    def runner_batch_size(self) -> int:
        return self._config.batch_size

    @property
    def sketch_compile_report(self) -> CompileReport | None:
        if self._sketch_compiled is None:
            return None
        return self._sketch_compiled.report

    @property
    def surrogate_train_seconds(self) -> float | None:
        if self._pinn_run is None:
            return None
        return self._pinn_run.train_seconds



    @property
    def derivative_requirements(self) -> DerivativeReqs:
        if self._mode == "tabular":
            return DerivativeReqs(
                provider_kind="none",
                lhs_order=0,
                lhs_source="field",
                supported_topologies=frozenset({DataTopology.TABULAR}),
            )
        return DerivativeReqs(
            provider_kind="finite_diff",
            max_atomic_order=2,
            lhs_order=1,
            needs_surrogate=False,
        )

    @property
    def derivative_source(self) -> Literal["finite_diff", "autograd", "none"]:
        if self._mode == "tabular":
            return "none"
        return "autograd" if self._config.pinn is not None else "finite_diff"

    @property
    def searched_vocabulary(self) -> dict[str, object]:
        tokens = self._require_engine().library
        library = self._engine_config.library
        return {
            "operators": list(library.operators),
            "state_vars": list(library.state_vars),
            "coord_vars": list(library.coord_vars),
            "token_names": list(tokens.names),
            "token_arities": [int(arity) for arity in tokens.arities],
        }

    def build_final_result(self) -> EvaluationResult:
        if self._pinn_run is not None:
            engine = self._require_engine()
            engine.state = self._pinn_run.finalize(engine)
        result = self._require_evaluator().evaluate_expression(self.best_expression)
        result_filter = _make_magnitude_filter(enabled=self._config.magnitude_filter)
        if result_filter is not None:
            result = result_filter(result)
        return result

    def build_result_target(self) -> Tensor:
        target = self._require_evaluator().lhs_target
        if not isinstance(target, Tensor):
            raise TypeError(
                "components.evaluator.lhs_target must be a Tensor, "
                f"got {type(target).__name__}."
            )
        return target.detach().clone()

    @property
    def state(self) -> dict[str, Any]:
        if self._engine is not None:
            engine_state = self._engine.state
        elif self._pending_state is not None:
            engine_state = self._pending_state
        else:
            raise RuntimeError("prepare() must be called before using the plugin.")
        payload: dict[str, Any] = {
            CONTROLLER_STATE_KEY: _serialize_controller_state(
                cast(Mapping[str, Tensor], engine_state.controller_state_dict)
            ),
            BASELINE_STATE_KEY: _serialize_baseline_state(engine_state.baseline_state),
            BEST_REWARD_KEY: float(engine_state.best_reward),
            BEST_EXPRESSION_KEY: engine_state.best_expression,
            _STATE_MODE: self._mode,
        }
        if engine_state.optimizer_state is not None:
            payload[OPTIMIZER_STATE_KEY] = engine_state.optimizer_state
        extras = dict(engine_state.extras or {})
        if self._mode == "tabular":
            extras[_TABULAR_FRONT] = self._front.state()
        if extras:
            payload[EXTRAS_KEY] = extras
        if engine_state.best_result_terms is not None:
            payload[BEST_RESULT_TERMS_KEY] = list(engine_state.best_result_terms)



            payload[BEST_RESULT_IS_VALID_KEY] = bool(engine_state.best_result_is_valid)
        if engine_state.best_result_coefficients is not None:
            payload[BEST_RESULT_COEFFICIENTS_KEY] = list(
                engine_state.best_result_coefficients,
            )
        state: dict[str, Any] = {
            "algorithm": ALGORITHM_NAME,
            ENGINE_STATE_KEY: payload,
        }
        if self._pinn_run is not None:
            state[_MODE2_STATE_KEY] = self._pinn_run.state_block(self._require_engine())
        elif self._pending_mode2 is not None:
            state[_MODE2_STATE_KEY] = copy.deepcopy(self._pending_mode2)
        return state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        pinn_mode = self._active_pinn is not None
        if not value:
            self._pending_state = None
            self._pending_mode2 = None
            self._restore_pending = False
            self._front = ParetoTracker()
            if pinn_mode:
                self._drop_prepared_run()
            elif self._engine is not None:
                torch.manual_seed(self._config.seed)
                self._engine = build_engine(self._engine_config, mode=self._mode)
                if self._tabular_deduplicator is not None:
                    self._engine.set_deduplicator(self._tabular_deduplicator)
            return
        if pinn_mode and _MODE2_STATE_KEY not in value:
            raise ValueError(
                "DISCOVER checkpoint carries no mode2 state, and config.pinn is "
                "set: the PINN mode cannot resume without its network."
            )
        if not pinn_mode and _MODE2_STATE_KEY in value:
            raise ValueError(
                "DISCOVER checkpoint carries a mode2 state, and this plugin does "
                "not run the PINN mode: the checkpoint's search ran on a network "
                "this plugin does not have."
            )
        raw_engine_state = value.get(ENGINE_STATE_KEY)
        if not isinstance(raw_engine_state, Mapping):
            raise TypeError("state must contain an engine_state mapping.")
        stored_mode = raw_engine_state.get(_STATE_MODE, "default")
        if stored_mode != self._mode:
            raise ValueError(
                f"DISCOVER checkpoint mode {stored_mode!r} does not match "
                f"plugin mode {self._mode!r}."
            )
        parsed = _parse_state_payload(value)
        if self._mode == "tabular":
            self._front = self._front_from_extras(parsed.extras)
        self._pending_state = parsed
        self._restore_pending = True
        if pinn_mode:
            self._pending_mode2 = copy.deepcopy(value[_MODE2_STATE_KEY])
            self._drop_prepared_run()
        elif self._engine is not None:
            self._engine.state = parsed

    def _drop_prepared_run(self) -> None:
        self._engine = None
        self._pinn_run = None
        self._evaluator = None

    def reseed(self) -> None:
        pass










    def list_plots(self) -> list[PlotInfo]:
        return _viz_helpers.list_plot_infos()

    def render_plot(self, name: str, ax: Axes) -> list[str]:
        return _viz_helpers.render(name, ax, self._recorder)

    def get_plot_data(self, name: str) -> dict[str, Any]:
        return _viz_helpers.get_data(name, self._recorder)

    def _coerce_evaluator(self, evaluator: object) -> _ExpressionEvaluator:
        if not hasattr(evaluator, "evaluate_expression"):
            raise TypeError(
                "DISCOVERPlugin requires components.evaluator exposing "
                f"evaluate_expression(); got {type(evaluator).__name__}. "
                "Assemble the platform with an evaluator (PlatformBuilder "
                "default) to run DISCOVER."
            )
        return cast(_ExpressionEvaluator, evaluator)

    def _coerce_tabular_evaluator(self, evaluator: object) -> _TabularEvaluator:
        if not hasattr(evaluator, "build_theta_matrix") or not hasattr(
            evaluator, "evaluate_terms"
        ):
            raise TypeError(
                "DISCOVERPlugin tabular mode requires components.evaluator "
                "exposing build_theta_matrix() and evaluate_terms()."
            )
        return cast(_TabularEvaluator, evaluator)

    def _update_front(self, results: list[EvaluationResult]) -> None:
        for result in results:
            coefficients = result.coefficients
            if not result.is_valid or coefficients is None or coefficients.numel() != 1:
                continue
            terms = result.terms or []
            if len(terms) != 1:
                continue
            self._front.offer(
                terms[0],
                result.complexity,
                result.nmse,
                float(coefficients.detach().reshape(-1)[0].item()),
            )

    def _log_front(self) -> None:
        if self._recorder is None:
            return
        entries = self._front.entries()
        self._recorder.log(
            PARETO_EXPRESSIONS_KEY,
            [entry.expression for entry in entries],
        )
        self._recorder.log(
            PARETO_COMPLEXITY_KEY,
            [entry.complexity for entry in entries],
        )
        self._recorder.log(PARETO_LOSS_KEY, [entry.loss for entry in entries])
        self._recorder.log(PARETO_SCALE_KEY, [entry.scale for entry in entries])

    def _front_from_extras(self, extras: dict[str, Any] | None) -> ParetoTracker:
        if extras is None:
            return ParetoTracker()
        raw = extras.get(_TABULAR_FRONT, [])
        if not isinstance(raw, list):
            raise TypeError("tabular_front checkpoint extra must be a list.")








        max_length = self._config.max_length
        kept = [row for row in raw if cast(int, row["complexity"]) <= max_length]
        if not self._config.magnitude_filter:
            return ParetoTracker.from_state(kept)
        tracker = ParetoTracker()
        for row in kept:
            scale = cast(float, row["scale"])
            if (
                magnitude_rejection(
                    torch.tensor([scale]),
                    selected_indices=None,
                )
                is not None
            ):
                continue
            tracker.offer(
                expression=cast(str, row["expression"]),
                complexity=cast(int, row["complexity"]),
                nmse=cast(float, row["loss"]),
                scale=scale,
            )
        return tracker

    def _require_engine(self) -> DiscoverEngine:
        if self._engine is None:
            raise RuntimeError("prepare() must be called before using the plugin.")
        return self._engine

    def _require_evaluator(self) -> _ExpressionEvaluator:
        if self._evaluator is None:
            raise RuntimeError("prepare() must be called before evaluate().")
        return self._evaluator


__all__ = ["DISCOVERPlugin", "DiscoverConfig"]
