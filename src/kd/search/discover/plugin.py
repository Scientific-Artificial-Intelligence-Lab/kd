
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, cast

import torch
from torch import Tensor

from kd.core.evaluator import EvaluationResult
from kd.core.platform.requirements import DerivativeReqs
from kd.search.discover import viz as _viz_helpers
from kd.search.discover.builder import _make_magnitude_filter, build_engine
from kd.search.discover.config import DiscoverConfig
from kd.search.discover.engine import DiscoverEngine, EngineState
from kd.search.discover.training.strategy import BaselineState
from kd.search.protocol import (
    IterativeSearchAlgorithm,
    PlatformComponents,
)
from kd.search.recorder import VizRecorder, log_whitelisted_metrics
from kd.viz.extension import PlotInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

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





















_LOGGED_METRICS: tuple[str, ...] = (

    "pg_loss",
    "entropy_loss",
    "total_loss",
    "baseline",
    "reward",
    "grad_norm",

    "reward_max",
    "best_reward",
    "n_valid",
    "n_eval_valid",
    "n_invalid_in_topk",
    "n_unique",
)


class _ExpressionEvaluator(Protocol):

    @property
    def lhs_target(self) -> Tensor:
        pass

    def evaluate_expression(self, expr: str) -> EvaluationResult:
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

    config_cls: ClassVar[type[DiscoverConfig]] = DiscoverConfig
    one_shot: ClassVar[bool] = False

    def __init__(self, config: DiscoverConfig | None = None) -> None:
        self._config = config or DiscoverConfig()
        self._evaluator: _ExpressionEvaluator | None = None
        self._engine: DiscoverEngine | None = None
        self._recorder: VizRecorder | None = None
        self._restore_pending: bool = False
        self._pending_state: EngineState | None = None
        torch.manual_seed(self._config.seed)

    def prepare(self, components: PlatformComponents) -> None:
        restore_state = self._pending_state
        if self._restore_pending and self._engine is not None:
            restore_state = self._engine.state
        torch.manual_seed(self._config.seed)
        self._evaluator = self._coerce_evaluator(components.evaluator)
        self._engine = build_engine(self._config)
        self._recorder = components.recorder
        if self._restore_pending and restore_state is not None:
            self._engine.state = restore_state


            self._engine.rebase_best(self._evaluator)
        self._restore_pending = False
        self._pending_state = None

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
        engine.receive_results(results)
        engine.update()
        log_whitelisted_metrics(
            self._recorder,
            _LOGGED_METRICS,
            engine.last_metrics,
            skip_missing=True,
        )

    def between_iterations(self) -> None:
        pass

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
    def derivative_requirements(self) -> DerivativeReqs:
        return DerivativeReqs(
            provider_kind="finite_diff",
            max_atomic_order=2,
            lhs_order=1,
            needs_surrogate=False,
        )

    def build_final_result(self) -> EvaluationResult:
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
        }
        if engine_state.optimizer_state is not None:
            payload[OPTIMIZER_STATE_KEY] = engine_state.optimizer_state
        if engine_state.extras is not None:
            payload[EXTRAS_KEY] = engine_state.extras
        if engine_state.best_result_terms is not None:
            payload[BEST_RESULT_TERMS_KEY] = list(engine_state.best_result_terms)



            payload[BEST_RESULT_IS_VALID_KEY] = bool(engine_state.best_result_is_valid)
        if engine_state.best_result_coefficients is not None:
            payload[BEST_RESULT_COEFFICIENTS_KEY] = list(
                engine_state.best_result_coefficients,
            )
        return {
            "algorithm": ALGORITHM_NAME,
            ENGINE_STATE_KEY: payload,
        }

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        if not value:
            self._pending_state = None
            self._restore_pending = False
            if self._engine is not None:
                torch.manual_seed(self._config.seed)
                self._engine = build_engine(self._config)
            return
        parsed = _parse_state_payload(value)
        self._pending_state = parsed
        self._restore_pending = True
        if self._engine is not None:
            self._engine.state = parsed










    def list_plots(self) -> list[PlotInfo]:
        return _viz_helpers.list_plot_infos()

    def render_plot(self, name: str, ax: Axes) -> None:
        _viz_helpers.render(name, ax, self._recorder)

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

    def _require_engine(self) -> DiscoverEngine:
        if self._engine is None:
            raise RuntimeError("prepare() must be called before using the plugin.")
        return self._engine

    def _require_evaluator(self) -> _ExpressionEvaluator:
        if self._evaluator is None:
            raise RuntimeError("prepare() must be called before evaluate().")
        return self._evaluator


__all__ = ["DISCOVERPlugin", "DiscoverConfig"]
