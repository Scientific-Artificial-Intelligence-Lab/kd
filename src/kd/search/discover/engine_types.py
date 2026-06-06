
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.nn import Parameter

from kd.core.evaluator import EvaluationResult
from kd.search.discover.core.batch import Batch
from kd.search.discover.tokens.library import Library
from kd.search.discover.training.strategy import BaselineState

if TYPE_CHECKING:
    from kd.search.discover.engine import DiscoverEngine

BoolArray = npt.NDArray[np.bool_]
Int64Array = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float32]


class Evaluator(Protocol):

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        pass


RewardAdapter = Callable[[EvaluationResult], float]
CycleCallback = Callable[[int, "DiscoverEngine"], None]
SearchProgressCallback = Callable[[int, dict[str, float], "DiscoverEngine"], None]


@runtime_checkable
class Generator(Protocol):

    @property
    def library(self) -> Library:
        pass

    def sample(self, batch_size: int) -> Batch:
        pass

    def make_neglogp_and_entropy(
        self,
        batch: Batch,
        entropy_gamma: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        pass

    @property
    def device(self) -> torch.device:
        pass

    def parameters(self) -> Iterator[Parameter]:
        pass

    def state_dict(self) -> dict[str, Any]:
        pass

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any:
        pass

    def train(self, mode: bool = True) -> Any:
        pass


@dataclass(frozen=True, slots=True)
class EngineState:

    controller_state_dict: dict[str, Any]
    baseline_state: BaselineState
    best_reward: float
    best_expression: str
    optimizer_state: dict[str, Any] | None = None
    extras: dict[str, Any] | None = None
    best_result_terms: list[str] | None = None
    best_result_coefficients: list[float] | None = None


@dataclass(slots=True)
class PendingState:

    batch: Batch
    valid_mask: BoolArray
    scatter_map: Int64Array
    unique_irs: list[str]
    unique_rewards: FloatArray | None = None
    unique_eval_valid_mask: BoolArray | None = None
    train_valid_mask: BoolArray | None = None
    full_rewards: Tensor | None = None
    unique_results: list[EvaluationResult] | None = None


__all__ = [
    "BoolArray",
    "CycleCallback",
    "EngineState",
    "Evaluator",
    "FloatArray",
    "Generator",
    "Int64Array",
    "PendingState",
    "RewardAdapter",
    "SearchProgressCallback",
]
