
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, runtime_checkable

from kd.core.equation.sketch import Sketch, sketch_to_dict
from kd.core.platform.sketch_compile import CompiledSketch, compile_sketch
from kd.core.verify import SKETCH_EXIT_VERIFY, VerifyPolicy
from kd.search.descriptor import InstrumentDescriptor

if TYPE_CHECKING:
    import torch
    from torch import Tensor
    from torch.nn import Module

    from kd.core.evaluator import EvaluationResult, Evaluator
    from kd.core.executor.context import ExecutionContext
    from kd.core.expr.executor import PythonExecutor
    from kd.core.expr.registry import FunctionRegistry
    from kd.core.platform.requirements import DerivativeReqs
    from kd.data.schema import PDEDataset
    from kd.models.trainer import TrainingResult
    from kd.search.recorder import VizRecorder

__all__ = [
    "DeadlineAwareSearchAlgorithm",
    "DerivativeSourceAlgorithm",
    "DiscoveryTask",
    "FacadeWiringContract",
    "IterativeSearchAlgorithm",
    "PlatformComponents",
    "ScoreContract",
    "SearchAlgorithm",
    "SearchedVocabularyAlgorithm",
    "SurrogateTrainer",
    "TerminatingSearchAlgorithm",
]


@dataclass(frozen=True, kw_only=True)
class DiscoveryTask:

    sketch: Sketch
    compiled: CompiledSketch
    payload: dict[str, Any]
    verify: VerifyPolicy

    @classmethod
    def from_sketch(
        cls, sketch: Sketch, *, verify: VerifyPolicy | None = None
    ) -> DiscoveryTask:
        return cls(
            sketch=sketch,
            compiled=compile_sketch(sketch),
            payload=sketch_to_dict(sketch),
            verify=SKETCH_EXIT_VERIFY if verify is None else verify,
        )


@dataclass(kw_only=True)
class PlatformComponents:

    dataset: PDEDataset
    executor: PythonExecutor
    evaluator: Evaluator | None = None
    context: ExecutionContext | None = None
    registry: FunctionRegistry
    recorder: VizRecorder | None = None
    task: DiscoveryTask | None = None
    device: torch.device | None = None
    deadline: float | None = None


@runtime_checkable
class ScoreContract(Protocol):

    score_kind: ClassVar[str]
    score_direction: ClassVar[Literal["min", "max"]]


@runtime_checkable
class FacadeWiringContract(ScoreContract, Protocol):

    config_cls: ClassVar[type[Any]]
    descriptor: ClassVar[InstrumentDescriptor]
    one_shot: ClassVar[bool]
    headline_coefficient_source: ClassVar[Literal["native", "platform_refit"]]
    sketch_lower_owner: ClassVar[Literal["platform", "native"]]
    constructor_kwargs: ClassVar[frozenset[str]]

    @property
    def runner_batch_size(self) -> int:
        ...

    @property
    def derivative_requirements(self) -> DerivativeReqs:
        ...


class SurrogateTrainer(Protocol):

    @property
    def artifacts(self) -> dict[str, dict[str, str | int]] | None:
        ...

    def train_surrogate(self, dataset: PDEDataset) -> tuple[Module, TrainingResult]:
        ...


@runtime_checkable
class SearchAlgorithm(Protocol):

    def prepare(self, components: PlatformComponents) -> None:
        ...

    def propose(self, n: int) -> list[str]:
        ...

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        ...

    def update(self, results: list[EvaluationResult]) -> None:
        ...

    def build_final_result(self) -> EvaluationResult:
        ...

    def build_result_target(self) -> Tensor:
        ...

    @property
    def best_score(self) -> float:
        ...

    @property
    def best_expression(self) -> str:
        ...

    @property
    def config(self) -> dict[str, Any]:
        ...

    @property
    def state(self) -> dict[str, Any]:
        ...

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        ...


@runtime_checkable
class IterativeSearchAlgorithm(SearchAlgorithm, Protocol):

    def between_iterations(self) -> None:
        ...


@runtime_checkable
class DerivativeSourceAlgorithm(SearchAlgorithm, Protocol):

    @property
    def derivative_source(self) -> Literal["finite_diff", "autograd", "none"]:
        ...


@runtime_checkable
class SearchedVocabularyAlgorithm(SearchAlgorithm, Protocol):

    @property
    def searched_vocabulary(self) -> dict[str, object]:
        ...


@runtime_checkable
class DeadlineAwareSearchAlgorithm(SearchAlgorithm, Protocol):

    def set_wall_clock_deadline(self, deadline: float) -> None:
        ...


@runtime_checkable
class TerminatingSearchAlgorithm(SearchAlgorithm, Protocol):

    @property
    def is_done(self) -> bool:
        ...
