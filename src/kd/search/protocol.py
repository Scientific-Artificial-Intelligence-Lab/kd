
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor

    from kd.core.evaluator import EvaluationResult, Evaluator
    from kd.core.executor.context import ExecutionContext
    from kd.core.expr.executor import PythonExecutor
    from kd.core.expr.registry import FunctionRegistry
    from kd.data.schema import PDEDataset
    from kd.search.recorder import VizRecorder


@dataclass(kw_only=True)
class PlatformComponents:

    dataset: PDEDataset
    executor: PythonExecutor
    evaluator: Evaluator | None = None
    context: ExecutionContext | None = None
    registry: FunctionRegistry
    recorder: VizRecorder | None = None


@runtime_checkable
class ScoreContract(Protocol):

    score_kind: ClassVar[str]
    score_direction: ClassVar[Literal["min", "max"]]


@runtime_checkable
class FacadeWiringContract(ScoreContract, Protocol):

    config_cls: ClassVar[type[Any]]
    one_shot: ClassVar[bool]

    @property
    def runner_batch_size(self) -> int:
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
class TerminatingSearchAlgorithm(SearchAlgorithm, Protocol):

    @property
    def is_done(self) -> bool:
        ...
