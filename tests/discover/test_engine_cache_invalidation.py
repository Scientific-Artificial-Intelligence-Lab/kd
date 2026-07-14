
from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.nn import Parameter

from kd.core.evaluator import EvaluationResult, Evaluator
from kd.core.executor.context import ExecutionContext
from kd.core.expr import FunctionRegistry, PythonExecutor
from kd.core.linear_solve import LeastSquaresSolver
from kd.data.derivatives.finite_diff import FiniteDiffProvider
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.search.discover.core.batch import Batch
from kd.search.discover.engine import DiscoverEngine
from kd.search.discover.evaluation.dedup import Deduplicator
from kd.search.discover.evaluation.reward import compute_reward
from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.validator import CandidateValidator
from kd.search.discover.training.strategy import RSPGStrategy


_LIBRARY_CONFIG = LibraryConfig(
    operators=["add"],
    state_vars=["u"],
    coord_vars=["x", "t"],
)


class _FixedAddGenerator:

    def __init__(self, library: Library) -> None:
        self._library = library
        self._param = Parameter(torch.tensor(0.0))
        add = library.name_to_index("add")
        u = library.name_to_index("u")
        self._row = np.asarray([add, u, u], dtype=np.int32)

    @property
    def library(self) -> Library:
        return self._library

    @property
    def device(self) -> torch.device:
        return self._param.device

    def sample(self, batch_size: int) -> Batch:
        actions = np.tile(self._row, (batch_size, 1))
        n_tokens = len(self._library.tokens)
        obs = np.zeros((batch_size, 4, actions.shape[1]), dtype=np.float32)
        priors = np.ones((batch_size, actions.shape[1], n_tokens), dtype=np.float32)
        lengths = np.full(batch_size, actions.shape[1], dtype=np.int32)
        return Batch(actions=actions, obs=obs, priors=priors, lengths=lengths)

    def make_neglogp_and_entropy(
        self,
        batch: Batch,
        entropy_gamma: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        scale = self._param.expand(batch.actions.shape[0])
        return scale, scale * 0.0

    def parameters(self) -> Iterator[Parameter]:
        yield self._param

    def state_dict(self) -> dict[str, Any]:
        return {"param": self._param.detach().clone()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._param.data.copy_(state_dict["param"])

    def train(self, mode: bool = True) -> None:
        pass


def _make_engine() -> DiscoverEngine:
    library = Library.from_config(_LIBRARY_CONFIG)
    return DiscoverEngine(
        generator=_FixedAddGenerator(library),
        strategy=RSPGStrategy(epsilon=1.0, baseline="R_e", entropy_weight=0.0),
        reward_adapter=compute_reward,
        validator=CandidateValidator(library, max_length=5),
        deduplicator=Deduplicator(library),
        batch_size=4,
    )


def _make_real_evaluator() -> Evaluator:
    x = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    u = xx + tt**2
    dataset = PDEDataset(
        name="tiny_discover_cache",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )
    provider = FiniteDiffProvider(dataset, max_order=1)
    context = ExecutionContext(dataset=dataset, derivative_provider=provider)
    registry = FunctionRegistry.create_default()
    return Evaluator(
        executor=PythonExecutor(registry),
        solver=LeastSquaresSolver(),
        context=context,
        lhs=provider.get_derivative("u", "t", order=1).flatten(),
    )


def test_run_iteration_clears_real_evaluator_term_cache_after_success() -> None:
    engine = _make_engine()
    evaluator = _make_real_evaluator()

    metrics = engine.run_iteration(evaluator)

    assert metrics["n_unique"] == 1.0
    assert metrics["n_eval_valid"] == 1.0
    assert evaluator._term_cache is not None
    assert len(evaluator._term_cache) == 0


def test_run_iteration_invalidates_term_cache_when_evaluator_raises() -> None:
    engine = _make_engine()
    evaluator = MagicMock(spec=Evaluator)
    evaluator.evaluate_expression.side_effect = RuntimeError("eval failed")

    with pytest.raises(RuntimeError, match="eval failed"):
        engine.run_iteration(evaluator)

    evaluator.invalidate_term_cache.assert_called_once_with()


def test_term_cache_is_bounded_across_multiple_iterations() -> None:
    engine = _make_engine()
    evaluator = _make_real_evaluator()

    for _ in range(3):
        metrics = engine.run_iteration(evaluator)
        assert metrics["n_eval_valid"] == 1.0
        assert evaluator._term_cache is not None
        assert len(evaluator._term_cache) == 0
