
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.callbacks import CheckpointCallback
from kd.search.eqgpt.backend import FakeGPTBackend
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.protocol import PlatformComponents, SearchAlgorithm
from kd.search.runner import ExperimentRunner
from tests.unit.search.eqgpt._state_fingerprint import state_fingerprint

_BATCH = 8
_SEED = 0


class _IterationRecorder:

    def __init__(self) -> None:
        self.indices: list[int] = []

    @property
    def should_stop(self) -> bool:
        return False

    def on_experiment_start(self, algorithm: SearchAlgorithm) -> None:
        pass

    def on_iteration_start(self, iteration: int, algorithm: SearchAlgorithm) -> None:
        self.indices.append(iteration)

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: SearchAlgorithm,
        candidates: list[str],
        results: list[Any],
    ) -> None:
        pass

    def on_experiment_end(self, algorithm: SearchAlgorithm) -> None:
        pass


def _components() -> PlatformComponents:
    x = torch.linspace(0.0, 1.0, 12)
    t = torch.linspace(0.0, 0.5, 6)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    dataset = PDEDataset(
        name="ckpt_tiny",
        task_type=TaskType.PDE,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=torch.sin(gx) * torch.cos(gt))},
        lhs_field="u",
        lhs_axis="t",
    )
    return PlatformBuilder(dataset, DerivativeReqs()).build()


def _plugin() -> EqGPTPlugin:
    config = EqGPTConfig(
        sparsity_alpha=0.02,
        seed=_SEED,
        samples_per_epoch=_BATCH,
        top_k=4,
        max_length=12,
    )
    return EqGPTPlugin(config, backend=FakeGPTBackend(57, seed=_SEED))


def _runner(max_iterations: int, callbacks: list | None = None) -> ExperimentRunner:
    return ExperimentRunner(
        algorithm=_plugin(),
        max_iterations=max_iterations,
        batch_size=_BATCH,
        callbacks=callbacks or [],
    )


@pytest.mark.integration
@pytest.mark.slow
def test_runner_checkpoint_resume_equals_straight(tmp_path: Path) -> None:
    components = _components()

    straight_runner = _runner(5)
    straight = straight_runner.run(components)

    ckpt_dir = tmp_path / "ckpt"
    _runner(2, [CheckpointCallback(directory=ckpt_dir, every_n=2)]).run(components)
    latest = max(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)







    starts = _IterationRecorder()
    resumed_runner = _runner(3, [starts])
    resumed_runner.load_checkpoint(latest)
    resumed = resumed_runner.run(components)

    assert starts.indices == [0, 1, 2]
    assert resumed.best_expression == straight.best_expression
    assert resumed.best_score == pytest.approx(straight.best_score, rel=0, abs=0)



    assert state_fingerprint(resumed_runner._algorithm) == state_fingerprint(
        straight_runner._algorithm
    )
