
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from kd.search.lifecycle import LifecycleState

from kd.core.platform.builder import PlatformBuilder
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.search.protocol import PlatformComponents
from kd.search.pysr import PySRPlugin
from kd.search.runner import ExperimentRunner

_SENTINEL_EXPRESSION = "u_x"


def _bridge_dataset() -> PDEDataset:
    x = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = 1.0 + xg * xg + tg * tg
    return PDEDataset(
        name="restore-bridge",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
        axis_order=["x", "t"],
        fields={"u": FieldData("u", u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _components() -> PlatformComponents:
    dataset = _bridge_dataset()
    return PlatformBuilder(dataset, PySRPlugin().derivative_requirements).build()


def _write_restore_checkpoint(components: PlatformComponents, path: Path) -> None:
    donor = PySRPlugin()
    donor.prepare(components)
    payload = donor.state
    payload["best_expression"] = _SENTINEL_EXPRESSION
    payload["best_score"] = 0.125
    payload["fitted"] = True
    donor.state = payload
    ExperimentRunner(algorithm=donor, max_iterations=0).save_checkpoint(path)


def _write_empty_state_checkpoint(path: Path) -> None:
    from kd.search.callbacks import CHECKPOINT_VERSION

    torch.save(
        {"version": CHECKPOINT_VERSION, "iteration": 0, "algorithm_state": {}},
        path,
    )


@pytest.mark.integration
def test_load_checkpoint_then_run_drives_restore_transition(tmp_path: Path) -> None:
    components = _components()
    ckpt = tmp_path / "restore.pt"
    _write_restore_checkpoint(components, ckpt)

    fresh = PySRPlugin()
    runner = ExperimentRunner(algorithm=fresh, max_iterations=0)
    runner.load_checkpoint(ckpt)
    runner.run(components)


    assert runner.lifecycle is not None
    assert runner.lifecycle.restored is True, (
        "resumed run drove a FRESH transition; the restore-armed flag set at "
        "load_checkpoint did not reach the per-run lifecycle machine"
    )
    assert runner.lifecycle.state is LifecycleState.DONE


    assert fresh.best_expression == _SENTINEL_EXPRESSION


@pytest.mark.integration
def test_run_without_load_is_fresh_transition(tmp_path: Path) -> None:
    components = _components()

    fresh = PySRPlugin()
    runner = ExperimentRunner(algorithm=fresh, max_iterations=0)
    runner.run(components)

    assert runner.lifecycle is not None
    assert runner.lifecycle.restored is False
    assert runner.lifecycle.state is LifecycleState.DONE
    assert fresh.best_expression == ""


@pytest.mark.integration
def test_load_checkpoint_empty_state_then_run_is_fresh(tmp_path: Path) -> None:
    components = _components()
    ckpt = tmp_path / "empty.pt"
    _write_empty_state_checkpoint(ckpt)

    fresh = PySRPlugin()
    runner = ExperimentRunner(algorithm=fresh, max_iterations=0)
    runner.load_checkpoint(ckpt)
    runner.run(components)

    assert runner.lifecycle is not None
    assert runner.lifecycle.restored is False, (
        "empty algorithm_state must classify FRESH, not RESTORE"
    )
    assert runner.lifecycle.state is LifecycleState.DONE
    assert fresh.best_expression == ""


@pytest.mark.integration
def test_same_runner_reuse_after_restore_is_fresh_no_error(tmp_path: Path) -> None:
    from kd.search.result import ExperimentResult

    components = _components()
    ckpt = tmp_path / "restore.pt"
    _write_restore_checkpoint(components, ckpt)

    plugin = PySRPlugin()
    runner = ExperimentRunner(algorithm=plugin, max_iterations=0)
    runner.load_checkpoint(ckpt)
    runner.run(components)
    assert runner.lifecycle is not None
    assert runner.lifecycle.restored is True
    assert plugin.best_expression == _SENTINEL_EXPRESSION


    result = runner.run(components)

    assert isinstance(result, ExperimentResult)
    assert runner.lifecycle.restored is False, (
        "reuse run inherited the prior RESTORE flag; the machine is not "
        "fresh-per-run"
    )
    assert runner.lifecycle.state is LifecycleState.DONE
    assert plugin.best_expression == "", "reuse must reset; stale best leaked"
