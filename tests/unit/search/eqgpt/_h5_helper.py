
from __future__ import annotations

import sys
from pathlib import Path

import torch

from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.eqgpt.backend import FakeGPTBackend
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.protocol import PlatformComponents
from tests.unit.search.eqgpt._state_fingerprint import (
    state_fingerprint,
    weights_fingerprint,
)

_SEED = 0
_BATCH = 8


def _dataset() -> PDEDataset:
    x = torch.linspace(0.0, 1.0, 12)
    t = torch.linspace(0.0, 0.5, 6)
    grid_x, grid_t = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(grid_x) * torch.cos(grid_t)
    return PDEDataset(
        name="h5_tiny",
        task_type=TaskType.PDE,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _components() -> PlatformComponents:
    return PlatformBuilder(_dataset(), DerivativeReqs()).build()


def _new_plugin() -> EqGPTPlugin:
    from kd.search.eqgpt.vocab import load_vocab

    w = load_vocab().word2id


    config = EqGPTConfig(
        sparsity_alpha=0.02,
        seed=_SEED,
        samples_per_epoch=_BATCH,
        top_k=4,
        max_length=12,
        variables=("t", "x"),
        masked_tokens=frozenset({w["uxxxx"], w["uxxxxx"]}),
    )
    return EqGPTPlugin(config, backend=FakeGPTBackend(57, seed=_SEED))


def _run_epochs(plugin: EqGPTPlugin, n: int) -> None:
    for _ in range(n):
        candidates = plugin.propose(_BATCH)
        plugin.update(plugin.evaluate(candidates))


def _emit(plugin: EqGPTPlugin) -> str:
    return f"{state_fingerprint(plugin)} {weights_fingerprint(plugin)}"


def _run_straight(epochs: int) -> str:
    plugin = _new_plugin()
    plugin.prepare(_components())
    initial = {k: v.clone() for k, v in plugin.state["backend_state"].items()}
    _run_epochs(plugin, epochs)
    state = plugin.state


    assert len(state["reward_history"]) == epochs, "reward_history did not grow"
    assert state["top_k"], "top_k pool is empty"
    assert any(
        not torch.equal(initial[k], v) for k, v in state["backend_state"].items()
    ), "fine-tuning changed no weight"
    return _emit(plugin)


def main() -> None:
    mode, path = sys.argv[1], Path(sys.argv[2])
    if mode == "straight5":
        print(_run_straight(5))
    elif mode == "straight2":
        print(_run_straight(2))
    elif mode == "phase1":
        plugin = _new_plugin()
        plugin.prepare(_components())
        _run_epochs(plugin, 2)
        torch.save(plugin.state, path)




        print(_emit(plugin))
    elif mode == "phase2":
        plugin = _new_plugin()
        plugin.state = torch.load(path, weights_only=False)
        plugin.prepare(_components())
        _run_epochs(plugin, 3)
        print(_emit(plugin))
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
