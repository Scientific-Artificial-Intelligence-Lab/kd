
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import torch

from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.eqgpt.backend import FakeGPTBackend
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.recorder import VizRecorder
from kd.search.runner import ExperimentRunner
from kd.viz.engine import VizEngine

_PLUGIN_STEMS = {
    "plugin_reward_convergence",
    "plugin_pool_reward_spread",
    "plugin_finetune_loss",
}


def _components(recorder: VizRecorder):
    x = torch.linspace(0.0, 1.0, 12)
    t = torch.linspace(0.0, 0.5, 6)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    dataset = PDEDataset(
        name="viz_e2e",
        task_type=TaskType.PDE,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=torch.sin(gx) * torch.cos(gt))},
        lhs_field="u",
        lhs_axis="t",
    )
    import dataclasses

    built = PlatformBuilder(dataset, DerivativeReqs()).build()
    return dataclasses.replace(built, recorder=recorder)


def _plugin() -> EqGPTPlugin:
    config = EqGPTConfig(
        sparsity_alpha=0.02,
        seed=0,
        samples_per_epoch=8,
        top_k=4,
        max_length=12,
        variables=("t", "x"),
    )
    return EqGPTPlugin(config, backend=FakeGPTBackend(57, seed=0))


def test_render_all_produces_the_three_eqgpt_plugin_plots(tmp_path) -> None:
    recorder = VizRecorder(enabled=True)
    plugin = _plugin()
    result = ExperimentRunner(algorithm=plugin, max_iterations=2).run(
        _components(recorder)
    )
    report = VizEngine(tmp_path).render_all(result, algorithm=plugin)
    plugin_stems = {p.stem for p in report.figures if p.stem.startswith("plugin_")}
    assert plugin_stems == _PLUGIN_STEMS
