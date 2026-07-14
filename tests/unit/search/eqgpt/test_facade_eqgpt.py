
from __future__ import annotations

import pytest
import torch

from kd.api import Model
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.eqgpt.backend import resolve_asset_path
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.sga.config import SGAConfig
from kd.viz.engine import VizEngine


def _pretrained_assets_available() -> bool:
    try:
        resolve_asset_path()
    except FileNotFoundError:
        return False
    return True






_skip_no_pretrained_assets = pytest.mark.skipif(
    not _pretrained_assets_available(),
    reason=(
        "pretrained EqGPT weights absent; the facade default path resolves them "
        "via resolve_asset_path() (repo-root ref_libs fallback or "
        "$KD_EQGPT_ASSET_DIR). Set KD_EQGPT_ASSET_DIR to run this e2e."
    ),
)


def _tiny_dataset() -> PDEDataset:
    x = torch.linspace(0.0, 1.0, 10)
    t = torch.linspace(0.0, 0.5, 5)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    return PDEDataset(
        name="facade_tiny",
        task_type=TaskType.PDE,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=torch.sin(gx) * torch.cos(gt))},
        lhs_field="u",
        lhs_axis="t",
    )


def test_eqgpt_registered_in_facade() -> None:
    from kd.api import _PLUGIN_CLASS_BY_ALGORITHM

    assert _PLUGIN_CLASS_BY_ALGORITHM.get("eqgpt") is EqGPTPlugin


def test_registered_eqgpt_plugin_is_eqgpt_reward_max() -> None:
    from kd.api import _PLUGIN_CLASS_BY_ALGORITHM

    cls = _PLUGIN_CLASS_BY_ALGORITHM.get("eqgpt")
    assert cls is not None
    assert cls.score_direction == "max"

    assert cls.score_kind == "EqGPT reward"


def test_model_accepts_eqgpt_algorithm_with_config() -> None:
    Model(algorithm="eqgpt", config=EqGPTConfig(sparsity_alpha=0.02))


def test_model_rejects_unknown_algorithm() -> None:
    with pytest.raises(NotImplementedError):
        Model(algorithm="not_an_algorithm")


def test_model_eqgpt_rejects_sga_only_params() -> None:
    with pytest.raises(TypeError):
        Model(algorithm="eqgpt", population=20)


def test_model_eqgpt_rejects_mismatched_config_type() -> None:
    with pytest.raises(TypeError):
        Model(algorithm="eqgpt", config=SGAConfig()).fit(_tiny_dataset())


@pytest.mark.slow
@_skip_no_pretrained_assets
def test_model_eqgpt_fit_runs(tmp_path) -> None:
    model = Model(
        algorithm="eqgpt", generations=3, config=EqGPTConfig(sparsity_alpha=0.02)
    )
    model.fit(_tiny_dataset())
    assert model.result_.iterations == 3
    assert model.best_expr_





    report = VizEngine(tmp_path).render_all(model.result_, algorithm=model.algorithm_)
    plugin_stems = {p.stem for p in report.figures if p.stem.startswith("plugin_")}
    assert plugin_stems == {
        "plugin_reward_convergence",
        "plugin_pool_reward_spread",
        "plugin_finetune_loss",
    }
