
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from kd.search.discover import viz as discover_viz
from kd.search.recorder import VizRecorder


@pytest.mark.unit
def test_reward_plots_bind_distinct_recorder_keys() -> None:
    recorder = VizRecorder(enabled=True)
    top_eps = [0.9, 0.95, 0.99]
    full = [0.2, 0.3, 0.4]
    for t, f in zip(top_eps, full, strict=True):
        recorder.log("reward", t)
        recorder.log("reward_full", f)

    top_data = discover_viz.get_data("reward_convergence", recorder)
    full_data = discover_viz.get_data("reward_full_mean", recorder)
    assert list(top_data["y"]) == pytest.approx(top_eps)
    assert list(full_data["y"]) == pytest.approx(full)
    assert top_data["ylabel"] == "reward"
    assert full_data["ylabel"] == "reward_full"


@pytest.mark.unit
def test_reward_titles_carry_reference_naming() -> None:
    infos = {info.name: info.title for info in discover_viz.list_plot_infos()}



    assert "Top" in infos["reward_convergence"]
    assert "Full-Batch" in infos["reward_full_mean"]
    assert infos["entropy_loss_decay"] == "Entropy Loss"
