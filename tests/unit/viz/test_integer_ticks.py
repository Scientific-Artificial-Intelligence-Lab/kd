
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

from kd.search.recorder import VizRecorder
from kd.viz.plots.comparison import render_overlaid_convergence
from kd.viz.plots.convergence import plot_convergence
from kd.viz.plots.pde_residual import _line_fallback
from kd.viz.plots.residual import _render_histogram, _render_spatial

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd.search.result import ExperimentResult


COUNTS = [5, 4, 4, 3, 3]
AICS = [-21.5, -21.5, -28.8, -28.8, -28.8]
LOSSES = [1e-1, 5e-2, 3e-2, 2e-2, 1.5e-2]
REWARDS = [0.1, 0.3, 0.5, 0.6, 0.7]
SPREAD_SERIES = {
    "pool_best": REWARDS,
    "pool_median": [0.05, 0.2, 0.4, 0.5, 0.6],
    "pool_worst": [0.0, 0.1, 0.2, 0.3, 0.4],
}


def recorder_of(series: dict[str, list[Any]]) -> VizRecorder:
    recorder = VizRecorder()
    for key, values in series.items():
        for value in values:
            recorder.log(key, value)
    return recorder


def assert_integer_ticks(ax: Axes, which: str) -> None:
    ax.figure.canvas.draw()
    axis = ax.xaxis if which == "x" else ax.yaxis
    low, high = ax.get_xlim() if which == "x" else ax.get_ylim()
    low, high = min(low, high), max(low, high)
    visible = [float(t) for t in axis.get_ticklocs() if low <= t <= high]
    assert visible, f"{which} axis drew no ticks"
    fractional = [t for t in visible if t != int(t)]
    assert not fractional, (
        f"{ax.get_xlabel() if which == 'x' else ax.get_ylabel()!r} "
        f"indexes a whole quantity but ticked at {fractional}"
    )


@pytest.fixture()
def ax() -> Any:
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)





def test_convergence_iteration_axis(
    ax: Axes, mock_experiment_result: ExperimentResult
) -> None:
    mock_experiment_result.recorder = recorder_of({"_best_score": AICS})
    plot_convergence(mock_experiment_result, ax)
    assert_integer_ticks(ax, "x")


def test_overlaid_convergence_iteration_axis(
    ax: Axes, mock_experiment_result: ExperimentResult
) -> None:
    mock_experiment_result.recorder = recorder_of({"_best_score": AICS})
    render_overlaid_convergence([mock_experiment_result], ax)
    assert_integer_ticks(ax, "x")


@pytest.mark.parametrize("n_samples", [9, 21])
def test_residual_histogram_count_axis(ax: Axes, n_samples: int) -> None:
    _render_histogram(ax, np.linspace(-1.0, 1.0, n_samples), [])
    assert_integer_ticks(ax, "y")


@pytest.mark.parametrize("side", [3, 4])
def test_residual_spatial_index_axes(side: int) -> None:
    figure, axes = plt.subplots(figsize=(12, 5))
    data = np.random.default_rng(0).normal(size=(side, side))
    _render_spatial(axes, data, (side, side), [])
    assert_integer_ticks(axes, "x")
    assert_integer_ticks(axes, "y")
    plt.close(figure)


def test_pde_residual_index_axis(ax: Axes) -> None:
    data = np.random.default_rng(0).normal(size=5)
    _line_fallback(ax, data, "residual")
    assert_integer_ticks(ax, "x")


@pytest.mark.parametrize("scores", [[], [float("inf")] * 3], ids=["empty", "all-inf"])
def test_convergence_no_data_panel_keeps_integer_axis(
    ax: Axes, mock_experiment_result: ExperimentResult, scores: list[float]
) -> None:
    mock_experiment_result.recorder = recorder_of({"_best_score": scores})
    plot_convergence(mock_experiment_result, ax)
    assert_integer_ticks(ax, "x")






ITERATION_PANELS = [
    ("kd.search.sga.viz", "population_diversity", {"n_unique": COUNTS}),
    ("kd.search.sga.viz", "complexity_evolution", {"gen_mean_complexity": COUNTS}),
    ("kd.search.sga.viz", "fitness_spread", {"pop_mean_aic": AICS}),
    ("kd.search.dlga.viz", "fitness_spread", {"gen_mean_fitness": LOSSES}),
    ("kd.search.dlga.viz", "population_diversity", {"n_unique": COUNTS}),
    ("kd.search.dlga.viz", "complexity_evolution", {"gen_mean_complexity": COUNTS}),
    ("kd.search.discover.viz", "reward_convergence", {"reward": REWARDS}),
    ("kd.search.discover.viz", "reward_full_mean", {"reward_full": REWARDS}),
    ("kd.search.discover.viz", "entropy_loss_decay", {"entropy_loss": LOSSES}),
    ("kd.search.discover.viz", "baseline_ewma", {"baseline": REWARDS}),
    ("kd.search.eqgpt.viz", "reward_convergence", {"pool_best": REWARDS}),
    ("kd.search.eqgpt.viz", "finetune_loss", {"finetune_loss": LOSSES}),


    ("kd.search.eqgpt.viz", "pool_reward_spread", SPREAD_SERIES),
    ("kd.search.llm4ed.viz", "pool_reward_spread", SPREAD_SERIES),
    ("kd.search.llm4ed.viz", "invalid_count", {"n_invalid": [2, 1, 1, 0, 1]}),
    ("kd.search.llm4ed.viz", "llm_calls", {"n_llm_calls": [3, 3, 3, 3, 3]}),
]


@pytest.mark.parametrize(
    ("module_path", "plot_name", "series"),
    ITERATION_PANELS,
    ids=[f"{path.split('.')[2]}-{name}" for path, name, _ in ITERATION_PANELS],
)
def test_plugin_iteration_axis(
    ax: Axes, module_path: str, plot_name: str, series: dict[str, list[Any]]
) -> None:
    import importlib

    module = importlib.import_module(module_path)
    module.render(plot_name, ax, recorder_of(series))
    assert_integer_ticks(ax, "x")


@pytest.mark.parametrize(
    ("module_path", "plot_name", "series"),
    [
        ("kd.search.sga.viz", "population_diversity", {"n_unique": COUNTS}),
        ("kd.search.dlga.viz", "population_diversity", {"n_unique": COUNTS}),
        ("kd.search.llm4ed.viz", "invalid_count", {"n_invalid": [2, 1, 1, 0, 1]}),


        ("kd.search.llm4ed.viz", "llm_calls", {"n_llm_calls": [1, 0, 0, 1, 0]}),
    ],
    ids=["sga-diversity", "dlga-diversity", "llm4ed-invalid", "llm4ed-calls"],
)
def test_plugin_count_axis(
    ax: Axes, module_path: str, plot_name: str, series: dict[str, list[Any]]
) -> None:
    import importlib

    module = importlib.import_module(module_path)
    module.render(plot_name, ax, recorder_of(series))
    assert_integer_ticks(ax, "y")


def test_surrogate_epoch_axis(ax: Axes) -> None:
    from kd.search.dlga.viz import render_surrogate

    recorder = recorder_of(
        {
            "surrogate_epoch": [[0, 1, 2, 3, 4]],
            "surrogate_train_loss": [LOSSES],
            "surrogate_val_loss": [[v * 1.2 for v in LOSSES]],
        }
    )
    render_surrogate(ax, recorder)
    assert_integer_ticks(ax, "x")


@pytest.mark.parametrize("complexities", [[1, 2, 3], [2, 4]], ids=["narrow", "gapped"])
def test_pysr_complexity_axis(ax: Axes, complexities: list[int]) -> None:
    from kd.search.pysr.viz import render

    losses = LOSSES[: len(complexities)]
    recorder = recorder_of(
        {
            "pareto_complexity": [complexities],
            "pareto_loss": [losses],
            "selected_complexity": [complexities[-1]],
            "selected_loss": [losses[-1]],
        }
    )
    render("pareto_front", ax, recorder)
    assert_integer_ticks(ax, "x")
