
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import Any

import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.axes import Axes

from kd.core.evaluator import EvaluationResult
from kd.search.discover import DISCOVERPlugin
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.viz.plots.comparison import render_overlaid_convergence

pytestmark = pytest.mark.unit

_N_SAMPLES = 12
_SCORE_SERIES = [1.0, 0.5, 0.25]


def _make_result(
    name: str,
    *,
    score_kind: str,
    score_direction: str,
    algorithm: str | None = None,
) -> ExperimentResult:
    recorder = VizRecorder()
    for score in _SCORE_SERIES:
        recorder.log("_best_score", score)
        recorder.log("_best_expr", f"expr_{name}")
    config: dict[str, Any] = {"max_iter": len(_SCORE_SERIES)}
    if algorithm is not None:
        config["algorithm"] = algorithm
    return ExperimentResult(
        best_expression=f"expr_{name}",
        best_score=_SCORE_SERIES[-1],
        iterations=len(_SCORE_SERIES),
        early_stopped=False,
        final_eval=EvaluationResult(
            mse=0.01,
            nmse=0.005,
            r2=0.99,
            score=-50.0,
            complexity=2,
            coefficients=torch.tensor([1.0, 0.5]),
            is_valid=True,
            error_message="",
            selected_indices=[0, 1],
            residuals=torch.zeros(_N_SAMPLES),
            terms=["u", "u_x"],
            expression=f"expr_{name}",
        ),
        actual=torch.linspace(0.0, 1.0, _N_SAMPLES),
        predicted=torch.linspace(0.0, 1.0, _N_SAMPLES),
        dataset_name="cross_pool",
        algorithm_name=name,
        config=config,
        recorder=recorder,
        score_kind=score_kind,
        score_direction=score_direction,
    )


def _has_mean_band(ax: Axes) -> bool:
    return "mean" in [line.get_label() for line in ax.lines]


def _discover_eqgpt_results() -> list[ExperimentResult]:
    return [
        _make_result(
            "discover_run",
            score_kind=DISCOVERPlugin.score_kind,
            score_direction=DISCOVERPlugin.score_direction,
            algorithm="discover",
        ),
        _make_result(
            "eqgpt_run",
            score_kind=EqGPTPlugin.score_kind,
            score_direction=EqGPTPlugin.score_direction,
            algorithm="eqgpt",
        ),
    ]


def test_discover_and_eqgpt_declare_distinct_identities() -> None:
    discover_identity = (DISCOVERPlugin.score_kind, DISCOVERPlugin.score_direction)
    eqgpt_identity = (EqGPTPlugin.score_kind, EqGPTPlugin.score_direction)
    assert discover_identity != eqgpt_identity, (
        f"DISCOVER {discover_identity} and EqGPT {eqgpt_identity} must be "
        "incommensurable (distinct identities) so their overlay does not pool"
    )


def test_discover_eqgpt_overlay_suppresses_band_and_warns() -> None:
    results = _discover_eqgpt_results()
    fig, ax = plt.subplots()
    try:
        warnings = render_overlaid_convergence(results, ax)
        assert not _has_mean_band(ax), (
            "DISCOVER reward and EqGPT reward are incommensurable; the mean±std "
            "band must be suppressed, not pooled"
        )
        assert warnings, "an incommensurable-identity overlay must emit a warning"
    finally:
        plt.close(fig)


def test_discover_eqgpt_overlay_keeps_individual_curves() -> None:
    results = _discover_eqgpt_results()
    fig, ax = plt.subplots()
    try:
        render_overlaid_convergence(results, ax)
        run_curves = [ln for ln in ax.lines if ln.get_label() != "mean"]
        assert len(run_curves) >= 2
    finally:
        plt.close(fig)


def test_discover_eqgpt_overlay_ylabel_falls_back_to_best_score() -> None:
    results = _discover_eqgpt_results()
    fig, ax = plt.subplots()
    try:
        render_overlaid_convergence(results, ax)
        assert ax.get_ylabel() == "Best Score"
    finally:
        plt.close(fig)
