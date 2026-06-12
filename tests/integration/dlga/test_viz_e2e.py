
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import kd
from kd.core.evaluator import EvaluationResult
from kd.search.dlga import DLGAConfig, DLGAPlugin
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.viz.extension import VizExtension

_EXPECTED_PLUGIN_PLOTS = (
    "fitness_spread",
    "population_diversity",
    "complexity_evolution",
)

_N_GENS = 5


def _populated_plugin() -> DLGAPlugin:
    recorder = VizRecorder(enabled=True)
    for i in range(_N_GENS):


        recorder.log("gen_best_fitness", float("inf") if i == 0 else 100.0 - 10.0 * i)
        recorder.log("n_unique", 20 - i)
        recorder.log("gen_mean_complexity", 1.0 + 0.5 * i)
        recorder.log("gen_mean_fitness", float("inf") if i == 0 else 120.0 - 8.0 * i)
        recorder.log("gen_best_nmse", float("inf") if i == 0 else 0.4 - 0.05 * i)
        recorder.log("n_valid", 0 if i == 0 else 18 + i)
        recorder.log("lhs_ut", 0 if i == 0 else 15)
        recorder.log("lhs_utt", 0 if i == 0 else 3)
    plugin = DLGAPlugin(DLGAConfig())
    plugin._recorder = recorder
    return plugin


def _minimal_result(recorder: VizRecorder) -> ExperimentResult:
    final_eval = EvaluationResult(
        mse=0.01,
        nmse=0.01,
        r2=0.99,
        aic=1.0,
        complexity=2,
        coefficients=torch.tensor([0.1, -0.5], dtype=torch.float64),
        is_valid=True,
        selected_indices=[0, 1],
        residuals=None,
        terms=["diff_x(u_x)", "diff_x(n2(u))"],
        expression="0.1*diff_x(u_x) - 0.5*diff_x(n2(u))",
        lhs_name="u_t",
    )
    actual = torch.linspace(0.0, 1.0, 20)
    return ExperimentResult(
        best_expression="0.1*diff_x(u_x) - 0.5*diff_x(n2(u))",
        best_score=1.0,
        iterations=_N_GENS,
        early_stopped=False,
        final_eval=final_eval,
        actual=actual,
        predicted=actual + 0.01,
        dataset_name="synthetic",
        algorithm_name="DLGAPlugin",
        config={"algorithm": "dlga"},
        recorder=recorder,
    )


@pytest.mark.integration
def test_dlga_vizextension_renders_via_engine(tmp_path: Path) -> None:
    plugin = _populated_plugin()
    result = _minimal_result(plugin._recorder)


    assert isinstance(plugin, VizExtension), (
        "DLGAPlugin must satisfy VizExtension (list_plots / render_plot / "
        "get_plot_data) so engine._render_plugin_plots renders it."
    )

    out_dir = tmp_path / "dlga_viz"
    engine = kd.VizEngine(output_dir=out_dir)
    report = engine.render_all(result, algorithm=plugin)


    expected_files = {f"plugin_{name}.svg" for name in _EXPECTED_PLUGIN_PLOTS}
    actual_files = {p.name for p in out_dir.iterdir() if p.is_file()}
    missing = expected_files - actual_files
    assert not missing, (
        f"Missing plugin SVG files: {sorted(missing)}. "
        f"Actual files: {sorted(actual_files)}"
    )


    figure_names = {p.name for p in report.figures}
    assert expected_files <= figure_names, (
        f"report.figures must include all 3 plugin plots; missing "
        f"{sorted(expected_files - figure_names)}"
    )


    html_path = out_dir / "report.html"
    assert html_path.is_file(), "render_all must produce an HTML report"
    assert html_path.stat().st_size > 0, "HTML report must be non-empty"


@pytest.mark.integration
def test_dlga_recorder_keys_are_within_allowed_whitelist(tmp_path: Path) -> None:
    del tmp_path
    from kd.search.dlga.plugin import _LOGGED_METRICS, _SURROGATE_METRICS

    plugin = _populated_plugin()
    recorder_keys = set(plugin._recorder.keys())

    allowed = set(_LOGGED_METRICS) | set(_SURROGATE_METRICS)
    stray = recorder_keys - allowed
    assert not stray, (
        f"recorder keys must be within the allowed whitelist "
        f"(_LOGGED_METRICS | _SURROGATE_METRICS); stray={sorted(stray)}."
    )


    assert set(_LOGGED_METRICS) <= recorder_keys, (
        f"the 8 generation metrics must be present; "
        f"missing={sorted(set(_LOGGED_METRICS) - recorder_keys)}."
    )
    plotted = {"gen_mean_fitness", "n_unique", "gen_mean_complexity"}
    assert plotted <= allowed, (
        f"plotted metrics must be a subset of the allowed whitelist; "
        f"stray={plotted - allowed}"
    )
