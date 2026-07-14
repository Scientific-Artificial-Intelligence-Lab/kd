
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import kd
from kd.core.evaluator import EvaluationResult
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.search.sga.config import SGAConfig
from kd.search.sga.plugin import SGAPlugin
from kd.viz.extension import VizExtension

_EXPECTED_PLUGIN_PLOTS = (
    "population_diversity",
    "complexity_evolution",
    "fitness_spread",
)

_N_GENS = 5



_MIN_RENDERED_SVG_BYTES = 2000


def _populated_plugin() -> SGAPlugin:
    recorder = VizRecorder(enabled=True)
    for i in range(_N_GENS):
        recorder.log("gen_mean_aic", float("inf") if i == 0 else 100.0 - 10.0 * i)
        recorder.log("n_unique", 20 - i)
        recorder.log("gen_mean_complexity", 1.0 + 0.5 * i)
        recorder.log("gen_best_aic", float("inf") if i == 0 else 80.0 - 8.0 * i)
        recorder.log("gen_best_nmse", float("inf") if i == 0 else 0.4 - 0.05 * i)
        recorder.log("n_valid", 0 if i == 0 else 18 + i)



        recorder.log("best_aic", 80.0 if i == 0 else 80.0 - 8.0 * i)
    plugin = SGAPlugin(SGAConfig())
    plugin._recorder = recorder
    return plugin


def _minimal_result(recorder: VizRecorder) -> ExperimentResult:
    final_eval = EvaluationResult(
        mse=0.01,
        nmse=0.01,
        r2=0.99,
        score=1.0,
        complexity=2,
        coefficients=torch.tensor([0.1, -0.5], dtype=torch.float64),
        is_valid=True,
        selected_indices=[0, 1],
        residuals=None,
        terms=["u_x", "u_xx"],
        expression="u_t = 0.1*u_x - 0.5*u_xx",
    )
    actual = torch.linspace(0.0, 1.0, 20)
    return ExperimentResult(
        best_expression="u_t = 0.1*u_x - 0.5*u_xx",
        best_score=1.0,
        iterations=_N_GENS,
        early_stopped=False,
        final_eval=final_eval,
        actual=actual,
        predicted=actual + 0.01,
        dataset_name="synthetic",
        algorithm_name="SGAPlugin",
        config={"algorithm": "sga"},
        recorder=recorder,
    )


@pytest.mark.integration
def test_sga_vizextension_renders_via_engine(tmp_path: Path) -> None:
    plugin = _populated_plugin()
    result = _minimal_result(plugin._recorder)


    assert isinstance(plugin, VizExtension), (
        "SGAPlugin must satisfy VizExtension (list_plots / render_plot / "
        "get_plot_data) so engine._render_plugin_plots renders it."
    )

    out_dir = tmp_path / "sga_viz"
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
def test_sga_tree_plots_render_via_engine(tmp_path: Path) -> None:
    from kd.search.sga.pde import PDE
    from kd.search.sga.tree import Node, Tree

    plugin = _populated_plugin()


    plugin._population = [
        PDE(
            [
                Tree(
                    Node(
                        "*",
                        2,
                        [Node("u", 0), Node("d", 2, [Node("u", 0), Node("x", 0)])],
                    )
                )
            ]
        )
    ]
    result = _minimal_result(plugin._recorder)

    out_dir = tmp_path / "sga_tree_viz"
    engine = kd.VizEngine(output_dir=out_dir)
    report = engine.render_all(result, algorithm=plugin)

    files = {p.name for p in out_dir.iterdir() if p.is_file()}
    assert "equation_tree.svg" in files, (
        f"universal equation tree must render; got {sorted(files)}"
    )
    assert "plugin_genome_tree.svg" in files, (
        f"SGA genome tree must render via the plugin; got {sorted(files)}"
    )



    assert (out_dir / "equation_tree.svg").stat().st_size > _MIN_RENDERED_SVG_BYTES
    assert (out_dir / "plugin_genome_tree.svg").stat().st_size > _MIN_RENDERED_SVG_BYTES



    assert report.report is not None, "render_all must produce an HTML report"
    html = report.report.read_text()
    assert "Equation Tree" in html, "equation tree heading missing from HTML report"
    assert "Genome Tree" in html, "genome tree heading missing from HTML report"


    tree_warnings = [w for w in report.warnings if "tree" in w.lower()]
    assert not tree_warnings, f"unexpected tree-plot warnings: {tree_warnings}"


@pytest.mark.integration
def test_sga_recorder_whitelist_covers_plotted_metrics() -> None:
    from kd.search.sga.plugin import _LOGGED_METRICS

    plotted = {"n_unique", "gen_mean_complexity", "gen_mean_aic"}
    assert plotted <= set(_LOGGED_METRICS), (
        f"plotted metrics must be a subset of the whitelist; "
        f"stray={plotted - set(_LOGGED_METRICS)}"
    )


@pytest.mark.integration
def test_sga_recorder_has_whitelist_fields() -> None:
    from kd.search.sga.plugin import _LOGGED_METRICS, _SURROGATE_METRICS

    plugin = _populated_plugin()
    recorder_keys = set(plugin._recorder.keys())
    per_gen_expected = set(_LOGGED_METRICS) | {"best_aic"}





    assert recorder_keys == per_gen_expected, (
        f"a GA-only recorder must carry exactly the 6 whitelist fields + legacy "
        f"'best_aic' (7 keys; surrogate keys are absent without training); "
        f"symmetric-difference={per_gen_expected ^ recorder_keys}"
    )
    assert recorder_keys.isdisjoint(_SURROGATE_METRICS), (
        f"a GA-only recorder must not carry any surrogate key; leaked "
        f"{sorted(recorder_keys & set(_SURROGATE_METRICS))}."
    )
