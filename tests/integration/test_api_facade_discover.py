
from __future__ import annotations

import html
from pathlib import Path

import pytest

import kd
from kd import DiscoverConfig, Model
from kd.search.discover import DISCOVERPlugin
from kd.search.result import ExperimentResult


@pytest.mark.smoke
@pytest.mark.integration
def test_facade_discover_fit_smoke() -> None:
    dataset = kd.generate_burgers_data(nx=32, nt=16, nu=0.1, seed=0)
    model = Model(
        algorithm="discover",
        generations=2,
        seed=0,
        verbose=False,
    )
    fitted = model.fit(dataset)


    assert fitted is model


    assert isinstance(model.result_, ExperimentResult)
    assert isinstance(model.best_expr_, str)
    assert isinstance(model.best_score_, float)


    assert isinstance(model.algorithm_, DISCOVERPlugin)


@pytest.mark.smoke
@pytest.mark.integration
def test_facade_discover_with_custom_config_smoke() -> None:
    dataset = kd.generate_burgers_data(nx=32, nt=16, nu=0.1, seed=0)
    cfg = DiscoverConfig(seed=7, batch_size=4)
    model = Model(
        algorithm="discover",
        generations=2,
        verbose=False,
        config=cfg,
    )
    model.fit(dataset)


    assert isinstance(model.algorithm_, DISCOVERPlugin)
    assert model.algorithm_._config.seed == 7
    assert model.algorithm_._config.batch_size == 4





_EXPECTED_DISCOVER_METRICS: frozenset[str] = frozenset(
    {
        "pg_loss",
        "entropy_loss",
        "total_loss",
        "baseline",
        "reward",
        "grad_norm",
        "reward_max",
        "best_reward",
        "n_valid",
        "n_eval_valid",
        "n_invalid_in_topk",
        "n_unique",
    },
)


@pytest.mark.integration
def test_discover_facade_records_engine_metrics() -> None:
    dataset = kd.generate_burgers_data(nx=32, nt=16, nu=0.1, seed=0)
    model = Model(
        algorithm="discover",
        generations=2,
        seed=0,
        verbose=False,
    )
    model.fit(dataset)

    recorder = model.result_.recorder
    keys = recorder.keys()
    missing = _EXPECTED_DISCOVER_METRICS - keys
    assert not missing, (
        f"After Model(algorithm='discover').fit(), result_.recorder is "
        f"missing whitelisted DISCOVER metrics: {sorted(missing)}. "
        f"recorder.keys() = {sorted(keys)}"
    )





    pg_series = recorder.get("pg_loss")
    expected_len = model.result_.iterations
    assert len(pg_series) == expected_len, (
        f"recorder.get('pg_loss') length = {len(pg_series)}; expected "
        f"{expected_len} (one append per completed generation)."
    )
    assert len(pg_series) >= 1, (
        f"recorder.get('pg_loss') length = {len(pg_series)}; expected >= 1 "
        "(at least one generation should run)."
    )






_EXPECTED_PLUGIN_PLOTS: frozenset[str] = frozenset(
    {
        "reward_convergence",
        "entropy_loss_decay",
        "baseline_ewma",
    },
)


@pytest.mark.integration
def test_render_all_includes_plugin_plots(tmp_path: Path) -> None:

    dataset = kd.generate_burgers_data(nx=32, nt=16, nu=0.1, seed=0)
    model = Model(
        algorithm="discover",
        generations=3,
        seed=0,
        verbose=False,
    )
    model.fit(dataset)


    from kd.viz.extension import VizExtension

    assert isinstance(model.algorithm_, VizExtension), (
        "DISCOVERPlugin must satisfy the VizExtension protocol after Stage "
        "3 GREEN (list_plots + render_plot + get_plot_data methods present)."
    )


    out_dir = tmp_path / "discover_viz"
    engine = kd.VizEngine(output_dir=out_dir)
    report = engine.render_all(
        model.result_,
        algorithm=model.algorithm_,
        dataset=dataset,
    )


    expected_plugin_files = {f"plugin_{name}.svg" for name in _EXPECTED_PLUGIN_PLOTS}
    actual_files = {p.name for p in out_dir.iterdir() if p.is_file()}
    missing = expected_plugin_files - actual_files
    assert not missing, (
        f"Missing plugin SVG files: {sorted(missing)}. "
        f"Actual files in {out_dir}: {sorted(actual_files)}"
    )



    figure_names = {p.name for p in report.figures}
    assert expected_plugin_files <= figure_names, (
        f"report.figures must include all 3 plugin plots. "
        f"Missing: {sorted(expected_plugin_files - figure_names)}. "
        f"Got figure names: {sorted(figure_names)}"
    )


    html_path = out_dir / "report.html"
    assert html_path.exists(), f"report.html missing at {html_path}"
    assert html_path.stat().st_size > 0, "report.html must not be empty"












    html_content = html_path.read_text()
    rendered_text = html.unescape(html_content)
    declared = model.algorithm_.list_plots()
    assert {info.name for info in declared} == _EXPECTED_PLUGIN_PLOTS

    for info in declared:
        assert info.title in rendered_text, (
            f"HTML report must reference plugin plot by its declared title "
            f"{info.title!r} (plot name {info.name!r}), but it was not found "
            f"in {html_path}."
        )
        assert info.description in rendered_text, (
            f"HTML report must carry the declared description for plot "
            f"{info.name!r}, but it was not found in {html_path}."
        )

        fallback_title = f"plugin_{info.name}".replace("_", " ").title()
        assert fallback_title not in rendered_text, (
            f"Declared title {info.title!r} must win over the filename-derived "
            f"fallback {fallback_title!r}, but the fallback is still rendered "
            f"in {html_path}."
        )
