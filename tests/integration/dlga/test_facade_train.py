
from __future__ import annotations

from pathlib import Path

import pytest

import kd
from kd.search.dlga import DLGAConfig


@pytest.fixture(scope="module")
def fitted_dlga_model() -> kd.Model:
    dataset = kd.generate_burgers_data(nx=16, nt=8, nu=0.1, seed=0)
    model = kd.Model(
        algorithm="dlga",
        generations=1,
        verbose=False,
        config=DLGAConfig(pop_size=4, seed=0, surrogate_max_epochs=5),
    )
    model.fit(dataset)
    return model


@pytest.mark.integration
def test_facade_dlga_trains_default_surrogate_without_crash() -> None:
    dataset = kd.generate_burgers_data(nx=16, nt=8, nu=0.1, seed=0)
    model = kd.Model(
        algorithm="dlga",
        generations=1,
        verbose=False,
        config=DLGAConfig(pop_size=4, seed=0, surrogate_max_epochs=5),
    )


    model.fit(dataset)

    assert model.result_ is not None


    assert model.best_expr_, "DLGA facade should recover a non-empty expression"


@pytest.mark.integration
def test_facade_dlga_records_surrogate_training_curve(
    fitted_dlga_model: kd.Model,
) -> None:
    model = fitted_dlga_model
    assert model.result_ is not None
    recorder = model.result_.recorder
    series = recorder.get("surrogate_train_loss")
    assert series, (
        "facade DLGA run must record a non-empty surrogate_train_loss series "
        "(the NN_1 training curve threaded through to result.recorder); got "
        f"{series!r}."
    )

    curve = series[-1]
    assert isinstance(curve, list) and curve, (
        f"surrogate_train_loss payload must be a non-empty list of per-epoch "
        f"losses; got {curve!r}."
    )


@pytest.mark.integration
def test_facade_dlga_engine_renders_surrogate_training_svg(
    fitted_dlga_model: kd.Model, tmp_path: Path
) -> None:
    model = fitted_dlga_model
    assert model.result_ is not None


    plugin = model.algorithm_
    assert plugin is not None

    out_dir = tmp_path / "dlga_surrogate_viz"
    engine = kd.VizEngine(output_dir=out_dir)
    engine.render_all(model.result_, algorithm=plugin)

    svg = out_dir / "plugin_surrogate_training.svg"
    assert svg.is_file(), (
        "VizEngine must render the 4th DLGA plugin plot to "
        "'plugin_surrogate_training.svg'. The engine swallows render exceptions "
        "into warnings, so a missing file is the only signal the plot failed. "
        f"Files present: {sorted(p.name for p in out_dir.iterdir())}."
    )
    assert svg.stat().st_size > 0, "the surrogate_training SVG must be non-empty"
