
from __future__ import annotations

import math
import sys
from typing import Any

import pytest
import torch

from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.search.pysr import PySRConfig, PySRPlugin





_N_X = 24
_N_T = 12


@pytest.fixture
def tiny_pde_dataset() -> PDEDataset:
    x = torch.linspace(0.0, 2.0 * math.pi, _N_X, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, _N_T, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(xg) * torch.exp(-tg)
    return PDEDataset(
        name="pysr-facade-test",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


class _RunnerSpy:

    last_max_iterations: int | None = None
    last_batch_size: int | None = None
    sentinel_result: Any = object()

    def __init__(
        self,
        algorithm: Any,
        max_iterations: int = 100,
        batch_size: int = 20,
        callbacks: Any = None,
    ) -> None:
        type(self).last_max_iterations = max_iterations
        type(self).last_batch_size = batch_size
        self._algorithm = algorithm

    def run(self, components: Any) -> Any:
        return type(self).sentinel_result







@pytest.mark.unit
def test_pysr_in_supported_algorithms() -> None:
    from kd.api import _SUPPORTED_ALGORITHMS

    assert "pysr" in _SUPPORTED_ALGORITHMS


@pytest.mark.unit
def test_pysr_model_construction_does_not_raise() -> None:
    from kd.api import Model

    m = Model(algorithm="pysr", verbose=False)
    assert m.algorithm == "pysr"


@pytest.mark.unit
def test_pysr_construction_does_not_boot_julia() -> None:
    from kd.api import Model

    pysr_already_loaded = "pysr" in sys.modules
    _ = Model(algorithm="pysr", verbose=False)
    if not pysr_already_loaded:
        assert "pysr" not in sys.modules, (
            "Model(algorithm='pysr') construction booted Julia / imported pysr"
        )







@pytest.mark.unit
def test_pysr_early_stop_mode_is_min() -> None:
    from kd.search.pysr.plugin import PySRPlugin

    assert PySRPlugin.score_direction == "min"









@pytest.mark.unit
def test_pysr_accepts_early_stop_min_mode() -> None:
    from kd.api import Model
    from kd.search.callbacks import EarlyStoppingCallback

    cb = EarlyStoppingCallback(mode="min", patience=5)
    m = Model(algorithm="pysr", verbose=False, callbacks=[cb])
    assert m.algorithm == "pysr"


@pytest.mark.unit
def test_pysr_rejects_early_stop_max_mode() -> None:
    from kd.api import Model
    from kd.search.callbacks import EarlyStoppingCallback

    cb = EarlyStoppingCallback(mode="max", patience=5)
    with pytest.raises(TypeError) as exc_info:
        Model(algorithm="pysr", verbose=False, callbacks=[cb])
    msg = str(exc_info.value)
    assert "pysr" in msg.lower(), f"error must mention 'pysr'; got: {msg!r}"







@pytest.mark.unit
def test_pysr_build_plugin_returns_pysr_plugin() -> None:
    from kd.api import Model

    m = Model(algorithm="pysr", verbose=False)
    plugin, batch_size = m._build_plugin()
    assert isinstance(plugin, PySRPlugin)
    assert isinstance(batch_size, int)
    assert batch_size > 0


@pytest.mark.unit
def test_pysr_build_plugin_does_not_boot_julia() -> None:
    from kd.api import Model

    pysr_already_loaded = "pysr" in sys.modules
    m = Model(algorithm="pysr", verbose=False)
    _ = m._build_plugin()
    if not pysr_already_loaded:
        assert "pysr" not in sys.modules


@pytest.mark.unit
def test_pysr_build_plugin_threads_user_config() -> None:
    from kd.api import Model

    cfg = PySRConfig(seed=123, niterations=9)
    m = Model(algorithm="pysr", verbose=False, config=cfg)
    plugin, _ = m._build_plugin()
    assert isinstance(plugin, PySRPlugin)
    assert plugin._config.seed == 123
    assert plugin._config.niterations == 9







@pytest.mark.unit
def test_build_pysr_config_default_maps_generations_to_niterations() -> None:
    from kd.api import Model

    m = Model(algorithm="pysr", generations=17, verbose=False)
    cfg = m._build_pysr_config()
    assert isinstance(cfg, PySRConfig)
    assert cfg.niterations == 17


@pytest.mark.unit
def test_build_pysr_config_default_threads_facade_seed() -> None:
    from kd.api import Model

    m = Model(algorithm="pysr", seed=42, verbose=False)
    cfg = m._build_pysr_config()
    assert cfg.seed == 42

    m_zero = Model(algorithm="pysr", seed=0, verbose=False)
    assert m_zero._build_pysr_config().seed == 0


@pytest.mark.unit
def test_build_pysr_config_uses_user_config_verbatim() -> None:
    from kd.api import Model

    cfg_in = PySRConfig(niterations=7)
    m = Model(algorithm="pysr", generations=50, verbose=False, config=cfg_in)
    cfg_out = m._build_pysr_config()
    assert isinstance(cfg_out, PySRConfig)
    assert cfg_out.niterations == 7


@pytest.mark.unit
def test_build_pysr_config_deep_copies_user_config() -> None:
    from kd.api import Model

    cfg_in = PySRConfig(niterations=7, seed=3)
    m = Model(algorithm="pysr", verbose=False, config=cfg_in)
    cfg_out = m._build_pysr_config()
    assert cfg_out is not cfg_in
    assert cfg_out == cfg_in


@pytest.mark.unit
def test_build_pysr_config_rejects_wrong_config_type() -> None:
    from kd.api import Model
    from kd.search.sga import SGAConfig

    m = Model(algorithm="pysr", verbose=False, config=SGAConfig())
    with pytest.raises(TypeError):
        m._build_pysr_config()







@pytest.mark.unit
@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("population", 50),
        ("depth", 6),
        ("width", 7),
        ("aic_ratio", 2.0),
        ("derivatives", "autograd"),
    ],
)
def test_pysr_rejects_sga_only_params(param: str, value: Any) -> None:
    from kd.api import Model

    with pytest.raises(TypeError) as exc_info:
        Model(algorithm="pysr", verbose=False, **{param: value})
    msg = str(exc_info.value)
    assert "pysr" in msg.lower(), f"error must mention 'pysr'; got: {msg!r}"


@pytest.mark.unit
def test_pysr_rejects_surrogate_model() -> None:
    from kd.api import Model

    surrogate = torch.nn.Linear(2, 1)
    with pytest.raises(TypeError) as exc_info:
        Model(algorithm="pysr", verbose=False, surrogate_model=surrogate)
    msg = str(exc_info.value)
    assert "surrogate_model" in msg, (
        f"error must name the offending 'surrogate_model' arg; got: {msg!r}"
    )


@pytest.mark.unit
def test_pysr_rejects_unknown_kwargs() -> None:
    from kd.api import Model

    with pytest.raises((TypeError, ValueError)):
        Model(algorithm="pysr", verbose=False, lam=0.1)


@pytest.mark.unit
def test_pysr_config_exclusivity_still_applies() -> None:
    from kd.api import Model

    with pytest.raises((TypeError, ValueError)):
        Model(
            algorithm="pysr",
            verbose=False,
            config=PySRConfig(),
            population=10,
        )







@pytest.mark.unit
def test_score_label_pysr_is_nmse() -> None:
    from kd.viz._labels import score_label

    assert score_label("pysr") == "NMSE"


@pytest.mark.unit
def test_api_score_label_pysr_delegates_to_viz() -> None:
    from typing import ClassVar

    from kd.api import _score_label as api_score_label
    from kd.search.pysr.plugin import PySRPlugin

    class _Stub:
        score_kind: ClassVar[str] = "NMSE"
        config = {"algorithm": "pysr"}

    assert api_score_label(_Stub()) == "NMSE"
    assert PySRPlugin.score_kind == "NMSE"
    assert api_score_label(_Stub()) == PySRPlugin.score_kind







@pytest.mark.unit
def test_pysr_fit_pins_max_iterations_to_one(
    tiny_pde_dataset: PDEDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kd.api as api

    _RunnerSpy.last_max_iterations = None
    monkeypatch.setattr(api, "ExperimentRunner", _RunnerSpy)

    m = api.Model(
        algorithm="pysr",
        generations=50,
        verbose=False,
        config=PySRConfig(niterations=5),
    )
    m.fit(tiny_pde_dataset)

    assert _RunnerSpy.last_max_iterations == 1, (
        "pysr fit must pin max_iterations=1 (one-shot plugin), got "
        f"{_RunnerSpy.last_max_iterations}"
    )


@pytest.mark.unit
def test_sga_fit_uses_generations_as_max_iterations(
    tiny_pde_dataset: PDEDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kd.api as api

    _RunnerSpy.last_max_iterations = None
    monkeypatch.setattr(api, "ExperimentRunner", _RunnerSpy)

    m = api.Model(
        algorithm="sga",
        generations=8,
        population=4,
        depth=3,
        width=3,
        seed=0,
        verbose=False,
    )
    m.fit(tiny_pde_dataset)

    assert _RunnerSpy.last_max_iterations == 8, (
        "SGA fit must keep max_iterations == generations, got "
        f"{_RunnerSpy.last_max_iterations}"
    )
