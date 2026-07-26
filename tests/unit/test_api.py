
from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytest
import torch

from kd.api import Model
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.data.synthetic import generate_burgers_data
from kd.search.iteration_events import IterationEvent, IterationEventEmitter
from kd.search.result import ExperimentResult
from kd.search.sga import SGAConfig






_NX = 32
_NT = 16
_NU = 0.1
_FAST_GENERATIONS = 3
_FAST_POPULATION = 5


@pytest.fixture
def small_burgers_dataset():
    return generate_burgers_data(nx=_NX, nt=_NT, nu=_NU, seed=0)


def _fast_model(**overrides) -> Model:
    kwargs = {
        "algorithm": "sga",
        "generations": _FAST_GENERATIONS,
        "population": _FAST_POPULATION,
        "depth": 3,
        "width": 3,
        "seed": 0,
        "verbose": False,
    }
    kwargs.update(overrides)
    return Model(**kwargs)







def test_model_init_defaults() -> None:
    m = Model()
    rep = repr(m)
    assert "fitted=False" in rep
    assert "algorithm='sga'" in rep
    assert "generations=50" in rep







def test_model_unfit_attribute_access() -> None:
    m = Model()
    with pytest.raises(RuntimeError) as exc_info:
        _ = m.best_expr_
    assert ".fit" in str(exc_info.value)

    with pytest.raises(RuntimeError):
        _ = m.best_score_

    with pytest.raises(RuntimeError):
        _ = m.result_

    with pytest.raises(RuntimeError):
        _ = m.algorithm_







def test_model_fit_returns_self(small_burgers_dataset) -> None:
    m = _fast_model()
    result = m.fit(small_burgers_dataset)
    assert result is m







def test_model_fit_burgers_smoke(small_burgers_dataset) -> None:
    m = _fast_model()
    m.fit(small_burgers_dataset)

    assert isinstance(m.best_expr_, str)
    assert m.best_expr_
    assert isinstance(m.best_score_, float)
    assert math.isfinite(m.best_score_)
    assert isinstance(m.result_, ExperimentResult)


def test_model_sga_fit_emits_iteration_events(small_burgers_dataset) -> None:
    events: list[IterationEvent] = []
    emitter = IterationEventEmitter(on_event=events.append)
    m = _fast_model(callbacks=[emitter])
    m.fit(small_burgers_dataset)

    assert len(events) >= 1
    assert events[-1].iteration == m.result_.iterations - 1







def test_model_repr_after_fit(small_burgers_dataset) -> None:
    m = _fast_model()
    m.fit(small_burgers_dataset)
    rep = repr(m)
    assert "fitted=True" in rep
    assert m.best_expr_ in rep



    assert "'sga'" in rep
    assert "SGAPlugin" not in rep







def test_model_verbose_silent(
    small_burgers_dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    m_silent = _fast_model(verbose=False)
    m_silent.fit(small_burgers_dataset)
    captured = capsys.readouterr()
    assert captured.out == ""

    m_loud = _fast_model(verbose=True)
    m_loud.fit(small_burgers_dataset)
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if "[kd]" in line]
    assert len(lines) >= 1







def test_model_unsupported_algorithm(small_burgers_dataset) -> None:
    with pytest.raises(NotImplementedError) as exc_info:
        Model(algorithm="gplearn", verbose=False)
    assert "gplearn" in str(exc_info.value)







@pytest.fixture
def dlga_pretrained_surrogate(small_burgers_dataset):
    from kd.models.field_model import FieldModel

    target_dtype = small_burgers_dataset.get_field("u").dtype
    model = FieldModel(
        coord_names=list(small_burgers_dataset.axis_order),
        field_names=["u"],
        hidden_sizes=[8],
        activation="tanh",
    ).to(dtype=target_dtype)
    return model


def test_model_dlga_init_does_not_raise() -> None:
    Model(algorithm="dlga", verbose=False)



def test_model_dlga_facade_seed_propagates_to_config() -> None:
    from kd.search.dlga import DLGAPlugin

    m = Model(algorithm="dlga", verbose=False, seed=42)
    plugin, _ = m._build_plugin()
    assert isinstance(plugin, DLGAPlugin)
    assert plugin._config.seed == 42


def test_model_dlga_fit_runs_through_facade(
    small_burgers_dataset, dlga_pretrained_surrogate
) -> None:
    from kd.search.dlga import DLGAConfig, DLGAPlugin

    cfg = DLGAConfig(pop_size=4, seed=0, epsilon=0.0)
    m = Model(
        algorithm="dlga",
        generations=2,
        verbose=False,
        config=cfg,


        surrogate_model=dlga_pretrained_surrogate,
    )
    m.fit(small_burgers_dataset)
    assert m.result_ is not None
    assert isinstance(m.algorithm_, DLGAPlugin)


def test_model_dlga_fit_emits_iteration_events(
    small_burgers_dataset, dlga_pretrained_surrogate
) -> None:
    from kd.search.dlga import DLGAConfig

    events: list[IterationEvent] = []
    emitter = IterationEventEmitter(on_event=events.append)
    cfg = DLGAConfig(pop_size=4, seed=0, epsilon=0.0)
    m = Model(
        algorithm="dlga",
        generations=2,
        verbose=False,
        config=cfg,
        surrogate_model=dlga_pretrained_surrogate,
        callbacks=[emitter],
    )
    m.fit(small_burgers_dataset)

    assert len(events) >= 1
    assert events[-1].iteration == m.result_.iterations - 1


def test_model_dlga_fit_records_preprocessing_and_surrogate_time(
    small_burgers_dataset: PDEDataset,
) -> None:
    from kd.search.dlga import DLGAConfig

    cfg = DLGAConfig(
        library=["u"],
        pop_size=2,
        seed=0,
        epsilon=0.0,
        lhs_auto_select=False,
        max_modules=1,
        max_module_length=1,
        surrogate_hidden_sizes=[4],
        surrogate_activation="tanh",
        surrogate_max_epochs=1,
        surrogate_patience=None,
        surrogate_val_ratio=0.0,
        surrogate_restore_best=False,
    )
    model = Model(
        algorithm="dlga",
        generations=1,
        verbose=False,
        config=cfg,
    )

    model.fit(small_burgers_dataset)

    record = model.result_.run_record
    assert record is not None
    cost = record.cost
    assert cost.preprocessing_seconds is not None
    assert cost.preprocessing_seconds >= 0.0
    assert cost.wallclock_seconds == (
        cost.search_seconds + cost.preprocessing_seconds
    )
    assert cost.surrogate_train_seconds is not None


def test_model_dlga_default_config(
    small_burgers_dataset, dlga_pretrained_surrogate
) -> None:
    m = Model(
        algorithm="dlga",
        generations=1,
        verbose=False,
        surrogate_model=dlga_pretrained_surrogate,
    )
    m.fit(small_burgers_dataset)
    assert m.result_ is not None


def test_model_dlga_repr_reflects_algorithm(dlga_pretrained_surrogate) -> None:
    m = Model(
        algorithm="dlga",
        verbose=False,
        surrogate_model=dlga_pretrained_surrogate,
    )
    rep = repr(m)
    assert "dlga" in rep.lower()


def test_model_dlga_repr_after_fit_reflects_algorithm(
    small_burgers_dataset, dlga_pretrained_surrogate
) -> None:
    m = Model(
        algorithm="dlga",
        generations=1,
        verbose=False,
        surrogate_model=dlga_pretrained_surrogate,
    )
    m.fit(small_burgers_dataset)
    rep = repr(m)
    assert "dlga" in rep.lower()


def test_model_sga_with_surrogate_model_raises() -> None:
    from kd.models.field_model import FieldModel

    cheap = FieldModel(
        coord_names=["x", "t"],
        field_names=["u"],
        hidden_sizes=[4],
        activation="tanh",
    )
    with pytest.raises(TypeError) as exc_info:
        Model(algorithm="sga", verbose=False, surrogate_model=cheap)
    msg = str(exc_info.value)
    assert "surrogate_model" in msg
    assert "dlga" in msg


def test_model_dlga_rejects_sga_only_facade_params() -> None:
    with pytest.raises(TypeError) as exc_info:
        Model(algorithm="dlga", verbose=False, population=99)
    msg = str(exc_info.value)
    assert "dlga" in msg.lower()
    assert "population" in msg
    assert "DLGAConfig" in msg


def test_model_dlga_rejects_sga_extra_kwargs() -> None:
    with pytest.raises(TypeError) as exc_info:
        Model(algorithm="dlga", verbose=False, lam=0.1)
    msg = str(exc_info.value)
    assert "dlga" in msg.lower()
    assert "lam" in msg


def test_model_sga_rejects_dlga_config() -> None:
    from kd.search.dlga import DLGAConfig

    m = Model(algorithm="sga", verbose=False, config=DLGAConfig())

    with pytest.raises(TypeError) as exc_info:
        m.fit(_make_tiny_dataset())
    msg = str(exc_info.value)
    assert "SGAConfig" in msg
    assert "DLGAConfig" in msg


def test_model_dlga_rejects_sga_config(dlga_pretrained_surrogate) -> None:
    m = Model(
        algorithm="dlga",
        verbose=False,
        config=SGAConfig(),
        surrogate_model=dlga_pretrained_surrogate,
    )
    with pytest.raises(TypeError) as exc_info:
        m.fit(_make_tiny_dataset())
    msg = str(exc_info.value)
    assert "DLGAConfig" in msg
    assert "SGAConfig" in msg


def _make_tiny_dataset() -> PDEDataset:
    nx, nt = 8, 6
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 0.1, nt, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(xg - tg)
    return PDEDataset(
        name="api-mismatch-test",
        task_type=TaskType.PDE,
        topology="grid",
        axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
        axis_order=["x", "t"],
        fields={"u": FieldData("u", u)},
        lhs_field="u",
        lhs_axis="t",
    )







def test_model_config_override(small_burgers_dataset) -> None:
    custom_pop = 7
    cfg = SGAConfig(num=custom_pop, depth=3, width=3, seed=0)
    m = Model(algorithm="sga", generations=2, verbose=False, config=cfg)
    m.fit(small_burgers_dataset)


    assert m.result_.config["num"] == custom_pop







def test_model_algorithm_attribute(small_burgers_dataset) -> None:
    from kd.search.protocol import SearchAlgorithm

    m = _fast_model()
    m.fit(small_burgers_dataset)
    assert isinstance(m.algorithm_, SearchAlgorithm)

    assert m.algorithm_.best_expression == m.best_expr_







def test_model_autograd_derivatives(small_burgers_dataset) -> None:
    m = Model(
        algorithm="sga",
        generations=2,
        population=3,
        depth=3,
        width=3,
        seed=0,
        verbose=False,
        derivatives="autograd",
        autograd_train_epochs=10,
    )
    m.fit(small_burgers_dataset)
    assert m.result_.config["use_autograd"] is True


def test_model_autograd_still_requires_uniform_grid() -> None:
    nx, nt = 32, 16

    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64) ** 2
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.randn(nx, nt, dtype=torch.float64)
    nonuniform_dataset = PDEDataset(
        name="nonuniform_for_autograd_test",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )

    m = Model(
        algorithm="sga",
        generations=2,
        population=3,
        depth=3,
        width=3,
        seed=0,
        verbose=False,
        derivatives="autograd",
        autograd_train_epochs=5,
    )
    with pytest.raises(ValueError, match="non-uniform|uniform"):
        m.fit(nonuniform_dataset)







def test_unknown_kwarg_raises_typeerror() -> None:
    with pytest.raises(TypeError) as exc_info:
        Model(populaton=10, verbose=False)
    msg = str(exc_info.value)
    assert "populaton" in msg

    assert "p_var" in msg or "p_mute" in msg or "lam" in msg







def test_collision_kwarg_raises_typeerror() -> None:
    with pytest.raises(TypeError) as exc_info:
        Model(population=20, num=99, verbose=False)
    msg = str(exc_info.value)
    assert "num" in msg
    assert "population" in msg







def test_config_override_with_facade_param_raises() -> None:
    cfg = SGAConfig(num=7, depth=3, width=3, seed=0)
    with pytest.raises(ValueError) as exc_info:
        Model(config=cfg, depth=8, verbose=False)
    msg = str(exc_info.value)
    assert "depth" in msg
    assert "config" in msg







def test_config_deepcopy_isolates_mutation(small_burgers_dataset) -> None:
    cfg = SGAConfig(num=5, depth=3, width=3, seed=0)
    m = Model(algorithm="sga", generations=2, verbose=False, config=cfg)

    cfg.num = 999
    m.fit(small_burgers_dataset)


    assert m.result_.config["num"] == 5







def _make_velocity_dataset() -> PDEDataset:
    nx, nt = 8, 6
    x = torch.linspace(0.0, 1.0, nx)
    t = torch.linspace(0.0, 1.0, nt)
    values = torch.zeros(nx, nt) + 0.1
    return PDEDataset(
        name="velocity_test",
        task_type=TaskType.PDE,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"velocity": FieldData(name="velocity", values=values)},

        lhs_field="",
        lhs_axis="t",
    )


def test_lhs_field_not_in_dataset_raises() -> None:
    dataset = _make_velocity_dataset()
    m = Model(
        algorithm="sga",
        generations=2,
        population=3,
        depth=3,
        width=3,
        seed=0,
        verbose=False,
    )
    with pytest.raises(ValueError) as exc_info:
        m.fit(dataset)
    msg = str(exc_info.value)
    assert "velocity" in msg
    assert "u" in msg







def test_invalid_derivatives_raises() -> None:
    with pytest.raises(ValueError) as exc_info:
        Model(derivatives="autogard", verbose=False)
    msg = str(exc_info.value)
    assert "autogard" in msg
    assert "finite_diff" in msg
    assert "autograd" in msg







def test_field_model_without_autograd_raises() -> None:

    class _DummyFieldModel:
        pass

    with pytest.raises(ValueError) as exc_info:
        Model(
            derivatives="finite_diff",
            field_model=_DummyFieldModel(),
            verbose=False,
        )
    msg = str(exc_info.value)
    assert "field_model" in msg
    assert "autograd" in msg







class _RecordingCallback:

    def __init__(self) -> None:
        self.start_calls = 0
        self.end_calls = 0
        self.iter_end_calls = 0

    @property
    def should_stop(self) -> bool:
        return False

    def on_experiment_start(self, algorithm: Any) -> None:
        self.start_calls += 1

    def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
        pass

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: Any,
        candidates: list[str],
        results: list[Any],
    ) -> None:
        self.iter_end_calls += 1

    def on_experiment_end(self, algorithm: Any) -> None:
        self.end_calls += 1


def test_callbacks_parameter_appended(small_burgers_dataset) -> None:
    cb = _RecordingCallback()
    m = _fast_model(callbacks=[cb])
    m.fit(small_burgers_dataset)
    assert cb.start_calls == 1
    assert cb.end_calls == 1
    assert cb.iter_end_calls >= 1







def test_failed_fit_resets_state(small_burgers_dataset) -> None:
    m = _fast_model()
    m.fit(small_burgers_dataset)
    assert isinstance(m.best_expr_, str)

    bad_dataset = _make_velocity_dataset()
    with pytest.raises(ValueError):
        m.fit(bad_dataset)



    with pytest.raises(RuntimeError):
        _ = m.best_expr_
    with pytest.raises(RuntimeError):
        _ = m.result_
    with pytest.raises(RuntimeError):
        _ = m.algorithm_







def test_repr_after_fit_uses_result_algorithm_name(small_burgers_dataset) -> None:
    m = _fast_model()
    m.fit(small_burgers_dataset)

    m.algorithm = "lying_value"
    rep = repr(m)



    assert "fitted=True" in rep
    assert "lying_value" not in rep
    expected_algo = m.result_.config.get("algorithm", m.result_.algorithm_name)
    assert f"'{expected_algo}'" in rep







def test_explicit_default_with_config_still_raises() -> None:
    cfg = SGAConfig(num=99, depth=3, width=3, seed=0)


    with pytest.raises(ValueError) as exc_info:
        Model(config=cfg, population=20, verbose=False)
    msg = str(exc_info.value)
    assert "population" in msg
    assert "config" in msg







def test_invalid_callback_raises_typeerror() -> None:
    with pytest.raises(TypeError) as exc_info:
        Model(callbacks=[42], verbose=False)
    msg = str(exc_info.value)
    assert "callbacks[0]" in msg
    assert "RunnerCallback" in msg














def _strip_lhs(dataset: PDEDataset, *, field: bool, axis: bool) -> PDEDataset:
    return dataclasses.replace(
        dataset,
        lhs_field="" if field else dataset.lhs_field,
        lhs_axis="" if axis else dataset.lhs_axis,
    )


def test_lhs_fallback_succeeds_when_both_empty(small_burgers_dataset) -> None:
    dataset = _strip_lhs(small_burgers_dataset, field=True, axis=True)
    assert dataset.lhs_field == ""
    assert dataset.lhs_axis == ""
    m = _fast_model()
    m.fit(dataset)
    assert m._fitted is True
    assert m.best_expr_


def test_lhs_fallback_succeeds_when_only_field_empty(small_burgers_dataset) -> None:
    dataset = _strip_lhs(small_burgers_dataset, field=True, axis=False)
    assert dataset.lhs_field == ""
    assert dataset.lhs_axis == "t"
    m = _fast_model()
    m.fit(dataset)
    assert m._fitted is True


def test_lhs_fallback_succeeds_when_only_axis_empty(small_burgers_dataset) -> None:
    dataset = _strip_lhs(small_burgers_dataset, field=False, axis=True)
    assert dataset.lhs_field == "u"
    assert dataset.lhs_axis == ""
    m = _fast_model()
    m.fit(dataset)
    assert m._fitted is True


def test_lhs_fallback_does_not_mutate_user_dataset(small_burgers_dataset) -> None:
    dataset = _strip_lhs(small_burgers_dataset, field=True, axis=True)
    m = _fast_model()
    m.fit(dataset)


    assert dataset.lhs_field == ""
    assert dataset.lhs_axis == ""


def test_lhs_fallback_writeback_reaches_plugin(small_burgers_dataset) -> None:
    dataset = _strip_lhs(small_burgers_dataset, field=True, axis=True)
    m = _fast_model()
    m.fit(dataset)
    plugin = m.algorithm_

    diff_ctx = plugin._diff_ctx
    assert diff_ctx is not None
    assert diff_ctx.lhs_axis == "t"














def _to_float32(dataset: PDEDataset) -> PDEDataset:
    assert dataset.axes is not None
    assert dataset.fields is not None
    new_axes = {
        name: dataclasses.replace(axis, values=axis.values.to(torch.float32))
        for name, axis in dataset.axes.items()
    }
    new_fields = {
        name: dataclasses.replace(field, values=field.values.to(torch.float32))
        for name, field in dataset.fields.items()
    }
    return dataclasses.replace(dataset, axes=new_axes, fields=new_fields)


def test_model_fit_with_float32_dataset(small_burgers_dataset) -> None:
    dataset = _to_float32(small_burgers_dataset)

    assert dataset.axes is not None
    for axis in dataset.axes.values():
        assert axis.values.dtype == torch.float32
    assert dataset.fields is not None
    for field in dataset.fields.values():
        assert field.values.dtype == torch.float32

    m = _fast_model()
    m.fit(dataset)
    assert m._fitted is True
    assert m.best_expr_







def test_vizengine_is_top_level_exported() -> None:
    import kd
    from kd.viz.engine import VizEngine

    assert kd.VizEngine is VizEngine


def test_top_level_byod_exports_present() -> None:
    import kd

    expected = (
        "Model",
        "PDEDataset",
        "AxisInfo",
        "FieldData",
        "TaskType",
        "DataTopology",
        "VizEngine",
        "preview",
        "load_chafee_infante",
        "load_kdv",
        "load_pde_compound",
        "load_pde_divide",
        "generate_advection_data",
        "generate_burgers_data",
        "generate_diffusion_data",
        "ExperimentResult",
        "SGAConfig",

        "law_signature",
        "verify_equation",
        "VerifyPolicy",
    )
    for symbol in expected:
        assert hasattr(kd, symbol), f"kd.{symbol} is missing"
        assert symbol in kd.__all__, f"kd.__all__ missing '{symbol}'"







def _make_tiny_order2_dataset() -> PDEDataset:
    nx, nt = 6, 5
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.randn(nx, nt, dtype=torch.float64)
    return PDEDataset.from_arrays(coords={"x": x, "t": t}, fields={"u": u}, lhs="u_tt")


@pytest.mark.parametrize("algorithm", ["sga", "dlga", "discover", "pysr"])
def test_facade_rejects_second_order_lhs_fail_loud(algorithm: str) -> None:
    ds = _make_tiny_order2_dataset()
    assert ds.lhs_order == 2
    m = Model(algorithm=algorithm, generations=1, verbose=False)
    with pytest.raises(NotImplementedError, match="lhs_order"):
        m.fit(ds)


def test_facade_accepts_first_order_lhs_passes_gate(small_burgers_dataset) -> None:
    assert small_burgers_dataset.lhs_order == 1
    m = Model(algorithm="sga", generations=2, population=4, verbose=False)
    m.fit(small_burgers_dataset)
    assert m.best_expr_ is not None
