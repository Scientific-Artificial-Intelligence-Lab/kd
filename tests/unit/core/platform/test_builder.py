
from __future__ import annotations

import math
import re
from typing import Any

import pytest
import torch
import torch.nn as nn

from kd.core.evaluator import Evaluator
from kd.core.executor.context import ExecutionContext
from kd.core.expr.executor import PythonExecutor
from kd.core.expr.registry import FunctionRegistry


from kd.core.platform.builder import (
    PlatformBuilder,
    _resolve_derivative_requirements,
)
from kd.core.platform.requirements import DerivativeReqs
from kd.data.derivatives.autograd import AutogradProvider
from kd.data.derivatives.finite_diff import FiniteDiffProvider
from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.search.dlga.surrogate import DLGASurrogateContext
from kd.search.protocol import PlatformComponents






def _make_dataset(
    *,
    field_name: str = "u",
    axis_field_name: str = "t",
    extra_fields: dict[str, FieldData] | None = None,
    extra_axes: dict[str, AxisInfo] | None = None,
    lhs_field: str = "",
    lhs_axis: str = "",
    n_x: int = 16,
    n_t: int = 8,
) -> PDEDataset:
    x = torch.linspace(0.0, 1.0, n_x, dtype=torch.float64)
    t = torch.linspace(0.0, 0.1, n_t, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(2.0 * math.pi * (xg - tg))

    axes: dict[str, AxisInfo] = {
        "x": AxisInfo(name="x", values=x),
        "t": AxisInfo(name="t", values=t),
    }
    if extra_axes is not None:
        axes.update(extra_axes)

    fields: dict[str, FieldData] = {field_name: FieldData(name=field_name, values=u)}
    if extra_fields is not None:
        fields.update(extra_fields)


    axis_order = ["x", "t"] + [name for name in axes if name not in ("x", "t")]
    del axis_field_name

    return PDEDataset(
        name="builder-test",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes=axes,
        axis_order=axis_order,
        fields=fields,
        lhs_field=lhs_field,
        lhs_axis=lhs_axis,
    )


@pytest.fixture
def small_dataset() -> PDEDataset:
    return _make_dataset()


@pytest.fixture
def small_dataset_with_lhs() -> PDEDataset:
    return _make_dataset(lhs_field="u", lhs_axis="t")


@pytest.fixture
def pretrained_model(small_dataset_with_lhs: PDEDataset) -> nn.Module:

    class _ConstantField(nn.Module):

        def __init__(self) -> None:
            super().__init__()

            self.linear = nn.Linear(2, 1)


            self.linear = self.linear.to(dtype=torch.float64)

        def forward(self, *, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            inp = torch.stack([x, t], dim=-1)
            return self.linear(inp).squeeze(-1)

    return _ConstantField()







def _is_autograd_context(ctx: ExecutionContext) -> bool:
    return isinstance(ctx.derivative_provider, AutogradProvider)


def _is_finite_diff_context(ctx: ExecutionContext) -> bool:
    return isinstance(ctx.derivative_provider, FiniteDiffProvider)







class TestDefaultReqsEquivalence:

    @pytest.mark.unit
    def test_build_returns_platform_components(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        builder = PlatformBuilder(small_dataset_with_lhs, DerivativeReqs())
        components = builder.build()
        assert isinstance(components, PlatformComponents)

    @pytest.mark.unit
    def test_required_components_are_not_none(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        components = PlatformBuilder(small_dataset_with_lhs, DerivativeReqs()).build()
        assert components.dataset is not None
        assert components.executor is not None
        assert components.evaluator is not None
        assert components.context is not None
        assert components.registry is not None

    @pytest.mark.unit
    def test_default_provider_is_finite_diff(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        components = PlatformBuilder(small_dataset_with_lhs, DerivativeReqs()).build()
        assert _is_finite_diff_context(components.context)

    @pytest.mark.unit
    def test_default_provider_max_order_is_two(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        components = PlatformBuilder(small_dataset_with_lhs, DerivativeReqs()).build()
        provider = components.context.derivative_provider


        max_order = getattr(provider, "max_order", None)
        if max_order is None:
            max_order = getattr(provider, "_max_order", None)
        assert max_order == 2, (
            f"FD provider max_order should be 2 (matches reqs.max_atomic_order), "
            f"got {max_order}"
        )

    @pytest.mark.unit
    def test_default_lhs_shape_is_dataset_flat(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        components = PlatformBuilder(small_dataset_with_lhs, DerivativeReqs()).build()
        n_total = small_dataset_with_lhs.fields["u"].values.numel()
        assert components.evaluator.lhs.shape == (n_total,)

    @pytest.mark.unit
    def test_default_context_is_plain_execution_context(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        components = PlatformBuilder(small_dataset_with_lhs, DerivativeReqs()).build()
        assert isinstance(components.context, ExecutionContext)

        assert not isinstance(components.context, DLGASurrogateContext), (
            "Default (needs_surrogate=False) must NOT wrap the context with "
            "DLGASurrogateContext — SGA Layer 1 raw u lock"
        )

    @pytest.mark.unit
    def test_default_evaluator_and_executor_types(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        components = PlatformBuilder(small_dataset_with_lhs, DerivativeReqs()).build()
        assert isinstance(components.evaluator, Evaluator)
        assert isinstance(components.executor, PythonExecutor)
        assert isinstance(components.registry, FunctionRegistry)

    @pytest.mark.unit
    def test_non_default_max_atomic_order_reaches_provider(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        reqs = DerivativeReqs(max_atomic_order=3)
        components = PlatformBuilder(small_dataset_with_lhs, reqs).build()
        provider = components.context.derivative_provider
        max_order = getattr(provider, "max_order", None)
        if max_order is None:
            max_order = getattr(provider, "_max_order", None)
        assert max_order == 3, (
            f"reqs.max_atomic_order=3 must reach FD provider (got {max_order}); "
            "hard-coded max_order=2 would silently break DLGA u_xxx path."
        )







class TestAutogradWithoutSurrogate:

    @pytest.mark.unit
    def test_build_does_not_raise(self, small_dataset_with_lhs: PDEDataset) -> None:
        reqs = DerivativeReqs(provider_kind="autograd", needs_surrogate=False)

        PlatformBuilder(small_dataset_with_lhs, reqs).build()

    @pytest.mark.unit
    def test_provider_is_autograd(self, small_dataset_with_lhs: PDEDataset) -> None:
        reqs = DerivativeReqs(provider_kind="autograd", needs_surrogate=False)
        components = PlatformBuilder(small_dataset_with_lhs, reqs).build()
        assert _is_autograd_context(components.context)

    @pytest.mark.unit
    def test_context_is_not_surrogate_context(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        reqs = DerivativeReqs(provider_kind="autograd", needs_surrogate=False)
        components = PlatformBuilder(small_dataset_with_lhs, reqs).build()
        assert isinstance(components.context, ExecutionContext)
        assert not isinstance(components.context, DLGASurrogateContext), (
            "needs_surrogate=False with autograd must NOT wrap context — "
            " SGA Layer 1 raw u invariant"
        )

    @pytest.mark.unit
    def test_sga_prepare_integration_default_reqs(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        from kd.search.sga.config import SGAConfig
        from kd.search.sga.plugin import SGAPlugin

        components = PlatformBuilder(small_dataset_with_lhs, DerivativeReqs()).build()

        plugin = SGAPlugin(SGAConfig(seed=0, num=4))

        plugin.prepare(components)










class TestAutogradWithProvidedSurrogate:

    @pytest.mark.unit
    def test_build_does_not_raise(
        self,
        small_dataset_with_lhs: PDEDataset,
        pretrained_model: nn.Module,
    ) -> None:
        reqs = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_model=pretrained_model,
        )
        PlatformBuilder(small_dataset_with_lhs, reqs).build()

    @pytest.mark.unit
    def test_context_is_surrogate_context(
        self,
        small_dataset_with_lhs: PDEDataset,
        pretrained_model: nn.Module,
    ) -> None:
        reqs = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_model=pretrained_model,
        )
        components = PlatformBuilder(small_dataset_with_lhs, reqs).build()
        assert isinstance(components.context, DLGASurrogateContext)

    @pytest.mark.unit
    def test_provider_uses_provided_model(
        self,
        small_dataset_with_lhs: PDEDataset,
        pretrained_model: nn.Module,
    ) -> None:
        reqs = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_model=pretrained_model,
        )
        components = PlatformBuilder(small_dataset_with_lhs, reqs).build()
        provider = components.context.derivative_provider

        assert provider.model is pretrained_model, (
            "Builder must pass the provided surrogate_model THROUGH to the provider "
            "instead of constructing a new instance — `is` identity required."
        )

    @pytest.mark.unit
    def test_get_variable_returns_nn_forward_not_raw_field(
        self,
        small_dataset_with_lhs: PDEDataset,
        pretrained_model: nn.Module,
    ) -> None:
        reqs = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_model=pretrained_model,
        )
        components = PlatformBuilder(small_dataset_with_lhs, reqs).build()
        ctx_value = components.context.get_variable(small_dataset_with_lhs.lhs_field)
        raw = small_dataset_with_lhs.fields[
            small_dataset_with_lhs.lhs_field
        ].values.flatten()

        assert ctx_value.shape == raw.shape
        assert not torch.allclose(ctx_value, raw, atol=1e-6), (
            "SurrogateContext.get_variable(lhs_field) must return the NN forward "
            "output, which (for our random-weight Linear) cannot equal raw u."
        )







class TestAutogradTrainsDefaultModel:

    @pytest.mark.unit
    def test_build_does_not_raise_when_model_is_none(
        self,
        small_dataset_with_lhs: PDEDataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}
        _install_fake_trainer(monkeypatch, captured)
        reqs = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_model=None,
            surrogate_train_kwargs={
                "max_epochs": 5,
                "patience": None,
                "val_ratio": 0.0,
            },
        )
        PlatformBuilder(small_dataset_with_lhs, reqs).build()

    @pytest.mark.unit
    def test_context_is_surrogate_context(
        self,
        small_dataset_with_lhs: PDEDataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}
        _install_fake_trainer(monkeypatch, captured)
        reqs = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_model=None,
            surrogate_train_kwargs={
                "max_epochs": 5,
                "patience": None,
                "val_ratio": 0.0,
            },
        )
        components = PlatformBuilder(small_dataset_with_lhs, reqs).build()
        assert isinstance(components.context, DLGASurrogateContext)

    @pytest.mark.unit
    def test_provider_holds_internally_built_model(
        self,
        small_dataset_with_lhs: PDEDataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}
        _install_fake_trainer(monkeypatch, captured)
        reqs = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_model=None,
            surrogate_train_kwargs={
                "max_epochs": 5,
                "patience": None,
                "val_ratio": 0.0,
            },
        )
        components = PlatformBuilder(small_dataset_with_lhs, reqs).build()
        provider_model = components.context.derivative_provider.model
        assert provider_model is not None
        assert isinstance(provider_model, nn.Module)

    @pytest.mark.unit
    def test_train_kwargs_are_forwarded(
        self,
        small_dataset_with_lhs: PDEDataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}
        _install_fake_trainer(monkeypatch, captured)
        user_kwargs = {
            "max_epochs": 7,
            "patience": None,
            "val_ratio": 0.0,
        }
        reqs = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_model=None,
            surrogate_train_kwargs=user_kwargs,
        )
        PlatformBuilder(small_dataset_with_lhs, reqs).build()
        assert "fit_kwargs" in captured, (
            "Fake trainer should record the kwargs passed to fit() — "
            "the builder either skipped training or routed it elsewhere."
        )
        for key, value in user_kwargs.items():
            assert captured["fit_kwargs"].get(key) == value, (
                f"surrogate_train_kwargs[{key}]={value!r} not forwarded to "
                f"FieldModelTrainer.fit (got {captured['fit_kwargs'].get(key)!r})"
            )

    @pytest.mark.unit
    def test_none_train_kwargs_actually_trains(
        self,
        small_dataset_with_lhs: PDEDataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}
        _install_fake_trainer(monkeypatch, captured)
        reqs = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_model=None,
            surrogate_train_kwargs=None,
        )
        components = PlatformBuilder(small_dataset_with_lhs, reqs).build()

        assert "model" in captured, (
            "When surrogate_train_kwargs=None and surrogate_model=None, builder "
            "MUST construct a default FieldModelTrainer (skipping training and "
            "returning an untrained NN is a silent regression — provider would "
            "deliver ~random values to evaluator)."
        )
        assert "fit_kwargs" in captured, (
            "Builder MUST call trainer.fit() even with surrogate_train_kwargs=None "
            "(builder supplies its own defaults). Skipping fit means the model is "
            "untrained — silent failure mode."
        )

        provider_model = components.context.derivative_provider.model
        assert provider_model is captured["model"], (
            "Builder must hand the trained model identity to AutogradProvider — "
            "constructing a SECOND nn.Module after training is a silent regression."
        )

    @pytest.mark.unit
    def test_train_kwargs_pass_through_not_nested(
        self,
        small_dataset_with_lhs: PDEDataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}
        _install_fake_trainer(monkeypatch, captured)
        user_kwargs = {
            "max_epochs": 7,
            "patience": None,
            "val_ratio": 0.0,
        }
        reqs = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_model=None,
            surrogate_train_kwargs=user_kwargs,
        )
        PlatformBuilder(small_dataset_with_lhs, reqs).build()

        for k in user_kwargs:
            assert k in captured["fit_kwargs"], (
                f"User key {k!r} must be a top-level fit_kwargs key, not nested in "
                f"a sub-dict. Got fit_kwargs keys: {list(captured['fit_kwargs'])}"
            )

        suspicious_wrappers = {"options", "config", "params", "kwargs", "user_kwargs"}
        wrapper_hits = suspicious_wrappers & set(captured["fit_kwargs"])
        assert not wrapper_hits, (
            f"fit_kwargs contains suspicious wrapper key(s) {wrapper_hits} — "
            f"user kwargs must be passed at top level, not nested. "
            f"Full fit_kwargs: {captured['fit_kwargs']}"
        )

    @pytest.mark.unit
    def test_evaluator_lhs_uses_autograd_provider(
        self,
        small_dataset_with_lhs: PDEDataset,
        pretrained_model: nn.Module,
    ) -> None:
        reqs = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_model=pretrained_model,
        )
        components = PlatformBuilder(small_dataset_with_lhs, reqs).build()

        provider = components.context.derivative_provider
        assert _is_autograd_context(components.context), (
            "Surrogate path must wire AutogradProvider; FD here would silently "
            "skip surrogate and provide raw u derivatives — DLGA recovery breaks."
        )

        expected = provider.get_derivative(
            components.dataset.lhs_field,
            components.dataset.lhs_axis,
            reqs.lhs_order,
        ).flatten()
        torch.testing.assert_close(
            components.evaluator.lhs, expected, rtol=1e-5, atol=1e-7
        )








class TestContextCarriesTrainingResult:

    @pytest.mark.unit
    def test_default_train_attaches_populated_training_result(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        from kd.models.trainer import TrainingResult

        reqs = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_model=None,
            surrogate_train_kwargs={
                "max_epochs": 5,
                "patience": None,
                "val_ratio": 0.0,
                "seed": 0,
            },
        )
        components = PlatformBuilder(small_dataset_with_lhs, reqs).build()

        training_result = getattr(components.context, "training_result", "MISSING")
        assert isinstance(training_result, TrainingResult), (
            "build() with needs_surrogate=True and no pre-trained model must "
            "attach the trainer's TrainingResult to context.training_result; "
            f"got {training_result!r}."
        )
        assert training_result.loss_history, (
            "context.training_result.loss_history must be non-empty after a "
            "default surrogate train (the curve the viz layer plots)."
        )
        assert len(training_result.loss_history) == training_result.epochs_run

    @pytest.mark.unit
    def test_provided_model_leaves_training_result_none(
        self,
        small_dataset_with_lhs: PDEDataset,
        pretrained_model: nn.Module,
    ) -> None:
        reqs = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_model=pretrained_model,
        )
        components = PlatformBuilder(small_dataset_with_lhs, reqs).build()
        training_result = getattr(components.context, "training_result", "MISSING")
        assert training_result is None, (
            "A provided pre-trained surrogate must leave "
            "context.training_result == None (nothing was trained); got "
            f"{training_result!r}."
        )

    @pytest.mark.unit
    def test_val_split_populates_val_loss_history_on_context(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        reqs = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_model=None,
            surrogate_train_kwargs={
                "max_epochs": 5,
                "patience": None,
                "val_ratio": 0.2,
                "seed": 0,
            },
        )
        components = PlatformBuilder(small_dataset_with_lhs, reqs).build()
        training_result = getattr(components.context, "training_result", "MISSING")
        assert training_result is not None and training_result != "MISSING"
        assert training_result.val_loss_history is not None, (
            "val_ratio>0 must thread a non-None val_loss_history through to the "
            "context.training_result."
        )
        assert len(training_result.val_loss_history) == len(
            training_result.loss_history
        )

    @pytest.mark.unit
    def test_sga_optin_plain_context_training_result_none(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        reqs = DerivativeReqs(provider_kind="autograd", needs_surrogate=False)
        components = PlatformBuilder(small_dataset_with_lhs, reqs).build()

        assert not isinstance(components.context, DLGASurrogateContext)
        training_result = getattr(components.context, "training_result", None)
        assert training_result is None, (
            "SGA opt-in (needs_surrogate=False) must leave no surrogate "
            f"training_result on the plain context; got {training_result!r}."
        )







class TestLHSResolveDefaults:

    @pytest.mark.unit
    def test_unset_lhs_defaults_to_u_t(self, small_dataset: PDEDataset) -> None:

        assert small_dataset.lhs_field == ""
        assert small_dataset.lhs_axis == ""
        components = PlatformBuilder(small_dataset, DerivativeReqs()).build()


        assert components.dataset is not None

    @pytest.mark.unit
    def test_explicit_lhs_field_not_overridden(self) -> None:
        ds = _make_dataset(
            field_name="v",
            extra_fields=None,
            lhs_field="v",
            lhs_axis="t",
        )
        components = PlatformBuilder(ds, DerivativeReqs()).build()
        assert components.dataset.lhs_field == "v", (
            "Explicit dataset.lhs_field must be preserved through the builder; "
            "the 'u' default applies only when lhs_field is empty."
        )
        assert components.dataset.lhs_axis == "t"







class TestLHSValidate:

    @pytest.mark.unit
    def test_missing_lhs_field_raises_with_field_list(self) -> None:
        ds = _make_dataset(
            field_name="u",
            extra_fields={"v": FieldData(name="v", values=torch.zeros(16, 8))},
        )




        ds.__dict__["lhs_field"] = "missing"

        with pytest.raises(ValueError) as exc_info:
            PlatformBuilder(ds, DerivativeReqs()).build()
        msg = str(exc_info.value)
        assert "missing" in msg, (
            f"Error must name the offending field ('missing'), got: {msg!r}"
        )
        assert "u" in msg or "v" in msg, (
            f"Error must list at least one available field (u/v), got: {msg!r}"
        )

    @pytest.mark.unit
    def test_missing_lhs_axis_raises_with_axis_list(self) -> None:
        ds = _make_dataset()
        ds.__dict__["lhs_axis"] = "missing"
        with pytest.raises(ValueError) as exc_info:
            PlatformBuilder(ds, DerivativeReqs()).build()
        msg = str(exc_info.value)
        assert "missing" in msg, (
            f"Error must name the offending axis ('missing'), got: {msg!r}"
        )

        assert "x" in msg or "t" in msg, (
            f"Error must list at least one available axis (x/t), got: {msg!r}"
        )

    @pytest.mark.unit
    def test_default_field_with_invalid_axis_raises_axis_error(self) -> None:
        ds = _make_dataset()

        ds.__dict__["lhs_axis"] = "missing"
        with pytest.raises(ValueError) as exc_info:
            PlatformBuilder(ds, DerivativeReqs()).build()
        msg = str(exc_info.value)

        assert "missing" in msg

        assert "x" in msg or "t" in msg

    @pytest.mark.unit
    def test_dataset_without_default_field_u_raises(self) -> None:
        ds = _make_dataset(field_name="rho")

        with pytest.raises(ValueError):
            PlatformBuilder(ds, DerivativeReqs()).build()







class TestLHSWriteback:

    @pytest.mark.unit
    def test_writeback_fills_default_lhs_field(self, small_dataset: PDEDataset) -> None:
        assert small_dataset.lhs_field == ""
        components = PlatformBuilder(small_dataset, DerivativeReqs()).build()
        assert components.dataset.lhs_field == "u", (
            "Builder must write the resolved default lhs_field back to "
            "components.dataset; SGAPlugin.prepare reads this directly."
        )

    @pytest.mark.unit
    def test_writeback_fills_default_lhs_axis(self, small_dataset: PDEDataset) -> None:
        assert small_dataset.lhs_axis == ""
        components = PlatformBuilder(small_dataset, DerivativeReqs()).build()
        assert components.dataset.lhs_axis == "t"

    @pytest.mark.unit
    def test_writeback_writes_to_a_real_field(self, small_dataset: PDEDataset) -> None:
        components = PlatformBuilder(small_dataset, DerivativeReqs()).build()
        ds = components.dataset
        assert ds.fields is not None
        assert ds.lhs_field in ds.fields, (
            f"components.dataset.lhs_field={ds.lhs_field!r} must reference an "
            f"actual field in components.dataset.fields={list(ds.fields)}"
        )

    @pytest.mark.unit
    def test_writeback_does_not_mutate_input_dataset(
        self, small_dataset: PDEDataset
    ) -> None:

        original_lhs_field = small_dataset.lhs_field
        original_lhs_axis = small_dataset.lhs_axis
        components = PlatformBuilder(small_dataset, DerivativeReqs()).build()

        assert small_dataset.lhs_field == original_lhs_field, (
            "Builder must not mutate the input dataset's lhs_field (use "
            "dataclasses.replace to produce a snapshot for downstream consumers)."
        )
        assert small_dataset.lhs_axis == original_lhs_axis

        assert components.dataset is not small_dataset, (
            "components.dataset must be a fresh snapshot via dataclasses.replace, "
            "not the same instance as the input."
        )

    @pytest.mark.unit
    def test_writeback_preserves_explicit_non_default_lhs_axis(self) -> None:

        n_x, n_tau = 8, 6
        x = torch.linspace(0.0, 1.0, n_x, dtype=torch.float64)
        tau = torch.linspace(0.0, 0.5, n_tau, dtype=torch.float64)
        xg, taug = torch.meshgrid(x, tau, indexing="ij")
        u = torch.sin(2.0 * math.pi * (xg - taug))
        ds = PDEDataset(
            name="builder-test",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x),
                "tau": AxisInfo(name="tau", values=tau),
            },
            axis_order=["x", "tau"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="tau",
        )
        components = PlatformBuilder(ds, DerivativeReqs()).build()
        assert components.dataset.lhs_axis == "tau", (
            "Explicit lhs_axis='tau' must be preserved through builder; "
            "default-fallback to 't' would be a regression."
        )

    @pytest.mark.unit
    def test_writeback_lhs_field_is_non_empty_string(
        self, small_dataset: PDEDataset
    ) -> None:
        components = PlatformBuilder(small_dataset, DerivativeReqs()).build()
        assert isinstance(components.dataset.lhs_field, str)
        assert components.dataset.lhs_field != "", (
            "Critical: post-build lhs_field must NEVER be empty — "
            "SGAPlugin.prepare reads this directly and ValueError-rejects "
            "empty strings."
        )
        assert isinstance(components.dataset.lhs_axis, str)
        assert components.dataset.lhs_axis != ""







class TestResolveDerivativeRequirements:

    @pytest.mark.unit
    def test_no_attribute_returns_default(self) -> None:

        class _PluginNoAttr:
            pass

        result = _resolve_derivative_requirements(_PluginNoAttr())
        assert result == DerivativeReqs()

    @pytest.mark.unit
    def test_property_returning_valid_reqs(self) -> None:

        custom = DerivativeReqs(
            provider_kind="autograd",
            max_atomic_order=3,
            needs_surrogate=False,
        )

        class _PluginWithProperty:
            @property
            def derivative_requirements(self) -> DerivativeReqs:
                return custom

        result = _resolve_derivative_requirements(_PluginWithProperty())
        assert result is custom or result == custom, (
            "Helper should return the property value verbatim (identity or equality)."
        )

    @pytest.mark.unit
    def test_method_not_property_raises_typeerror(self) -> None:

        class _PluginWithMethod:
            def derivative_requirements(self) -> DerivativeReqs:
                return DerivativeReqs()

        with pytest.raises(TypeError) as exc_info:
            _resolve_derivative_requirements(_PluginWithMethod())
        msg = str(exc_info.value)

        assert "_PluginWithMethod" in msg, (
            f"TypeError must name the plugin class (got: {msg!r})"
        )

        assert re.search(r"property|method|callable", msg, re.IGNORECASE), (
            f"TypeError must mention property/method/callable distinction "
            f"(got: {msg!r})"
        )

    @pytest.mark.unit
    def test_wrong_type_dict_raises_typeerror(self) -> None:

        class _PluginReturnsDict:
            @property
            def derivative_requirements(self) -> dict:
                return {"provider_kind": "autograd"}

        with pytest.raises(TypeError) as exc_info:
            _resolve_derivative_requirements(_PluginReturnsDict())
        msg = str(exc_info.value)
        assert "DerivativeReqs" in msg, (
            f"TypeError must mention the expected type 'DerivativeReqs' (got: {msg!r})"
        )
        assert "dict" in msg, (
            f"TypeError must mention the wrong actual type 'dict' (got: {msg!r})"
        )

    @pytest.mark.unit
    def test_subclass_instance_accepted(self) -> None:

        class _ExtendedReqs(DerivativeReqs):
            pass

        custom = _ExtendedReqs()

        class _PluginSubclass:
            @property
            def derivative_requirements(self) -> DerivativeReqs:
                return custom

        result = _resolve_derivative_requirements(_PluginSubclass())
        assert result is custom or isinstance(result, _ExtendedReqs)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("plugin_kind", "expect_default"),
        [
            ("no_attr", True),
            ("property_default", False),
            ("property_custom", False),
        ],
    )
    def test_truth_table_smoke(self, plugin_kind: str, expect_default: bool) -> None:
        if plugin_kind == "no_attr":

            class _P:
                pass

            plugin: Any = _P()
        elif plugin_kind == "property_default":

            class _P2:
                @property
                def derivative_requirements(self) -> DerivativeReqs:
                    return DerivativeReqs()

            plugin = _P2()
        elif plugin_kind == "property_custom":

            class _P3:
                @property
                def derivative_requirements(self) -> DerivativeReqs:
                    return DerivativeReqs(provider_kind="autograd")

            plugin = _P3()
        else:
            raise AssertionError(f"unknown plugin_kind {plugin_kind}")

        result = _resolve_derivative_requirements(plugin)
        assert isinstance(result, DerivativeReqs)
        if expect_default:
            assert result == DerivativeReqs()







def _install_fake_trainer(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
) -> None:

    class _FakeFitResult:
        def __init__(self) -> None:
            self.final_loss = 0.0
            self.epochs_run = 0
            self.early_stopped = False
            self.val_loss = None
            self.best_val_loss = None
            self.best_epoch = None
            self.best_restored = False

    class _FakeTrainer:
        def __init__(self, model: nn.Module, *args: Any, **kwargs: Any) -> None:
            captured["init_args"] = args
            captured["init_kwargs"] = kwargs
            captured["model"] = model
            self._model = model

        def fit(self, *args: Any, **kwargs: Any) -> _FakeFitResult:
            captured["fit_args"] = args
            captured["fit_kwargs"] = kwargs
            return _FakeFitResult()


    monkeypatch.setattr(
        "kd.core.platform.builder.FieldModelTrainer",
        _FakeTrainer,
        raising=True,
    )










skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device available"
)


class TestDeviceKnob:
    def test_device_none_fd_context_on_cpu(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        comps = PlatformBuilder(
            small_dataset_with_lhs, DerivativeReqs(), device=None
        ).build()
        assert comps.context is not None
        assert comps.context.device == torch.device("cpu")

    def test_device_cpu_fd_context_and_lhs_on_cpu(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        comps = PlatformBuilder(
            small_dataset_with_lhs, DerivativeReqs(), device="cpu"
        ).build()
        assert comps.context is not None and comps.evaluator is not None
        assert comps.context.device == torch.device("cpu")

        assert comps.evaluator.lhs.device == comps.context.device

    def test_device_none_autograd_coords_on_cpu(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        reqs = DerivativeReqs(provider_kind="autograd", needs_surrogate=False)
        comps = PlatformBuilder(
            small_dataset_with_lhs, reqs, device=None
        ).build()
        provider = comps.context.derivative_provider
        coord = next(iter(provider.coords.values()))
        assert coord.device == torch.device("cpu")

    @skip_no_cuda
    def test_device_cuda_fd_context_and_lhs_on_cuda(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        comps = PlatformBuilder(
            small_dataset_with_lhs, DerivativeReqs(), device="cuda"
        ).build()
        assert comps.context is not None and comps.evaluator is not None
        assert comps.context.device.type == "cuda"
        assert comps.evaluator.lhs.device == comps.context.device

    @skip_no_cuda
    def test_device_cuda_autograd_coords_are_leaf_on_cuda(
        self, small_dataset_with_lhs: PDEDataset
    ) -> None:
        reqs = DerivativeReqs(provider_kind="autograd", needs_surrogate=False)
        comps = PlatformBuilder(
            small_dataset_with_lhs, reqs, device="cuda"
        ).build()
        provider = comps.context.derivative_provider
        coord = next(iter(provider.coords.values()))

        assert coord.is_cuda
        assert coord.requires_grad is True
        assert coord.is_leaf is True
