
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.models.field_model import FieldModel
from kd.models.trainer import TrainingResult
from kd.search.protocol import PlatformComponents
from kd.search.sga.config import SGAConfig
from kd.search.sga.pde import PDE
from kd.search.sga.plugin import SGAPlugin

_DTYPE = torch.float64
_NX = 8
_NT = 5


def _sweep_config(**overrides: Any) -> SGAConfig:
    base: dict[str, Any] = {
        "normalize": 2,
        "d_tol": 0.5,
        "maxit": 5,
        "str_iters": 10,
        "lam": 0.0,
    }
    base.update(overrides)
    return SGAConfig(**base)








def _tiny_dataset() -> PDEDataset:
    x = torch.linspace(0.0, 6.0, _NX, dtype=_DTYPE)
    t = torch.linspace(0.0, 1.0, _NT, dtype=_DTYPE)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(gx) * torch.exp(-gt)
    return PDEDataset(
        name="sga-budget-passthrough",
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


def _mock_components(dataset: PDEDataset) -> PlatformComponents:
    context = MagicMock()
    context.dataset = dataset

    def get_variable(name: str) -> torch.Tensor:
        if dataset.fields and name in dataset.fields:
            values: torch.Tensor = dataset.fields[name].values
            return values
        if dataset.axes and name in dataset.axes:
            axis_values: torch.Tensor = dataset.axes[name].values
            return axis_values
        raise KeyError(name)

    context.get_variable = get_variable
    context.derivative_provider = MagicMock()
    return PlatformComponents(
        dataset=dataset,
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=context,
        registry=MagicMock(),
    )


class _SpyTrainer:

    calls: list[dict[str, Any]] = []

    def __init__(self, model: FieldModel, lr: float = 1e-3, **_: Any) -> None:
        self._model = model
        self._lr = lr

    def fit(self, *args: Any, **kwargs: Any) -> TrainingResult:
        type(self).calls.append({"args": tuple(args), **dict(kwargs)})



        coords = kwargs.get("coords")
        if coords:
            data_dtype = next(iter(coords.values())).dtype
            self._model = self._model.to(dtype=data_dtype)



        seed = int(kwargs.get("seed", 0))
        cuda_devices = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            for module in self._model.modules():
                reset_fn = getattr(module, "reset_parameters", None)
                if reset_fn is not None:
                    reset_fn()
        return TrainingResult(
            final_loss=0.0,
            epochs_run=0,
            early_stopped=False,
            val_loss=None,
            loss_history=[],
            val_loss_history=None,
        )


@pytest.fixture
def spy_trainer(monkeypatch: pytest.MonkeyPatch) -> type[_SpyTrainer]:
    _SpyTrainer.calls = []
    monkeypatch.setattr(
        "kd.search.sga.plugin.FieldModelTrainer", _SpyTrainer, raising=True
    )
    return _SpyTrainer


def _autograd_config(**overrides: Any) -> SGAConfig:
    base: dict[str, Any] = {
        "num": 4,
        "depth": 2,
        "width": 2,
        "seed": 42,
        "use_autograd": True,
        "autograd_train_epochs": 7,
    }
    base.update(overrides)
    return SGAConfig(**base)







class TestDefaultBudgetPassthrough:

    @pytest.mark.unit
    def test_fit_receives_patience_none_and_val_ratio_zero(
        self,
        spy_trainer: type[_SpyTrainer],
    ) -> None:
        plugin = SGAPlugin(_autograd_config())
        plugin.prepare(_mock_components(_tiny_dataset()))

        assert spy_trainer.calls, "prepare() must call trainer.fit exactly once"
        kwargs = spy_trainer.calls[-1]
        assert "patience" in kwargs, (
            "trainer.fit must receive an explicit 'patience' kwarg (the plugin "
            "currently omits it, so fit inherits patience=100 — dead knob)."
        )
        assert kwargs["patience"] is None, (
            f"default autograd budget must pass patience=None (v1 fixed-step); "
            f"got {kwargs['patience']!r}"
        )
        assert "val_ratio" in kwargs, (
            "trainer.fit must receive an explicit 'val_ratio' kwarg (the plugin "
            "currently omits it, so fit inherits val_ratio=0.2 — 20% held out)."
        )
        assert kwargs["val_ratio"] == 0.0, (
            f"default autograd budget must pass val_ratio=0.0 (full-data "
            f"train); got {kwargs['val_ratio']!r}"
        )

    @pytest.mark.unit
    def test_fit_still_receives_max_epochs_and_seed(
        self,
        spy_trainer: type[_SpyTrainer],
    ) -> None:
        plugin = SGAPlugin(_autograd_config(autograd_train_epochs=7, seed=42))
        plugin.prepare(_mock_components(_tiny_dataset()))

        kwargs = spy_trainer.calls[-1]
        assert kwargs.get("max_epochs") == 7, (
            f"max_epochs must still be forwarded; got {kwargs.get('max_epochs')!r}"
        )
        assert kwargs.get("seed") == 42, (
            f"seed must still be forwarded; got {kwargs.get('seed')!r}"
        )







class TestExplicitBudgetPassthrough:

    @pytest.mark.unit
    def test_explicit_patience_and_val_ratio_reach_fit(
        self,
        spy_trainer: type[_SpyTrainer],
    ) -> None:
        plugin = SGAPlugin(
            _autograd_config(
                autograd_train_patience=5,
                autograd_train_val_ratio=0.2,
            )
        )
        plugin.prepare(_mock_components(_tiny_dataset()))

        kwargs = spy_trainer.calls[-1]
        assert kwargs.get("patience") == 5, (
            f"explicit patience must reach fit; got {kwargs.get('patience')!r}"
        )
        assert kwargs.get("val_ratio") == pytest.approx(0.2), (
            f"explicit val_ratio must reach fit; got {kwargs.get('val_ratio')!r}"
        )

    @pytest.mark.unit
    def test_facade_passthrough_reaches_trainer_fit(
        self,
        spy_trainer: type[_SpyTrainer],
    ) -> None:
        import kd

        model = kd.Model(
            algorithm="sga",
            derivatives="autograd",
            autograd_train_patience=5,
            autograd_train_val_ratio=0.25,
            autograd_train_epochs=8,
            generations=1,
            population=3,
            depth=2,
            width=2,
            seed=0,
            verbose=False,
        )
        model.fit(_tiny_dataset())

        assert spy_trainer.calls, "Model.fit must reach trainer.fit via the plugin"
        kwargs = spy_trainer.calls[-1]
        assert kwargs.get("patience") == 5, (
            f"facade autograd_train_patience must reach fit; got "
            f"{kwargs.get('patience')!r}"
        )
        assert kwargs.get("val_ratio") == pytest.approx(0.25), (
            f"facade autograd_train_val_ratio must reach fit; got "
            f"{kwargs.get('val_ratio')!r}"
        )
        assert kwargs.get("max_epochs") == 8, (
            f"facade autograd_train_epochs must reach fit; got "
            f"{kwargs.get('max_epochs')!r}"
        )

    @pytest.mark.unit
    def test_pretrained_field_model_skips_trainer_entirely(
        self,
        spy_trainer: type[_SpyTrainer],
    ) -> None:
        dataset = _tiny_dataset()
        model = FieldModel(
            coord_names=["x", "t"],
            field_names=["u"],
            hidden_sizes=[4, 4],
        ).to(dtype=_DTYPE)
        model.eval()




        plugin = SGAPlugin(_autograd_config(field_model=model))
        plugin.prepare(_mock_components(dataset))

        assert spy_trainer.calls == [], (
            "a pre-trained field_model must skip trainer.fit entirely; spy "
            f"recorded {len(spy_trainer.calls)} call(s)."
        )
























def _empty_support_init_inputs() -> tuple[
    PDE, dict[str, torch.Tensor], torch.Tensor, torch.Tensor
]:
    from kd.search.sga.tree import Node, Tree

    n = 24
    giant_scale = 1e10
    gen = torch.Generator().manual_seed(7)
    u = giant_scale * torch.randn(n, dtype=_DTYPE, generator=gen)

    y = (1e-6 * torch.randn(n, dtype=_DTYPE, generator=gen)).unsqueeze(1)
    data_dict = {"u": u}
    default_terms = u.unsqueeze(1)
    pde = PDE(terms=[Tree(root=Node(name="u", arity=0, children=[]))])
    return pde, data_dict, default_terms, y


class TestInitResampleGate:

    @pytest.mark.unit
    def test_empty_support_candidate_is_rejected_by_resample_gate(self) -> None:
        from kd.search.sga.plugin import _is_valid_aic, _safe_evaluate_aic

        pde, data_dict, default_terms, y = _empty_support_init_inputs()
        config = _sweep_config()

        aic, _pruned = _safe_evaluate_aic(
            pde, data_dict, default_terms, y, config, None
        )
        assert not _is_valid_aic(aic), (
            f"an empty-support candidate must be REJECTED by the init resample "
            f"gate (_is_valid_aic False → resampled); got aic={aic!r}, "
            f"_is_valid_aic={_is_valid_aic(aic)}. Today the empty-support back "
            f"door leaves a finite in-range AIC, so the gate wrongly accepts it."
        )

    @pytest.mark.unit
    def test_empty_support_recipe_is_genuinely_empty_today(self) -> None:
        from kd.search.sga.train import evaluate_candidate

        pde, data_dict, default_terms, y = _empty_support_init_inputs()
        config = _sweep_config()






        assert all(torch.isfinite(v).all() for v in data_dict.values())
        assert torch.isfinite(default_terms).all()
        assert torch.isfinite(y).all()

        result = evaluate_candidate(pde, data_dict, default_terms, y, config, None)
        assert result.selected_indices == [], (
            f"the gate-input candidate must STRidge-empty its support today "
            f"(else the resample-gate test is vacuous); got "
            f"{result.selected_indices}"
        )
