
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from kd.models.field_model import FieldModel
from kd.models.trainer import FieldModelTrainer




skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device available"
)




_LR = 1e-2
_MAX_EPOCHS = 100
_SEED = 7





def _make_sin_data(
    n: int = 32, dtype: torch.dtype = torch.float32
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    x = torch.linspace(0.0, 2.0 * math.pi, n, dtype=dtype)
    u = torch.sin(x)
    return {"x": x}, {"u": u}


def _build_trainer(lr: float = _LR) -> tuple[FieldModel, FieldModelTrainer]:
    model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
    trainer = FieldModelTrainer(model, lr=lr)
    return model, trainer


def _attach_device_observer(model: FieldModel) -> list[str]:
    observed: list[str] = []

    def _hook(module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        observed.append(next(module.parameters()).device.type)

    model.register_forward_pre_hook(_hook)
    return observed


def _rel_l2(actual: Tensor, reference: Tensor) -> float:
    num = torch.linalg.vector_norm(actual - reference)
    den = torch.linalg.vector_norm(reference) + 1e-12
    return (num / den).item()







def test_cpu_default_no_device_stays_on_cpu() -> None:
    model, trainer = _build_trainer()
    coords, targets = _make_sin_data(32)

    result = trainer.fit(
        coords,
        targets,
        max_epochs=_MAX_EPOCHS,
        patience=None,
        val_ratio=0.0,
        seed=_SEED,
    )


    assert next(model.parameters()).device.type == "cpu"

    assert math.isfinite(result.final_loss)
    assert result.final_loss < 0.5 * result.loss_history[0]

    preds = model(**coords)
    assert preds["u"].device.type == "cpu"
    assert torch.isfinite(preds["u"]).all()


def test_default_forward_only_on_cpu() -> None:
    model, trainer = _build_trainer()
    coords, targets = _make_sin_data(32)
    observed = _attach_device_observer(model)

    trainer.fit(
        coords,
        targets,
        max_epochs=_MAX_EPOCHS,
        patience=None,
        val_ratio=0.0,
        seed=_SEED,
    )

    assert observed, "expected training forwards to be observed"
    assert set(observed) == {"cpu"}, (
        f"default fit touched non-CPU device: {set(observed)}"
    )







@skip_no_cuda
@pytest.mark.numerical
def test_gpu_train_returns_model_on_entry_device() -> None:
    model, trainer = _build_trainer()
    coords, targets = _make_sin_data(32)

    result = trainer.fit(
        coords,
        targets,
        max_epochs=_MAX_EPOCHS,
        patience=None,
        val_ratio=0.0,
        seed=_SEED,
        device="cuda",
    )


    assert math.isfinite(result.final_loss)
    assert result.final_loss < 0.5 * result.loss_history[0]

    assert next(model.parameters()).device.type == "cpu"

    preds = model(**coords)
    assert preds["u"].device.type == "cpu"
    assert torch.isfinite(preds["u"]).all()







@skip_no_cuda
@pytest.mark.numerical
def test_gpu_forward_actually_runs_on_cuda() -> None:
    model, trainer = _build_trainer()
    coords, targets = _make_sin_data(32)
    observed = _attach_device_observer(model)

    result = trainer.fit(
        coords,
        targets,
        max_epochs=_MAX_EPOCHS,
        patience=None,
        val_ratio=0.0,
        seed=_SEED,
        device="cuda",
    )


    assert "cuda" in observed, f"training never ran on CUDA; observed={set(observed)}"

    assert math.isfinite(result.final_loss)
    assert result.final_loss < 0.5 * result.loss_history[0]







@skip_no_cuda
@pytest.mark.numerical
def test_cpu_gpu_prediction_loose_parity() -> None:
    coords, targets = _make_sin_data(32)

    model_cpu, trainer_cpu = _build_trainer()
    result_cpu = trainer_cpu.fit(
        coords,
        targets,
        max_epochs=_MAX_EPOCHS,
        patience=None,
        val_ratio=0.0,
        seed=_SEED,
    )
    pred_cpu = model_cpu(**coords)["u"].detach()

    model_gpu, trainer_gpu = _build_trainer()
    result_gpu = trainer_gpu.fit(
        coords,
        targets,
        max_epochs=_MAX_EPOCHS,
        patience=None,
        val_ratio=0.0,
        seed=_SEED,
        device="cuda",
    )
    pred_gpu = model_gpu(**coords)["u"].detach()

    assert math.isfinite(result_cpu.final_loss) and math.isfinite(result_gpu.final_loss)
    assert torch.isfinite(pred_cpu).all() and torch.isfinite(pred_gpu).all()
    assert _rel_l2(pred_gpu, pred_cpu) < 0.2







@skip_no_cuda
@pytest.mark.numerical
def test_cpu_data_with_gpu_device_val_split() -> None:
    model, trainer = _build_trainer()
    coords, targets = _make_sin_data(40)

    result = trainer.fit(
        coords,
        targets,
        max_epochs=_MAX_EPOCHS,
        patience=None,
        val_ratio=0.25,
        seed=_SEED,
        device="cuda",
    )

    assert math.isfinite(result.final_loss)
    assert result.val_loss is not None and math.isfinite(result.val_loss)

    assert next(model.parameters()).device.type == "cpu"







@skip_no_cuda
@pytest.mark.numerical
def test_norm_buffers_return_to_entry_device() -> None:
    model, trainer = _build_trainer()
    coords, targets = _make_sin_data(32)

    trainer.fit(
        coords,
        targets,
        max_epochs=_MAX_EPOCHS,
        patience=None,
        val_ratio=0.0,
        seed=_SEED,
        device="cuda",
    )

    buffers = dict(model.named_buffers())
    assert buffers, "model should expose normalization buffers"
    for name, buf in buffers.items():
        assert buf.device.type == "cpu", f"buffer {name!r} not restored to CPU"
        assert torch.isfinite(buf).all(), f"buffer {name!r} not finite"



    torch.testing.assert_close(
        buffers["coord_x_mean"], coords["x"].mean(), rtol=1e-3, atol=1e-4
    )


    preds = model(**coords)
    assert torch.isfinite(preds["u"]).all()







def test_cuda_requested_when_unavailable_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model, trainer = _build_trainer()
    coords, targets = _make_sin_data(16)

    with pytest.raises((ValueError, RuntimeError), match=r"(?i)cuda|available|device"):
        trainer.fit(
            coords,
            targets,
            max_epochs=5,
            patience=None,
            val_ratio=0.0,
            seed=_SEED,
            device="cuda",
        )


def test_bogus_device_string_raises() -> None:
    model, trainer = _build_trainer()
    coords, targets = _make_sin_data(16)

    with pytest.raises((ValueError, RuntimeError), match=r"(?i)device|expected"):
        trainer.fit(
            coords,
            targets,
            max_epochs=5,
            patience=None,
            val_ratio=0.0,
            seed=_SEED,
            device="not_a_real_device",
        )







@skip_no_cuda
@pytest.mark.numerical
def test_same_device_same_seed_determinism() -> None:
    coords, targets = _make_sin_data(32)

    def _run() -> tuple[float, Tensor]:
        model, trainer = _build_trainer()
        result = trainer.fit(
            coords,
            targets,
            max_epochs=_MAX_EPOCHS,
            patience=None,
            val_ratio=0.0,
            seed=_SEED,
            device="cuda",
        )
        return result.final_loss, model(**coords)["u"].detach()

    loss_a, pred_a = _run()
    loss_b, pred_b = _run()

    assert math.isfinite(loss_a) and math.isfinite(loss_b)
    torch.testing.assert_close(torch.tensor(loss_a), torch.tensor(loss_b))
    torch.testing.assert_close(pred_a, pred_b)







@skip_no_cuda
@pytest.mark.numerical
def test_entry_cuda_returns_on_cuda() -> None:
    model, trainer = _build_trainer()
    model.to("cuda")
    coords, targets = _make_sin_data(32)

    result = trainer.fit(
        coords,
        targets,
        max_epochs=_MAX_EPOCHS,
        patience=None,
        val_ratio=0.0,
        seed=_SEED,
        device="cuda",
    )

    assert math.isfinite(result.final_loss)
    assert next(model.parameters()).device.type == "cuda", "entry device not restored"







@skip_no_cuda
@pytest.mark.numerical
def test_restore_best_with_device() -> None:
    model, trainer = _build_trainer()
    coords, targets = _make_sin_data(40)

    result = trainer.fit(
        coords,
        targets,
        max_epochs=_MAX_EPOCHS,
        patience=None,
        val_ratio=0.25,
        seed=_SEED,
        restore_best=True,
        device="cuda",
    )


    assert result.best_restored is True
    assert result.best_epoch is not None
    assert result.best_val_loss is not None and math.isfinite(result.best_val_loss)

    assert next(model.parameters()).device.type == "cpu"
    preds = model(**coords)
    assert torch.isfinite(preds["u"]).all()







@skip_no_cuda
@pytest.mark.numerical
def test_torch_device_object_accepted() -> None:
    model, trainer = _build_trainer()
    coords, targets = _make_sin_data(32)
    observed = _attach_device_observer(model)

    result = trainer.fit(
        coords,
        targets,
        max_epochs=_MAX_EPOCHS,
        patience=None,
        val_ratio=0.0,
        seed=_SEED,
        device=torch.device("cuda"),
    )

    assert "cuda" in observed, f"torch.device('cuda') not honoured: {set(observed)}"
    assert math.isfinite(result.final_loss)
    assert next(model.parameters()).device.type == "cpu"


def test_explicit_cpu_device_string() -> None:
    model, trainer = _build_trainer()
    coords, targets = _make_sin_data(32)
    observed = _attach_device_observer(model)

    result = trainer.fit(
        coords,
        targets,
        max_epochs=_MAX_EPOCHS,
        patience=None,
        val_ratio=0.0,
        seed=_SEED,
        device="cpu",
    )

    assert math.isfinite(result.final_loss)
    assert result.final_loss < 0.5 * result.loss_history[0]
    assert next(model.parameters()).device.type == "cpu"
    assert set(observed) == {"cpu"}







@skip_no_cuda
@pytest.mark.numerical
def test_float64_cuda_dtype_device_smoke() -> None:
    model, trainer = _build_trainer()
    coords, targets = _make_sin_data(32, dtype=torch.float64)
    observed = _attach_device_observer(model)

    result = trainer.fit(
        coords,
        targets,
        max_epochs=_MAX_EPOCHS,
        patience=None,
        val_ratio=0.0,
        seed=_SEED,
        device="cuda",
    )

    assert "cuda" in observed, "float64 training never ran on CUDA"
    assert math.isfinite(result.final_loss)
    param = next(model.parameters())
    assert param.dtype == torch.float64
    assert param.device.type == "cpu"
    preds = model(**coords)
    assert preds["u"].dtype == torch.float64
    assert torch.isfinite(preds["u"]).all()
