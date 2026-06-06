
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from kd.search.discover.config import PINNConfig
from kd.search.discover.pinn.model import PINNModel, PretrainResult














EXPECTED_PARAM_COUNT = 3021

_CPU_DEVICE = torch.device("cpu")

COORD_NAMES = ["x", "t"]
FIELD_NAMES = ["u"]


SMALL_CONFIG = PINNConfig(
    number_layer=8,
    n_hidden=20,
    activation="tanh",
    pretrain_epoch=100,
    early_stop_patience=5,
    lr=0.01,
)


ZERO_EPOCH_CONFIG = PINNConfig(
    number_layer=8,
    n_hidden=20,
    activation="tanh",
    pretrain_epoch=0,
    early_stop_patience=5,
    lr=0.01,
)

N_POINTS = 200







@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def model(device: torch.device) -> PINNModel:
    return PINNModel(
        coord_names=COORD_NAMES,
        field_names=FIELD_NAMES,
        config=SMALL_CONFIG,
        device=device,
    )


@pytest.fixture
def sample_coords(device: torch.device) -> dict[str, Tensor]:
    torch.manual_seed(42)
    return {
        "x": torch.randn(N_POINTS, device=device, requires_grad=True),
        "t": torch.randn(N_POINTS, device=device, requires_grad=True),
    }


def _make_sine_data(
    n_train: int = 200,
    n_val: int = 50,
    device: torch.device = _CPU_DEVICE,
    seed: int = 42,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    torch.manual_seed(seed)

    x_train = torch.rand(n_train, device=device) * 2 - 1
    t_train = torch.rand(n_train, device=device) * 2 - 1
    u_train = torch.sin(torch.pi * x_train) * torch.cos(torch.pi * t_train)


    x_val = torch.rand(n_val, device=device) * 2 - 1
    t_val = torch.rand(n_val, device=device) * 2 - 1
    u_val = torch.sin(torch.pi * x_val) * torch.cos(torch.pi * t_val)

    coords = {"x": x_train, "t": t_train}
    targets = {"u": u_train}
    val_coords = {"x": x_val, "t": t_val}
    val_targets = {"u": u_val}
    return coords, targets, val_coords, val_targets







class TestPINNModelConstruction:

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_parameter_count(self, model: PINNModel) -> None:
        total = sum(p.numel() for p in model.parameters())
        assert total == EXPECTED_PARAM_COUNT

    @pytest.mark.unit
    def test_construction_with_all_activations(self, device: torch.device) -> None:
        for act in ("tanh", "sin", "relu"):
            cfg = PINNConfig(number_layer=2, n_hidden=10, activation=act)
            m = PINNModel(COORD_NAMES, FIELD_NAMES, cfg, device)
            assert m is not None

    @pytest.mark.unit
    def test_multi_field_output(self, device: torch.device) -> None:
        cfg = PINNConfig(number_layer=2, n_hidden=10)
        m = PINNModel(["x", "t"], ["u", "v"], cfg, device)
        x = torch.randn(10, device=device)
        t = torch.randn(10, device=device)
        out = m(x=x, t=t)
        assert set(out.keys()) == {"u", "v"}
        for v in out.values():
            assert v.shape == (10,)







class TestPINNModelForward:

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_output_shape(
        self, model: PINNModel, sample_coords: dict[str, Tensor]
    ) -> None:
        out = model(**sample_coords)
        assert "u" in out
        assert out["u"].shape == (N_POINTS,)

    @pytest.mark.unit
    def test_output_has_grad_fn(
        self, model: PINNModel, sample_coords: dict[str, Tensor]
    ) -> None:
        out = model(**sample_coords)
        assert out["u"].grad_fn is not None

    @pytest.mark.unit
    def test_autograd_first_derivative(
        self, model: PINNModel, sample_coords: dict[str, Tensor]
    ) -> None:
        out = model(**sample_coords)
        (du_dx,) = torch.autograd.grad(
            out["u"],
            sample_coords["x"],
            grad_outputs=torch.ones_like(out["u"]),
            create_graph=True,
        )
        assert du_dx.shape == (N_POINTS,)
        assert du_dx.grad_fn is not None

    @pytest.mark.unit
    def test_autograd_second_derivative(
        self, model: PINNModel, sample_coords: dict[str, Tensor]
    ) -> None:
        out = model(**sample_coords)
        (du_dx,) = torch.autograd.grad(
            out["u"],
            sample_coords["x"],
            grad_outputs=torch.ones_like(out["u"]),
            create_graph=True,
        )
        (d2u_dx2,) = torch.autograd.grad(
            du_dx,
            sample_coords["x"],
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True,
        )
        assert d2u_dx2.shape == (N_POINTS,)

    @pytest.mark.unit
    def test_forward_keyword_interface(self, model: PINNModel) -> None:
        coords = {
            "x": torch.randn(50, requires_grad=True),
            "t": torch.randn(50, requires_grad=True),
        }
        out = model(**coords)
        assert isinstance(out, dict)
        assert "u" in out
        assert out["u"].shape == (50,)

    @pytest.mark.unit
    def test_no_grad_inputs_still_produce_output(
        self, model: PINNModel, device: torch.device
    ) -> None:
        x = torch.randn(30, device=device)
        t = torch.randn(30, device=device)
        out = model(x=x, t=t)
        assert out["u"].shape == (30,)

    @pytest.mark.unit
    def test_output_dtype_float32(
        self, model: PINNModel, sample_coords: dict[str, Tensor]
    ) -> None:
        out = model(**sample_coords)
        assert out["u"].dtype == torch.float32

    @pytest.mark.unit
    def test_single_point_forward(self, model: PINNModel) -> None:
        out = model(x=torch.tensor([0.5]), t=torch.tensor([0.5]))
        assert out["u"].shape == (1,)







class TestPINNModelDevice:

    @pytest.mark.unit
    def test_to_device_moves_all_params(self, device: torch.device) -> None:
        model = PINNModel(COORD_NAMES, FIELD_NAMES, SMALL_CONFIG, device)
        model = model.to(device)
        for p in model.parameters():
            assert p.device == device

        for name, buf in model.named_buffers():
            assert buf.device == device, f"Buffer {name} on wrong device"







class TestPINNModelPretrain:

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_loss_decreases(self, device: torch.device) -> None:
        model = PINNModel(COORD_NAMES, FIELD_NAMES, SMALL_CONFIG, device)
        coords, targets, val_coords, val_targets = _make_sine_data(device=device)


        with torch.no_grad():
            out_init = model(**coords)
        initial_mse = ((out_init["u"] - targets["u"]) ** 2).mean().item()

        result = model.pretrain(coords, targets, val_coords, val_targets, SMALL_CONFIG)

        assert isinstance(result, PretrainResult)

        assert result.train_loss < initial_mse * 0.5, (
            f"Pretrain did not reduce loss enough: {result.train_loss:.4f} "
            f"vs initial {initial_mse:.4f}"
        )

    @pytest.mark.unit
    def test_pretrain_returns_result_dataclass(self, device: torch.device) -> None:
        model = PINNModel(COORD_NAMES, FIELD_NAMES, SMALL_CONFIG, device)
        coords, targets, val_coords, val_targets = _make_sine_data(device=device)

        result = model.pretrain(coords, targets, val_coords, val_targets, SMALL_CONFIG)

        assert hasattr(result, "train_loss")
        assert hasattr(result, "val_loss")
        assert hasattr(result, "epochs_run")
        assert hasattr(result, "stopped_early")
        assert isinstance(result.train_loss, float)
        assert isinstance(result.val_loss, float)
        assert isinstance(result.epochs_run, int)
        assert isinstance(result.stopped_early, bool)

    @pytest.mark.unit
    def test_early_stopping_triggers(self, device: torch.device) -> None:

        early_cfg = PINNConfig(
            number_layer=2,
            n_hidden=10,
            activation="tanh",
            pretrain_epoch=5000,
            early_stop_patience=3,
            lr=0.01,
        )
        model = PINNModel(COORD_NAMES, FIELD_NAMES, early_cfg, device)
        coords, targets, val_coords, val_targets = _make_sine_data(device=device)

        result = model.pretrain(coords, targets, val_coords, val_targets, early_cfg)


        assert result.stopped_early is True
        assert result.epochs_run < 500, (
            f"Early stopping too late: {result.epochs_run} epochs "
            f"(patience={early_cfg.early_stop_patience})"
        )

    @pytest.mark.unit
    def test_zero_epochs_returns_immediately(self, device: torch.device) -> None:
        model = PINNModel(COORD_NAMES, FIELD_NAMES, ZERO_EPOCH_CONFIG, device)
        coords, targets, val_coords, val_targets = _make_sine_data(device=device)

        result = model.pretrain(
            coords, targets, val_coords, val_targets, ZERO_EPOCH_CONFIG
        )

        assert result.epochs_run == 0
        assert result.stopped_early is False

    @pytest.mark.unit
    def test_normalization_stats_applied(self, device: torch.device) -> None:
        model = PINNModel(COORD_NAMES, FIELD_NAMES, SMALL_CONFIG, device)
        coords, targets, val_coords, val_targets = _make_sine_data(device=device)

        model.pretrain(coords, targets, val_coords, val_targets, SMALL_CONFIG)



        buffers = dict(model.named_buffers())
        has_nonidentity = False
        for name, buf in buffers.items():
            if "mean" in name and buf.item() != 0.0:
                has_nonidentity = True
            if "std" in name and buf.item() != 1.0:
                has_nonidentity = True
        assert has_nonidentity, (
            "All normalization buffers are identity (mean=0, std=1) — "
            "pretrain did not compute normalization stats"
        )

    @pytest.mark.unit
    def test_pretrain_output_on_training_data_is_reasonable(
        self, device: torch.device
    ) -> None:
        cfg = PINNConfig(
            number_layer=4,
            n_hidden=20,
            activation="tanh",
            pretrain_epoch=500,
            early_stop_patience=50,
            lr=0.005,
        )
        model = PINNModel(COORD_NAMES, FIELD_NAMES, cfg, device)
        coords, targets, val_coords, val_targets = _make_sine_data(
            n_train=300, device=device
        )


        zero_baseline = (targets["u"] ** 2).mean().item()

        result = model.pretrain(coords, targets, val_coords, val_targets, cfg)

        with torch.no_grad():
            out = model(**coords)
        mse = ((out["u"] - targets["u"]) ** 2).mean().item()

        assert mse < zero_baseline * 0.2, (
            f"Pretrain MSE {mse:.4f} not much better than "
            f"zero baseline {zero_baseline:.4f}"
        )
        assert result.train_loss < zero_baseline * 0.2

    @pytest.mark.unit
    def test_pretrain_restores_best_weights(self, device: torch.device) -> None:
        early_cfg = PINNConfig(
            number_layer=2,
            n_hidden=10,
            activation="tanh",
            pretrain_epoch=5000,
            early_stop_patience=3,
            lr=0.01,
        )
        model = PINNModel(COORD_NAMES, FIELD_NAMES, early_cfg, device)
        coords, targets, val_coords, val_targets = _make_sine_data(device=device)

        result = model.pretrain(coords, targets, val_coords, val_targets, early_cfg)
        assert result.stopped_early is True


        with torch.no_grad():
            out = model(**val_coords)
        val_mse = ((out["u"] - val_targets["u"]) ** 2).mean().item()

        assert abs(val_mse - result.val_loss) < 0.1, (
            f"Model val MSE {val_mse:.4f} doesn't match reported best "
            f"{result.val_loss:.4f} — weights may not be restored to best epoch"
        )







class TestPINNModelCheckpoint:

    @pytest.mark.unit
    def test_save_load_roundtrip(
        self, model: PINNModel, tmp_path: Path, device: torch.device
    ) -> None:
        ckpt_path = tmp_path / "model.ckpt"
        model.save_checkpoint(ckpt_path)

        model2 = PINNModel(COORD_NAMES, FIELD_NAMES, SMALL_CONFIG, device)
        model2.load_checkpoint(ckpt_path)


        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters(), strict=False
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2), f"Parameter {n1} differs after load"

    @pytest.mark.unit
    def test_loaded_model_same_output(
        self,
        model: PINNModel,
        sample_coords: dict[str, Tensor],
        tmp_path: Path,
        device: torch.device,
    ) -> None:
        ckpt_path = tmp_path / "model.ckpt"

        with torch.no_grad():
            out_before = model(**sample_coords)

        model.save_checkpoint(ckpt_path)

        model2 = PINNModel(COORD_NAMES, FIELD_NAMES, SMALL_CONFIG, device)
        model2.load_checkpoint(ckpt_path)

        with torch.no_grad():
            out_after = model2(**sample_coords)

        assert torch.allclose(out_before["u"], out_after["u"], atol=1e-7)

    @pytest.mark.unit
    def test_checkpoint_preserves_normalization(
        self, tmp_path: Path, device: torch.device
    ) -> None:
        model = PINNModel(COORD_NAMES, FIELD_NAMES, SMALL_CONFIG, device)
        coords, targets, val_coords, val_targets = _make_sine_data(device=device)


        model.pretrain(coords, targets, val_coords, val_targets, SMALL_CONFIG)

        ckpt_path = tmp_path / "pretrained.ckpt"
        model.save_checkpoint(ckpt_path)

        model2 = PINNModel(COORD_NAMES, FIELD_NAMES, SMALL_CONFIG, device)
        model2.load_checkpoint(ckpt_path)


        with torch.no_grad():
            out1 = model(**coords)
            out2 = model2(**coords)
        assert torch.allclose(out1["u"], out2["u"], atol=1e-7)


        for name, buf1 in model.named_buffers():
            buf2 = dict(model2.named_buffers())[name]
            assert torch.allclose(buf1, buf2), f"Buffer {name} differs"
