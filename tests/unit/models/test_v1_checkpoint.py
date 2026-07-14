
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from torch import Tensor

from kd.models.field_model import FieldModel
from kd.models.v1_checkpoint import load_v1_field_model





_HIDDEN_LAYERS = 6
_NEURONS = 60
_IN_DIM = 2
_OUT_DIM = 1


def _make_v1_state_dict(
    *,
    in_dim: int = _IN_DIM,
    out_dim: int = _OUT_DIM,
    seed: int = 0,
) -> OrderedDict[str, Tensor]:
    gen = torch.Generator().manual_seed(seed)
    sd: OrderedDict[str, Tensor] = OrderedDict()
    sd["Layers.0.weight"] = torch.randn(_NEURONS, in_dim, generator=gen)
    sd["Layers.0.bias"] = torch.randn(_NEURONS, generator=gen)
    for i in range(1, _HIDDEN_LAYERS):
        sd[f"Layers.{i}.weight"] = torch.randn(_NEURONS, _NEURONS, generator=gen)
        sd[f"Layers.{i}.bias"] = torch.randn(_NEURONS, generator=gen)
    sd[f"Layers.{_HIDDEN_LAYERS}.weight"] = torch.randn(
        out_dim, _NEURONS, generator=gen
    )
    sd[f"Layers.{_HIDDEN_LAYERS}.bias"] = torch.randn(out_dim, generator=gen)
    return sd


def _raw_v1_forward(sd: OrderedDict[str, Tensor], x: Tensor) -> Tensor:
    h = x
    for i in range(_HIDDEN_LAYERS):
        w = sd[f"Layers.{i}.weight"]
        b = sd[f"Layers.{i}.bias"]
        h = torch.sin(h @ w.t() + b)
    w = sd[f"Layers.{_HIDDEN_LAYERS}.weight"]
    b = sd[f"Layers.{_HIDDEN_LAYERS}.bias"]
    return (h @ w.t() + b).squeeze(-1)


def _save(sd: object, path: Path) -> Path:
    torch.save(sd, path)
    return path







class TestCheckpointSmoke:

    @pytest.mark.smoke
    def test_callable(self) -> None:
        assert callable(load_v1_field_model)

    @pytest.mark.smoke
    def test_exported_from_models(self) -> None:
        from kd import models

        assert callable(models.load_v1_field_model)







class TestSyntheticLoad:

    @pytest.mark.unit
    def test_returns_field_model(self, tmp_path: Path) -> None:
        path = _save(_make_v1_state_dict(), tmp_path / "net.pkl")
        model = load_v1_field_model(path)
        assert isinstance(model, FieldModel)
        assert model.n_coords == _IN_DIM
        assert model.n_fields == _OUT_DIM

    @pytest.mark.numerical
    def test_forward_equals_raw_net(self, tmp_path: Path) -> None:
        sd = _make_v1_state_dict(seed=3)
        path = _save(sd, tmp_path / "net.pkl")
        model = load_v1_field_model(path)

        pts = torch.tensor(
            [[1.0, 0.5], [5.0, 1.0], [10.0, 2.0], [2.5, 0.01]],
            dtype=torch.float32,
        )
        coord_names = model.coord_names
        coords = {name: pts[:, i] for i, name in enumerate(coord_names)}
        out = model(**coords)[model.field_names[0]]
        expected = _raw_v1_forward(sd, pts)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.numerical
    def test_forward_output_shape_and_finite(self, tmp_path: Path) -> None:
        path = _save(_make_v1_state_dict(), tmp_path / "net.pkl")
        model = load_v1_field_model(path)
        n = 7
        coords = {name: torch.randn(n) for name in model.coord_names}
        out = model(**coords)[model.field_names[0]]
        assert out.shape == (n,)
        assert torch.isfinite(out).all()







class TestSyntheticNegative:

    @pytest.mark.unit
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_v1_field_model(tmp_path / "absent.pkl")

    @pytest.mark.unit
    def test_wrong_input_shape_raises(self, tmp_path: Path) -> None:
        sd = _make_v1_state_dict(in_dim=3)
        path = _save(sd, tmp_path / "net.pkl")
        with pytest.raises(ValueError):
            load_v1_field_model(path)

    @pytest.mark.unit
    def test_wrong_output_dim_raises(self, tmp_path: Path) -> None:
        sd = _make_v1_state_dict(out_dim=2)
        path = _save(sd, tmp_path / "net.pkl")
        with pytest.raises(ValueError):
            load_v1_field_model(path)

    @pytest.mark.unit
    def test_missing_key_raises(self, tmp_path: Path) -> None:
        sd = _make_v1_state_dict()
        del sd[f"Layers.{_HIDDEN_LAYERS}.weight"]
        path = _save(sd, tmp_path / "net.pkl")
        with pytest.raises(ValueError):
            load_v1_field_model(path)

    @pytest.mark.unit
    def test_unexpected_extra_key_raises(self, tmp_path: Path) -> None:
        sd = _make_v1_state_dict()
        sd["Layers.7.weight"] = torch.randn(_NEURONS, _NEURONS)
        sd["Layers.7.bias"] = torch.randn(_NEURONS)
        path = _save(sd, tmp_path / "net.pkl")
        with pytest.raises(ValueError):
            load_v1_field_model(path)

    @pytest.mark.unit
    def test_wrong_hidden_width_raises(self, tmp_path: Path) -> None:
        sd = _make_v1_state_dict()
        sd["Layers.3.weight"] = torch.randn(64, _NEURONS)
        sd["Layers.3.bias"] = torch.randn(64)
        path = _save(sd, tmp_path / "net.pkl")
        with pytest.raises(ValueError):
            load_v1_field_model(path)

    @pytest.mark.numerical
    def test_non_tensor_value_raises(self, tmp_path: Path) -> None:
        sd = _make_v1_state_dict()
        sd["Layers.0.bias"] = 5
        path = _save(sd, tmp_path / "net.pkl")
        with pytest.raises(ValueError):
            load_v1_field_model(path)
