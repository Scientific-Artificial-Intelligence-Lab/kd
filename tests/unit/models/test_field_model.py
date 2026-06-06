
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from kd.models.field_model import FieldModel






@pytest.fixture()
def single_field_model() -> FieldModel:
    return FieldModel(
        coord_names=["x", "t"],
        field_names=["u"],
        hidden_sizes=[32, 32],
        activation="tanh",
    )


@pytest.fixture()
def multi_field_model() -> FieldModel:
    return FieldModel(
        coord_names=["x", "t"],
        field_names=["u", "v"],
        hidden_sizes=[32, 32],
        activation="tanh",
    )


@pytest.fixture()
def sample_coords() -> dict[str, Tensor]:
    torch.manual_seed(0)
    return {
        "x": torch.randn(50),
        "t": torch.randn(50),
    }







class TestSingleFieldForward:

    def test_returns_dict_with_field_key(
        self, single_field_model: FieldModel, sample_coords: dict[str, Tensor]
    ) -> None:
        out = single_field_model(**sample_coords)
        assert isinstance(out, dict)
        assert "u" in out

    def test_output_shape_matches_input(
        self, single_field_model: FieldModel, sample_coords: dict[str, Tensor]
    ) -> None:
        out = single_field_model(**sample_coords)
        assert out["u"].shape == sample_coords["x"].shape

    def test_no_extra_keys(
        self, single_field_model: FieldModel, sample_coords: dict[str, Tensor]
    ) -> None:
        out = single_field_model(**sample_coords)
        assert set(out.keys()) == {"u"}







class TestMultiFieldForward:

    def test_returns_both_fields(
        self, multi_field_model: FieldModel, sample_coords: dict[str, Tensor]
    ) -> None:
        out = multi_field_model(**sample_coords)
        assert "u" in out
        assert "v" in out

    def test_each_field_shape(
        self, multi_field_model: FieldModel, sample_coords: dict[str, Tensor]
    ) -> None:
        out = multi_field_model(**sample_coords)
        for name in ("u", "v"):
            assert out[name].shape == sample_coords["x"].shape

    def test_fields_are_different(
        self, multi_field_model: FieldModel, sample_coords: dict[str, Tensor]
    ) -> None:
        out = multi_field_model(**sample_coords)

        assert not torch.allclose(out["u"], out["v"])







class TestPassThroughMode:

    def test_forward_without_normalization(
        self, single_field_model: FieldModel, sample_coords: dict[str, Tensor]
    ) -> None:
        out = single_field_model(**sample_coords)
        assert out["u"].shape == sample_coords["x"].shape
        assert torch.isfinite(out["u"]).all()







class TestNormalization:

    def test_output_changes_with_normalization(
        self, single_field_model: FieldModel, sample_coords: dict[str, Tensor]
    ) -> None:
        out_before = single_field_model(**sample_coords)

        single_field_model.set_normalization(
            coord_stats={
                "x": (torch.tensor(0.0), torch.tensor(1.0)),
                "t": (torch.tensor(0.0), torch.tensor(1.0)),
            },
            field_stats={
                "u": (torch.tensor(5.0), torch.tensor(2.0)),
            },
        )
        out_after = single_field_model(**sample_coords)


        assert not torch.allclose(out_before["u"], out_after["u"])

    def test_normalization_in_computation_graph(
        self, single_field_model: FieldModel
    ) -> None:
        single_field_model.set_normalization(
            coord_stats={
                "x": (torch.tensor(0.0), torch.tensor(1.0)),
                "t": (torch.tensor(0.0), torch.tensor(1.0)),
            },
            field_stats={
                "u": (torch.tensor(0.0), torch.tensor(1.0)),
            },
        )
        x = torch.randn(10, requires_grad=True)
        t = torch.randn(10, requires_grad=True)
        out = single_field_model(x=x, t=t)

        loss = out["u"].sum()
        loss.backward()
        assert x.grad is not None
        assert t.grad is not None

    def test_std_zero_fallback(self) -> None:
        model = FieldModel(
            coord_names=["x"],
            field_names=["u"],
            hidden_sizes=[16],
            activation="tanh",
        )
        model.set_normalization(
            coord_stats={"x": (torch.tensor(3.0), torch.tensor(0.0))},
            field_stats={"u": (torch.tensor(1.0), torch.tensor(0.0))},
        )
        x = torch.randn(10)
        out = model(x=x)
        assert torch.isfinite(out["u"]).all()

    def test_multi_field_normalization(
        self, multi_field_model: FieldModel, sample_coords: dict[str, Tensor]
    ) -> None:
        multi_field_model.set_normalization(
            coord_stats={
                "x": (torch.tensor(1.0), torch.tensor(2.0)),
                "t": (torch.tensor(-1.0), torch.tensor(0.5)),
            },
            field_stats={
                "u": (torch.tensor(10.0), torch.tensor(3.0)),
                "v": (torch.tensor(-5.0), torch.tensor(1.0)),
            },
        )
        out = multi_field_model(**sample_coords)
        assert out["u"].shape == sample_coords["x"].shape
        assert out["v"].shape == sample_coords["x"].shape
        assert torch.isfinite(out["u"]).all()
        assert torch.isfinite(out["v"]).all()







class TestStateDictPersistence:

    def test_buffers_in_state_dict(self, single_field_model: FieldModel) -> None:
        single_field_model.set_normalization(
            coord_stats={
                "x": (torch.tensor(1.0), torch.tensor(2.0)),
                "t": (torch.tensor(3.0), torch.tensor(4.0)),
            },
            field_stats={
                "u": (torch.tensor(5.0), torch.tensor(6.0)),
            },
        )
        sd = single_field_model.state_dict()

        buffer_keys = [k for k in sd if "mean" in k or "std" in k]
        assert len(buffer_keys) > 0

    def test_load_state_dict_restores_normalization(
        self, sample_coords: dict[str, Tensor]
    ) -> None:
        torch.manual_seed(42)
        model1 = FieldModel(["x", "t"], ["u"], hidden_sizes=[16, 16])
        model1.set_normalization(
            coord_stats={
                "x": (torch.tensor(1.0), torch.tensor(2.0)),
                "t": (torch.tensor(3.0), torch.tensor(4.0)),
            },
            field_stats={
                "u": (torch.tensor(5.0), torch.tensor(6.0)),
            },
        )
        out1 = model1(**sample_coords)


        sd = model1.state_dict()
        model2 = FieldModel(["x", "t"], ["u"], hidden_sizes=[16, 16])
        model2.load_state_dict(sd)
        out2 = model2(**sample_coords)

        torch.testing.assert_close(out1["u"], out2["u"])







class TestActivations:

    @pytest.mark.parametrize("act", ["tanh", "sin", "relu"])
    def test_activation_produces_finite_output(
        self, act: str, sample_coords: dict[str, Tensor]
    ) -> None:
        model = FieldModel(
            coord_names=["x", "t"],
            field_names=["u"],
            hidden_sizes=[16, 16],
            activation=act,
        )
        out = model(**sample_coords)
        assert torch.isfinite(out["u"]).all()

    def test_different_activations_give_different_outputs(
        self, sample_coords: dict[str, Tensor]
    ) -> None:
        torch.manual_seed(0)
        model_tanh = FieldModel(["x", "t"], ["u"], hidden_sizes=[16], activation="tanh")
        torch.manual_seed(0)
        model_sin = FieldModel(["x", "t"], ["u"], hidden_sizes=[16], activation="sin")

        out_tanh = model_tanh(**sample_coords)
        out_sin = model_sin(**sample_coords)

        assert not torch.allclose(out_tanh["u"], out_sin["u"])







class TestValidation:

    def test_unknown_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="activation"):
            FieldModel(
                coord_names=["x", "t"],
                field_names=["u"],
                activation="unknown_act",
            )

    def test_empty_coord_names_raises(self) -> None:
        with pytest.raises(ValueError, match="coord_names"):
            FieldModel(
                coord_names=[],
                field_names=["u"],
            )

    def test_empty_field_names_raises(self) -> None:
        with pytest.raises(ValueError, match="field_names"):
            FieldModel(
                coord_names=["x"],
                field_names=[],
            )

    def test_empty_hidden_sizes_raises(self) -> None:
        with pytest.raises(ValueError, match="hidden_sizes"):
            FieldModel(
                coord_names=["x"],
                field_names=["u"],
                hidden_sizes=[],
            )







class TestAutogradCompatibility:

    def test_gradient_flows_through_forward(
        self, single_field_model: FieldModel
    ) -> None:
        x = torch.randn(20, requires_grad=True)
        t = torch.randn(20, requires_grad=True)
        out = single_field_model(x=x, t=t)


        (du_dx,) = torch.autograd.grad(
            outputs=out["u"],
            inputs=x,
            grad_outputs=torch.ones_like(out["u"]),
            create_graph=True,
        )
        assert du_dx.shape == x.shape
        assert torch.isfinite(du_dx).all()

    def test_gradient_with_normalization(self) -> None:
        model = FieldModel(["x", "t"], ["u"], hidden_sizes=[16, 16])
        model.set_normalization(
            coord_stats={
                "x": (torch.tensor(0.0), torch.tensor(1.0)),
                "t": (torch.tensor(0.0), torch.tensor(1.0)),
            },
            field_stats={
                "u": (torch.tensor(0.0), torch.tensor(1.0)),
            },
        )
        x = torch.randn(20, requires_grad=True)
        t = torch.randn(20, requires_grad=True)
        out = model(x=x, t=t)

        (du_dx,) = torch.autograd.grad(
            outputs=out["u"],
            inputs=x,
            grad_outputs=torch.ones_like(out["u"]),
            create_graph=True,
        )
        assert du_dx.shape == x.shape
        assert torch.isfinite(du_dx).all()

    def test_second_order_derivative(self, single_field_model: FieldModel) -> None:
        x = torch.randn(20, requires_grad=True)
        t = torch.randn(20, requires_grad=True)
        out = single_field_model(x=x, t=t)


        (du_dx,) = torch.autograd.grad(
            outputs=out["u"],
            inputs=x,
            grad_outputs=torch.ones_like(out["u"]),
            create_graph=True,
        )

        (d2u_dx2,) = torch.autograd.grad(
            outputs=du_dx,
            inputs=x,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True,
        )
        assert d2u_dx2.shape == x.shape
        assert torch.isfinite(d2u_dx2).all()







class TestEdgeCases:

    def test_single_coord(self) -> None:
        model = FieldModel(["x"], ["u"], hidden_sizes=[16])
        x = torch.randn(30)
        out = model(x=x)
        assert out["u"].shape == (30,)

    def test_many_coords(self) -> None:
        model = FieldModel(["x", "y", "z", "t"], ["u"], hidden_sizes=[16])
        coords = {name: torch.randn(20) for name in ["x", "y", "z", "t"]}
        out = model(**coords)
        assert out["u"].shape == (20,)

    def test_single_data_point(self) -> None:
        model = FieldModel(["x", "t"], ["u"], hidden_sizes=[16])
        out = model(x=torch.tensor([1.0]), t=torch.tensor([2.0]))
        assert out["u"].shape == (1,)

    def test_deep_network(self) -> None:
        model = FieldModel(
            ["x", "t"],
            ["u"],
            hidden_sizes=[32, 32, 32, 32],
            activation="tanh",
        )
        coords = {"x": torch.randn(10), "t": torch.randn(10)}
        out = model(**coords)
        assert out["u"].shape == (10,)
        assert torch.isfinite(out["u"]).all()







class TestConstructionSupplemental:

    def test_default_params_construction(self) -> None:
        model = FieldModel(coord_names=["x", "t"], field_names=["u"])
        assert model.n_coords == 2
        assert model.n_fields == 1


        assert len(model.trunk) == 4

        first_linear = model.trunk[0]
        assert isinstance(first_linear, torch.nn.Linear)
        assert first_linear.in_features == 2
        assert first_linear.out_features == 64

    def test_custom_hidden_sizes_changes_architecture(self) -> None:
        model = FieldModel(
            coord_names=["x"], field_names=["u"], hidden_sizes=[128, 64, 32]
        )

        assert len(model.trunk) == 6
        linear_layers = [m for m in model.trunk if isinstance(m, torch.nn.Linear)]
        assert linear_layers[0].in_features == 1
        assert linear_layers[0].out_features == 128
        assert linear_layers[1].in_features == 128
        assert linear_layers[1].out_features == 64
        assert linear_layers[2].in_features == 64
        assert linear_layers[2].out_features == 32

        assert model.head.in_features == 32

    def test_coord_and_field_names_stored(self) -> None:
        model = FieldModel(coord_names=["a", "b", "c"], field_names=["p", "q"])
        assert model.coord_names == ["a", "b", "c"]
        assert model.field_names == ["p", "q"]
        assert model.n_coords == 3
        assert model.n_fields == 2







class TestBatchSizeVariations:

    @pytest.mark.parametrize("batch_size", [1, 5, 100, 1000])
    def test_various_batch_sizes(self, batch_size: int) -> None:
        model = FieldModel(["x", "t"], ["u", "v"], hidden_sizes=[16])
        coords = {"x": torch.randn(batch_size), "t": torch.randn(batch_size)}
        out = model(**coords)
        for name in ["u", "v"]:
            assert out[name].shape == (batch_size,)
            assert torch.isfinite(out[name]).all()







class TestNormalizationSupplemental:

    def test_std_zero_buffer_value_is_one(self) -> None:
        model = FieldModel(["x"], ["u"], hidden_sizes=[16])
        model.set_normalization(
            coord_stats={"x": (torch.tensor(5.0), torch.tensor(0.0))},
            field_stats={"u": (torch.tensor(3.0), torch.tensor(0.0))},
        )

        assert model.coord_x_std.item() == 1.0
        assert model.field_u_std.item() == 1.0

        assert model.coord_x_mean.item() == 5.0
        assert model.field_u_mean.item() == 3.0

    def test_normalization_buffers_follow_device(self) -> None:
        model = FieldModel(["x"], ["u"], hidden_sizes=[16])
        model.set_normalization(
            coord_stats={"x": (torch.tensor(1.0), torch.tensor(2.0))},
            field_stats={"u": (torch.tensor(3.0), torch.tensor(4.0))},
        )

        model = model.to(torch.device("cpu"))
        assert model.coord_x_mean.device == torch.device("cpu")
        assert model.coord_x_std.device == torch.device("cpu")
        assert model.field_u_mean.device == torch.device("cpu")
        assert model.field_u_std.device == torch.device("cpu")


        assert model.coord_x_mean.item() == 1.0
        assert model.coord_x_std.item() == 2.0
        assert model.field_u_mean.item() == 3.0
        assert model.field_u_std.item() == 4.0

    def test_normalization_identity_is_passthrough(self) -> None:
        torch.manual_seed(99)
        model = FieldModel(["x", "t"], ["u"], hidden_sizes=[16])
        x = torch.randn(20)
        t = torch.randn(20)

        out_before = model(x=x, t=t)["u"].detach().clone()


        model.set_normalization(
            coord_stats={
                "x": (torch.tensor(0.0), torch.tensor(1.0)),
                "t": (torch.tensor(0.0), torch.tensor(1.0)),
            },
            field_stats={"u": (torch.tensor(0.0), torch.tensor(1.0))},
        )
        out_after = model(x=x, t=t)["u"]
        torch.testing.assert_close(out_before, out_after)

    def test_set_normalization_overwrites_previous(self) -> None:
        model = FieldModel(["x"], ["u"], hidden_sizes=[16])
        model.set_normalization(
            coord_stats={"x": (torch.tensor(1.0), torch.tensor(2.0))},
            field_stats={"u": (torch.tensor(3.0), torch.tensor(4.0))},
        )
        model.set_normalization(
            coord_stats={"x": (torch.tensor(10.0), torch.tensor(20.0))},
            field_stats={"u": (torch.tensor(30.0), torch.tensor(40.0))},
        )
        assert model.coord_x_mean.item() == 10.0
        assert model.coord_x_std.item() == 20.0
        assert model.field_u_mean.item() == 30.0
        assert model.field_u_std.item() == 40.0
