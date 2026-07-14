
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from kd.models.field_model import FieldModel

pytestmark = pytest.mark.unit


_A = (1.1915, 1.5957, 0.5, 0.0218)
_B = (2.3830, 0.0, 1.0)


_STEADY_HIDDEN = [50, 50, 50, 50, 50]


def _reference_rational(x: Tensor) -> Tensor:
    a = torch.tensor(_A, dtype=torch.float32)
    b = torch.tensor(_B, dtype=torch.float32)
    n = a[0] + x * (a[1] + x * (a[2] + a[3] * x))
    d = b[0] + x * (b[1] + b[2] * x)
    return n / d


def _activation_modules(model: FieldModel) -> list[nn.Module]:
    return [m for m in model.trunk if not isinstance(m, nn.Linear)]







class TestConstructionAndForward:
    def test_constructs_steady_5x50_rational(self) -> None:
        model = FieldModel(
            coord_names=["x", "y"],
            field_names=["u"],
            hidden_sizes=_STEADY_HIDDEN,
            activation="rational",
        )
        assert model.n_coords == 2
        assert model.n_fields == 1

    def test_forward_finite_on_scatter(self) -> None:
        model = FieldModel(
            coord_names=["x", "y"],
            field_names=["u"],
            hidden_sizes=_STEADY_HIDDEN,
            activation="rational",
        )
        torch.manual_seed(0)
        x = torch.randn(32)
        y = torch.randn(32)
        out = model(x=x, y=y)
        assert set(out) == {"u"}
        assert out["u"].shape == (32,)
        assert torch.isfinite(out["u"]).all()

    def test_autograd_grad_finite(self) -> None:
        model = FieldModel(
            coord_names=["x", "y"],
            field_names=["u"],
            hidden_sizes=[50, 50],
            activation="rational",
        )
        x = torch.randn(24, requires_grad=True)
        y = torch.randn(24, requires_grad=True)
        out = model(x=x, y=y)
        (du_dx,) = torch.autograd.grad(
            out["u"], x, grad_outputs=torch.ones_like(out["u"]), create_graph=True
        )
        assert du_dx.shape == x.shape
        assert torch.isfinite(du_dx).all()







class TestRationalEquivalence:
    def test_matches_reference_formula_at_default_init(self) -> None:
        model = FieldModel(
            coord_names=["x"],
            field_names=["u"],
            hidden_sizes=[8],
            activation="rational",
        )
        act = _activation_modules(model)[0]
        x = torch.tensor(
            [-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0], dtype=torch.float32
        )
        torch.testing.assert_close(act(x), _reference_rational(x))

    def test_no_nan_guard_at_forced_denominator_root(self) -> None:
        model = FieldModel(
            coord_names=["x"],
            field_names=["u"],
            hidden_sizes=[8],
            activation="rational",
        )
        act = _activation_modules(model)[0]
        denom = next(p for p in act.parameters() if p.numel() == 3)
        with torch.no_grad():
            denom.copy_(torch.tensor([0.0, 0.0, 1.0], dtype=denom.dtype))
        out = act(torch.tensor([0.0], dtype=torch.float32))
        assert torch.isinf(out).all()







class TestPerLayerParameters:
    def test_five_distinct_trainable_instances(self) -> None:
        model = FieldModel(
            coord_names=["x", "y"],
            field_names=["u"],
            hidden_sizes=_STEADY_HIDDEN,
            activation="rational",
        )
        acts = _activation_modules(model)
        assert len(acts) == len(_STEADY_HIDDEN)


        assert len({id(a) for a in acts}) == len(acts)


        param_id_sets = []
        for act in acts:
            params = list(act.parameters())
            assert params, "a parametric Rational must expose trainable parameters"
            assert all(p.requires_grad for p in params)
            param_id_sets.append({id(p) for p in params})

        for i in range(len(param_id_sets)):
            for j in range(i + 1, len(param_id_sets)):
                assert param_id_sets[i].isdisjoint(param_id_sets[j])







def test_unknown_activation_still_raises() -> None:
    with pytest.raises(ValueError, match="activation"):
        FieldModel(
            coord_names=["x", "y"],
            field_names=["u"],
            hidden_sizes=[8],
            activation="definitely_not_a_real_activation",
        )
