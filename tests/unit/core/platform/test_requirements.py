
from __future__ import annotations

import dataclasses
import re
from dataclasses import replace
from typing import get_args, get_type_hints

import pytest
import torch

from kd.core.platform.requirements import DerivativeReqs






class TestDefaults:

    @pytest.mark.unit
    def test_construct_no_args(self) -> None:
        req = DerivativeReqs()
        assert req is not None

    @pytest.mark.unit
    def test_default_provider_kind(self) -> None:
        req = DerivativeReqs()
        assert req.provider_kind == "finite_diff"

    @pytest.mark.unit
    def test_default_max_atomic_order(self) -> None:
        req = DerivativeReqs()
        assert req.max_atomic_order == 2

    @pytest.mark.unit
    def test_default_lhs_order(self) -> None:
        req = DerivativeReqs()
        assert req.lhs_order == 1

    @pytest.mark.unit
    def test_default_needs_surrogate(self) -> None:
        req = DerivativeReqs()
        assert req.needs_surrogate is False

    @pytest.mark.unit
    def test_default_surrogate_model(self) -> None:
        req = DerivativeReqs()
        assert req.surrogate_model is None

    @pytest.mark.unit
    def test_default_surrogate_train_kwargs(self) -> None:
        req = DerivativeReqs()
        assert req.surrogate_train_kwargs is None







class TestFrozen:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("field_name", "new_value"),
        [
            ("provider_kind", "autograd"),
            ("max_atomic_order", 3),
            ("lhs_order", 2),
            ("needs_surrogate", True),
            ("surrogate_model", None),
            ("surrogate_train_kwargs", {"lr": 1e-3}),
        ],
    )
    def test_field_assignment_raises(self, field_name: str, new_value: object) -> None:
        req = DerivativeReqs()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(req, field_name, new_value)

    @pytest.mark.unit
    def test_dataclass_params_frozen_flag(self) -> None:
        params = DerivativeReqs.__dataclass_params__
        assert params.frozen is True

    @pytest.mark.unit
    def test_instance_is_hashable(self) -> None:
        a = DerivativeReqs()
        b = DerivativeReqs()
        assert hash(a) == hash(b)

        bag = {a, b, DerivativeReqs(provider_kind="autograd")}
        assert len(bag) == 2







class TestPostInitConstraint:

    @pytest.mark.unit
    def test_surrogate_with_finite_diff_raises(self) -> None:
        with pytest.raises(ValueError):
            DerivativeReqs(needs_surrogate=True, provider_kind="finite_diff")

    @pytest.mark.unit
    def test_error_message_is_semantically_meaningful(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            DerivativeReqs(needs_surrogate=True, provider_kind="finite_diff")
        msg = str(excinfo.value)

        assert "needs_surrogate" in msg
        assert "provider_kind" in msg
        assert "autograd" in msg

        assert re.search(r"requires|must|needs", msg, re.IGNORECASE), (
            f"Error message '{msg}' lacks semantic connector word "
            "(requires/must/needs); cheat-proof check failed."
        )


        assert msg.index("needs_surrogate") < msg.index("autograd"), (
            f"Error message '{msg}' has 'autograd' before 'needs_surrogate'; "
            "semantic structure should describe the constraint direction."
        )

    @pytest.mark.unit
    def test_default_does_not_raise(self) -> None:
        DerivativeReqs()

    @pytest.mark.unit
    def test_none_provider_with_surrogate_train_kwargs_raises(self) -> None:
        with pytest.raises(ValueError, match="surrogate_train_kwargs"):
            DerivativeReqs(
                provider_kind="none",
                surrogate_train_kwargs={"max_epochs": 1},
            )

    @pytest.mark.unit
    def test_none_provider_with_surrogate_arch_kwargs_raises(self) -> None:
        with pytest.raises(ValueError, match="surrogate_arch_kwargs"):
            DerivativeReqs(
                provider_kind="none",
                surrogate_arch_kwargs={"hidden_layers": (16,)},
            )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("needs_surrogate", "provider_kind"),
        [
            (False, "finite_diff"),
            (False, "autograd"),
            (True, "autograd"),
        ],
    )
    def test_valid_combinations_do_not_raise(
        self, needs_surrogate: bool, provider_kind: str
    ) -> None:
        DerivativeReqs(
            needs_surrogate=needs_surrogate,
            provider_kind=provider_kind,
        )







class TestReplacePath:

    @pytest.mark.unit
    def test_replace_only_surrogate_flag_raises(self) -> None:
        base = DerivativeReqs()
        with pytest.raises(ValueError):
            replace(base, needs_surrogate=True)

    @pytest.mark.unit
    def test_replace_both_fields_succeeds(self) -> None:
        base = DerivativeReqs()
        result = replace(base, provider_kind="autograd", needs_surrogate=True)
        assert result.provider_kind == "autograd"
        assert result.needs_surrogate is True

    @pytest.mark.unit
    def test_replace_on_autograd_base_succeeds(self) -> None:
        base = DerivativeReqs(provider_kind="autograd")
        result = replace(base, needs_surrogate=True)
        assert result.provider_kind == "autograd"
        assert result.needs_surrogate is True

    @pytest.mark.unit
    def test_replace_demoting_provider_kind_to_finite_diff_raises(self) -> None:
        base = DerivativeReqs(provider_kind="autograd", needs_surrogate=True)
        with pytest.raises(ValueError):
            replace(base, provider_kind="finite_diff")







class TestProviderKindStrictness:

    @pytest.mark.unit
    def test_invalid_string_with_surrogate_still_raises(self) -> None:
        with pytest.raises(ValueError):

            DerivativeReqs(
                provider_kind="invalid",
                needs_surrogate=True,
            )







class TestSurrogateModelField:

    @pytest.mark.unit
    def test_accepts_none(self) -> None:
        req = DerivativeReqs(surrogate_model=None)
        assert req.surrogate_model is None

    @pytest.mark.unit
    def test_accepts_torch_module(self) -> None:
        model = torch.nn.Linear(1, 1)
        req = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_model=model,
        )
        assert req.surrogate_model is model

    @pytest.mark.unit
    def test_surrogate_train_kwargs_dict(self) -> None:
        kwargs = {"lr": 1e-3, "epochs": 100}
        req = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_train_kwargs=kwargs,
        )
        assert req.surrogate_train_kwargs == kwargs







class TestEquality:

    @pytest.mark.unit
    def test_equal_when_all_fields_match(self) -> None:
        a = DerivativeReqs()
        b = DerivativeReqs()
        assert a == b

    @pytest.mark.unit
    def test_equal_with_explicit_defaults(self) -> None:
        a = DerivativeReqs()
        b = DerivativeReqs(
            provider_kind="finite_diff",
            max_atomic_order=2,
            lhs_order=1,
            needs_surrogate=False,
            surrogate_model=None,
            surrogate_train_kwargs=None,
        )
        assert a == b

    @pytest.mark.unit
    def test_unequal_when_provider_kind_differs(self) -> None:
        a = DerivativeReqs(provider_kind="finite_diff")
        b = DerivativeReqs(provider_kind="autograd")
        assert a != b

    @pytest.mark.unit
    def test_unequal_when_max_atomic_order_differs(self) -> None:
        a = DerivativeReqs(max_atomic_order=2)
        b = DerivativeReqs(max_atomic_order=3)
        assert a != b

    @pytest.mark.unit
    def test_unequal_when_lhs_order_differs(self) -> None:
        a = DerivativeReqs(lhs_order=1)
        b = DerivativeReqs(lhs_order=2)
        assert a != b

    @pytest.mark.unit
    def test_unequal_when_needs_surrogate_differs(self) -> None:
        a = DerivativeReqs(provider_kind="autograd", needs_surrogate=False)
        b = DerivativeReqs(provider_kind="autograd", needs_surrogate=True)
        assert a != b

    @pytest.mark.unit
    def test_unequal_when_surrogate_train_kwargs_differs(self) -> None:
        a = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_train_kwargs={"lr": 1e-3},
        )
        b = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=True,
            surrogate_train_kwargs={"lr": 1e-4},
        )
        assert a != b







class TestTypeContract:

    @pytest.mark.unit
    def test_provider_kind_literal_values(self) -> None:
        hints = get_type_hints(DerivativeReqs)
        provider_kind_type = hints["provider_kind"]

        assert provider_kind_type is not str

        args = get_args(provider_kind_type)
        assert set(args) == {"finite_diff", "autograd", "none"}, (
            f"Expected Literal['finite_diff', 'autograd', 'none'], got "
            f"{provider_kind_type}"
        )







class TestFieldSet:

    @pytest.mark.unit
    def test_field_set_is_exactly_eight(self) -> None:
        names = {f.name for f in dataclasses.fields(DerivativeReqs)}
        assert names == {
            "provider_kind",
            "max_atomic_order",
            "lhs_order",
            "needs_surrogate",
            "surrogate_model",
            "surrogate_train_kwargs",
            "surrogate_arch_kwargs",
            "supported_topologies",
        }







class TestPluginShapes:

    @pytest.mark.unit
    def test_sga_default_shape(self) -> None:
        reqs = DerivativeReqs()
        assert reqs.provider_kind == "finite_diff"
        assert reqs.max_atomic_order == 2
        assert reqs.lhs_order == 1
        assert reqs.needs_surrogate is False
        assert reqs.surrogate_model is None

    @pytest.mark.unit
    def test_sga_use_autograd_shape(self) -> None:
        reqs = DerivativeReqs(provider_kind="autograd", needs_surrogate=False)
        assert reqs.provider_kind == "autograd"
        assert reqs.needs_surrogate is False

    @pytest.mark.unit
    def test_dlga_stage1_shape(self) -> None:
        model = torch.nn.Linear(2, 1)
        reqs = DerivativeReqs(
            provider_kind="autograd",
            max_atomic_order=3,
            lhs_order=1,
            needs_surrogate=True,
            surrogate_model=model,
            surrogate_train_kwargs=None,
        )
        assert reqs.max_atomic_order == 3
        assert reqs.surrogate_model is model
        assert reqs.needs_surrogate is True

    @pytest.mark.unit
    def test_dlga_stage1_with_default_training_kwargs(self) -> None:
        reqs = DerivativeReqs(
            provider_kind="autograd",
            max_atomic_order=3,
            needs_surrogate=True,
            surrogate_model=None,
            surrogate_train_kwargs={"lr": 1e-3, "epochs": 5000, "patience": None},
        )
        assert reqs.surrogate_model is None
        assert reqs.surrogate_train_kwargs is not None
        assert reqs.surrogate_train_kwargs["epochs"] == 5000

    @pytest.mark.unit
    def test_pde_find_shape(self) -> None:
        reqs = DerivativeReqs(
            provider_kind="finite_diff",
            max_atomic_order=3,
            lhs_order=1,
            needs_surrogate=False,
        )
        assert reqs.provider_kind == "finite_diff"
        assert reqs.max_atomic_order == 3







class TestSurrogateModelIndependence:

    @pytest.mark.unit
    def test_model_with_surrogate_false_is_allowed(self) -> None:
        model = torch.nn.Linear(1, 1)
        reqs = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=False,
            surrogate_model=model,
        )
        assert reqs.surrogate_model is model
        assert reqs.needs_surrogate is False

    @pytest.mark.unit
    def test_train_kwargs_with_surrogate_false_is_allowed(self) -> None:
        reqs = DerivativeReqs(
            provider_kind="autograd",
            needs_surrogate=False,
            surrogate_train_kwargs={"lr": 1e-3},
        )
        assert reqs.surrogate_train_kwargs == {"lr": 1e-3}
