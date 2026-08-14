
from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from kd.api import Model
from kd.search.sga import SGAConfig


def _sga_config(**kwargs: Any) -> SGAConfig:
    model = Model(algorithm="sga", verbose=False, **kwargs)
    return model._build_config()


def _as_comparable(config: SGAConfig) -> dict[str, Any]:
    out = dataclasses.asdict(config)
    out["field_model"] = None if out["field_model"] is None else "<set>"
    return out





_CORPUS: list[tuple[str, dict[str, Any], SGAConfig]] = [
    ("bare", {}, SGAConfig()),
    (
        "facade_knobs",
        {"population": 30, "depth": 3, "width": 6, "aic_ratio": 2.0},
        SGAConfig(num=30, depth=3, width=6, aic_ratio=2.0),
    ),
    (
        "forwarded_scalars",
        {"lam": 0.5, "d_tol": 0.1, "maxit": 30, "str_iters": 7},
        SGAConfig(lam=0.5, d_tol=0.1, maxit=30, str_iters=7),
    ),




    ("norm_order", {"normalize": 0}, SGAConfig(normalize=0)),
    ("literal_field", {"dedup_mode": "dual"}, SGAConfig(dedup_mode="dual")),
    (
        "probabilities",
        {"p_var": 0.6, "p_mute": 0.4, "p_cro": 0.3, "p_rep": 0.2},
        SGAConfig(p_var=0.6, p_mute=0.4, p_cro=0.3, p_rep=0.2),
    ),
    (
        "autograd_budget",
        {
            "derivatives": "autograd",
            "autograd_train_epochs": 8,
            "autograd_train_lr": 0.005,
            "autograd_train_patience": 5,
            "autograd_train_val_ratio": 0.25,
        },
        SGAConfig(
            use_autograd=True,
            autograd_train_epochs=8,
            autograd_train_lr=0.005,
            autograd_train_patience=5,
            autograd_train_val_ratio=0.25,
        ),
    ),
    ("seed", {"seed": 7}, SGAConfig(seed=7)),


    ("generations_is_not_a_config_field", {"generations": 3}, SGAConfig()),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "call_kwargs", "expected"),
    _CORPUS,
    ids=[row[0] for row in _CORPUS],
)
def test_sga_facade_call_resolves_to_the_same_config(
    label: str, call_kwargs: dict[str, Any], expected: SGAConfig
) -> None:
    del label
    assert _as_comparable(_sga_config(**call_kwargs)) == _as_comparable(expected)


@pytest.mark.unit
def test_sga_forwards_every_config_field_that_is_not_facade_mapped() -> None:
    facade_mapped = {"num", "depth", "width", "aic_ratio", "seed", "use_autograd"}
    forwardable = {
        field.name for field in dataclasses.fields(SGAConfig)
    } - facade_mapped
    assert forwardable == {
        "p_var",
        "p_mute",
        "p_cro",
        "p_rep",
        "lam",
        "d_tol",
        "maxit",
        "str_iters",
        "normalize",
        "dedup_mode",
        "field_model",
        "autograd_train_epochs",
        "autograd_train_lr",
        "autograd_train_patience",
        "autograd_train_val_ratio",
    }


@pytest.mark.unit
def test_sga_forwards_a_live_field_model_object_untouched() -> None:
    from kd.models.field_model import FieldModel

    field_model = FieldModel(coord_names=["x", "t"], field_names=["u"])
    config = _sga_config(derivatives="autograd", field_model=field_model)
    assert config.field_model is field_model


@pytest.mark.unit
@pytest.mark.parametrize(
    ("call_kwargs", "fragment"),
    [
        ({"num": 20}, "population"),
        ({"use_autograd": True}, "derivatives"),
        ({"random_unknown": 1}, "random_unknown"),
    ],
    ids=["num_collides_with_population", "use_autograd_collides", "unknown_name"],
)
def test_sga_rejection_messages_keep_naming_the_remedy(
    call_kwargs: dict[str, Any], fragment: str
) -> None:
    with pytest.raises(TypeError) as exc_info:
        Model(algorithm="sga", verbose=False, **call_kwargs)
    assert fragment in str(exc_info.value)
