
from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pytest

from kd.api import Model
from kd.harness import PlanEntry
from kd.search.discover.config import PINNConfig
from kd.search.pysindy import PySINDyConfig


def _build_pysindy_with_dict_config() -> None:
    model = Model(
        algorithm="pysindy",
        config={"threshold": 0.2},
        verbose=False,
    )
    model._build_pysindy_config()


def _build_bare_eqgpt() -> None:
    Model(algorithm="eqgpt", verbose=False)._build_eqgpt_config()


def _reserved_plan_key() -> None:
    PlanEntry(
        instrument="sga",
        dataset_ref="burgers",
        seed=0,
        model_kwargs={"seed": 3},
    )


_REJECTIONS: list[
    tuple[str, Callable[[], object], type[Exception], tuple[str, ...]]
] = [
    (
        "unknown_name",
        lambda: Model(
            algorithm="pysindy", verbose=False, thresold_typo=0.2
        ),
        TypeError,
        ("pysindy", "thresold_typo", "PySINDyConfig", "instrument_schemas"),
    ),
    (
        "field_owned_by_two_other_algorithms",
        lambda: Model(algorithm="pysindy", verbose=False, max_length=12),
        TypeError,
        ("max_length", "pysindy", "PySINDyConfig", "discover", "eqgpt"),
    ),
    (
        "sga_facade_parameter_on_non_sga",
        lambda: Model(algorithm="pysindy", verbose=False, population=12),
        TypeError,
        ("pysindy", "population", "PySINDyConfig"),
    ),
    (
        "closed_eqgpt_asset_path_receives_json_string",
        lambda: Model(
            algorithm="eqgpt",
            verbose=False,
            sparsity_alpha=0.02,
            weights_path="/tmp/weights.pt",
        ),
        TypeError,
        ("eqgpt", "weights_path", "Path", "str", "KD_EQGPT_ASSET_DIR"),
    ),

    (
        "sga_wrong_json_scalar_kind",
        lambda: Model(algorithm="sga", verbose=False, lam="0.5"),
        TypeError,
        ("sga", "lam", "float", "str"),
    ),

    (
        "sga_norm_order_rejects_bool",
        lambda: Model(algorithm="sga", verbose=False, normalize=True),
        TypeError,
        ("sga", "normalize", "int", "bool"),
    ),
    (
        "container_field_receives_scalar",
        lambda: Model(algorithm="pysindy", verbose=False, terms="u"),
        TypeError,
        ("pysindy", "terms", "tuple", "str"),
    ),
    (
        "container_element_has_wrong_kind",
        lambda: Model(
            algorithm="eqgpt",
            verbose=False,
            sparsity_alpha=0.02,
            masked_tokens=[True],
        ),
        TypeError,
        ("eqgpt", "masked_tokens", "0", "int", "bool"),
    ),

    (
        "sga_literal_non_member",
        lambda: Model(algorithm="sga", verbose=False, dedup_mode="dedupe"),
        ValueError,
        ("sga", "dedup_mode", "dedupe", "pre_prune"),
    ),
    (
        "closed_non_nullable_field_receives_none",
        lambda: Model(algorithm="discover", verbose=False, library=None),
        TypeError,
        ("discover", "library", "nullable", "config="),
    ),
    (
        "config_object_has_wrong_type",
        _build_pysindy_with_dict_config,
        TypeError,
        ("pysindy", "PySINDyConfig", "dict"),
    ),
    (
        "config_and_field_are_both_passed",
        lambda: Model(
            algorithm="pysindy",
            config=PySINDyConfig(),
            verbose=False,
            threshold=0.2,
        ),
        ValueError,
        ("config=", "threshold", "generations"),
    ),
    (
        "facade_owned_config_name",
        lambda: Model(algorithm="pysr", verbose=False, niterations=100),
        TypeError,
        ("pysr", "niterations", "generations"),
    ),
    (
        "eqgpt_required_field_missing",
        _build_bare_eqgpt,
        TypeError,
        ("EqGPTConfig", "sparsity_alpha", "sparsity_alpha=", "config="),
    ),





    (
        "bogus_algorithm_with_non_sga_kwarg",
        lambda: Model(algorithm="bogus", verbose=False, threshold=1),
        NotImplementedError,
        ("bogus", "Supported algorithms"),
    ),



    (
        "bogus_algorithm_with_config_and_unknown_kwarg",
        lambda: Model(
            algorithm="bogus", verbose=False, config=PySINDyConfig(), typo=1
        ),
        ValueError,
        ("config=", "typo"),
    ),
    (
        "discover_pinn_via_kwargs",
        lambda: Model(
            algorithm="discover", verbose=False, pinn=PINNConfig()
        ),
        TypeError,
        ("discover", "pinn", "MODE2"),
    ),
    (
        "plan_model_kwargs_reserved_key",
        _reserved_plan_key,
        ValueError,
        ("seed",),
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "invoke", "exception_type", "fragments"),
    _REJECTIONS,
    ids=[row[0] for row in _REJECTIONS],
)
def test_ordered_rejection_table(
    label: str,
    invoke: Callable[[], object],
    exception_type: type[Exception],
    fragments: tuple[str, ...],
) -> None:
    del label
    with pytest.raises(exception_type) as exc_info:
        invoke()
    message = str(exc_info.value)
    for fragment in fragments:
        assert fragment in message


@pytest.mark.unit
def test_json_list_is_converted_to_the_declared_container() -> None:
    config = Model(
        algorithm="pysindy", verbose=False, terms=["u", "u_x"]
    )._build_pysindy_config()
    assert config.terms == ("u", "u_x")
    assert type(config.terms) is tuple


@pytest.mark.unit
def test_json_dict_is_snapshotted_at_model_construction() -> None:
    optimizer_kwargs = {"alpha": 1, "nested": {"values": [2]}}
    model = Model(
        algorithm="pysindy",
        verbose=False,
        extra_optimizer_kwargs=optimizer_kwargs,
    )

    optimizer_kwargs["alpha"] = 99
    optimizer_kwargs["nested"]["values"].append(3)

    config = model._build_pysindy_config()
    assert config.extra_optimizer_kwargs == {
        "alpha": 1,
        "nested": {"values": [2]},
    }


@pytest.mark.unit
def test_non_json_python_scalar_is_forwarded_without_facade_validation() -> None:
    value = np.float64(0.25)
    config = Model(algorithm="sga", verbose=False, lam=value)._build_config()
    assert config.lam is value


@pytest.mark.unit
def test_discover_kwargs_warning_waits_for_successful_config_construction() -> None:
    model = Model(algorithm="discover", verbose=False, n_iterations=5)
    with pytest.warns(UserWarning) as caught:
        config = model._build_discover_config()
    assert config.n_iterations == 5
    assert "n_iterations" in str(caught[0].message)


@pytest.mark.unit
def test_invalid_discover_kwargs_do_not_warn_before_the_config_error() -> None:
    model = Model(algorithm="discover", verbose=False, stability_selection=11)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as exc_info:
            model._build_discover_config()
    assert "stability_queue_capacity" in str(exc_info.value)
    assert caught == []


@pytest.mark.unit
def test_sga_autograd_rejects_json_field_model_value() -> None:
    with pytest.raises(TypeError) as exc_info:
        Model(
            algorithm="sga",
            derivatives="autograd",
            field_model="not-a-model",
            verbose=False,
        )
    message = str(exc_info.value)
    for fragment in ("sga", "field_model", "FieldModel", "str"):
        assert fragment in message
