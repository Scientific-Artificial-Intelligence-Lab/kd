
from __future__ import annotations

from typing import Any

import pytest
import torch

from kd.api import _SUPPORTED_ALGORITHMS, Model
from kd.search.discover import DiscoverConfig, DISCOVERPlugin
from kd.search.dlga import DLGAConfig
from kd.search.sga import SGAConfig






@pytest.mark.unit
def test_discover_in_supported_algorithms() -> None:
    assert "discover" in _SUPPORTED_ALGORITHMS


@pytest.mark.unit
def test_discover_init_default_config() -> None:
    m = Model(algorithm="discover", verbose=False)


    assert m._config_override is None


@pytest.mark.unit
def test_discover_default_generations_and_verbose() -> None:
    m = Model(algorithm="discover")
    assert m.generations == 50
    assert m.verbose is True







@pytest.mark.unit
def test_discover_with_explicit_config() -> None:
    cfg = DiscoverConfig(seed=7)
    m = Model(algorithm="discover", verbose=False, config=cfg)
    assert isinstance(m._config_override, DiscoverConfig)
    assert m._config_override.seed == 7


@pytest.mark.unit
def test_discover_config_deep_copy() -> None:
    cfg = DiscoverConfig(seed=3, repeat_tokens=["add"])
    m = Model(algorithm="discover", verbose=False, config=cfg)


    cfg.repeat_tokens.append("mul")
    snapshot = m._config_override
    assert snapshot is not None
    assert snapshot.repeat_tokens == ["add"]


@pytest.mark.unit
def test_discover_config_not_aliased() -> None:
    cfg = DiscoverConfig(seed=11)
    m = Model(algorithm="discover", verbose=False, config=cfg)
    assert m._config_override is not cfg


@pytest.mark.unit
def test_discover_rejects_sga_config() -> None:
    with pytest.raises(TypeError) as exc_info:
        m = Model(algorithm="discover", verbose=False, config=SGAConfig())

        m._build_plugin()
    msg = str(exc_info.value)
    assert "discover" in msg.lower()

    assert "DiscoverConfig" in msg or "SGAConfig" in msg


@pytest.mark.unit
def test_discover_rejects_dlga_config() -> None:
    with pytest.raises(TypeError) as exc_info:
        m = Model(algorithm="discover", verbose=False, config=DLGAConfig())
        m._build_plugin()
    msg = str(exc_info.value)
    assert "discover" in msg.lower()
    assert "DiscoverConfig" in msg or "DLGAConfig" in msg







@pytest.mark.unit
def test_discover_rejects_surrogate_model() -> None:
    cheap = torch.nn.Linear(1, 1)
    with pytest.raises(TypeError) as exc_info:
        Model(algorithm="discover", verbose=False, surrogate_model=cheap)
    msg = str(exc_info.value)
    assert "surrogate_model" in msg
    assert "discover" in msg.lower()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwarg_name", "kwarg_value"),
    [
        ("population", 20),
        ("depth", 3),
        ("width", 5),
        ("aic_ratio", 0.5),
        ("derivatives", "autograd"),
    ],
)
def test_discover_rejects_sga_only_facade_params(
    kwarg_name: str, kwarg_value: Any
) -> None:
    kwargs: dict[str, Any] = {
        "algorithm": "discover",
        "verbose": False,
        kwarg_name: kwarg_value,
    }
    with pytest.raises(TypeError) as exc_info:
        Model(**kwargs)
    msg = str(exc_info.value)
    assert "discover" in msg.lower()
    assert kwarg_name in msg


@pytest.mark.unit
def test_discover_rejects_unknown_kwarg() -> None:
    with pytest.raises(TypeError) as exc_info:



        Model(algorithm="discover", verbose=False, lam=0.1)
    msg = str(exc_info.value)
    assert "discover" in msg.lower()
    assert "lam" in msg


@pytest.mark.unit
def test_discover_rejects_truly_unknown_kwarg() -> None:
    with pytest.raises(TypeError) as exc_info:
        Model(algorithm="discover", verbose=False, random_unknown=1)
    msg = str(exc_info.value)


    assert "random_unknown" in msg







@pytest.mark.unit
def test_discover_build_plugin_returns_discover_plugin_instance() -> None:
    m = Model(algorithm="discover", verbose=False)
    plugin, batch_size = m._build_plugin()
    assert isinstance(plugin, DISCOVERPlugin)



    assert isinstance(batch_size, int)
    assert batch_size > 0


@pytest.mark.unit
def test_discover_build_plugin_uses_default_config_when_no_override() -> None:
    m = Model(algorithm="discover", verbose=False)
    plugin, _ = m._build_plugin()




    assert isinstance(plugin._config, DiscoverConfig)


@pytest.mark.unit
def test_discover_build_plugin_threads_user_config() -> None:
    cfg = DiscoverConfig(seed=42)
    m = Model(algorithm="discover", verbose=False, config=cfg)
    plugin, _ = m._build_plugin()
    assert isinstance(plugin, DISCOVERPlugin)
    assert isinstance(plugin._config, DiscoverConfig)
    assert plugin._config.seed == 42


@pytest.mark.unit
def test_discover_facade_seed_propagates_to_config() -> None:
    m = Model(algorithm="discover", verbose=False, seed=42)
    plugin, _ = m._build_plugin()
    assert isinstance(plugin, DISCOVERPlugin)
    assert plugin._config.seed == 42


@pytest.mark.unit
def test_discover_repr_unfit_mentions_algorithm() -> None:
    m = Model(algorithm="discover", verbose=False)
    rep = repr(m)
    assert "discover" in rep.lower()
    assert "fitted=False" in rep

















@pytest.mark.unit
def test_discover_rejects_early_stop_min_mode() -> None:
    from kd.search.callbacks import EarlyStoppingCallback

    cb = EarlyStoppingCallback(mode="min", patience=10)
    with pytest.raises(TypeError) as exc_info:
        Model(algorithm="discover", verbose=False, callbacks=[cb])
    msg = str(exc_info.value)
    assert "discover" in msg.lower(), (
        f"Error message must mention 'discover' so the user can identify "
        f"the incompatible algorithm; got: {msg!r}"
    )


    assert "mode='max'" in msg or 'mode="max"' in msg or "mode=max" in msg, (
        f"Error message must suggest the fix (mode='max'); got: {msg!r}"
    )


@pytest.mark.unit
def test_discover_accepts_early_stop_max_mode() -> None:
    from kd.search.callbacks import EarlyStoppingCallback

    cb = EarlyStoppingCallback(mode="max", patience=10)

    m = Model(algorithm="discover", verbose=False, callbacks=[cb])
    assert m.algorithm == "discover"


@pytest.mark.unit
def test_sga_accepts_early_stop_min_mode() -> None:
    from kd.search.callbacks import EarlyStoppingCallback

    cb = EarlyStoppingCallback(mode="min", patience=10)
    m = Model(algorithm="sga", verbose=False, callbacks=[cb])
    assert m.algorithm == "sga"







@pytest.mark.unit
def test_dlga_accepts_early_stop_min_mode() -> None:
    from kd.search.callbacks import EarlyStoppingCallback

    cb = EarlyStoppingCallback(mode="min", patience=10)
    m = Model(algorithm="dlga", verbose=False, callbacks=[cb])
    assert m.algorithm == "dlga"







@pytest.mark.unit
def test_discover_rejects_early_stop_default_mode() -> None:
    from kd.search.callbacks import EarlyStoppingCallback


    cb = EarlyStoppingCallback(patience=10)
    with pytest.raises(TypeError) as exc_info:
        Model(algorithm="discover", verbose=False, callbacks=[cb])
    msg = str(exc_info.value)
    assert "discover" in msg.lower(), (
        f"Error message must mention 'discover'; got: {msg!r}"
    )














@pytest.mark.unit
def test_sga_rejects_early_stop_max_mode() -> None:
    from kd.search.callbacks import EarlyStoppingCallback

    cb = EarlyStoppingCallback(mode="max", patience=10)
    with pytest.raises(TypeError) as exc_info:
        Model(algorithm="sga", verbose=False, callbacks=[cb])
    msg = str(exc_info.value)
    assert "sga" in msg.lower(), (
        f"Error message must mention 'sga' so the user can identify the "
        f"incompatible algorithm; got: {msg!r}"
    )
    assert "mode='min'" in msg or 'mode="min"' in msg or "mode=min" in msg, (
        f"Error message must suggest the fix (mode='min'); got: {msg!r}"
    )


@pytest.mark.unit
def test_dlga_rejects_early_stop_max_mode() -> None:
    from kd.search.callbacks import EarlyStoppingCallback

    cb = EarlyStoppingCallback(mode="max", patience=10)
    with pytest.raises(TypeError) as exc_info:
        Model(algorithm="dlga", verbose=False, callbacks=[cb])
    msg = str(exc_info.value)
    assert "dlga" in msg.lower(), f"Error message must mention 'dlga'; got: {msg!r}"
    assert "mode='min'" in msg or 'mode="min"' in msg or "mode=min" in msg, (
        f"Error message must suggest the fix (mode='min'); got: {msg!r}"
    )


@pytest.mark.unit
def test_early_stop_mode_map_covers_all_supported_algorithms() -> None:
    from kd.api import _EARLY_STOP_MODE_BY_ALGORITHM, _SUPPORTED_ALGORITHMS

    missing = [
        a for a in _SUPPORTED_ALGORITHMS if a not in _EARLY_STOP_MODE_BY_ALGORITHM
    ]
    assert not missing, f"algorithms without an early-stop mode: {missing}"







@pytest.mark.unit
def test_discover_rejects_early_stop_min_mode_in_mixed_list() -> None:
    from kd.search.callbacks import EarlyStoppingCallback, LoggingCallback

    logger = LoggingCallback(every_n=1)
    early_stop = EarlyStoppingCallback(mode="min", patience=10)
    with pytest.raises(TypeError) as exc_info:

        Model(
            algorithm="discover",
            verbose=False,
            callbacks=[logger, early_stop],
        )
    msg = str(exc_info.value)
    assert "discover" in msg.lower(), (
        f"Error message must mention 'discover' even when the offending "
        f"callback is not at index 0; got: {msg!r}"
    )







@pytest.mark.unit
@pytest.mark.parametrize(
    ("algo", "expected"),
    [
        ("sga", "AIC"),
        ("dlga", "DLGA fitness"),
        ("discover", "reward"),
        ("unknown", "Score"),
        ("", "Score"),
    ],
)
def test_api_score_label_matches_viz_helper(algo: str, expected: str) -> None:
    from kd.api import _score_label as api_score_label
    from kd.viz._labels import score_label as viz_score_label


    class _Stub:
        config = {"algorithm": algo}

    api_result = api_score_label(_Stub())
    viz_result = viz_score_label(algo)

    assert api_result == expected, (
        f"api._score_label({algo!r}) returned {api_result!r}, expected {expected!r}"
    )
    assert viz_result == expected, (
        f"viz.score_label({algo!r}) returned {viz_result!r}, expected {expected!r}"
    )
    assert api_result == viz_result, (
        f"api and viz helpers diverged for algo={algo!r}: "
        f"api={api_result!r}, viz={viz_result!r}"
    )







@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_name",
    ["DISCOVER", "Discover", "SGA", "Sga", "DLGA", "Dlga", "pysr", ""],
)
def test_model_rejects_unsupported_algorithm_at_init(bad_name: str) -> None:
    from kd import Model

    with pytest.raises(NotImplementedError) as exc:
        Model(algorithm=bad_name)
    msg = str(exc.value)
    assert "Supported algorithms" in msg, (
        f"NotImplementedError must hint at the canonical whitelist; got: {msg!r}"
    )


@pytest.mark.unit
def test_model_rejects_uppercase_discover_before_early_stop_guard() -> None:
    from kd import Model
    from kd.search.callbacks import EarlyStoppingCallback

    with pytest.raises(NotImplementedError) as exc:
        Model(
            algorithm="DISCOVER",
            callbacks=[EarlyStoppingCallback(mode="min", patience=10)],
        )
    assert "Supported algorithms" in str(exc.value)


@pytest.mark.unit
def test_model_accepts_supported_lowercase_algorithm() -> None:
    from kd import Model


    Model(algorithm="sga")
    Model(algorithm="dlga")
    Model(algorithm="discover")














@pytest.mark.unit
def test_facade_rejects_pinn_config() -> None:
    from kd.search.discover.config import PINNConfig

    with pytest.raises(TypeError, match="pinn"):
        Model(algorithm="discover", config=DiscoverConfig(pinn=PINNConfig()))


@pytest.mark.unit
def test_facade_warns_on_changed_n_iterations() -> None:
    with pytest.warns(UserWarning, match="n_iterations"):
        m = Model(algorithm="discover", config=DiscoverConfig(n_iterations=999))

    assert m.algorithm == "discover"


@pytest.mark.unit
def test_facade_warns_on_stability_selection() -> None:
    with pytest.warns(UserWarning, match="stability"):
        m = Model(algorithm="discover", config=DiscoverConfig(stability_selection=5))
    assert m.algorithm == "discover"


@pytest.mark.unit
def test_facade_warns_on_stability_queue_capacity() -> None:
    with pytest.warns(UserWarning, match="stability"):
        m = Model(
            algorithm="discover",
            config=DiscoverConfig(stability_queue_capacity=128),
        )
    assert m.algorithm == "discover"


@pytest.mark.unit
def test_facade_preset_is_usable_with_only_a_warning() -> None:
    with pytest.warns(UserWarning, match="n_iterations"):
        m = Model(
            algorithm="discover",
            generations=10,
            config=DiscoverConfig.burgers_preset(),
        )
    assert m.algorithm == "discover"


@pytest.mark.unit
def test_facade_accepts_facade_effective_config() -> None:
    import warnings


    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Model(algorithm="discover", config=DiscoverConfig(seed=7, batch_size=4))


@pytest.mark.unit
def test_facade_accepts_no_config() -> None:
    Model(algorithm="discover", generations=10)
