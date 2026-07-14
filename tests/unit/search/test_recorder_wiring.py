
from __future__ import annotations

from typing import Any

import pytest


from kd.search.recorder import log_whitelisted_metrics


class SpyRecorder:

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def log(self, key: str, value: Any) -> None:
        self.calls.append((key, value))





@pytest.mark.smoke
def test_helper_is_importable_and_callable() -> None:
    assert callable(log_whitelisted_metrics)





@pytest.mark.unit
def test_none_recorder_is_noop_strict() -> None:


    log_whitelisted_metrics(
        None, ("a", "missing"), {"a": 1}, skip_missing=False
    )


@pytest.mark.unit
def test_none_recorder_is_noop_lenient() -> None:
    log_whitelisted_metrics(None, ("a", "b"), {}, skip_missing=True)





@pytest.mark.unit
def test_strict_missing_key_raises_keyerror() -> None:
    spy = SpyRecorder()
    with pytest.raises(KeyError):
        log_whitelisted_metrics(
            spy, ("present", "absent"), {"present": 1.0}, skip_missing=False
        )


@pytest.mark.unit
def test_strict_default_is_strict() -> None:

    spy = SpyRecorder()
    with pytest.raises(KeyError):
        log_whitelisted_metrics(spy, ("x",), {})


@pytest.mark.unit
def test_strict_logs_all_present_in_whitelist_order() -> None:
    spy = SpyRecorder()
    whitelist = ("gen_best_aic", "n_valid", "n_unique")

    metrics: dict[str, float | int] = {
        "n_unique": 5,
        "gen_best_aic": 1.5,
        "n_valid": 3,
    }
    log_whitelisted_metrics(spy, whitelist, metrics, skip_missing=False)
    assert [key for key, _ in spy.calls] == list(whitelist)





@pytest.mark.unit
def test_lenient_skips_missing_keeps_present_in_order() -> None:
    spy = SpyRecorder()
    whitelist = ("pg_loss", "absent1", "reward", "absent2", "grad_norm")
    metrics = {"pg_loss": 0.1, "reward": 2.0, "grad_norm": 9.0}
    log_whitelisted_metrics(spy, whitelist, metrics, skip_missing=True)
    assert [key for key, _ in spy.calls] == ["pg_loss", "reward", "grad_norm"]


@pytest.mark.unit
def test_lenient_empty_metrics_logs_nothing() -> None:
    spy = SpyRecorder()
    log_whitelisted_metrics(spy, ("a", "b", "c"), {}, skip_missing=True)
    assert spy.calls == []





@pytest.mark.unit
def test_only_whitelisted_keys_logged_extras_ignored() -> None:
    spy = SpyRecorder()
    whitelist = ("b", "a")
    metrics = {"a": 1, "b": 2, "c_extra": 3, "d_extra": 4}
    log_whitelisted_metrics(spy, whitelist, metrics, skip_missing=False)
    logged_keys = [key for key, _ in spy.calls]
    assert logged_keys == ["b", "a"]
    assert "c_extra" not in logged_keys
    assert "d_extra" not in logged_keys


@pytest.mark.unit
def test_whitelist_accepts_generator_iterable() -> None:
    spy = SpyRecorder()
    metrics = {"x": 1, "y": 2}
    log_whitelisted_metrics(
        spy, (k for k in ("x", "y")), metrics, skip_missing=False
    )
    assert [key for key, _ in spy.calls] == ["x", "y"]


@pytest.mark.unit
def test_empty_whitelist_logs_nothing() -> None:
    spy = SpyRecorder()
    log_whitelisted_metrics(spy, (), {"a": 1}, skip_missing=False)
    assert spy.calls == []





@pytest.mark.unit
def test_values_forwarded_unmodified_scalar_types() -> None:
    spy = SpyRecorder()
    whitelist = ("i", "f")
    metrics: dict[str, int | float] = {"i": 7, "f": 3.25}
    log_whitelisted_metrics(spy, whitelist, metrics, skip_missing=False)
    logged = dict(spy.calls)
    assert logged["i"] == 7
    assert isinstance(logged["i"], int)
    assert logged["f"] == pytest.approx(3.25)
    assert isinstance(logged["f"], float)


@pytest.mark.unit
def test_list_value_forwarded_by_identity() -> None:


    spy = SpyRecorder()
    pareto = [1.0, 2.0, None, 4.0]
    metrics = {"pareto_loss": pareto}
    log_whitelisted_metrics(
        spy, ("pareto_loss",), metrics, skip_missing=False
    )
    (key, value), = spy.calls
    assert key == "pareto_loss"
    assert value is pareto


@pytest.mark.unit
def test_mixed_scalar_and_list_values_preserve_order_and_content() -> None:


    spy = SpyRecorder()
    whitelist = (
        "pareto_complexity",
        "pareto_loss",
        "selected_complexity",
        "selected_loss",
    )
    metrics: dict[str, Any] = {
        "pareto_complexity": [1, 2, 3],
        "pareto_loss": [0.1, 0.2, 0.3],
        "selected_complexity": 2,
        "selected_loss": 0.2,
    }
    log_whitelisted_metrics(spy, whitelist, metrics, skip_missing=False)
    assert [key for key, _ in spy.calls] == list(whitelist)
    logged = dict(spy.calls)
    assert logged["pareto_complexity"] == [1, 2, 3]
    assert logged["selected_complexity"] == 2
