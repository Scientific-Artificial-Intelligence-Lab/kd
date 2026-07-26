
from __future__ import annotations

from typing import Any

import pytest

import kd
from kd.api import Model


@pytest.fixture(scope="module")
def tiny_dataset() -> Any:
    return kd.generate_burgers_data(nx=16, nt=8, nu=0.1, seed=0)


def test_default_device_is_none() -> None:
    assert Model().device is None


def test_device_cpu_is_stored() -> None:
    assert Model(device="cpu").device == "cpu"


@pytest.mark.parametrize("bad", [123, "not-a-device"])
def test_invalid_device_raises_value_error_naming_device(bad: object) -> None:
    with pytest.raises(ValueError, match="device"):
        Model(device=bad)


def test_device_is_forwarded_to_platform_builder(
    tiny_dataset: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class _SpyBuilder:
        def __init__(self, dataset: Any, reqs: Any, *, device: Any = None) -> None:
            captured["device"] = device

        def build(self) -> Any:
            return object()

    monkeypatch.setattr(
        "kd.core.platform.builder.PlatformBuilder", _SpyBuilder
    )
    Model(algorithm="sga", device="cpu")._build_components(tiny_dataset)
    assert captured["device"] == "cpu"


def test_device_none_forwards_none_to_builder(
    tiny_dataset: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class _SpyBuilder:
        def __init__(self, dataset: Any, reqs: Any, *, device: Any = None) -> None:
            captured["device"] = device

        def build(self) -> Any:
            return object()

    monkeypatch.setattr(
        "kd.core.platform.builder.PlatformBuilder", _SpyBuilder
    )
    Model(algorithm="sga")._build_components(tiny_dataset)
    assert captured["device"] is None


def test_device_does_not_leak_into_extra_kwargs() -> None:


    model = Model(algorithm="sga", device="cpu")
    assert "device" not in getattr(model, "_extra_kwargs", {})
