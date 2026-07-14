
from __future__ import annotations

import pytest

from kd.api import Model
from kd.search.llm4ed.config import Llm4edConfig
from kd.search.llm4ed.plugin import Llm4edPlugin

from ._plugin_helpers import GOOD, FakeProvider, heat_dataset, make_config

pytestmark = pytest.mark.unit


def test_llm4ed_registered_in_facade() -> None:
    from kd.api import _PLUGIN_CLASS_BY_ALGORITHM

    assert _PLUGIN_CLASS_BY_ALGORITHM.get("llm4ed") is Llm4edPlugin


def test_registered_llm4ed_plugin_is_sparse_reward_max() -> None:
    from kd.api import _PLUGIN_CLASS_BY_ALGORITHM

    cls = _PLUGIN_CLASS_BY_ALGORITHM.get("llm4ed")
    assert cls is not None
    assert cls.score_direction == "max"
    assert cls.score_kind == "LLM4ED sparse reward"


def test_model_accepts_llm4ed_with_default_config() -> None:
    Model(algorithm="llm4ed")


def test_model_accepts_llm4ed_with_config() -> None:
    Model(algorithm="llm4ed", config=Llm4edConfig())


def test_model_rejects_sga_only_params_on_llm4ed() -> None:
    with pytest.raises(TypeError):
        Model(algorithm="llm4ed", population=20)


def test_model_rejects_provider_on_non_llm4ed() -> None:
    with pytest.raises(TypeError):
        Model(algorithm="sga", provider=FakeProvider(GOOD))


def test_model_rejects_mismatched_config_type() -> None:
    from kd.search.sga.config import SGAConfig

    with pytest.raises(TypeError):
        Model(algorithm="llm4ed", config=SGAConfig(), provider=FakeProvider(GOOD)).fit(
            heat_dataset()
        )


def test_model_llm4ed_fit_recovers_u_xx() -> None:
    model = Model(
        algorithm="llm4ed",
        generations=2,
        config=make_config(),
        provider=FakeProvider(GOOD),
    )
    model.fit(heat_dataset())

    assert model.best_expr_ == "u_xx"
    assert model.best_score_ == pytest.approx(0.989, abs=5e-3)
    assert model.result_.iterations == 2


def test_model_llm4ed_fit_then_resume_recovers_u_xx(tmp_path) -> None:
    first = Model(
        algorithm="llm4ed",
        generations=2,
        config=make_config(),
        provider=FakeProvider(GOOD),
        checkpoint_dir=tmp_path,
    )
    first.fit(heat_dataset())
    checkpoint = tmp_path / "checkpoint_final.pt"
    assert checkpoint.is_file(), "facade did not write a final checkpoint"

    resumed = Model(
        algorithm="llm4ed",
        generations=2,
        config=make_config(),
        provider=FakeProvider(GOOD),
    )
    resumed.fit(heat_dataset(), resume_from=checkpoint)
    assert resumed.best_expr_ == "u_xx"
    assert resumed.best_score_ == pytest.approx(0.989, abs=5e-3)


def test_model_llm4ed_uses_injected_provider() -> None:
    provider = FakeProvider(GOOD)
    model = Model(
        algorithm="llm4ed",
        generations=1,
        config=make_config(),
        provider=provider,
    )
    model.fit(heat_dataset())


    assert provider.requests, "injected provider was never called"

    assert provider.requests[0].params.temperature == pytest.approx(0.8)
