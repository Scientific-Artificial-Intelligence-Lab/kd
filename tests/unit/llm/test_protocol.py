
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys

import pytest

from kd.llm import (
    BudgetedProvider,
    LLMBackendError,
    LLMBudgetExhausted,
    LLMError,
    LLMParams,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMTapeMismatchError,
    LLMUsage,
    OpenAICompatProvider,
    TapeRecordingProvider,
    TapeReplayProvider,
)
from tests.unit.llm._fakes import FakeProvider






@pytest.mark.smoke
def test_all_providers_conform_to_protocol(tmp_path: object) -> None:
    tape = os.path.join(str(tmp_path), "tape.jsonl")
    inner = OpenAICompatProvider(model="gpt-x", base_url="http://localhost:1/v1")
    assert isinstance(inner, LLMProvider)
    assert isinstance(BudgetedProvider(inner, max_calls=3), LLMProvider)
    assert isinstance(TapeRecordingProvider(inner, path=tape), LLMProvider)
    assert isinstance(TapeReplayProvider(path=tape), LLMProvider)
    assert isinstance(FakeProvider(), LLMProvider)







def test_import_kd_llm_does_not_eagerly_import_openai() -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import kd; import kd.llm; "
            "kd.llm.OpenAICompatProvider(model='m', base_url='http://x/v1'); "
            "assert 'openai' not in sys.modules, 'openai imported eagerly'; "
            "print('ok')",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout







class TestErrorTree:
    def test_base_is_runtime_error(self) -> None:
        assert issubclass(LLMError, RuntimeError)

    def test_subtypes_descend_from_llm_error(self) -> None:
        for exc in (LLMBackendError, LLMBudgetExhausted, LLMTapeMismatchError):
            assert issubclass(exc, LLMError)

    def test_subtypes_are_distinct(self) -> None:
        siblings = (LLMBackendError, LLMBudgetExhausted, LLMTapeMismatchError)
        for a in siblings:
            for b in siblings:
                if a is not b:
                    assert not issubclass(a, b), f"{a.__name__} <: {b.__name__}"







class TestValueTypes:
    def test_request_carries_prompt_seed_params_not_model(self) -> None:
        req = LLMRequest(
            prompt="p", seed=7, params=LLMParams(temperature=0.5, max_tokens=64)
        )
        assert req.prompt == "p"
        assert req.seed == 7
        assert req.params.temperature == 0.5
        assert req.params.max_tokens == 64

        assert not hasattr(req, "model")

    def test_response_carries_text_model_usage(self) -> None:
        usage = LLMUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        resp = LLMResponse(text="out", model="served-model", usage=usage)
        assert resp.text == "out"
        assert resp.model == "served-model"
        assert resp.usage == usage

    def test_usage_total_tokens_may_be_none(self) -> None:
        usage = LLMUsage(prompt_tokens=5, completion_tokens=6, total_tokens=None)
        assert usage.total_tokens is None

    def test_response_usage_may_be_none(self) -> None:
        resp = LLMResponse(text="out", model="m", usage=None)
        assert resp.usage is None

    def test_value_types_are_frozen(self) -> None:
        req = LLMRequest(
            prompt="p", seed=0, params=LLMParams(temperature=0.1, max_tokens=8)
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            req.seed = 1

    def test_value_types_have_structural_equality(self) -> None:
        a = LLMRequest(
            prompt="p", seed=1, params=LLMParams(temperature=0.2, max_tokens=16)
        )
        b = LLMRequest(
            prompt="p", seed=1, params=LLMParams(temperature=0.2, max_tokens=16)
        )
        assert a == b







class TestLLMParamsValidation:

    def test_negative_temperature_rejected(self) -> None:
        with pytest.raises(ValueError):
            LLMParams(temperature=-0.1, max_tokens=64)

    def test_nan_temperature_rejected(self) -> None:
        with pytest.raises(ValueError):
            LLMParams(temperature=float("nan"), max_tokens=8)

    def test_zero_max_tokens_rejected(self) -> None:
        with pytest.raises(ValueError):
            LLMParams(temperature=1.0, max_tokens=0)

    def test_negative_max_tokens_rejected(self) -> None:
        with pytest.raises(ValueError):
            LLMParams(temperature=1.0, max_tokens=-5)

    def test_boundary_values_accepted(self) -> None:
        params = LLMParams(temperature=0.0, max_tokens=1)
        assert params.temperature == 0.0
        assert params.max_tokens == 1
