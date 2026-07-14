
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from kd.llm import openai_compat as oc_mod
from kd.llm.openai_compat import DEFAULT_API_KEY_ENV_VAR, OpenAICompatProvider
from kd.llm.protocol import LLMBackendError, LLMResponse, LLMUsage
from tests.unit.llm._fakes import (
    BoomError,
    RecordingChatClient,
    TransientError,
    make_request,
    make_sdk_usage,
    make_sdk_usage_partial,
)


def _provider_with_client(
    client: Any, *, max_retries_per_call: int = 5
) -> tuple[OpenAICompatProvider, list[float]]:
    delays: list[float] = []
    provider = OpenAICompatProvider(
        model="gpt-x",
        base_url="http://localhost:1/v1",
        max_retries_per_call=max_retries_per_call,
        backoff_initial_seconds=1.0,
        backoff_factor=2.0,
        client=client,
        retryable_exceptions=(TransientError,),
        sleep_fn=delays.append,
    )
    return provider, delays


def _fake_openai_sdk() -> Any:

    class _ErrError(Exception):
        pass

    def _openai(**kwargs: Any) -> Any:
        return SimpleNamespace(chat=None, _kwargs=kwargs)

    return SimpleNamespace(
        OpenAI=_openai,
        APIConnectionError=_ErrError,
        APITimeoutError=_ErrError,
        RateLimitError=_ErrError,
        InternalServerError=_ErrError,
    )







class TestFailLoud:
    def test_default_api_key_env_var_name(self) -> None:
        assert DEFAULT_API_KEY_ENV_VAR == "OPENAI_API_KEY"

    def test_missing_sdk_fails_loud_at_prepare_with_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        def _raise_missing_sdk() -> Any:
            raise ImportError(oc_mod._MISSING_SDK_MESSAGE)

        monkeypatch.setattr(oc_mod, "_import_sdk", _raise_missing_sdk)
        provider = OpenAICompatProvider(
            model="gpt-x", base_url="http://localhost:1/v1"
        )
        with pytest.raises(ImportError, match="llm4ed"):
            provider.prepare()

    def test_missing_api_key_fails_loud_naming_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(oc_mod, "_import_sdk", _fake_openai_sdk)
        monkeypatch.delenv(DEFAULT_API_KEY_ENV_VAR, raising=False)
        provider = OpenAICompatProvider(
            model="gpt-x", base_url="http://localhost:1/v1"
        )
        with pytest.raises(ValueError, match=DEFAULT_API_KEY_ENV_VAR):
            provider.prepare()

    def test_rejects_negative_max_retries(self) -> None:
        with pytest.raises(ValueError):
            OpenAICompatProvider(
                model="m", base_url="u", max_retries_per_call=-1
            )

    def test_rejects_shrinking_backoff_factor(self) -> None:
        with pytest.raises(ValueError):
            OpenAICompatProvider(model="m", base_url="u", backoff_factor=0.5)

    def test_rejects_negative_backoff_initial(self) -> None:
        with pytest.raises(ValueError):
            OpenAICompatProvider(
                model="m", base_url="u", backoff_initial_seconds=-1.0
            )







class TestRetry:
    def test_succeeds_after_transient_failures(self) -> None:
        client = RecordingChatClient(fail_times=2, content="u_xx - u")
        provider, _ = _provider_with_client(client, max_retries_per_call=5)
        resp = provider.complete(make_request(prompt="prompt", seed=7))
        assert isinstance(resp, LLMResponse)
        assert resp.text == "u_xx - u"
        assert len(client.calls) == 3

    def test_retries_preserve_seed_model_and_params(self) -> None:
        client = RecordingChatClient(fail_times=2)
        provider, _ = _provider_with_client(client, max_retries_per_call=5)
        provider.complete(
            make_request(
                prompt="prompt", seed=7, temperature=0.8, max_tokens=128
            )
        )
        assert [c["seed"] for c in client.calls] == [7, 7, 7]
        assert {c["model"] for c in client.calls} == {"gpt-x"}
        assert {c["temperature"] for c in client.calls} == {0.8}
        assert {c["max_tokens"] for c in client.calls} == {128}

    def test_exponential_backoff_delays(self) -> None:
        client = RecordingChatClient(fail_times=2)
        provider, delays = _provider_with_client(client, max_retries_per_call=5)
        provider.complete(make_request(prompt="prompt", seed=1))

        assert delays == [1.0, 2.0]

    def test_bounded_exhaustion_raises_not_recurses(self) -> None:
        client = RecordingChatClient(fail_times=99)
        provider, delays = _provider_with_client(client, max_retries_per_call=2)
        with pytest.raises(LLMBackendError):
            provider.complete(make_request(prompt="prompt", seed=1))
        assert len(client.calls) == 3
        assert delays == [1.0, 2.0]

    def test_seed_varies_per_distinct_call_not_per_retry(self) -> None:
        client = RecordingChatClient(fail_times=0)
        provider, _ = _provider_with_client(client, max_retries_per_call=2)
        provider.complete(make_request(prompt="prompt", seed=10))
        provider.complete(make_request(prompt="prompt", seed=11))
        assert [c["seed"] for c in client.calls] == [10, 11]







class TestMalformed:
    def test_non_retryable_error_is_not_retried(self) -> None:
        def _create(**kwargs: Any) -> Any:
            raise BoomError("permanent")

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=_create)
            )
        )
        delays: list[float] = []
        provider = OpenAICompatProvider(
            model="m",
            base_url="u",
            client=client,
            retryable_exceptions=(TransientError,),
            sleep_fn=delays.append,
        )
        with pytest.raises(BoomError):
            provider.complete(make_request(prompt="prompt", seed=1))
        assert delays == []

    def test_empty_choices_fails_loud_not_retryable(self) -> None:
        def _create(**kwargs: Any) -> Any:
            return SimpleNamespace(choices=[])

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=_create)
            )
        )
        delays: list[float] = []
        provider = OpenAICompatProvider(
            model="m",
            base_url="u",
            client=client,
            retryable_exceptions=(TransientError,),
            sleep_fn=delays.append,
        )
        with pytest.raises(LLMBackendError, match="choices"):
            provider.complete(make_request(prompt="prompt", seed=1))
        assert delays == []

    def test_none_content_fails_loud_not_retryable(self) -> None:
        def _create(**kwargs: Any) -> Any:
            message = SimpleNamespace(content=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=_create)
            )
        )
        delays: list[float] = []
        provider = OpenAICompatProvider(
            model="m",
            base_url="u",
            client=client,
            retryable_exceptions=(TransientError,),
            sleep_fn=delays.append,
        )
        with pytest.raises(LLMBackendError):
            provider.complete(make_request(prompt="prompt", seed=1))
        assert delays == []







class TestResponseMapping:
    def test_returns_response_with_text_model_usage_populated(self) -> None:
        client = RecordingChatClient(
            fail_times=0,
            content="u_x + u_xxx",
            model="served-model-xyz",
            usage=make_sdk_usage(
                prompt_tokens=11, completion_tokens=22, total_tokens=33
            ),
        )
        provider, _ = _provider_with_client(client)
        resp = provider.complete(
            make_request(prompt="prompt", seed=3, max_tokens=256)
        )
        assert isinstance(resp, LLMResponse)
        assert resp.text == "u_x + u_xxx"

        assert client.calls[0]["max_tokens"] == 256

        assert resp.model == "served-model-xyz"



        assert isinstance(resp.usage, LLMUsage)
        assert resp.usage.prompt_tokens == 11
        assert resp.usage.completion_tokens == 22
        assert resp.usage.total_tokens == 33

    def test_defensive_read_when_response_lacks_model_and_usage(self) -> None:
        client = RecordingChatClient(fail_times=0, content="u_xx - u*u_x")
        provider, _ = _provider_with_client(client)
        resp = provider.complete(make_request(prompt="prompt", seed=5))
        assert isinstance(resp, LLMResponse)
        assert resp.text == "u_xx - u*u_x"
        assert resp.model == "gpt-x"
        assert resp.usage is None

    def test_defensive_usage_mapping_tolerates_missing_token_fields(
        self,
    ) -> None:
        client = RecordingChatClient(
            fail_times=0,
            usage=make_sdk_usage_partial(completion_tokens=None),
        )
        provider, _ = _provider_with_client(client)
        resp = provider.complete(make_request(prompt="prompt", seed=5))

        assert isinstance(resp.usage, LLMUsage)
        assert resp.usage.prompt_tokens == 0
        assert resp.usage.completion_tokens == 0
        assert resp.usage.total_tokens is None

    def test_present_none_model_falls_back_to_configured_model(self) -> None:
        client = RecordingChatClient(fail_times=0, model=None)
        provider, _ = _provider_with_client(client)
        resp = provider.complete(make_request(prompt="prompt", seed=5))
        assert resp.model == "gpt-x"
