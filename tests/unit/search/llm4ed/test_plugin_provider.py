
from __future__ import annotations

from collections.abc import Callable

import pytest

from kd.llm import (
    BudgetedProvider,
    LLMBackendError,
    LLMParams,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    OpenAICompatProvider,
)
from kd.search.llm4ed.plugin import Llm4edPlugin
from tests.unit.search.llm4ed._plugin_helpers import (
    ALL_INVALID,
    GOOD,
    MIX,
    FakeProvider,
    components_for,
    make_config,
    prepared,
    raises,
)






def test_fake_provider_satisfies_llmprovider_by_name_and_by_call_shape() -> None:
    fake = FakeProvider(MIX)

    assert isinstance(fake, LLMProvider)



    request = LLMRequest(
        prompt="hello", seed=5, params=LLMParams(temperature=0.8, max_tokens=16)
    )
    response = fake.complete(request)
    assert isinstance(response, LLMResponse)
    assert fake.requests[-1] is request
    assert response.model == "fake" and response.usage is None







def test_injected_provider_is_used_as_is() -> None:
    fake = FakeProvider(MIX)
    plugin, _ = prepared(provider=fake)
    plugin.propose(4)
    assert fake.requests, "injected provider received no calls -- not used as-is"
    assert plugin._provider is fake










def test_default_provider_chain_is_budgeted_openai_compat() -> None:



    config = make_config(max_llm_calls_per_run=17, base_url="https://api.example/v1")
    plugin = Llm4edPlugin(config, provider=None)
    chain = plugin._build_default_provider()
    assert isinstance(chain, BudgetedProvider)
    assert isinstance(chain._inner, OpenAICompatProvider)
    assert chain._max_calls == 17
    assert chain.calls_made == 0


    assert chain._inner._base_url == "https://api.example/v1"


def test_default_provider_chain_rejects_none_base_url() -> None:







    plugin = Llm4edPlugin(make_config(base_url=None), provider=None)
    with pytest.raises(ValueError):
        plugin._build_default_provider()








def test_budget_exhaustion_yields_partial_batch_without_raising() -> None:
    inner = FakeProvider(ALL_INVALID)
    budget = 2
    provider = BudgetedProvider(inner, max_calls=budget)

    config = make_config(max_llm_calls_per_propose=50)
    plugin, _ = prepared(config=config, provider=provider)

    batch = plugin.propose(4)

    assert batch == []



    assert len(inner.requests) == budget


def test_budget_exhaustion_is_a_clean_noop_round() -> None:
    inner = FakeProvider(ALL_INVALID)
    provider = BudgetedProvider(inner, max_calls=1)
    plugin, _ = prepared(
        config=make_config(max_llm_calls_per_propose=50), provider=provider
    )
    batch = plugin.propose(4)
    assert plugin.evaluate(batch) == []
    plugin.update(plugin.evaluate(batch))
    assert plugin.is_done is False


def test_non_budget_provider_error_propagates() -> None:


    plugin, _ = prepared(provider=FakeProvider(raises(LLMBackendError("down"))))
    with pytest.raises(LLMBackendError):
        plugin.propose(4)







def test_sampling_params_come_from_config_on_every_call() -> None:
    fake = FakeProvider(ALL_INVALID)
    config = make_config(temperature=0.55, max_tokens=321, max_llm_calls_per_propose=3)
    plugin, _ = prepared(config=config, provider=fake)
    plugin.propose(4)
    assert len(fake.requests) == 3


    for request in fake.requests:
        assert request.params.temperature == 0.55
        assert request.params.max_tokens == 321








def test_llm_seed_is_base_plus_monotonic_counter_per_call() -> None:
    fake = FakeProvider(ALL_INVALID)
    config = make_config(seed=100, max_llm_calls_per_propose=3)
    plugin, _ = prepared(config=config, provider=fake)

    plugin.propose(4)
    plugin.propose(4)

    assert fake.seeds == [100 + i for i in range(6)]









class _UsageProvider:

    def __init__(
        self, text: str, usage_for: Callable[[int], LLMUsage | None]
    ) -> None:
        self._text = text
        self._usage_for = usage_for
        self.requests: list[LLMRequest] = []
        self.reported_in = 0
        self.reported_out = 0
        self.reporting_calls = 0

    def prepare(self) -> None:
        pass

    def complete(self, request: LLMRequest) -> LLMResponse:
        index = len(self.requests)
        self.requests.append(request)
        usage = self._usage_for(index)
        if usage is not None:
            self.reported_in += usage.prompt_tokens
            self.reported_out += usage.completion_tokens
            self.reporting_calls += 1
        return LLMResponse(text=self._text, model="fake-usage", usage=usage)


def _fixed_usage(_index: int) -> LLMUsage:
    return LLMUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13)


def test_llm_token_totals_complete_when_every_call_reports_usage() -> None:



    provider = _UsageProvider(GOOD, _fixed_usage)
    plugin, _ = prepared(provider=provider)
    plugin.propose(4)
    assert provider.reporting_calls == len(provider.requests) > 0
    assert plugin.llm_token_totals == {
        "tokens_in": provider.reported_in,
        "tokens_out": provider.reported_out,
    }


def test_llm_token_totals_none_under_mixed_usage() -> None:




    provider = _UsageProvider(
        ALL_INVALID,
        lambda index: _fixed_usage(index) if index % 2 == 0 else None,
    )
    plugin, _ = prepared(
        config=make_config(max_llm_calls_per_propose=4), provider=provider
    )
    plugin.propose(4)
    assert 0 < provider.reporting_calls < len(provider.requests)
    assert plugin.llm_token_totals is None


def test_llm_token_totals_none_when_no_call_reports_usage() -> None:


    fake = FakeProvider(ALL_INVALID)
    plugin, _ = prepared(provider=fake)
    plugin.propose(4)
    assert fake.requests
    assert plugin.llm_token_totals is None


def test_prepare_resets_token_totals_on_fresh_path() -> None:
    provider = _UsageProvider(GOOD, _fixed_usage)
    plugin, components = prepared(provider=provider)
    plugin.propose(4)
    assert plugin.llm_token_totals is not None

    plugin.prepare(components)
    assert plugin.llm_token_totals is None


def test_prepare_resets_token_totals_on_restore_path() -> None:
    source = _UsageProvider(GOOD, _fixed_usage)
    plugin, _ = prepared(provider=source)
    plugin.propose(4)
    assert plugin.llm_token_totals is not None
    payload = plugin.state




    resumed = Llm4edPlugin(make_config(), provider=_UsageProvider(GOOD, _fixed_usage))
    resumed.state = payload
    resumed.prepare(components_for())
    assert resumed.llm_token_totals is None


def test_checkpoint_state_payload_carries_no_token_keys() -> None:


    provider = _UsageProvider(GOOD, _fixed_usage)
    plugin, _ = prepared(provider=provider)
    plugin.propose(4)
    payload = plugin.state
    assert "tokens_in" not in payload
    assert "tokens_out" not in payload
    assert "tokens_cached" not in payload
