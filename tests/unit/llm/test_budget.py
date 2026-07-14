
from __future__ import annotations

import pytest

from kd.llm.budget import BudgetedProvider
from kd.llm.protocol import LLMBackendError, LLMBudgetExhausted
from tests.unit.llm._fakes import (
    AlwaysFailingProvider,
    FakeProvider,
    RaisingProvider,
    make_request,
)


class TestBudgetEnforcement:
    def test_negative_max_calls_rejected(self) -> None:
        with pytest.raises(ValueError):
            BudgetedProvider(FakeProvider(), max_calls=-1)

    def test_negative_initial_calls_rejected(self) -> None:
        with pytest.raises(ValueError):
            BudgetedProvider(FakeProvider(), max_calls=2, initial_calls=-1)

    def test_n_successes_then_exhaustion(self) -> None:
        inner = FakeProvider()
        provider = BudgetedProvider(inner, max_calls=3)
        for i in range(3):
            provider.complete(make_request(prompt="p", seed=i))
        assert provider.calls_made == 3
        assert len(inner.requests) == 3
        with pytest.raises(LLMBudgetExhausted):
            provider.complete(make_request(prompt="p", seed=99))

        assert len(inner.requests) == 3


        assert provider.calls_made == 3

    def test_exhaustion_message_names_used_and_limit(self) -> None:
        provider = BudgetedProvider(FakeProvider(), max_calls=2, initial_calls=5)
        with pytest.raises(LLMBudgetExhausted) as excinfo:
            provider.complete(make_request(seed=0))
        message = str(excinfo.value)
        assert "5" in message
        assert "2" in message

    def test_rejected_call_does_not_reach_inner(self) -> None:
        inner = RaisingProvider()
        provider = BudgetedProvider(inner, max_calls=0)
        with pytest.raises(LLMBudgetExhausted):
            provider.complete(make_request(seed=0))
        assert inner.complete_calls == 0


class TestCountOnAdmission:
    def test_admitted_call_costs_budget_even_when_inner_fails(self) -> None:
        inner = AlwaysFailingProvider()
        provider = BudgetedProvider(inner, max_calls=2)
        for i in range(2):
            with pytest.raises(LLMBackendError):
                provider.complete(make_request(seed=i))

        assert provider.calls_made == 2
        assert inner.complete_calls == 2

        with pytest.raises(LLMBudgetExhausted):
            provider.complete(make_request(seed=2))
        assert inner.complete_calls == 2


class TestPassthrough:
    def test_within_budget_response_is_identical(self) -> None:
        inner = FakeProvider()
        provider = BudgetedProvider(inner, max_calls=5)
        request = make_request(prompt="passthrough", seed=4, temperature=0.3)
        direct = FakeProvider().complete(request)
        wrapped = provider.complete(request)
        assert wrapped == direct

    def test_prepare_forwards_to_inner(self) -> None:
        inner = FakeProvider()
        BudgetedProvider(inner, max_calls=1).prepare()
        assert inner.prepared is True


class TestComposability:
    def test_wraps_another_provider(self) -> None:
        inner = FakeProvider()
        provider = BudgetedProvider(
            BudgetedProvider(inner, max_calls=1), max_calls=5
        )
        provider.complete(make_request(seed=0))
        with pytest.raises(LLMBudgetExhausted):
            provider.complete(make_request(seed=1))


class TestRestoreSeam:
    def test_initial_calls_resumes_not_resets(self) -> None:
        inner = FakeProvider()
        provider = BudgetedProvider(inner, max_calls=5, initial_calls=4)
        assert provider.calls_made == 4
        provider.complete(make_request(seed=0))
        assert provider.calls_made == 5
        with pytest.raises(LLMBudgetExhausted):
            provider.complete(make_request(seed=1))
        assert len(inner.requests) == 1

    def test_initial_calls_at_limit_exhausts_immediately(self) -> None:
        inner = RaisingProvider()
        provider = BudgetedProvider(inner, max_calls=3, initial_calls=3)
        with pytest.raises(LLMBudgetExhausted):
            provider.complete(make_request(seed=0))
        assert inner.complete_calls == 0
