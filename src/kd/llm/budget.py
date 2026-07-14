
from __future__ import annotations

from kd.llm.protocol import (
    LLMBudgetExhausted,
    LLMProvider,
    LLMRequest,
    LLMResponse,
)


class BudgetedProvider:

    def __init__(
        self,
        inner: LLMProvider,
        *,
        max_calls: int,
        initial_calls: int = 0,
    ) -> None:
        if max_calls < 0:
            raise ValueError("max_calls must be >= 0")
        if initial_calls < 0:
            raise ValueError("initial_calls must be >= 0")
        self._inner = inner
        self._max_calls = max_calls
        self._calls_made = initial_calls

    @property
    def calls_made(self) -> int:
        return self._calls_made

    def prepare(self) -> None:
        self._inner.prepare()

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self._calls_made >= self._max_calls:
            raise LLMBudgetExhausted(
                "LLM call budget exhausted: "
                f"used {self._calls_made}, limit {self._max_calls}"
            )
        self._calls_made += 1
        return self._inner.complete(request)
