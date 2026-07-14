
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable






class LLMError(RuntimeError):
    pass


class LLMBackendError(LLMError):
    pass


class LLMBudgetExhausted(LLMError):
    pass


class LLMTapeMismatchError(LLMError):
    pass







@dataclass(frozen=True)
class LLMParams:

    temperature: float
    max_tokens: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature", float(self.temperature))
        if not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0.0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")


@dataclass(frozen=True)
class LLMRequest:

    prompt: str
    seed: int
    params: LLMParams


@dataclass(frozen=True)
class LLMUsage:

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int | None


@dataclass(frozen=True)
class LLMResponse:

    text: str
    model: str
    usage: LLMUsage | None







@runtime_checkable
class LLMProvider(Protocol):

    def prepare(self) -> None:
        ...

    def complete(self, request: LLMRequest) -> LLMResponse:
        ...
