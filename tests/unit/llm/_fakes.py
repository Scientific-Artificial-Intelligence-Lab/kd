
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Final

from kd.llm.protocol import (
    LLMBackendError,
    LLMParams,
    LLMRequest,
    LLMResponse,
    LLMUsage,
)


def make_request(
    *,
    prompt: str = "propose an equation",
    seed: int = 0,
    temperature: float = 0.8,
    max_tokens: int = 128,
) -> LLMRequest:
    return LLMRequest(
        prompt=prompt,
        seed=seed,
        params=LLMParams(temperature=temperature, max_tokens=max_tokens),
    )


class TransientError(Exception):
    pass


class BoomError(Exception):
    pass




_OMIT: Final[Any] = object()


class RecordingChatClient:

    def __init__(
        self,
        *,
        fail_times: int = 0,
        content: str = "u_xx - u",
        model: Any = _OMIT,
        usage: Any = _OMIT,
    ) -> None:
        self._fail_times = fail_times
        self._content = content
        self._model = model
        self._usage = usage
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if len(self.calls) <= self._fail_times:
            raise TransientError("simulated transient failure")
        message = SimpleNamespace(content=self._content)
        attrs: dict[str, Any] = {"choices": [SimpleNamespace(message=message)]}
        if self._model is not _OMIT:
            attrs["model"] = self._model
        if self._usage is not _OMIT:
            attrs["usage"] = self._usage
        return SimpleNamespace(**attrs)


def make_sdk_usage(
    *, prompt_tokens: int, completion_tokens: int, total_tokens: int | None
) -> Any:
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def make_sdk_usage_partial(
    *,
    prompt_tokens: Any = _OMIT,
    completion_tokens: Any = _OMIT,
    total_tokens: Any = _OMIT,
) -> Any:
    attrs: dict[str, Any] = {}
    if prompt_tokens is not _OMIT:
        attrs["prompt_tokens"] = prompt_tokens
    if completion_tokens is not _OMIT:
        attrs["completion_tokens"] = completion_tokens
    if total_tokens is not _OMIT:
        attrs["total_tokens"] = total_tokens
    return SimpleNamespace(**attrs)


class FakeProvider:

    def __init__(
        self, *, model: str = "fake/model-v1", emit_usage: bool = True
    ) -> None:
        self._model = model
        self._emit_usage = emit_usage
        self.requests: list[LLMRequest] = []
        self.prepared = False

    def prepare(self) -> None:
        self.prepared = True

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        usage = (
            LLMUsage(
                prompt_tokens=len(request.prompt),
                completion_tokens=request.seed + 1,
                total_tokens=None,
            )
            if self._emit_usage
            else None
        )
        return LLMResponse(
            text=(
                f"completion::{request.prompt}::seed={request.seed}"
                f"::T={request.params.temperature}"
            ),
            model=self._model,
            usage=usage,
        )


class RaisingProvider:

    def __init__(self) -> None:
        self.complete_calls = 0

    def prepare(self) -> None:
        pass

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.complete_calls += 1
        raise AssertionError("inner.complete must not be called here")


class AlwaysFailingProvider:

    def __init__(self) -> None:
        self.complete_calls = 0

    def prepare(self) -> None:
        pass

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.complete_calls += 1
        raise LLMBackendError("simulated inner failure")
