
from __future__ import annotations

import importlib
import logging
import os
import time
from collections.abc import Callable, Mapping
from typing import Any, Final

from kd.llm.protocol import LLMBackendError, LLMRequest, LLMResponse, LLMUsage

logger = logging.getLogger(__name__)


DEFAULT_API_KEY_ENV_VAR: Final[str] = "OPENAI_API_KEY"


class _EmptyCompletionError(Exception):
    pass


def _import_sdk() -> Any:
    return importlib.import_module("openai")


def _retryable_exceptions_from_sdk(sdk: Any) -> tuple[type[BaseException], ...]:
    return (
        sdk.APIConnectionError,
        sdk.APITimeoutError,
        sdk.RateLimitError,
        sdk.InternalServerError,
    )


class OpenAICompatProvider:

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env_var: str = DEFAULT_API_KEY_ENV_VAR,
        extra_body: Mapping[str, Any] | None = None,
        max_retries_per_call: int = 5,
        backoff_initial_seconds: float = 1.0,
        backoff_factor: float = 2.0,
        timeout: float = 60.0,
        client: Any | None = None,
        retryable_exceptions: tuple[type[BaseException], ...] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries_per_call < 0:
            raise ValueError("max_retries_per_call must be >= 0")
        if backoff_factor < 1.0:
            raise ValueError(
                "backoff_factor must be >= 1.0 for a non-shrinking backoff"
            )
        if backoff_initial_seconds < 0.0:
            raise ValueError("backoff_initial_seconds must be >= 0.0")

        self._model = model
        self._base_url = base_url
        self._api_key_env_var = api_key_env_var
        self._extra_body = extra_body
        self._max_retries_per_call = max_retries_per_call
        self._backoff_initial_seconds = backoff_initial_seconds
        self._backoff_factor = backoff_factor
        self._timeout = timeout
        self._sleep_fn = sleep_fn

        self._client: Any | None = client
        self._retryable_exceptions = retryable_exceptions
        self._prepared = False

    def prepare(self) -> None:
        if self._prepared:
            return
        if self._client is None:
            sdk = _import_sdk()
            api_key = self._resolve_api_key()
            self._client = sdk.OpenAI(
                base_url=self._base_url,
                api_key=api_key,
                timeout=self._timeout,
                max_retries=0,
            )
            if self._retryable_exceptions is None:
                self._retryable_exceptions = _retryable_exceptions_from_sdk(sdk)
        elif self._retryable_exceptions is None:
            self._retryable_exceptions = _retryable_exceptions_from_sdk(_import_sdk())
        self._prepared = True

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not self._prepared:
            self.prepare()
        client = self._client
        retryable = self._retryable_exceptions
        assert client is not None and retryable is not None
        retryable = (_EmptyCompletionError, *retryable)

        last_exc: BaseException | None = None
        total_attempts = self._max_retries_per_call + 1
        for attempt in range(total_attempts):
            try:
                return self._call_api(client, request)
            except LLMBackendError:
                raise
            except retryable as exc:
                last_exc = exc
                if attempt == self._max_retries_per_call:
                    break
                delay = self._backoff_initial_seconds * (self._backoff_factor**attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s; backing off %.2fs",
                    attempt + 1,
                    total_attempts,
                    exc,
                    delay,
                )
                self._sleep_fn(delay)
        raise LLMBackendError(
            f"OpenAI-compatible LLM provider exhausted {total_attempts} attempt(s)"
        ) from last_exc

    def _resolve_api_key(self) -> str:
        key = os.environ.get(self._api_key_env_var)
        if not key:
            raise ValueError(
                "The OpenAI-compatible LLM provider requires an API key in the "
                f"{self._api_key_env_var} environment variable; set it or "
                "inject a pre-authenticated client."
            )
        return key

    def _call_api(self, client: Any, request: LLMRequest) -> LLMResponse:
        extra_kwargs = (
            {} if self._extra_body is None else {"extra_body": self._extra_body}
        )
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=request.params.temperature,
            max_tokens=request.params.max_tokens,
            seed=request.seed,
            **extra_kwargs,
        )
        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMBackendError(
                "OpenAI-compatible LLM provider received a response with no choices"
            )
        model = getattr(response, "model", None)
        if not model:
            raise LLMBackendError(
                "OpenAI-compatible LLM provider received a response with no "
                "model id; the served model is recorded on every response and "
                "tape entry, so it is never substituted with the configured one"
            )
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if content is None:
            raise _EmptyCompletionError(
                "OpenAI-compatible LLM provider received an empty completion"
            )
        return LLMResponse(
            text=str(content),
            model=str(model),
            usage=self._map_usage(getattr(response, "usage", None)),
        )

    @staticmethod
    def _map_usage(usage: Any | None) -> LLMUsage | None:
        if usage is None:
            return None
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if prompt_tokens is None or completion_tokens is None:
            return None
        total_tokens = getattr(usage, "total_tokens", None)
        return LLMUsage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=None if total_tokens is None else int(total_tokens),
        )
