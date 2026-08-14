
from __future__ import annotations

import copy
import pickle
from typing import Any

from kd.llm import LLMProvider, LLMRequest, LLMResponse
from kd.search.llm4ed.plugin import Llm4edPlugin
from tests.unit.search.llm4ed._fake_backend import FakeLlm4edBackend


class RecordingProvider:

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.requests: list[LLMRequest] = []

    def prepare(self) -> None:
        self._inner.prepare()

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self._inner.complete(request)

    @property
    def seeds(self) -> list[int]:
        return [request.seed for request in self.requests]

    @property
    def prompts(self) -> list[str]:
        return [request.prompt for request in self.requests]


def build_fake_transport(
    *, model: str, base_url: str | None = None
) -> LLMProvider:
    return FakeLlm4edBackend()


def pickle_roundtrip(state: dict[str, Any]) -> dict[str, Any]:
    return pickle.loads(pickle.dumps(state))


def canonical_state(plugin: Llm4edPlugin) -> dict[str, Any]:
    state: dict[str, Any] = copy.deepcopy(plugin.state)
    pool = state["elite_pool"]
    pool["items"] = sorted(
        pool["items"], key=lambda item: (item["score"], item["expression"])
    )
    return state
