
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from kd.llm.protocol import (
    LLMParams,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMTapeMismatchError,
    LLMUsage,
)


def request_hash(request: LLMRequest) -> str:
    payload = {
        "prompt": request.prompt,
        "seed": request.seed,
        "params": {
            "temperature": request.params.temperature,
            "max_tokens": request.params.max_tokens,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def request_to_json(request: LLMRequest) -> dict[str, Any]:
    return {
        "prompt": request.prompt,
        "params": {
            "temperature": request.params.temperature,
            "max_tokens": request.params.max_tokens,
        },
        "seed": request.seed,
    }


def request_from_json(data: dict[str, Any]) -> LLMRequest:
    params = data["params"]
    return LLMRequest(
        prompt=data["prompt"],
        seed=data["seed"],
        params=LLMParams(
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
        ),
    )


def usage_to_json(usage: LLMUsage) -> dict[str, int | None]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def response_to_json(response: LLMResponse) -> dict[str, Any]:
    return {
        "text": response.text,
        "model": response.model,
        "usage": (
            None
            if response.usage is None
            else usage_to_json(response.usage)
        ),
    }


def response_from_json(data: dict[str, Any]) -> LLMResponse:
    usage_data = data.get("usage")
    usage = (
        None
        if usage_data is None
        else LLMUsage(
            prompt_tokens=usage_data["prompt_tokens"],
            completion_tokens=usage_data["completion_tokens"],
            total_tokens=usage_data.get("total_tokens"),
        )
    )
    return LLMResponse(text=data["text"], model=data["model"], usage=usage)


def load_tape_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise LLMTapeMismatchError(f"LLM tape file not found: {path}")
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise LLMTapeMismatchError(
                f"LLM tape contains invalid JSON at line {line_number}: {path}"
            ) from exc
    return entries


class TapeRecordingProvider:

    def __init__(self, inner: LLMProvider, *, path: str | Path) -> None:
        self._inner = inner
        self._path = Path(path)

    def prepare(self) -> None:
        self._inner.prepare()

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self._inner.complete(request)
        entry = {
            "kind": "llm_call",
            "request": request_to_json(request),
            "response": response_to_json(response),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return response


class TapeReplayProvider:

    def __init__(self, *, path: str | Path, initial_position: int = 0) -> None:
        if initial_position < 0:
            raise ValueError("initial_position must be >= 0")
        self._path = Path(path)
        self._position = initial_position
        self._entries: list[dict[str, Any]] | None = None

    @property
    def position(self) -> int:
        return self._position

    def prepare(self) -> None:
        return None

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self._entries is None:
            self._entries = load_tape_entries(self._path)
        entries = self._entries
        if self._position >= len(entries):
            raise LLMTapeMismatchError(
                f"LLM tape exhausted at position {self._position}"
            )

        entry = entries[self._position]
        if entry.get("kind") != "llm_call":
            raise LLMTapeMismatchError(
                f"LLM tape entry at position {self._position} is not an llm_call"
            )
        recorded_request = request_from_json(entry["request"])
        if request_hash(request) != request_hash(recorded_request):
            raise LLMTapeMismatchError(
                f"LLM tape request mismatch at position {self._position}"
            )

        self._position += 1
        return response_from_json(entry["response"])
