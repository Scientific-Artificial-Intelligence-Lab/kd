
from __future__ import annotations

import hashlib
import random
from typing import Final

from kd.llm import LLMRequest, LLMResponse
from kd.search.llm4ed.prompts import (
    EVOLUTION,
    OPTIMIZE,
    classify_prompt,
)








_VALID_EQUATIONS: Final[tuple[str, ...]] = (
    "u_xx - u + u^3",
    "u_xx - u*u_x",
    "u*u_x + u_xx",
    "u_x + u_xxx",
    "u^2 + u_x*u_xx",
    "u*u_xx - u_x",
    "u_xxx - u*u_x",
    "u_xx + u^2",
    "u_x*u_xx + u",
    "x*u_x - u",
    "u_xxx - u",
    "u^3 - u_x",
    "u*u - u_xx",
    "u_x^2 + u_xx",
    "x*u - u_xxx",
    "u_xx*u_xxx - u",
    "u^2*u_x + u",
    "u_x*u_xxx - u_xx",
)


_INVALID_EQUATIONS: Final[tuple[str, ...]] = (
    "sin(u)",
    "u_xxxx",
    "u^6",
    "u_t - u",
)

_MIN_VALID: Final[int] = 4
_MAX_VALID: Final[int] = 6


class FakeLlm4edBackend:

    def __init__(self, *, model: str = "fake", base_url: str | None = None) -> None:
        self.model = model
        self.base_url = base_url

    def prepare(self) -> None:
        pass

    def complete(self, request: LLMRequest) -> LLMResponse:
        prompt = request.prompt
        seed = request.seed
        rng = self._rng_for(prompt, seed)
        candidates = _scripted_candidates(rng)
        kind = classify_prompt(prompt)
        if kind == EVOLUTION:
            text = _format_evolution(candidates, rng)
        elif kind == OPTIMIZE:
            text = _format_optimize(candidates)
        else:
            text = _format_initialization(candidates)
        return LLMResponse(text=text, model=self.model, usage=None)

    @staticmethod
    def _rng_for(prompt: str, seed: int) -> random.Random:


        digest = hashlib.sha256(f"{seed}\x00{prompt}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))


def _scripted_candidates(rng: random.Random) -> list[str]:
    valid = list(_VALID_EQUATIONS)
    rng.shuffle(valid)
    n_valid = rng.randint(_MIN_VALID, min(_MAX_VALID, len(valid)))
    chosen_valid = valid[:n_valid]

    invalid = list(_INVALID_EQUATIONS)
    rng.shuffle(invalid)
    n_invalid = rng.randint(1, min(2, len(invalid)))
    chosen_invalid = invalid[:n_invalid]

    mixed = chosen_valid + chosen_invalid
    rng.shuffle(mixed)
    return mixed


def _format_initialization(candidates: list[str]) -> str:
    return "\n".join(f"{i + 1}. {eq}" for i, eq in enumerate(candidates))


def _format_optimize(candidates: list[str]) -> str:
    return "\n".join(f"<res>{eq}</res>" for eq in candidates)


def _format_evolution(candidates: list[str], rng: random.Random) -> str:
    lines: list[str] = []
    for eq in candidates:
        sel_a = rng.choice(_VALID_EQUATIONS)
        sel_b = rng.choice(_VALID_EQUATIONS)
        cross = rng.choice(_VALID_EQUATIONS)
        lines.append(f"<select>{{{sel_a}}}</select>")
        lines.append(f"<select>{{{sel_b}}}</select>")
        lines.append(f"<cross>{cross}</cross>")
        lines.append(f"<res>{eq}</res>")
    return "\n".join(lines)
