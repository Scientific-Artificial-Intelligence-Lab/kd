
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import torch

from kd.search.eqgpt.backend import GPTBackend
from kd.search.eqgpt.vocab import DIV_ID, E_ID, FIRST_TERM_ID, MUL_ID, PLUS_ID, Vocab

_OPERATOR_IDS: Final[tuple[int, ...]] = (E_ID, PLUS_ID, MUL_ID, DIV_ID)


DEFAULT_START_WORDS: Final[tuple[str, ...]] = ("S", "ut", "+")

DEFAULT_MAX_LENGTH: Final[int] = 49
DEFAULT_EXPLORATION_RATE: Final[float] = 0.2


class AllTokensMaskedError(ValueError):
    pass


@dataclass(frozen=True)
class SamplingConfig:

    start_tokens: tuple[int, ...]
    masked_tokens: frozenset[int] = field(default_factory=frozenset)
    exploration_rate: float = DEFAULT_EXPLORATION_RATE
    max_length: int = DEFAULT_MAX_LENGTH

    def __post_init__(self) -> None:
        if len(self.start_tokens) < 1:
            raise ValueError("start_tokens must be non-empty")
        if not 0.0 <= self.exploration_rate <= 1.0:
            raise ValueError(
                f"exploration_rate must be in [0, 1], got {self.exploration_rate}"
            )

        if self.max_length <= len(self.start_tokens):
            raise ValueError(
                f"max_length ({self.max_length}) must exceed len(start_tokens) "
                f"({len(self.start_tokens)})"
            )


def default_start_tokens(vocab: Vocab) -> tuple[int, ...]:
    return tuple(vocab.word2id[word] for word in DEFAULT_START_WORDS)


def dimension_masked_tokens(vocab: Vocab, variables: Sequence[str]) -> frozenset[int]:
    variable_set = set(variables)
    masked: set[int] = set()
    for idx, word in enumerate(vocab.id2word):
        if "t" in word:
            masked.add(idx)
    if "x" not in variable_set:
        for idx, word in enumerate(vocab.id2word):
            if "x" in word:
                masked.add(idx)
    if "y" not in variable_set:
        for idx, word in enumerate(vocab.id2word):
            if "y" in word or "Laplace" in word or "Div" in word:
                masked.add(idx)
    if "z" not in variable_set:
        for idx, word in enumerate(vocab.id2word):
            if "z" in word or "Div" in word:
                masked.add(idx)
    return frozenset(masked)


class Sampler:

    def __init__(
        self, backend: GPTBackend, vocab: Vocab, config: SamplingConfig
    ) -> None:
        self.backend = backend
        self.vocab = vocab
        self.config = config

    def _legal_ids(self, *, term_position: bool) -> list[int]:
        candidates: range | tuple[int, ...] = (
            range(FIRST_TERM_ID, self.vocab.size) if term_position else _OPERATOR_IDS
        )
        return [tok for tok in candidates if tok not in self.config.masked_tokens]

    def _draw_next(self, prefix: list[int]) -> int:
        term_position = len(prefix) % 2 == 1
        legal = self._legal_ids(term_position=term_position)
        if not legal:
            kind = "term" if term_position else "operator"
            raise AllTokensMaskedError(
                f"no legal {kind} token at position {len(prefix)}"
            )
        if torch.rand(()).item() < self.config.exploration_rate:
            idx = int(torch.randint(0, len(legal), (1,)).item())
            return legal[idx]
        logits = self.backend.next_token_logits(prefix)
        legal_idx = torch.as_tensor(legal, dtype=torch.long)
        probs = torch.softmax(logits[legal_idx], dim=0)
        choice = int(torch.multinomial(probs, 1).item())
        return legal[choice]

    def _draw_sentence(self) -> list[int]:
        sentence = list(self.config.start_tokens)
        while len(sentence) < self.config.max_length:
            token = self._draw_next(sentence)
            sentence.append(token)
            if token == E_ID:
                break
        return sentence

    def sample(self, *, seed: int) -> list[int]:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            return self._draw_sentence()

    def sample_batch(self, n: int, *, seed: int) -> list[list[int]]:
        if n == 0:
            return []
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            return [self._draw_sentence() for _ in range(n)]
