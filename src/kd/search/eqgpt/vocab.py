
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Final



PAD_ID: Final[int] = 0
E_ID: Final[int] = 1
PLUS_ID: Final[int] = 2
MUL_ID: Final[int] = 3
DIV_ID: Final[int] = 4
S_ID: Final[int] = 5
FIRST_TERM_ID: Final[int] = 6



VOCAB_ASSET_FILENAME: Final[str] = "dict_datas_0725.json"
VOCAB_SIZE: Final[int] = 57
VOCAB_SHA256: Final[str] = (
    "6cec0ef67d61fc385e30826a7799e7cd42ff4759c729f51eab409ad2f128fc30"
)
VOCAB_SIZE_BYTES: Final[int] = 1228


class UnknownTokenError(KeyError):
    pass


@dataclass(frozen=True)
class Vocab:

    word2id: Mapping[str, int]
    id2word: Sequence[str]

    @property
    def size(self) -> int:
        return len(self.id2word)

    def encode(self, words: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for word in words:
            if word not in self.word2id:
                raise UnknownTokenError(word)
            ids.append(self.word2id[word])
        return ids

    def decode(self, ids: Sequence[int]) -> list[str]:
        words: list[str] = []
        for word_id in ids:


            if not (0 <= word_id < len(self.id2word)):
                raise IndexError(f"id out of range: {word_id}")
            words.append(self.id2word[word_id])
        return words

    def encode_sentence(
        self, terms: Sequence[str], *, pad_to: int | None = None
    ) -> list[int]:
        ids = [S_ID, *self.encode(terms), E_ID]
        if pad_to is not None:
            if len(ids) > pad_to:
                raise ValueError(
                    f"framed sentence length {len(ids)} exceeds pad_to={pad_to}"
                )
            ids = ids + [PAD_ID] * (pad_to - len(ids))
        return ids

    def decode_sentence(self, ids: Sequence[int]) -> list[str]:
        words = self.decode(ids)
        start = 1 if words and words[0] == self.id2word[S_ID] else 0
        try:
            end = words.index(self.id2word[E_ID])
        except ValueError:
            end = len(words)
        return words[start:end]


def vocab_asset_path() -> Path:
    return Path(
        str(resources.files(__package__).joinpath("_assets", VOCAB_ASSET_FILENAME))
    )


def load_vocab() -> Vocab:
    raw = json.loads(vocab_asset_path().read_text(encoding="utf-8"))
    word2id = dict(raw["word2id"])
    id2word = list(raw["id2word"])
    if len(id2word) != VOCAB_SIZE or len(word2id) != VOCAB_SIZE:
        raise ValueError(
            f"vocab asset entry count mismatch: expected {VOCAB_SIZE}, got "
            f"id2word={len(id2word)}, word2id={len(word2id)}"
        )
    if id2word[PAD_ID] != "<pad>":
        raise ValueError(
            f"vocab asset pad slot mismatch: id2word[{PAD_ID}]={id2word[PAD_ID]!r}"
            ", expected '<pad>'"
        )
    for idx, word in enumerate(id2word):
        if word2id.get(word) != idx:
            raise ValueError(
                f"vocab asset word2id/id2word mismatch at id {idx} ({word!r})"
            )
    return Vocab(word2id=word2id, id2word=id2word)
