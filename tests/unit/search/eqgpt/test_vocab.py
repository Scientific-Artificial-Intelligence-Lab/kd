
from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from collections.abc import Callable
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import pytest

import kd
import kd.search as ksearch
from kd.search.eqgpt import vocab as vmod
from kd.search.eqgpt.vocab import (
    UnknownTokenError,
    Vocab,
    load_vocab,
    vocab_asset_path,
)






@pytest.mark.smoke
def test_vocab_asset_sha256_is_locked() -> None:
    digest = hashlib.sha256(vocab_asset_path().read_bytes()).hexdigest()
    assert digest == vmod.VOCAB_SHA256


def test_vocab_asset_byte_size_is_locked() -> None:
    assert vocab_asset_path().stat().st_size == vmod.VOCAB_SIZE_BYTES


def test_vocab_asset_declared_in_package_data() -> None:
    root = Path(__file__).resolve().parents[4]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    globs = package_data.get("kd.search.eqgpt", [])
    assert any(fnmatch(f"_assets/{vmod.VOCAB_ASSET_FILENAME}", g) for g in globs), (
        f"vendored vocab not covered by package-data globs {globs}"
    )


def test_vocab_has_expected_entry_count() -> None:
    vocab = load_vocab()
    assert vocab.size == vmod.VOCAB_SIZE == 57


def test_special_token_ids_match_pretrained_order() -> None:
    vocab = load_vocab()
    assert vocab.word2id["<pad>"] == vmod.PAD_ID == 0
    assert vocab.word2id["E"] == vmod.E_ID == 1
    assert vocab.word2id["+"] == vmod.PLUS_ID == 2
    assert vocab.word2id["*"] == vmod.MUL_ID == 3
    assert vocab.word2id["/"] == vmod.DIV_ID == 4
    assert vocab.word2id["S"] == vmod.S_ID == 5


def test_terms_start_at_first_term_id() -> None:
    vocab = load_vocab()

    assert vocab.id2word[vmod.FIRST_TERM_ID] == "ut"
    specials = {"<pad>", "E", "+", "*", "/", "S"}
    assert set(vocab.id2word[: vmod.FIRST_TERM_ID]) == specials


def test_word2id_and_id2word_are_mutual_inverses() -> None:
    vocab = load_vocab()
    assert len(vocab.word2id) == len(vocab.id2word)
    for idx, word in enumerate(vocab.id2word):
        assert vocab.word2id[word] == idx
    for word, idx in vocab.word2id.items():
        assert vocab.id2word[idx] == word


def test_key_wave_term_is_in_vocabulary() -> None:

    vocab = load_vocab()
    assert "(uux)xx" in vocab.word2id


def test_vocabulary_is_plugin_private() -> None:
    forbidden = {
        "Vocab",
        "load_vocab",
        "vocab_asset_path",
        "word2id",
        "id2word",
        "UnknownTokenError",
        "VOCAB_SIZE",
    }
    assert forbidden.isdisjoint(set(getattr(kd, "__all__", [])))
    assert forbidden.isdisjoint(set(getattr(ksearch, "__all__", [])))
    assert "eqgpt" not in set(getattr(ksearch, "__all__", []))
    for name in forbidden:
        assert not hasattr(kd, name), f"kd.{name} leaks a plugin-private symbol"







def test_encode_decode_roundtrip_for_every_entry() -> None:
    vocab = load_vocab()
    for word in vocab.id2word:
        assert vocab.decode(vocab.encode([word])) == [word]


def test_encode_matches_word2id_ids() -> None:
    vocab = load_vocab()
    words = ["ut", "+", "ux"]
    assert vocab.encode(words) == [vocab.word2id[w] for w in words]


def test_encode_decode_roundtrip_for_full_sentence() -> None:
    vocab = load_vocab()
    ids = [vmod.S_ID, 6, vmod.PLUS_ID, 8, vmod.E_ID]
    assert vocab.encode(vocab.decode(ids)) == ids


def test_encode_sentence_frames_with_start_and_end() -> None:
    vocab = load_vocab()
    ids = vocab.encode_sentence(["ut", "+", "ux"])
    assert ids[0] == vmod.S_ID
    assert ids[-1] == vmod.E_ID
    assert ids == [
        vmod.S_ID,
        vocab.word2id["ut"],
        vmod.PLUS_ID,
        vocab.word2id["ux"],
        vmod.E_ID,
    ]


def test_encode_sentence_pads_to_context_length() -> None:
    vocab = load_vocab()
    ids = vocab.encode_sentence(["ut", "+", "ux"], pad_to=50)
    assert len(ids) == 50
    assert ids[:5] == [
        vmod.S_ID,
        vocab.word2id["ut"],
        vmod.PLUS_ID,
        vocab.word2id["ux"],
        vmod.E_ID,
    ]
    assert set(ids[5:]) == {vmod.PAD_ID}


def test_decode_sentence_inverts_encode_sentence() -> None:
    vocab = load_vocab()
    terms = ["ut", "+", "uxxx", "+", "(uux)xx", "+", "ux"]
    assert vocab.decode_sentence(vocab.encode_sentence(terms, pad_to=50)) == terms







def test_encode_unknown_token_raises_loudly() -> None:
    vocab = load_vocab()
    with pytest.raises(UnknownTokenError):
        vocab.encode(["ut", "definitely_not_a_token", "ux"])


def test_encode_does_not_silently_skip_unknown_token() -> None:
    vocab = load_vocab()
    try:
        result = vocab.encode(["ut", "not_a_token"])
    except UnknownTokenError:
        return
    pytest.fail(f"unknown token was not rejected; got {result!r}")


def test_encode_sentence_overflow_raises() -> None:
    vocab = load_vocab()
    with pytest.raises((ValueError, IndexError)):

        vocab.encode_sentence(["ut", "+", "ux"], pad_to=3)


@pytest.mark.parametrize("bad_id", [vmod.VOCAB_SIZE + 100, -1])
def test_decode_rejects_out_of_range_id(bad_id: int) -> None:


    vocab = load_vocab()
    with pytest.raises((IndexError, KeyError, ValueError)):
        vocab.decode([bad_id])


def test_vocab_dataclass_is_frozen() -> None:
    vocab = load_vocab()
    assert isinstance(vocab, Vocab)
    with pytest.raises(dataclasses.FrozenInstanceError):
        vocab.word2id = {}


def _valid_vocab_dict() -> dict[str, Any]:
    raw = json.loads(vocab_asset_path().read_text(encoding="utf-8"))
    return {"word2id": dict(raw["word2id"]), "id2word": list(raw["id2word"])}


def _corrupt_count(d: dict[str, Any]) -> dict[str, Any]:
    d["word2id"] = {"<pad>": 0, "E": 1}
    d["id2word"] = ["<pad>", "E"]
    return d


def _corrupt_pad(d: dict[str, Any]) -> dict[str, Any]:
    d["id2word"][0] = "NOTPAD"
    d["word2id"].pop("<pad>")
    d["word2id"]["NOTPAD"] = 0
    return d


def _corrupt_inverse(d: dict[str, Any]) -> dict[str, Any]:
    d["word2id"]["ut"], d["word2id"]["u"] = 7, 6
    return d


@pytest.mark.parametrize("corrupt", [_corrupt_count, _corrupt_pad, _corrupt_inverse])
def test_load_vocab_rejects_corrupted_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(corrupt(_valid_vocab_dict())), encoding="utf-8")
    monkeypatch.setattr(vmod, "vocab_asset_path", lambda: bad)
    with pytest.raises(ValueError, match="vocab asset"):
        load_vocab()
