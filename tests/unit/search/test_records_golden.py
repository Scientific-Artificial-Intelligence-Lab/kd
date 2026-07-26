
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import kd.search.records as records
from tests.unit.search.test_records import (
    _full_evidence,
    _minimal_evidence,
    _run_record_d,
)
from tests.unit.search.test_run_spec import _run_spec






def _golden_minimal() -> records.EvidenceRecord:
    return _minimal_evidence()


def _golden_full() -> records.EvidenceRecord:
    return _full_evidence()


def _golden_coeff_negative_zero() -> records.EvidenceRecord:
    return _minimal_evidence(coefficients=[-0.0])


def _golden_coeff_positive_zero() -> records.EvidenceRecord:
    return _minimal_evidence(coefficients=[0.0])


def _golden_seed_none() -> records.EvidenceRecord:
    return _minimal_evidence(seed=None)


def _golden_residual_stats_none_moments() -> records.EvidenceRecord:
    return _minimal_evidence(
        residual_stats=records.ResidualStats(mean=None, std=None, max_abs=None, n=0)
    )




_GOLDEN_PINS: list[tuple[Callable[[], records.EvidenceRecord], str]] = [
    (
        _golden_minimal,
        "89e751e3d8bfc8c8e1ff1e0b72a6b1fd8938ae3e1c1d91c3e174a3f0bccc7dc7",
    ),
    (
        _golden_full,
        "6194fec3f92dff68b5045ab099653603d28b5a01f0345b746deca1ae08c3d993",
    ),
    (
        _golden_coeff_negative_zero,
        "fa8a6b55157235c5a8ebe1dd3bf3583875245fdc3cda3da1cb3219d948516a75",
    ),
    (
        _golden_coeff_positive_zero,
        "3d71e696e6e4d6080a9b5c4dfcd67db56937c5072e40334a878aada6d13bfacd",
    ),
    (
        _golden_seed_none,
        "1ceec8c96d57d4ae711e0bd45ea3f24f23088a7a0401f2014bb61a2e9671f2c1",
    ),
    (
        _golden_residual_stats_none_moments,
        "effb0c64796d8fbbb385063bc35cfafe4837785650d8b505eeceb39e5843e59a",
    ),
]


@pytest.mark.parametrize(
    ("builder", "pinned_hex"),
    _GOLDEN_PINS,
    ids=[builder.__name__ for builder, _ in _GOLDEN_PINS],
)
def test_a8_golden_hash_matches_pin(
    builder: Callable[[], records.EvidenceRecord], pinned_hex: str
) -> None:

    assert builder().content_hash() == pinned_hex


def test_a8_golden_vectors_are_pairwise_distinct() -> None:

    hashes = {builder().content_hash() for builder, _ in _GOLDEN_PINS}
    assert len(hashes) == len(_GOLDEN_PINS)







def _golden_run_spec_minimal() -> Any:
    return _run_spec(config={"algorithm": "sga"}, library_fingerprint=None)


def _golden_run_spec_with_path() -> Any:
    return _run_spec(
        config={"algorithm": "pysindy", "checkpoint_dir": Path("/tmp/ckpt")},
        library_fingerprint="0123456789abcdef",
    )


_RUN_SPEC_GOLDEN_PINS: list[tuple[str, Callable[[], Any], str]] = [
    (
        "minimal",
        _golden_run_spec_minimal,
        "4d28c61c5344ba566b585e816189702a3982e869953cb002e18e98eae31e9e41",
    ),
    (
        "with_path",
        _golden_run_spec_with_path,
        "a913991d13a6517184c3f440b5db3fc7a448b8d6b323aa8824a7baca0c22ae91",
    ),
]


@pytest.mark.parametrize(
    ("builder", "pinned_hex"),
    [(builder, pinned) for _name, builder, pinned in _RUN_SPEC_GOLDEN_PINS],
    ids=[name for name, _b, _p in _RUN_SPEC_GOLDEN_PINS],
)
def test_d5_golden_run_spec_hash_matches_pin(
    builder: Callable[[], Any], pinned_hex: str
) -> None:


    assert builder().run_spec_hash == pinned_hex


def test_d5_golden_run_specs_are_pairwise_distinct() -> None:


    hashes = {builder().run_spec_hash for _name, builder, _hex in _RUN_SPEC_GOLDEN_PINS}
    assert len(hashes) == len(_RUN_SPEC_GOLDEN_PINS)












_RECORD_GOLDEN_HASH = (
    "9a8eabb100cf5abebc02432898c943bb9442bdb827073f096069827247ae2184"
)


def test_item1_golden_record_hash_matches_pin() -> None:

    assert _run_record_d().record_hash == _RECORD_GOLDEN_HASH
