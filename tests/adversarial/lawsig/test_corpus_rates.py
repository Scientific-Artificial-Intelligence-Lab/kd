
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parents[2]
_TESTS_PATH = str(_TESTS_DIR)
sys.path.insert(0, _TESTS_PATH)

try:
    from adversarial.lawsig.corpus import (
        CORPUS,
        REQUIRED_FAMILIES,
    )
    from adversarial.lawsig.rates import evaluate_corpus
finally:
    sys.path.remove(_TESTS_PATH)


@pytest.mark.unit
def test_corpus_has_no_false_merges() -> None:
    result = evaluate_corpus()

    assert result.overall.false_merge_count == 0, (
        "false-merge pairs: "
        + ", ".join(result.overall.false_merge_pair_ids)
    )


@pytest.mark.unit
def test_every_false_split_family_is_declared() -> None:
    result = evaluate_corpus()
    undeclared_pair_ids = [
        pair_id
        for family in result.families.values()
        if not family.declared
        for pair_id in family.false_split_pair_ids
    ]

    assert not undeclared_pair_ids, (
        "undeclared false-split pairs: " + ", ".join(undeclared_pair_ids)
    )


@pytest.mark.unit
def test_corpus_has_no_labeling_violations() -> None:
    result = evaluate_corpus()
    details = "; ".join(
        f"{violation.pair_id}: {violation.message}"
        for violation in result.labeling_violations
    )

    assert not result.labeling_violations, f"labeling violations: {details}"


@pytest.mark.unit
def test_corpus_has_minimum_size_and_pair_metadata() -> None:
    assert len(CORPUS) >= 40

    missing_family = [pair.pair_id for pair in CORPUS if not pair.family.strip()]
    missing_rationale = [
        pair.pair_id for pair in CORPUS if not pair.rationale.strip()
    ]
    assert not missing_family, "pairs missing family: " + ", ".join(missing_family)
    assert not missing_rationale, (
        "pairs missing rationale: " + ", ".join(missing_rationale)
    )


@pytest.mark.unit
def test_corpus_spans_required_families() -> None:
    present_families = {pair.family for pair in CORPUS}
    missing_families = sorted(REQUIRED_FAMILIES - present_families)

    assert not missing_families, (
        "missing adversarial families: " + ", ".join(missing_families)
    )
