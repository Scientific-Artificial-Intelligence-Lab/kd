
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from .corpus import (
    COEFF_ATOL,
    CORPUS,
    KNOWN_SPLIT_FAMILIES,
    CorpusPair,
    Expected,
)

from kd.core.equation.signature import LawSignature, law_signature
from kd.core.equation.types import Equation


@dataclass(frozen=True)
class PairRates:

    pair_count: int
    false_merge_count: int
    false_split_count: int
    false_merge_rate: float
    false_split_rate: float
    false_merge_pair_ids: tuple[str, ...]
    false_split_pair_ids: tuple[str, ...]


@dataclass(frozen=True)
class FamilyRates(PairRates):

    declared: bool


@dataclass(frozen=True)
class LabelingViolation:

    pair_id: str
    family: str
    expected: Expected
    observed: Expected | None
    message: str
    exception_type: str | None = None


@dataclass(frozen=True)
class CorpusEvaluation:

    families: dict[str, FamilyRates]
    overall: PairRates
    labeling_violations: list[LabelingViolation]


@dataclass
class _RateAccumulator:
    pair_count: int = 0
    false_merge_pair_ids: list[str] = field(default_factory=list)
    false_split_pair_ids: list[str] = field(default_factory=list)


_Evaluation: TypeAlias = tuple[Expected | None, tuple[LabelingViolation, ...]]
_MERGE_EXPECTATIONS = frozenset({Expected.MERGE_EXACT, Expected.MERGE_STRUCT})


def _violation(
    pair: CorpusPair,
    message: str,
    *,
    observed: Expected | None = None,
    exception: Exception | None = None,
) -> LabelingViolation:
    return LabelingViolation(
        pair_id=pair.pair_id,
        family=pair.family,
        expected=pair.expected,
        observed=observed,
        message=message,
        exception_type=type(exception).__name__ if exception is not None else None,
    )


def _evaluate_raises(pair: CorpusPair) -> _Evaluation:
    try:
        law_signature(pair.eq_a)
    except ValueError:
        return None, ()
    except Exception as exc:
        message = f"eq_a raised unexpected {type(exc).__name__}: {exc}"
        return None, (_violation(pair, message, exception=exc),)
    return None, (_violation(pair, "eq_a did not raise ValueError"),)


def _capture_signature(
    pair: CorpusPair,
    equation: Equation,
    side: str,
) -> tuple[LawSignature | None, LabelingViolation | None]:
    try:
        return law_signature(equation), None
    except Exception as exc:
        message = f"{side} raised unexpected {type(exc).__name__}: {exc}"
        return None, _violation(pair, message, exception=exc)


def _classify(a: LawSignature, b: LawSignature) -> Expected:
    if a.structure_key != b.structure_key:
        return Expected.SPLIT
    coeff_close = len(a.coefficients) == len(b.coefficients) and all(
        abs(a_value - b_value) <= COEFF_ATOL
        for a_value, b_value in zip(a.coefficients, b.coefficients, strict=True)
    )
    return Expected.MERGE_EXACT if coeff_close else Expected.MERGE_STRUCT


def _evaluate_pair(pair: CorpusPair) -> _Evaluation:
    if pair.expected is Expected.RAISES:
        return _evaluate_raises(pair)
    if pair.eq_b is None:
        return None, (_violation(pair, "eq_b is required for this outcome"),)

    signature_a, violation_a = _capture_signature(pair, pair.eq_a, "eq_a")
    signature_b, violation_b = _capture_signature(pair, pair.eq_b, "eq_b")
    violations = tuple(
        item for item in (violation_a, violation_b) if item is not None
    )
    if violations:
        return None, violations
    assert signature_a is not None and signature_b is not None
    try:
        observed = _classify(signature_a, signature_b)
    except Exception as exc:
        message = f"comparison raised unexpected {type(exc).__name__}: {exc}"
        return None, (_violation(pair, message, exception=exc),)
    if (
        pair.expected in _MERGE_EXPECTATIONS
        and observed in _MERGE_EXPECTATIONS
        and observed is not pair.expected
    ):
        message = f"expected {pair.expected.value}, observed {observed.value}"
        return observed, (_violation(pair, message, observed=observed),)
    return observed, ()


def _to_pair_rates(accumulator: _RateAccumulator) -> PairRates:
    pair_count = accumulator.pair_count
    false_merge_count = len(accumulator.false_merge_pair_ids)
    false_split_count = len(accumulator.false_split_pair_ids)
    denominator = float(pair_count) if pair_count else 1.0
    return PairRates(
        pair_count=pair_count,
        false_merge_count=false_merge_count,
        false_split_count=false_split_count,
        false_merge_rate=false_merge_count / denominator,
        false_split_rate=false_split_count / denominator,
        false_merge_pair_ids=tuple(accumulator.false_merge_pair_ids),
        false_split_pair_ids=tuple(accumulator.false_split_pair_ids),
    )


def _record_outcome(
    accumulator: _RateAccumulator,
    pair: CorpusPair,
    observed: Expected | None,
) -> None:
    accumulator.pair_count += 1
    if pair.expected is Expected.SPLIT and observed in _MERGE_EXPECTATIONS:
        accumulator.false_merge_pair_ids.append(pair.pair_id)
    elif pair.expected in _MERGE_EXPECTATIONS and observed is Expected.SPLIT:
        accumulator.false_split_pair_ids.append(pair.pair_id)


def evaluate_corpus() -> CorpusEvaluation:
    overall_accumulator = _RateAccumulator()
    family_accumulators: dict[str, _RateAccumulator] = {}
    violations: list[LabelingViolation] = []
    for pair in CORPUS:
        observed, pair_violations = _evaluate_pair(pair)
        family_accumulator = family_accumulators.setdefault(
            pair.family, _RateAccumulator()
        )
        _record_outcome(overall_accumulator, pair, observed)
        _record_outcome(family_accumulator, pair, observed)
        violations.extend(pair_violations)

    families = {
        family: FamilyRates(
            **vars(_to_pair_rates(family_accumulators[family])),
            declared=family in KNOWN_SPLIT_FAMILIES,
        )
        for family in sorted(family_accumulators)
    }
    return CorpusEvaluation(
        families=families,
        overall=_to_pair_rates(overall_accumulator),
        labeling_violations=violations,
    )


__all__ = [
    "CorpusEvaluation",
    "FamilyRates",
    "LabelingViolation",
    "PairRates",
    "evaluate_corpus",
]
