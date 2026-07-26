
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from torch import Tensor

from kd.core.equation.construct import PIVOT_UNITY_RTOL
from kd.core.equation.projection import active_law
from kd.core.equation.rendering import render_lhs_label
from kd.core.equation.signature import (
    LawAgreement,
    LawSignature,
    compare_laws,
    law_signature,
)
from kd.core.equation.types import Equation, Evolution, Form, Homogeneous, Scalar
from kd.core.executor.context import ExecutionContext
from kd.core.expr.executor import PythonExecutor
from kd.data.schema import compute_dataset_fingerprint


@dataclass(frozen=True)
class VerifyPolicy:

    nmse_max: float | None = None
    coeff_atol: float = 1e-2
    empirical_atol: float = 1e-2
    pivot_unity_rtol: float = PIVOT_UNITY_RTOL

    def __post_init__(self) -> None:
        values = [self.coeff_atol, self.empirical_atol, self.pivot_unity_rtol]
        if self.nmse_max is not None:
            values.append(self.nmse_max)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("verification policy thresholds must be finite")


@dataclass(frozen=True)
class VerificationReport:

    signature: LawSignature
    form: Form
    dataset_name: str
    dataset_fingerprint: str
    mse: float
    nmse: float
    r2: float | None
    residual_mean: float
    residual_std: float
    residual_max_abs: float
    n_samples: int
    inactive_coefficient_mass: float
    normalizer_term: str
    normalizer_variance: float
    policy: VerifyPolicy
    passed: bool | None

    def __post_init__(self) -> None:
        values = [
            self.mse,
            self.nmse,
            self.residual_mean,
            self.residual_std,
            self.residual_max_abs,
            self.inactive_coefficient_mass,
            self.normalizer_variance,
        ]
        if self.r2 is not None:
            values.append(self.r2)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("verification report measurements must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "signature": self.signature.to_dict(),
            "form": self.form.value,
            "dataset_name": self.dataset_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "mse": self.mse,
            "nmse": self.nmse,
            "r2": self.r2,
            "residual_mean": self.residual_mean,
            "residual_std": self.residual_std,
            "residual_max_abs": self.residual_max_abs,
            "n_samples": self.n_samples,
            "inactive_coefficient_mass": self.inactive_coefficient_mass,
            "normalizer_term": self.normalizer_term,
            "normalizer_variance": self.normalizer_variance,
            "policy": asdict(self.policy),
            "passed": self.passed,
        }


def _coefficient_value(coefficient: object, term_ir: str) -> float:
    if not isinstance(coefficient, Scalar):
        raise NotImplementedError(
            f"{type(coefficient).__name__} coefficient for {term_ir!r} is reserved"
        )
    return float(coefficient.value)


def _inactive_mass(eq: Equation) -> float:
    if eq.active_indices is None:
        return 0.0
    active = set(eq.active_indices)
    mass = math.fsum(
        abs(_coefficient_value(coefficient, term_ir))
        for index, (term_ir, coefficient) in enumerate(eq.terms)
        if index not in active
    )
    if not math.isfinite(mass):
        raise ValueError("inactive coefficient mass must be finite")
    return mass


def _execute_column(
    term_ir: str,
    *,
    executor: PythonExecutor,
    context: ExecutionContext,
) -> Tensor:
    try:
        return executor.execute(term_ir, context).value.detach().double()
    except Exception as exc:
        raise ValueError(f"failed to execute verification term {term_ir!r}") from exc


def _active_columns(
    eq: Equation,
    *,
    executor: PythonExecutor,
    context: ExecutionContext,
) -> tuple[list[float], list[Tensor]]:
    if not eq.terms:
        raise ValueError("active law must contain at least one term")
    coefficients = [
        _coefficient_value(coefficient, term_ir)
        for term_ir, coefficient in eq.terms
    ]
    if any(not math.isfinite(value) for value in coefficients):
        raise ValueError("active law coefficients must be finite")
    columns = [
        _execute_column(term_ir, executor=executor, context=context)
        for term_ir, _coefficient in eq.terms
    ]
    return coefficients, columns


def _weighted_sum(coefficients: list[float], columns: list[Tensor]) -> Tensor:
    total = coefficients[0] * columns[0]
    for coefficient, column in zip(
        coefficients[1:], columns[1:], strict=True
    ):
        total = total + coefficient * column
    return total


def _normalizer_variance(target: Tensor, *, term: str) -> float:
    variance = float(target.var(correction=0))
    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError(
            f"verification normalizer variance for column {term!r} must be "
            f"finite and > 0 (got {variance!r}); a constant column "
            "-- e.g. a time derivative on steady/equilibrium data -- has "
            "no usable variance scale"
        )
    return variance


def _measure(residual: Tensor, variance: float) -> tuple[float, ...]:
    mse = float((residual * residual).mean())
    nmse = mse / variance
    metrics = (
        mse,
        nmse,
        float(residual.mean()),
        float(residual.std(correction=0)),
        float(residual.abs().max()),
    )
    if any(not math.isfinite(value) for value in metrics):
        raise ValueError("verification metrics must be finite")
    return metrics


def verify_equation(
    eq: Equation,
    *,
    executor: PythonExecutor,
    context: ExecutionContext,
    policy: VerifyPolicy = VerifyPolicy(),
) -> VerificationReport:
    projected = active_law(eq)
    inactive_coefficient_mass = _inactive_mass(eq)
    signature = law_signature(eq)
    coefficients, columns = _active_columns(
        projected, executor=executor, context=context
    )

    if isinstance(projected, Evolution):
        normalizer_term = render_lhs_label(projected.lhs_spec)
        target = _execute_column(
            normalizer_term, executor=executor, context=context
        )
        residual = _weighted_sum(coefficients, columns) - target
        variance = _normalizer_variance(target, term=normalizer_term)
    elif isinstance(projected, Homogeneous):
        normalizer_term = projected.terms[0][0]
        residual = _weighted_sum(coefficients, columns)
        variance = _normalizer_variance(
            coefficients[0] * columns[0], term=normalizer_term
        )
    else:
        raise TypeError(f"unsupported equation type: {type(projected).__name__}")

    mse, nmse, residual_mean, residual_std, residual_max_abs = _measure(
        residual, variance
    )
    r2 = 1.0 - nmse if isinstance(projected, Evolution) else None
    passed = None if policy.nmse_max is None else nmse <= policy.nmse_max
    return VerificationReport(
        signature=signature,
        form=projected.form,
        dataset_name=context.dataset.name,
        dataset_fingerprint=compute_dataset_fingerprint(context.dataset),
        mse=mse,
        nmse=nmse,
        r2=r2,
        residual_mean=residual_mean,
        residual_std=residual_std,
        residual_max_abs=residual_max_abs,
        n_samples=residual.numel(),
        inactive_coefficient_mass=inactive_coefficient_mass,
        normalizer_term=normalizer_term,
        normalizer_variance=variance,
        policy=policy,
        passed=passed,
    )


def empirical_agreement(
    rep_a: VerificationReport,
    rep_b: VerificationReport,
    *,
    policy: VerifyPolicy = VerifyPolicy(),
) -> bool | None:
    if (
        rep_a.dataset_name != rep_b.dataset_name
        or rep_a.dataset_fingerprint != rep_b.dataset_fingerprint
    ):
        return None
    if rep_a.normalizer_term != rep_b.normalizer_term:
        return None
    return abs(rep_a.nmse - rep_b.nmse) <= policy.empirical_atol


def law_agreement(
    a: LawSignature,
    b: LawSignature,
    *,
    policy: VerifyPolicy = VerifyPolicy(),
) -> LawAgreement:
    return compare_laws(a, b, coeff_atol=policy.coeff_atol)


VERIFICATION_ARTIFACT_TAG = "kd-verification-v1"


def write_verification_artifact(
    report: VerificationReport,
    *,
    evidence_hash: str,
    path: str | Path,
) -> Path:
    if not evidence_hash:
        raise ValueError("evidence_hash must be a non-empty string")
    payload = {
        "artifact": VERIFICATION_ARTIFACT_TAG,
        "evidence_hash": evidence_hash,
        "report": report.to_dict(),
    }
    target = Path(path)
    target.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    return target


__all__ = [
    "VERIFICATION_ARTIFACT_TAG",
    "VerificationReport",
    "VerifyPolicy",
    "empirical_agreement",
    "law_agreement",
    "verify_equation",
    "write_verification_artifact",
]
