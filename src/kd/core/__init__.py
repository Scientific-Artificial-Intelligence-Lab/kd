
from typing import TYPE_CHECKING

from kd.core.evaluator import EvaluationResult, Evaluator
from kd.core.metrics import (
    ScorerFn,
    aic,
    aic_no_n,
    aicc,
    bic,
    make_aic_scorer,
    make_bic_scorer,
    make_sga_scorer,
    nmse,
)
from kd.core.safety import safe_div, safe_exp, safe_log
from kd.core.verify import (
    VERIFICATION_ARTIFACT_TAG,
    VerificationReport,
    VerifyPolicy,
    empirical_agreement,
    law_agreement,
    verify_equation,
    write_verification_artifact,
)

if TYPE_CHECKING:
    from kd.core.integrator import IntegrationResult, integrate_pde

_LAZY_INTEGRATOR = {"IntegrationResult", "integrate_pde"}


def __getattr__(name: str) -> object:





    if name in _LAZY_INTEGRATOR:
        from kd.core import integrator

        return getattr(integrator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_INTEGRATOR)


__all__ = [
    "safe_div",
    "safe_exp",
    "safe_log",
    "Evaluator",
    "EvaluationResult",
    "IntegrationResult",
    "integrate_pde",
    "ScorerFn",
    "aic",
    "aic_no_n",
    "aicc",
    "bic",
    "make_aic_scorer",
    "make_bic_scorer",
    "make_sga_scorer",
    "nmse",
    "VERIFICATION_ARTIFACT_TAG",
    "VerificationReport",
    "VerifyPolicy",
    "empirical_agreement",
    "law_agreement",
    "verify_equation",
    "write_verification_artifact",
]
