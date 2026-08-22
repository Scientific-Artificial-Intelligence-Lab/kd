
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from kd.core.equation.types import Equation
from kd.core.executor.context import ExecutionContext
from kd.core.expr.executor import PythonExecutor
from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs
from kd.core.verify import VerificationReport, VerifyPolicy, verify_equation
from kd.data.derivatives.finite_diff import MAX_SUPPORTED_ORDER
from kd.data.schema import DataTopology, PDEDataset

logger = logging.getLogger(__name__)



VerificationStatus = Literal["verified", "failed", "not_evaluated"]






_DEFAULT_MAX_ATOMIC_ORDER: int = MAX_SUPPORTED_ORDER
_DEFAULT_PROVIDER_KIND: str = "finite_diff"


@dataclass(frozen=True, kw_only=True)
class VerifyExecution:

    executor: PythonExecutor
    context: ExecutionContext
    provider_kind: str


VerifyContextFactory = Callable[[PDEDataset], "VerifyExecution"]


@dataclass(frozen=True, kw_only=True)
class MemberVerification:

    status: VerificationStatus
    stage: Literal["context_build", "verify"] | None
    error_type: str | None
    error_message: str | None
    report: VerificationReport | None


class _DefaultVerifyContextFactory:

    provider_kind: str = _DEFAULT_PROVIDER_KIND

    def __call__(self, dataset: PDEDataset) -> VerifyExecution:
        if dataset.topology is DataTopology.TABULAR:





            reqs = DerivativeReqs(
                provider_kind="none",
                lhs_order=0,
                lhs_source="field",
                supported_topologies=frozenset({DataTopology.TABULAR}),
            )
            components = PlatformBuilder(dataset, reqs).build()
            context = components.context
            if context is None:
                raise ValueError(
                    "default verify context factory expected a non-None "
                    "execution context for the tabular field-target bundle"
                )
            return VerifyExecution(
                executor=components.executor,
                context=context,
                provider_kind="none",
            )
        components = PlatformBuilder(
            dataset, DerivativeReqs(max_atomic_order=_DEFAULT_MAX_ATOMIC_ORDER)
        ).build()
        context = components.context
        if context is None:
            raise ValueError(
                "default verify context factory expected a non-None execution "
                "context (finite_diff provider), got a provider-less light bundle"
            )
        return VerifyExecution(
            executor=components.executor,
            context=context,
            provider_kind=_DEFAULT_PROVIDER_KIND,
        )


default_verify_context_factory = _DefaultVerifyContextFactory()


def verify_members(
    equations: Mapping[int, Equation],
    execution: VerifyExecution | None,
    *,
    policy: VerifyPolicy,
) -> dict[int, MemberVerification]:
    result: dict[int, MemberVerification] = {}
    if execution is None:
        for entry_index in equations:
            result[entry_index] = MemberVerification(
                status="not_evaluated",
                stage=None,
                error_type=None,
                error_message=None,
                report=None,
            )
        return result
    for entry_index, equation in equations.items():
        try:
            report = verify_equation(
                equation,
                executor=execution.executor,
                context=execution.context,
                policy=policy,
            )
        except Exception as exc:
            result[entry_index] = MemberVerification(
                status="failed",
                stage="verify",
                error_type=type(exc).__name__,
                error_message=str(exc),
                report=None,
            )
        else:
            result[entry_index] = MemberVerification(
                status="verified",
                stage=None,
                error_type=None,
                error_message=None,
                report=report,
            )
    return result


__all__ = [
    "MemberVerification",
    "VerificationStatus",
    "VerifyContextFactory",
    "VerifyExecution",
    "default_verify_context_factory",
    "verify_members",
]
