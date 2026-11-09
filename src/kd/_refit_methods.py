
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
import torch

from kd._evaluate_classify import (
    TermRejection,
    TermValidation,
    TermValidationReport,
    classify_terms,
)
from kd._fixed_terms import lowered_components, measure_fixed, merge_fixed
from kd._refit_frame import (
    AUTOGRAD_IMPLEMENTATION,
    FD2_IMPLEMENTATION,
    FD_IMPLEMENTATION,
    FrameDeclaration,
    RefitFrame,
    Trim,
    build_components,
    make_frame,
    require_source_binding,
    resolve_rows,
    unmodeled_field_reasons,
)
from kd._registry import Registry, extension_dependencies, registering_builtins
from kd._weak_form import (
    METHOD_VERSION,
    WeakFormDiagnostics,
    WeakFormOptions,
    WeakFrame,
    WeakRefusal,
    WeakTerm,
    check_weak_dataset,
    classify_weak_terms,
    raise_weak_refusals,
    resolve_weak_frame,
    solve_weak,
    weak_diagnostics,
    weak_zero_model,
)
from kd.core.equation._sketch_terms import lower_term
from kd.core.metrics import nmse as _metrics_nmse
from kd.core.platform.builder import resolve_lhs_defaults

if TYPE_CHECKING:
    from torch import Tensor

    from kd.core.evaluator import EvaluationResult, Evaluator
    from kd.data.schema import PDEDataset
    from kd.models import StoredSurrogate
    from kd.search.protocol import PlatformComponents


@dataclass(frozen=True)
class FrameFit:

    coefficients: tuple[float, ...]
    mse: float
    nmse: float
    r2: float
    condition_number: float | None
    weak: WeakFormDiagnostics | None = None


class ResolvedFrame(Protocol):

    @property
    def record(self) -> RefitFrame: ...
    @property
    def refused(self) -> Mapping[str, str]: ...
    def raise_refused(self) -> None:
        ...

    def fit(
        self, terms: Sequence[str], *, fixed: Mapping[str, float] | None = None
    ) -> FrameFit: ...
    def zero_model(self) -> tuple[float, float]:
        ...


class RefitMethod(FrameDeclaration, Protocol):

    @property
    def description(self) -> str: ...
    @property
    def dependencies(self) -> tuple[tuple[str, str], ...]: ...

    def resolve(
        self,
        dataset: PDEDataset,
        terms: Sequence[str],
        *,
        lhs_order: int,
        max_order: int,
        trim: Trim,
        rows: Tensor | None,
        surrogate: StoredSurrogate | None,
        weak: WeakFormOptions | None,
    ) -> ResolvedFrame: ...


_METHODS: Registry[RefitMethod] = Registry(
    "refit_method", implementation_of=lambda entry: entry.implementation
)


def register_refit_method(name: str, method: RefitMethod) -> None:
    if method.name != name:
        raise ValueError(
            f"refit method name {method.name!r} differs from registration {name!r}"
        )
    _METHODS.register(name, method, dependencies=method.dependencies)


def refit_methods() -> tuple[str, ...]:
    return _METHODS.names()


def refit_method(name: str) -> RefitMethod:
    try:
        return _METHODS.get(name)
    except KeyError as error:
        raise ValueError(str(error)) from error


@dataclass(frozen=True)
class _IdentifiedFrame:
    record: RefitFrame
    resolved: ResolvedFrame

    @property
    def refused(self) -> Mapping[str, str]:
        return self.resolved.refused

    def raise_refused(self) -> None:
        self.resolved.raise_refused()

    def fit(
        self, terms: Sequence[str], *, fixed: Mapping[str, float] | None = None
    ) -> FrameFit:
        return self.resolved.fit(terms, fixed=fixed)

    def zero_model(self) -> tuple[float, float]:
        return self.resolved.zero_model()


def resolve_refit_frame(
    dataset: PDEDataset,
    terms: Sequence[str],
    *,
    method: RefitMethod,
    lhs_order: int,
    max_order: int,
    trim: Trim,
    rows: Tensor | None,
    surrogate: StoredSurrogate | None,
    weak: WeakFormOptions | None,
) -> ResolvedFrame:
    resolved = method.resolve(
        dataset,
        terms,
        lhs_order=lhs_order,
        max_order=max_order,
        trim=trim,
        rows=rows,
        surrogate=surrogate,
        weak=weak,
    )
    identities = extension_dependencies("refit_method", method.name)
    identity, *components = identities
    if identity["source"] == "builtin":
        return resolved
    identity.pop("kind")
    record = dataclasses.replace(
        resolved.record,
        derivative_method=method.name,
        implementation=method.implementation,
        extension={"method": identity, "components": components},
    )
    return _IdentifiedFrame(record, resolved)


class InvalidTermsError(ValueError):

    def __init__(self, rejected: list[TermRejection]) -> None:
        self.rejected = list(rejected)
        summary = ", ".join(f"{rej.term!r} ({rej.reason})" for rej in self.rejected)
        super().__init__(f"Rejected terms: {summary}")


class EvaluationFailedError(RuntimeError):
    pass


@dataclass(frozen=True)
class _StrongFrame:
    record: RefitFrame
    components: PlatformComponents
    report: TermValidationReport
    rows: Tensor | None

    @property
    def refused(self) -> Mapping[str, str]:
        return {r.term: r.reason for r in self.report.rejected}

    def raise_refused(self) -> None:
        if self.report.rejected:
            raise InvalidTermsError(self.report.rejected)

    def fit(
        self, terms: Sequence[str], *, fixed: Mapping[str, float] | None = None
    ) -> FrameFit:
        fit = (
            fit_fixed(self.components, list(terms), fixed, rows=self.rows)
            if fixed
            else fit_valid(self.components, list(terms), rows=self.rows)
        )
        assert fit.coefficients is not None
        return FrameFit(
            tuple(fit.coefficients.tolist()),
            fit.mse,
            fit.nmse,
            fit.r2,
            fit.condition_number,
        )

    def zero_model(self) -> tuple[float, float]:
        evaluator = cast("Evaluator", self.components.evaluator)
        target = evaluator.lhs_target
        selected = self.rows
        if selected is not None:
            if selected.dtype == torch.bool:
                selected = selected.flatten().nonzero().flatten()
            target = target.index_select(0, selected.to(target.device))
        mse = float((target**2).mean().item())
        return mse, _metrics_nmse(mse, float(target.var(correction=0).item()))


@dataclass(frozen=True)
class StrongFormMethod:

    name: str
    implementation: str
    description: str
    field_source: str
    stencil_trim: bool
    surrogate: Literal["refused", "required"]
    provider_kind: str
    dependencies: tuple[tuple[str, str], ...]
    condition_number_basis: str = "raw_columns"
    row_selection: bool = True
    weak_options: bool = False

    def resolve(
        self,
        dataset: PDEDataset,
        terms: Sequence[str],
        *,
        lhs_order: int,
        max_order: int,
        trim: Trim,
        rows: Tensor | None,
        surrogate: StoredSurrogate | None,
        weak: WeakFormOptions | None,
    ) -> ResolvedFrame:
        if surrogate is not None:
            require_source_binding(dataset, surrogate)
        components = build_components(
            dataset,
            max_order=max_order,
            lhs_order=lhs_order,
            provider_kind=self.provider_kind,
            surrogate=surrogate,
        )
        report = classify_terms(list(terms), components, max_order=max_order)
        if surrogate is not None:
            report = _with_rejections(
                report, unmodeled_field_reasons(report.valid, components.dataset)
            )
        margins, selected, rows_sha256 = resolve_rows(
            components, trim=trim, rows=rows, terms=report.valid, lhs_order=lhs_order
        )
        record = make_frame(
            components.dataset,
            method=self,
            lhs_order=lhs_order,
            trim=margins,
            n_samples=_n_samples(components, selected),
            rows_sha256=rows_sha256,
            surrogate=surrogate,
        )
        return _StrongFrame(record, components, report, selected)


@dataclass(frozen=True)
class _WeakFrame:
    record: RefitFrame
    frame: WeakFrame
    members: Mapping[str, WeakTerm]
    refusals: tuple[WeakRefusal, ...]

    @property
    def refused(self) -> Mapping[str, str]:
        return {r.term: r.reason for r in self.refusals}

    def raise_refused(self) -> None:
        raise_weak_refusals(self.refusals)

    def fit(
        self, terms: Sequence[str], *, fixed: Mapping[str, float] | None = None
    ) -> FrameFit:
        held = {} if fixed is None else fixed
        try:
            solution = solve_weak(
                self.frame, terms, [self.members[term] for term in terms], fixed=held
            )
        except np.linalg.LinAlgError as error:
            raise EvaluationFailedError(
                f"weak least squares failed: {error}"
            ) from error
        if not math.isfinite(solution.mse):
            raise EvaluationFailedError("weak least squares produced a non-finite MSE")
        return FrameFit(
            coefficients=tuple(float(c) for c in solution.coefficients),
            mse=solution.mse,
            nmse=solution.nmse,
            r2=solution.r2,
            condition_number=solution.condition_number,
            weak=weak_diagnostics(
                self.frame,
                rank=solution.rank,
                n_columns=sum(term not in held for term in terms),
            ),
        )

    def zero_model(self) -> tuple[float, float]:
        return weak_zero_model(self.frame)


@dataclass(frozen=True)
class WeakFormMethod:

    name: str = "weak"
    implementation: str = METHOD_VERSION
    description: str = "Weak-form integration on raw-field windows."
    field_source: str = "observed"
    condition_number_basis: str = "unit_l2_columns"
    row_selection: bool = False
    stencil_trim: bool = False
    surrogate: Literal["refused", "required"] = "refused"
    weak_options: bool = True
    dependencies: tuple[tuple[str, str], ...] = ()

    def resolve(
        self,
        dataset: PDEDataset,
        terms: Sequence[str],
        *,
        lhs_order: int,
        max_order: int,
        trim: Trim,
        rows: Tensor | None,
        surrogate: StoredSurrogate | None,
        weak: WeakFormOptions | None,
    ) -> ResolvedFrame:
        resolved = resolve_lhs_defaults(dataset)
        spatial_axis, time_axis = check_weak_dataset(resolved, lhs_order)
        members, refusals = classify_weak_terms(
            terms, resolved, spatial_axis=spatial_axis, max_order=max_order
        )
        frame = resolve_weak_frame(
            resolved,
            WeakFormOptions() if weak is None else weak,
            spatial_axis=spatial_axis,
            time_axis=time_axis,
            lhs_order=lhs_order,
        )
        record = make_frame(
            resolved,
            method=self,
            lhs_order=lhs_order,
            trim=dict.fromkeys(cast(list[str], resolved.axis_order), 0),
            n_samples=frame.n_windows,
        )
        return _WeakFrame(record, frame, members, tuple(refusals))


def _with_rejections(
    report: TermValidationReport, reasons: Mapping[str, str]
) -> TermValidationReport:
    if not reasons:
        return report
    results = [
        TermValidation(r.term, False, reasons[r.term]) if r.term in reasons else r
        for r in report.results
    ]
    rejected = [TermRejection(r.term, r.reason or "") for r in results if not r.ok]
    return TermValidationReport(
        results=results,
        valid=[r.term for r in results if r.ok],
        rejected=rejected,
        ok=not rejected,
    )


def _n_samples(components: PlatformComponents, rows: Tensor | None) -> int:
    if rows is None:
        return components.dataset.get_field(components.dataset.lhs_field).numel()
    if rows.dtype == torch.bool:
        return int(rows.sum().item())
    return int(rows.numel())


def fit_fixed(
    components: PlatformComponents,
    terms: list[str],
    fixed: Mapping[str, float],
    *,
    rows: Tensor | None,
) -> EvaluationResult:
    base = cast("Evaluator", components.evaluator)
    free = [term for term in terms if term not in fixed]
    if free:
        fit = fit_valid(lowered_components(components, fixed), free, rows=rows)
        result = merge_fixed(fit, terms, fixed, base=base, rows=rows)
    else:
        values = [fixed[term] for term in terms]
        result = measure_fixed(base, terms, values, rows=rows, complexity=0)
    if not math.isfinite(result.mse):
        raise EvaluationFailedError(
            "the equation under fixed coefficients has a non-finite MSE"
        )
    return result


def fit_valid(
    components: PlatformComponents,
    terms: list[str],
    *,
    rows: Tensor | None,
) -> EvaluationResult:
    evaluator = cast("Evaluator", components.evaluator)
    irs = [lower_term(term) for term in terms]
    result = evaluator.evaluate_terms(irs, skip_invalid=False, rows=rows)
    if not result.is_valid:
        raise EvaluationFailedError(
            result.error_message or "Evaluation produced an invalid result"
        )
    return dataclasses.replace(result, terms=list(terms))


with registering_builtins():
    register_refit_method(
        "finite_diff",
        StrongFormMethod(
            name="finite_diff",
            implementation=FD_IMPLEMENTATION,
            description="Finite differences of the observed field.",
            field_source="observed",
            stencil_trim=True,
            surrogate="refused",
            provider_kind="finite_diff",
            dependencies=(
                ("derivative_provider", "finite_diff"),
                ("sparse_solver", "ols"),
            ),
        ),
    )
    register_refit_method(
        "autograd",
        StrongFormMethod(
            name="autograd",
            implementation=AUTOGRAD_IMPLEMENTATION,
            description="Autograd through a stored surrogate bound to the input.",
            field_source="surrogate",
            stencil_trim=False,
            surrogate="required",
            provider_kind="autograd",
            dependencies=(
                ("derivative_provider", "autograd"),
                ("sparse_solver", "ols"),
            ),
        ),
    )
    register_refit_method("weak", WeakFormMethod())
    register_refit_method(
        "fd2",
        StrongFormMethod(
            name="fd2",
            implementation=FD2_IMPLEMENTATION,
            description=(
                "Three-point second-order finite differences of the observed "
                "field; diagnostic profile."
            ),
            field_source="observed",
            stencil_trim=True,
            surrogate="refused",
            provider_kind="fd2",
            dependencies=(
                ("derivative_provider", "fd2"),
                ("sparse_solver", "ols"),
            ),
        ),
    )
