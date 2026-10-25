"""Agent-facing narrow entry for evaluating candidate PDE terms.

Responsibility: measuring fixed candidate terms against one dataset. That is
term validation and fitting, refits in a recorded measurement frame
(finite differences, autograd through a stored model, or the weak form),
drop-one ablation and batched scoring on one shared frame, and the two
verification signals (block dispersion, bounded forward error). The frame
rules live in ``kd._refit_frame`` and the weak-form numerics in
``kd._weak_form``; this module owns the public entries and result types.

This module is the single, stateless, fail-loud front door for "score this list
of candidate terms against a dataset". It replaces the 8-12 line hand-assembly of
provider -> context -> registry -> executor -> solver -> ``Evaluator`` that
callers (notably LLM agents) would otherwise repeat, and it
closes two agent-hostile traps:

1. ``skip_invalid=True`` silently dropping bad terms with only ``logger.debug``.
   Here, lenient mode drops with exactly ONE ``logger.warning`` naming each
   dropped term + reason; strict mode raises ``InvalidTermsError`` with a
   COMPLETE rejection report (all bad terms in one shot, not fail-at-first).
2. Invalid evaluations returning a penalty-sentinel result (``mse=1e10``,
   ``is_valid=False``). Here, an invalid fit raises ``EvaluationFailedError`` —
   if ``evaluate_terms`` returns at all, ``result.is_valid`` is guaranteed True.

Term syntax (canonical funcall IR or product notation, ENFORCED)
----------------------------------------------------------------
Each term must be a SINGLE term, in canonical funcall IR, e.g.
``"diff_x(u)"``, ``"mul(u, diff_x(u))"``, or a terminal token like ``"u_xx"``,
or in the product notation of ``kd.Sketch``: explicit multiplication and
positive integer powers of such terms (``"u*u_x"``, ``"u**2"``). A term in
product notation is lowered to its funcall IR
(``kd.core.equation._sketch_terms.lower_term``: ``u*u_x`` is ``mul(u,u_x)``,
``u**2`` the raw product ``mul(u,u)``, never the clipped ``n2(u)``; operand
order, nesting and literals are kept), which is what is executed, while every
result names the term in the caller's spelling and order. The contract is
ENFORCED, not merely documented: every term is first passed
through the platform's own ``split_terms`` (the same canonicality gate the
LINEAR evaluator uses), so two classes of non-canonical input are rejected up
front rather than silently mis-fitted as one column:

- Infix sums and divisions, unary and boolean operators (``"u + u_xx"``,
  ``"-u"``, ``"u^2"``) → ``syntax`` rejection. The executor *can* eval a
  BinOp string as a single column, which would hand back a bogus
  one-coefficient fit; the syntax layer blocks it.
- A top-level additive expression (``"add(u, u_xx)"``, ``"sub(u, u)"``) →
  ``composite`` rejection: it is really MULTIPLE terms. The reason lists the
  split parts so an agent can resubmit them separately (self-repair).

``diff_x`` / ``diff2_x`` ... are *open-form* derivative calls computed
on-the-fly: they are NOT gated by ``max_order`` (the on-the-fly FD
scheme itself caps at order 3). ``max_order`` gates only *terminal* tokens
(``u_xx`` / ``u_xxx`` cache columns), NOT open-form ``diff3_x(u)`` calls. To
force a third-order *terminal* token to resolve, raise ``max_order`` — and a
terminal token beyond the current ``max_order`` is rejected with an actionable
hint telling the agent which ``max_order`` to retry with.

Pure constants / scalars (e.g. ``"1"``) are rejected: a numeric literal is a
scalar, not a column, so it would broadcast to the wrong length and detonate the
fit unattributed. They surface as a ``shape`` rejection naming the term. The
constant column is spelled ``one``.

Rejection categories (applied in order)
---------------------------------------
``syntax`` (non-canonical) → ``composite`` (top-level add/sub) → ``execution``
(parse/exec error, incl. the max_order hint) → ``shape`` (wrong column length)
→ ``non-finite`` (NaN/Inf) → ``all-zero`` → ``tautology`` (duplicates the LHS
target). An allocation failure (CUDA OOM or CPU) propagates instead of becoming
a rejection.

LHS-tautology guard
-------------------
A term whose column is EXACTLY the LHS regression target (e.g. ``"diff_t(u)"``
under the default ``lhs_order=1``, where the LHS is ``u_t``) is rejected as a
``tautology``: fitting the target against itself yields the trivial
coefficient-1 / nmse≈0 "solution", which for an agent loop is the first
reward-hacking optimum (the predecessor SGA shipped explicit LHS defenses against this).
The guard detects EXACT duplicates only (bitwise ``torch.equal`` — the candidate
and the LHS come from the same finite-difference provider, so a genuine
tautology is bitwise-equal); near-duplicates are NOT detected. The guard is
relative to ``lhs_order``: ``diff_t(u)`` is a tautology under ``lhs_order=1`` but
a legitimate RHS term under ``lhs_order=2`` (the telegraph/wave case where the
LHS is ``u_tt``), so ``validate_terms`` accepts the same ``lhs_order`` keyword to
keep the two-call pattern consistent.

Two-call agent pattern
----------------------
Agents that want the structured report WITHOUT exceptions call
``validate_terms`` first (pure per-term classification, JSON-dumpable report),
inspect ``report.valid`` / ``report.rejected``, then call ``evaluate_terms`` on
the survivors. Agents that prefer exceptions call ``evaluate_terms`` directly.

Cost note (v1)
--------------
``evaluate_terms`` executes each term TWICE: once in the pre-validation pass
(``executor.execute`` under ``torch.no_grad``) to build the complete rejection
report, then again inside ``evaluator.evaluate_terms`` when fitting survivors.
This double execution is an accepted v1 cost — the validation pass guarantees a
complete, fail-loud report before any fit is attempted.

Statelessness
-------------
The caller's ``dataset`` object is never mutated. ``PlatformBuilder`` resolves
LHS defaults on an internal ``dataclasses.replace`` copy; nothing is written back
to the user's instance. Repeated calls with the same inputs are deterministic.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from scipy.linalg import norm

from kd._evaluate_classify import (
    TermRejection,
    TermValidation,
    TermValidationReport,
    classify_terms,
)
from kd._fixed_terms import (
    FixedConstraint,
    FixedPlacement,
    lowered_components,
    measure_fixed,
    merge_fixed,
    place_fixed,
    resolve_constraint,
)
from kd._refit_frame import (
    RefitFrame,
    Trim,
    build_components,
    check_frame_arguments,
    make_frame,
    require_source_binding,
    resolve_rows,
    unmodeled_field_reasons,
)
from kd._weak_form import (
    WeakDesignError,
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
from kd.core.equation import Evolution, LhsSpec, active_law, render_lhs_label
from kd.core.equation._sketch_terms import lower_term
from kd.core.expr.constant_fit import DEFAULT_STARTS, ConstantRefit, refit_literals
from kd.core.expr.registry import FunctionRegistry
from kd.core.expr.term_key import structure_term_key
from kd.core.metrics import nmse as _metrics_nmse
from kd.core.platform.builder import resolve_lhs_defaults
from kd.core.small_terms import small_term_indices
from kd.data import TabularDataset, interface_mask
from kd.data.schema import DataTopology, compute_dataset_fingerprint

if TYPE_CHECKING:
    from torch import Tensor

    from kd.core.equation import Sketch
    from kd.core.equation.sketch_input import SketchSpec
    from kd.core.evaluator import EvaluationResult, Evaluator
    from kd.data.schema import PDEDataset
    from kd.models import StoredSurrogate
    from kd.search.protocol import PlatformComponents
    from kd.search.result import ExperimentResult
    from kd.viz.report import ReportResult

logger = logging.getLogger(__name__)



__all__ = [
    "AblationResult",
    "AblationRow",
    "ComparisonResult",
    "ConstantRefit",
    "EvaluationFailedError",
    "ForwardError",
    "InvalidTermsError",
    "RefitFrame",
    "RefitResult",
    "TermRejection",
    "TermSetEntry",
    "TermSetScores",
    "TermValidation",
    "TermValidationReport",
    "TrajectoryForwardError",
    "WeakFormDiagnostics",
    "WeakFormOptions",
    "ablate_terms",
    "block_dispersion",
    "evaluate_terms",
    "compare_results",
    "forward_error",
    "forward_error_trajectories",
    "refit_constants",
    "refit_terms",
    "score_term_sets",
    "validate_terms",
]


@dataclass(frozen=True)
class ForwardError:
    """Outcome of ``forward_error``.

    Attributes:
        value: ``||pred - u||_2 / ||u||_2`` over the full grid, or None when
            the equation could not be integrated or was not measured.
        reason: None when ``value`` is set; otherwise ``"diverged"``,
            ``"solve-failed"`` (scipy ``solve_ivp`` failed), ``"refused"``
            (the equation or dataset is unsupported) or ``"timeout"`` (the
            ``max_seconds`` deadline passed: unmeasured, not a numerical
            outcome).
        warning: Why integration was refused, failed or stopped, empty on a
            clean run.
        terms: The submitted terms.
        coefficients: The submitted coefficients, one per term.
        input_identity: ``compute_dataset_fingerprint`` of the comparison
            input with its LHS defaults resolved.
        interval: ``(t_first, t_last)`` of the integration, None when the
            input is not a grid.
        boundary: Per spatial axis, ``"periodic"`` or ``"held_initial"`` (both
            endpoints held at their initial values); None when the input is
            not a grid.
    """

    value: float | None
    reason: str | None
    warning: str
    terms: tuple[str, ...] = ()
    coefficients: tuple[float, ...] = ()
    input_identity: str | None = None
    interval: tuple[float, float] | None = None
    boundary: Mapping[str, str] | None = None


@dataclass
class ComparisonResult:
    """Structure refits on one dataset, separate from native search scores.

    ``evaluations`` contains successful refits; ``exclusions`` names every
    run that could not be compared and explains why. Labels are supplied by
    the caller, so a preset answer can be labelled as such in every view.
    ``fixed`` is the constraint every run was refitted under, as law term to
    coefficient (``neg(...)`` folded into the value); empty without one.
    """

    results: dict[str, ExperimentResult]
    evaluations: dict[str, EvaluationResult]
    exclusions: dict[str, str]
    lhs_label: str
    reference: str | None
    fixed: Mapping[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "Unified platform evaluation (structure refit, same NMSE)",
            f"{'Run':<24} {'NMSE':>12} {'R^2':>10} Structure",
        ]
        if self.fixed:
            pins = ", ".join(f"{term} = {value}" for term, value in self.fixed.items())
            lines.insert(1, f"Refitted with fixed coefficients: {pins}")
        for name in self.results:
            if name in self.exclusions:
                lines.append(f"{name}: excluded - {self.exclusions[name]}")
            else:
                ev = self.evaluations[name]
                lines.append(
                    f"{name:<24} {ev.nmse:>12.4g} {ev.r2:>10.4f} "
                    + " + ".join(cast(list[str], ev.terms))
                )
        return "\n".join(lines)

    def render(self, output_path: str | Path, *, note: str = "") -> ReportResult:
        """Save the equation table, normalized progress and refit-NMSE sheet."""
        import matplotlib.pyplot as plt

        from kd.viz.plots import plot_structure_comparison
        from kd.viz.report import ReportResult

        figure, notes = plot_structure_comparison(self, note=note)
        path = Path(output_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, bbox_inches="tight")
        finally:
            plt.close(figure)
        return ReportResult(figures=[path], warnings=notes)


def compare_results(
    dataset: PDEDataset,
    results: Mapping[str, ExperimentResult],
    *,
    max_order: int = 3,
    fixed: Mapping[str, float] | None = None,
    sketch: Sketch | SketchSpec | None = None,
) -> ComparisonResult:
    """Refit each active evolution-law structure on the same FD features.

    The shared target is the dataset's LHS. This compares discovered
    structures, not the algorithms' native scores or original coefficients.
    Missing equations, other forms/LHSs, and refused term evaluations are
    explicitly excluded. Other errors propagate. Inputs are never mutated.

    With ``fixed`` or ``sketch`` (see ``evaluate_terms``) every run is
    refitted under the same fixed coefficients: a fixed term the run's
    structure lacks is added to it, so a run searched without the sketch is
    compared on the constrained structure rather than excluded. The
    constraint is recorded in ``ComparisonResult.fixed``. A refused
    constraint raises before any run is refitted.
    """
    resolved = resolve_lhs_defaults(dataset)
    lhs = LhsSpec(resolved.lhs_field, resolved.lhs_axis, resolved.lhs_order)
    constraint = resolve_constraint(
        resolved, fixed=fixed, sketch=sketch, lhs_order=lhs.order
    )
    comparison = ComparisonResult(
        dict(results),
        {},
        {},
        render_lhs_label(lhs),
        dataset.ground_truth,
        {} if constraint is None else dict(constraint.law),
    )
    for name, result in results.items():
        equation = result.equation
        if not isinstance(equation, Evolution):
            comparison.exclusions[name] = "no evolution equation to compare"
            continue
        if equation.lhs_spec != lhs:
            comparison.exclusions[name] = (
                f"LHS={render_lhs_label(equation.lhs_spec)} != {comparison.lhs_label}"
            )
            continue
        terms = [term for term, _ in active_law(equation).terms]
        if not terms:
            comparison.exclusions[name] = "equation has no terms to refit"
            continue
        try:
            comparison.evaluations[name] = evaluate_terms(
                resolved, terms, max_order=max_order, fixed=fixed, sketch=sketch
            )
        except (InvalidTermsError, EvaluationFailedError) as error:
            comparison.exclusions[name] = f"unified refit refused: {error}"
    return comparison






_DEFAULT_MAX_ORDER = 2



CV_DENOM_EPS = 1e-12







class InvalidTermsError(ValueError):
    """Raised when one or more terms are rejected and cannot be fitted.

    Carries the COMPLETE rejection report so an agent sees every bad term in a
    single round trip (strict mode), and is also raised when EVERY term is
    rejected under ``skip_invalid=True`` (a silent empty fit is forbidden).

    Attributes:
        rejected: Every rejected term with its classification.
    """

    def __init__(self, rejected: list[TermRejection]) -> None:
        self.rejected = list(rejected)
        summary = ", ".join(f"{rej.term!r} ({rej.reason})" for rej in self.rejected)
        super().__init__(f"Rejected terms: {summary}")


class EvaluationFailedError(RuntimeError):
    """Raised when the final fit is invalid (solver failure / non-finite MSE).

    The penalty-sentinel result (``mse=1e10``, ``is_valid=False``) must never
    reach the caller as a return value; this exception is raised instead,
    carrying the underlying diagnostic message.
    """







def validate_terms(
    dataset: PDEDataset,
    terms: Sequence[str],
    *,
    max_order: int = _DEFAULT_MAX_ORDER,
    lhs_order: int | None = None,
) -> TermValidationReport:
    """Classify each term as valid / rejected WITHOUT fitting.

    Builds the platform via ``PlatformBuilder`` and, for each term, runs the
    canonicality gate (``split_terms``) then — if canonical — executes it once
    (under ``torch.no_grad``) to classify it. Categories, in order: ``syntax``
    (non-canonical), ``composite`` (top-level add/sub), ``execution`` (parse/exec
    error, with max_order hint; an allocation failure propagates), ``shape``
    (wrong length), ``non-finite`` (NaN/Inf), ``all-zero``, ``tautology``
    (duplicates the LHS target). No least-squares fit is performed and the
    dataset is not mutated.

    Args:
        dataset: The PDE dataset to evaluate against (never mutated).
        terms: Candidate term strings (canonical funcall IR or product
            notation); the report names each in the caller's spelling.
        max_order: Maximum atomic derivative order made resolvable
            (``-> DerivativeReqs.max_atomic_order``). Gates terminal tokens such
            as ``u_xx``; open-form ``diff*_x`` calls are not gated.
        lhs_order: LHS derivative order the tautology guard compares against
            (``1`` -> ``u_t``, ``2`` -> ``u_tt``). Default ``None`` DERIVES the
            order from ``dataset.lhs_order`` (the single source of truth). An
            explicit value is a deliberate OVERRIDE (request semantics) — it may
            differ from ``dataset.lhs_order`` to probe a what-if target. Must
            match the ``lhs_order`` passed to the subsequent ``evaluate_terms``
            call so the two-call pattern stays consistent (a tautology under one
            order is not one under another).

    Returns:
        A ``TermValidationReport`` partitioning the terms into valid / rejected,
        JSON-dumpable via ``to_dict()``.

    Raises:
        ValueError: If ``terms`` is empty.
        ValueError: Propagated from ``PlatformBuilder`` when the resolved LHS
            field/axis is absent from the dataset, when the resolved
            ``lhs_order > max_order`` (the LHS is read from the same precomputed
            derivative cache that ``max_order`` bounds), or when
            ``max_order > 3`` (provider cap).
    """
    term_list = _require_nonempty(terms)
    lhs_order_resolved = _resolve_lhs_order(dataset, lhs_order)
    components = build_components(
        dataset, max_order=max_order, lhs_order=lhs_order_resolved
    )
    return classify_terms(term_list, components, max_order=max_order)


def evaluate_terms(
    dataset: PDEDataset,
    terms: Sequence[str],
    *,
    skip_invalid: bool = False,
    max_order: int = _DEFAULT_MAX_ORDER,
    lhs_order: int | None = None,
    rows: Tensor | None = None,
    fixed: Mapping[str, float] | None = None,
    sketch: Sketch | SketchSpec | None = None,
) -> EvaluationResult:
    """Evaluate candidate terms against a dataset, fail-loud.

    Pre-validates ALL terms (complete report), then fits the valid ones via the
    real platform ``Evaluator`` (OLS-on-Theta through ``LeastSquaresSolver``).

    Args:
        dataset: The PDE dataset to evaluate against (never mutated).
        terms: Candidate term strings (canonical funcall IR or product
            notation). ``result.terms`` keeps the caller's spelling and order.
        skip_invalid: When ``False`` (default, strict), ANY rejected term raises
            ``InvalidTermsError`` carrying every rejection and nothing is fitted.
            When ``True`` (lenient), rejected terms are dropped with exactly one
            ``logger.warning`` and the survivors are fitted; if ALL terms are
            rejected, ``InvalidTermsError`` is still raised (no silent empty fit).
        max_order: Maximum atomic derivative order (``-> max_atomic_order``).
        lhs_order: LHS derivative order targeted by the fit (``1`` -> ``u_t``,
            ``2`` -> ``u_tt``; ``-> DerivativeReqs.lhs_order``). Default ``None``
            DERIVES from ``dataset.lhs_order`` (single source of truth); an
            explicit value is a deliberate OVERRIDE (request semantics, may
            differ from ``dataset.lhs_order``).
        rows: Grid-shaped boolean mask or flat int32/int64 sample indices.
            Derivatives and term validation run on the complete grid first;
            fitting, residuals and metrics use only selected rows. Indices
            preserve order and duplicates. An empty selection is an error.
            ``None`` uses the original full-grid fit. The dataset is unchanged.
        fixed: Term to coefficient for terms whose coefficient is held, not
            fitted. Keys are written as ``terms`` are (``u*u_x`` and
            ``mul(u,u_x)`` fix one column). A key matches the input term with
            the same law identity (``mul(u,u_x)`` fixes ``mul(u_x,u)``;
            ``neg(...)`` folds into the sign) and is appended, in the key's
            spelling, after the input terms when none has it. Each fixed
            column times its value is deducted from the LHS target, the other
            terms are fitted to the rest, and the result reports the full
            equation:
            coefficients in result order with each fixed value as given (in
            the data's dtype, so exactly on float64 data), and the MSE, NMSE
            and R-squared of the full equation on the LHS itself. With every
            term fixed nothing is solved. The result does
            not mark which terms were fixed; the caller holds ``fixed``.
            Refused: a zero or non-finite value, a key without a law identity,
            a key that is the LHS in any spelling or sign, two keys naming
            one column (two spellings of one term included, even with equal
            values), an input term that is an alias spelling of a fixed
            column (shares a jet monomial with it) or has no jet expansion,
            and one key matching several input terms. A rejected fixed term
            raises even under ``skip_invalid=True``.
        sketch: A ``kd.Sketch`` whose pinned terms are the fixed
            coefficients; its anchored terms and holes do not change the
            fit. A shorthand Sketch is resolved on this fit's LHS (``lhs_order``
            included) without modifying ``dataset``; a Sketch written for
            another LHS is refused. Not with ``fixed``.

    Returns:
        An ``EvaluationResult`` with ``is_valid is True`` (honest metrics).
        This entry always measures ``condition_number`` (candidate-library
        collinearity). The measured matrix is the one the solver was handed:
        under ``skip_invalid=True`` that is the SURVIVING columns, not every
        string in ``terms`` — a term dropped as ``shape`` / ``all-zero`` /
        ``tautology`` is not in the number. Strict mode fits every submitted
        term or raises, so there the two coincide. Under fixed coefficients
        it is the matrix of the fitted (free) columns, None when every term
        is fixed; ``complexity`` and ``score`` count the free columns.

    Raises:
        ValueError: If ``terms`` is empty.
        ValueError: Propagated from ``PlatformBuilder`` when the resolved LHS
            field/axis is absent from the dataset, when the resolved
            ``lhs_order > max_order`` (the LHS is read from the same precomputed
            derivative cache that ``max_order`` bounds), or when
            ``max_order > 3`` (provider cap).
        ValueError: A refused ``fixed`` or ``sketch`` (see those arguments).
        InvalidTermsError: If terms are rejected (see ``skip_invalid``), or an
            input term cannot be fitted beside the fixed terms.
        EvaluationFailedError: If the final fit is invalid (the penalty sentinel
            is never returned).
    """
    term_list = _require_nonempty(terms)
    lhs_order_resolved = _resolve_lhs_order(dataset, lhs_order)
    constraint = _constraint(
        dataset, fixed=fixed, sketch=sketch, lhs_order=lhs_order_resolved
    )
    placement = _placed(constraint, term_list)
    components = build_components(
        dataset, max_order=max_order, lhs_order=lhs_order_resolved
    )
    if placement is None:
        report = classify_terms(term_list, components, max_order=max_order)
        _handle_rejections(report, skip_invalid=skip_invalid)
        return _fit_valid(components, report.valid, rows=rows)
    union = [*term_list, *placement.appended]
    report = classify_terms(union, components, max_order=max_order)
    _refuse_rejected_fixed(report, placement.values)
    _handle_rejections(report, skip_invalid=skip_invalid)
    return _fit_fixed(components, report.valid, placement.values, rows=rows)


@dataclass(frozen=True)
class RefitResult:
    """Outcome of ``refit_terms``.

    Attributes:
        terms: The kept terms, in input order and input spelling, then any
            fixed term absent from the input, in the caller's spelling.
        coefficients: Refitted coefficients aligned with ``terms``; a fixed
            term's coefficient is its fixed value.
        dropped: ``(term, input coefficient)`` for every pruned term, in input
            order; empty when nothing was pruned.
        nmse: Residual MSE over the population variance of the LHS target:
            the selected rows of the LHS under ``finite_diff`` and
            ``autograd``, the integrated weak target under ``weak``. Under
            fixed coefficients it is still that target, not the target minus
            the fixed terms.
        r2: R-squared of the refit on the same target.
        condition_number: Condition number of the fitted-term matrix (the
            free terms under fixed coefficients), measured on the basis
            ``frame.condition_number_basis`` names; None when every term is
            fixed.
        lhs_order: The LHS derivative order the fit targeted, resolved from
            ``dataset.lhs_order`` when the caller passed ``None``.
        derivative_source: The method's name: ``"finite_diff"``,
            ``"autograd"`` or ``"weak"``.
        mse: Residual MSE of the refit on the same target as ``nmse``.
        weak: Window, correction and solve diagnostics of a weak fit, None
            otherwise.
        frame: How the refit measured (``RefitFrame``). ``refit_terms``
            always sets it; the default exists for hand-built results.
        fixed: The terms of ``terms`` whose coefficient was fixed, in result
            order; empty without ``fixed`` or ``sketch``.
    """

    terms: tuple[str, ...]
    coefficients: tuple[float, ...]
    dropped: tuple[tuple[str, float], ...]
    nmse: float
    r2: float
    condition_number: float | None
    lhs_order: int
    derivative_source: str
    mse: float | None = None
    weak: WeakFormDiagnostics | None = None
    frame: RefitFrame | None = None
    fixed: tuple[str, ...] = ()


def refit_terms(
    dataset: PDEDataset,
    terms: Sequence[str],
    coefficients: Sequence[float] | None = None,
    *,
    small_term_rtol: float = 0.0,
    lhs_order: int | None = None,
    max_order: int = 3,
    method: str = "finite_diff",
    trim: Trim = 0,
    rows: Tensor | None = None,
    surrogate: StoredSurrogate | None = None,
    weak: WeakFormOptions | None = None,
    fixed: Mapping[str, float] | None = None,
    sketch: Sketch | SketchSpec | None = None,
) -> RefitResult:
    """Prune small terms by their input coefficients, then refit the rest once.

    Pruning uses ``kd.core.small_terms.small_term_indices``, the same rule as
    ``Model(small_term_rtol=...)``. With ``m`` the largest finite ``|c|`` among
    the input coefficients, a finite coefficient with
    ``|c| < small_term_rtol * m`` is dropped; a non-finite coefficient is kept
    and never sets ``m``; the largest term always survives. Pruning happens
    before the fit, so a pruned term is never validated or executed.

    ``method`` chooses where columns and target come from; nothing falls back
    to another method, and ``result.frame`` records the resolved choice:

    - ``"finite_diff"`` (default): the platform stencils on the complete
      ``dataset``, then row selection. With ``small_term_rtol=0``,
      ``trim=0`` and ``rows=None`` the result equals
      ``evaluate_terms(dataset, terms, max_order=max_order,
      lhs_order=lhs_order)`` bit for bit.
    - ``"autograd"``: field values, RHS derivatives and the LHS all come from
      ``surrogate``, a stored ``FieldModel`` whose identity binds it to
      ``dataset`` (``AutogradProvider`` + ``SurrogateContext``). A term on a
      field other than the LHS field is rejected: the model supplies only
      the LHS field.
    - ``"weak"``: integration by parts against compactly supported test
      functions on the raw field (``kd-weak-v1``, defined in
      ``kd._weak_form``): one real field, one spatial axis, a uniform grid,
      first-order evolution, and the terms ``one``, the field, its square and
      cube (``n2``/``n3`` of it, or the raw products ``mul(u, u)`` and
      ``mul(mul(u, u), u)`` in either bracketing, each under its own label),
      pure spatial derivatives of order 1 to 3 (``max_order`` bounds every
      spelling) and ``mul`` of the field with its first derivative.
      ``nmse`` is the weak residual MSE over the population variance of the
      integrated target, not a full-grid finite-difference NMSE, and
      ``condition_number`` is measured on unit-L2 columns. ``result.weak``
      holds the windows, correction and rank. Window rows overlap, so their
      count is not a count of independent observations.

    Rows: ``trim=k`` drops ``k`` grid points at both ends of every axis, a
    mapping drops them per axis (unnamed axes keep everything), and
    ``"by_order"`` drops the stencil footprint of the terms and the LHS on
    every axis (nested derivatives accumulate; a periodic axis needs none
    unless the coordinate-leaf rule fires). Trimming selects rows after
    differentiating the complete input; it never crops the dataset first.
    ``rows`` takes ``evaluate_terms(rows=...)`` semantics. A nonzero ``trim``
    together with ``rows`` is refused.

    The fit executes the input IR verbatim; spellings are not algebraically
    canonicalized. A nested derivative such as ``diff2_x(diff_x(add(u, x)))``
    is a different finite-difference column from ``u_xxx``: on one KdV
    candidate the refitted coefficient of that term differs by 6.7% between
    the two spellings. To refit on a smoothed field, the caller smooths it
    first (``kd.data.savgol_smooth`` or ``kd.data.lowpass_smooth``) and passes
    the smoothed dataset. On noisy data a finite-difference refit can be much
    worse than the coefficients an algorithm reports from surrogate
    derivatives: the true Burgers terms on a field with 1% relative noise
    refit to an NMSE of about 0.9.

    Fixed coefficients (``fixed`` or ``sketch``, as in ``evaluate_terms``):
    each fixed column times its value is deducted from the target, the free
    terms are fitted to the rest on the same rows or windows, and the result
    reports the full equation on the original target, so a constrained and an
    unconstrained refit of one support are measured on one ruler. A fixed
    term absent from ``terms`` is appended; ``result.fixed`` names the fixed
    terms. Pruning never drops a fixed term and never takes its value as the
    reference magnitude (its entry in ``coefficients`` is ignored); the
    constraint is placed on the kept terms, so a pruned term is never checked
    against it. The weak form deducts the fixed columns from the integrated
    target the same way; a fixed term outside the weak class is refused like
    any other term.

    The function judges nothing: it takes no reference equation and returns
    no verdict.

    Args:
        dataset: The PDE dataset to fit against (never mutated).
        terms: Candidate terms in canonical funcall IR or product notation,
            as in ``evaluate_terms``; ``result.terms`` and ``result.dropped``
            keep the caller's spelling.
        coefficients: One input coefficient per term, used only for pruning;
            required when ``small_term_rtol > 0``.
        small_term_rtol: Relative pruning tolerance in ``[0, 1)``.
        lhs_order: As in ``evaluate_terms``; ``None`` takes
            ``dataset.lhs_order``.
        max_order: As in ``evaluate_terms``. The default 3 keeps third-order
            terminal tokens such as ``u_xxx`` resolvable.
        method: ``"finite_diff"``, ``"autograd"`` or ``"weak"``.
        trim: ``0`` (default), a nonnegative integer, a mapping from axis
            name to a nonnegative integer, or ``"by_order"``. Not with
            ``autograd`` as ``"by_order"`` (no stencils).
        rows: Grid-shaped boolean mask or flat integer sample indices.
        surrogate: The stored surrogate ``autograd`` differentiates
            (``kd.models.load_stored_surrogate``); refused by other methods.
        weak: ``WeakFormOptions`` for ``method="weak"`` (None takes the
            defaults: automatic widths, no correction); refused otherwise.
        fixed: Term to fixed coefficient; see ``evaluate_terms``.
        sketch: A ``kd.Sketch`` whose pinned terms are the fixed
            coefficients; see ``evaluate_terms``. A shorthand Sketch binds to
            the LHS at the resolved ``lhs_order``.

    Returns:
        A ``RefitResult`` with ``mse`` and ``frame`` set, and ``weak`` for a
        weak fit.

    Raises:
        ValueError: If ``small_term_rtol`` is outside ``[0, 1)`` (NaN
            included), if ``coefficients`` is given with a length other than
            ``len(terms)`` (at any tolerance), or if ``small_term_rtol > 0``
            without ``coefficients``; for an unknown ``method`` or a refused
            argument combination; for a surrogate without an identity or
            bound to another dataset; a refused ``fixed`` or ``sketch``.
            Errors from ``evaluate_terms`` (``ValueError``,
            ``InvalidTermsError``, ``EvaluationFailedError``) propagate
            unchanged.
            Under ``weak``: a derivative order above ``max_order``, invalid
            widths, an axis without a spectral corner or legal width, fewer
            windows than columns, an identically zero column or a degenerate
            target variance.
        NotImplementedError: As in ``evaluate_terms``: a tabular dataset or a
            resolved LHS order of 0; only grid datasets with a first- or
            second-order LHS are fitted. Under ``weak``: a term outside the
            class or an unsupported dataset (see ``method``).
        EvaluationFailedError: The final fit is invalid (weak: the SVD did
            not converge).

    Two spellings of one column (``mul(u,u_x)`` and ``mul(u_x,u)``) are two
    exactly collinear columns: the fit returns the minimum-norm split between
    them with ``condition_number`` infinite, as ``evaluate_terms`` does.
    """
    constraint = _constraint(dataset, fixed=fixed, sketch=sketch, lhs_order=lhs_order)
    held = (
        frozenset() if constraint is None else place_fixed(constraint, terms).positions
    )
    kept, dropped = _prune_small_terms(terms, coefficients, small_term_rtol, held)
    placement = _placed(constraint, kept)
    check_frame_arguments(method, trim=trim, rows=rows, surrogate=surrogate, weak=weak)
    values = {} if placement is None else placement.values
    appended = () if placement is None else placement.appended
    term_list = _require_nonempty([*kept, *appended])
    lhs_order_resolved = _resolve_lhs_order(dataset, lhs_order)
    if method == "weak":
        members, refusals = _weak_members(
            dataset, term_list, lhs_order=lhs_order_resolved, max_order=max_order
        )
        raise_weak_refusals(refusals)
        frame, record = _weak_frame(dataset, weak, lhs_order=lhs_order_resolved)
        return _weak_result(term_list, members, frame, record, dropped, values)
    strong = _strong_frame(
        dataset,
        term_list,
        method=method,
        lhs_order=lhs_order_resolved,
        max_order=max_order,
        trim=trim,
        rows=rows,
        surrogate=surrogate,
    )
    _handle_rejections(strong.report, skip_invalid=False)
    if not values:
        fit = _fit_valid(strong.components, strong.report.valid, rows=strong.rows)
        return _strong_result(term_list, fit, strong.frame, dropped)
    fit = _fit_fixed(strong.components, term_list, values, rows=strong.rows)
    held_terms = tuple(term for term in term_list if term in values)
    return _strong_result(term_list, fit, strong.frame, dropped, fixed=held_terms)


@dataclass(frozen=True)
class _StrongFrame:
    """One resolved finite-difference or autograd frame and its components."""

    components: PlatformComponents
    report: TermValidationReport
    rows: Tensor | None
    frame: RefitFrame


def _strong_frame(
    dataset: PDEDataset,
    terms: list[str],
    *,
    method: str,
    lhs_order: int,
    max_order: int,
    trim: Trim,
    rows: Tensor | None,
    surrogate: StoredSurrogate | None,
) -> _StrongFrame:
    """Build components, validate ``terms`` once and resolve the fitting rows.

    ``terms`` is the union the frame serves; ``trim="by_order"`` reads the
    stencil footprints of its valid members.
    """
    if surrogate is not None:
        require_source_binding(dataset, surrogate)
    components = build_components(
        dataset, max_order=max_order, lhs_order=lhs_order, surrogate=surrogate
    )
    report = classify_terms(terms, components, max_order=max_order)
    if surrogate is not None:
        report = _with_rejections(
            report, unmodeled_field_reasons(report.valid, components.dataset)
        )
    margins, selected, rows_sha256 = resolve_rows(
        components, trim=trim, rows=rows, terms=report.valid
    )
    frame = make_frame(
        components.dataset,
        method=method,
        lhs_order=lhs_order,
        trim=margins,
        n_samples=_n_samples(components, selected),
        rows_sha256=rows_sha256,
        surrogate=surrogate,
    )
    return _StrongFrame(components, report, selected, frame)


def _with_rejections(
    report: TermValidationReport, reasons: Mapping[str, str]
) -> TermValidationReport:
    """``report`` with the valid terms named in ``reasons`` rejected."""
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


def _weak_members(
    dataset: PDEDataset, terms: Sequence[str], *, lhs_order: int, max_order: int
) -> tuple[dict[str, WeakTerm], list[WeakRefusal]]:
    """Check the dataset for the weak form and classify ``terms``."""
    resolved = resolve_lhs_defaults(dataset)
    spatial_axis, _ = check_weak_dataset(resolved, lhs_order)
    return classify_weak_terms(
        terms, resolved, spatial_axis=spatial_axis, max_order=max_order
    )


def _weak_frame(
    dataset: PDEDataset, options: WeakFormOptions | None, *, lhs_order: int
) -> tuple[WeakFrame, RefitFrame]:
    """Resolve the weak windows, target and sigma once, and their record."""
    resolved = resolve_lhs_defaults(dataset)
    spatial_axis, time_axis = check_weak_dataset(resolved, lhs_order)
    frame = resolve_weak_frame(
        resolved,
        WeakFormOptions() if options is None else options,
        spatial_axis=spatial_axis,
        time_axis=time_axis,
    )
    record = make_frame(
        resolved,
        method="weak",
        lhs_order=lhs_order,
        trim=dict.fromkeys(cast(list[str], resolved.axis_order), 0),
        n_samples=frame.n_windows,
    )
    return frame, record


def _weak_result(
    terms: Sequence[str],
    members: Mapping[str, WeakTerm],
    frame: WeakFrame,
    record: RefitFrame,
    dropped: tuple[tuple[str, float], ...],
    fixed: Mapping[str, float] | None = None,
) -> RefitResult:
    """Fit ``terms`` on the weak ``frame``; a failed solve fails loud.

    A term in ``fixed`` keeps its value (``solve_weak``).
    """
    held = {} if fixed is None else fixed
    try:
        solution = solve_weak(
            frame, terms, [members[term] for term in terms], fixed=held
        )
    except np.linalg.LinAlgError as error:
        raise EvaluationFailedError(f"weak least squares failed: {error}") from error
    if not math.isfinite(solution.mse):
        raise EvaluationFailedError("weak least squares produced a non-finite MSE")
    return RefitResult(
        terms=tuple(terms),
        coefficients=tuple(float(c) for c in solution.coefficients),
        dropped=dropped,
        nmse=solution.nmse,
        r2=solution.r2,
        condition_number=solution.condition_number,
        lhs_order=record.lhs_order,
        derivative_source="weak",
        mse=solution.mse,
        weak=weak_diagnostics(
            frame,
            rank=solution.rank,
            n_columns=sum(term not in held for term in terms),
        ),
        frame=record,
        fixed=tuple(term for term in terms if term in held),
    )


def _n_samples(components: PlatformComponents, rows: Tensor | None) -> int:
    """Number of fitted rows for a row selection (None is the full grid)."""
    if rows is None:
        return components.dataset.get_field(components.dataset.lhs_field).numel()
    if rows.dtype == torch.bool:
        return int(rows.sum().item())
    return int(rows.numel())


def _strong_result(
    terms: Sequence[str],
    fit: EvaluationResult,
    frame: RefitFrame,
    dropped: tuple[tuple[str, float], ...],
    *,
    fixed: tuple[str, ...] = (),
) -> RefitResult:
    assert fit.coefficients is not None
    return RefitResult(
        terms=tuple(terms),
        coefficients=tuple(fit.coefficients.tolist()),
        dropped=dropped,
        nmse=fit.nmse,
        r2=fit.r2,
        condition_number=fit.condition_number,
        lhs_order=frame.lhs_order,
        derivative_source=frame.derivative_method,
        mse=fit.mse,
        frame=frame,
        fixed=fixed,
    )


@dataclass(frozen=True)
class AblationRow:
    """One drop-one refit of ``ablate_terms``.

    Attributes:
        dropped_term: The term left out.
        retained_terms: The other terms, in input order; empty for the zero
            model.
        coefficients: Their refitted coefficients.
        mse: Residual MSE on the baseline's target and rows or windows.
        nmse: ``mse`` over the population variance of that target.
        delta_mse: ``mse - baseline.mse``.
        relative_rise: ``delta_mse / baseline.mse``, None on a zero baseline.
    """

    dropped_term: str
    retained_terms: tuple[str, ...]
    coefficients: tuple[float, ...]
    mse: float
    nmse: float
    delta_mse: float
    relative_rise: float | None


@dataclass(frozen=True)
class AblationResult:
    """Outcome of ``ablate_terms``.

    Attributes:
        baseline: The fit of every term; its ``frame`` (and ``weak`` for the
            weak form) is the one measurement setting every row reuses.
        rows: One drop-one refit per term, in input order.
        relative_rise_reason: ``"zero_baseline"`` when the baseline MSE is
            exactly zero and no relative rise is defined, otherwise None.
    """

    baseline: RefitResult
    rows: tuple[AblationRow, ...]
    relative_rise_reason: str | None


@dataclass(frozen=True)
class TermSetEntry:
    """One term set of ``score_term_sets``: its fit or why it has none.

    ``failure_kind`` is ``"refused"`` (a term was rejected or is outside the
    method's class) or ``"fit_failed"`` (the fit itself failed).
    """

    terms: tuple[str, ...]
    fit: RefitResult | None
    failure: str | None
    failure_kind: str | None


@dataclass(frozen=True)
class TermSetScores:
    """Outcome of ``score_term_sets``: the shared frame and one entry per set."""

    frame: RefitFrame
    entries: tuple[TermSetEntry, ...]


def ablate_terms(
    dataset: PDEDataset,
    terms: Sequence[str],
    *,
    lhs_order: int | None = None,
    max_order: int = 3,
    method: str = "finite_diff",
    trim: Trim = 0,
    rows: Tensor | None = None,
    surrogate: StoredSurrogate | None = None,
    weak: WeakFormOptions | None = None,
) -> AblationResult:
    """Refit the support, then refit it once without each term.

    One frame is resolved for the whole support (``trim="by_order"`` reads
    every term's footprint; weak widths and sigma are selected once) and
    every reduced fit reuses its rows or windows, target and correction.
    Dropping the only term scores the zero model on the same target. The
    rise of a necessary term is large; a heuristic threshold on it is the
    caller's decision, not a verdict this function returns.

    Args:
        dataset: The PDE dataset to fit against (never mutated).
        terms: The support; duplicate column identities
            (``kd.core.expr.term_key.structure_term_key``) are refused.
        lhs_order, max_order, method, trim, rows, surrogate, weak: As in
            ``refit_terms``.

    Returns:
        An ``AblationResult``.

    Raises:
        ValueError: Duplicate terms, an empty support, or any argument error
            of ``refit_terms``.
        InvalidTermsError, NotImplementedError: A term the method refuses
            (as in ``refit_terms``).
        EvaluationFailedError: The baseline fit failed, or a reduced fit
            failed; the message names the dropped term. A failed reduced fit
            never becomes zero importance.
    """
    check_frame_arguments(method, trim=trim, rows=rows, surrogate=surrogate, weak=weak)
    term_list = _require_nonempty(terms)
    _refuse_duplicates(term_list)
    frame = _union_frame(
        dataset,
        term_list,
        method=method,
        lhs_order=_resolve_lhs_order(dataset, lhs_order),
        max_order=max_order,
        trim=trim,
        rows=rows,
        surrogate=surrogate,
        weak=weak,
    )
    _raise_refused(frame)
    baseline = _fit_in(frame, term_list)
    assert baseline.mse is not None
    zero = baseline.mse == 0.0
    ablation_rows = []
    for index, term in enumerate(term_list):
        retained = term_list[:index] + term_list[index + 1:]
        try:
            reduced = _reduced_fit(frame, retained)
        except (EvaluationFailedError, WeakDesignError) as error:
            raise EvaluationFailedError(
                f"the ablation fit without {term!r} failed: {error}"
            ) from error
        coefficients, mse, nmse = reduced
        delta = mse - baseline.mse
        ablation_rows.append(
            AblationRow(
                dropped_term=term,
                retained_terms=tuple(retained),
                coefficients=coefficients,
                mse=mse,
                nmse=nmse,
                delta_mse=delta,
                relative_rise=None if zero else delta / baseline.mse,
            )
        )
    return AblationResult(
        baseline=baseline,
        rows=tuple(ablation_rows),
        relative_rise_reason="zero_baseline" if zero else None,
    )


def score_term_sets(
    dataset: PDEDataset,
    term_sets: Sequence[Sequence[str]],
    *,
    lhs_order: int | None = None,
    max_order: int = 3,
    method: str = "finite_diff",
    trim: Trim = 0,
    rows: Tensor | None = None,
    surrogate: StoredSurrogate | None = None,
    weak: WeakFormOptions | None = None,
) -> TermSetScores:
    """Fit several fixed supports in one frame resolved for their union.

    Every set is fitted on the same rows or windows, target and correction
    (``trim="by_order"`` covers the footprint of every term in any set). A
    set with a refused term, or whose fit fails, gets a classified
    ``failure`` and no fit; it is never scored as zero.

    Args:
        dataset: The PDE dataset to fit against (never mutated).
        term_sets: The supports, each non-empty.
        lhs_order, max_order, method, trim, rows, surrogate, weak: As in
            ``refit_terms``.

    Returns:
        ``TermSetScores`` with one entry per set, in input order.

    Raises:
        ValueError: No sets, an empty set, or an argument error of
            ``refit_terms`` (including a failure to resolve the frame).
        NotImplementedError: A dataset the method does not support.
    """
    check_frame_arguments(method, trim=trim, rows=rows, surrogate=surrogate, weak=weak)
    if not term_sets:
        raise ValueError("score_term_sets needs at least one term set")
    sets = [tuple(_require_nonempty(term_set)) for term_set in term_sets]
    union = list(dict.fromkeys(term for term_set in sets for term in term_set))
    frame = _union_frame(
        dataset,
        union,
        method=method,
        lhs_order=_resolve_lhs_order(dataset, lhs_order),
        max_order=max_order,
        trim=trim,
        rows=rows,
        surrogate=surrogate,
        weak=weak,
    )
    return TermSetScores(
        frame=frame.record,
        entries=tuple(_score_one(frame, term_set) for term_set in sets),
    )


@dataclass(frozen=True)
class _UnionFrame:
    """One frame resolved for a union of terms, and the terms it refuses."""

    record: RefitFrame
    refused: dict[str, str]
    strong: _StrongFrame | None = None
    weak: WeakFrame | None = None
    members: Mapping[str, WeakTerm] | None = None
    weak_refusals: tuple[WeakRefusal, ...] = ()


def _union_frame(
    dataset: PDEDataset,
    terms: list[str],
    *,
    method: str,
    lhs_order: int,
    max_order: int,
    trim: Trim,
    rows: Tensor | None,
    surrogate: StoredSurrogate | None,
    weak: WeakFormOptions | None,
) -> _UnionFrame:
    if method == "weak":
        members, refusals = _weak_members(
            dataset, terms, lhs_order=lhs_order, max_order=max_order
        )
        frame, record = _weak_frame(dataset, weak, lhs_order=lhs_order)
        return _UnionFrame(
            record=record,
            refused={r.term: r.reason for r in refusals},
            weak=frame,
            members=members,
            weak_refusals=tuple(refusals),
        )
    strong = _strong_frame(
        dataset,
        terms,
        method=method,
        lhs_order=lhs_order,
        max_order=max_order,
        trim=trim,
        rows=rows,
        surrogate=surrogate,
    )
    refused = {r.term: r.reason for r in strong.report.rejected}
    return _UnionFrame(record=strong.frame, refused=refused, strong=strong)


def _raise_refused(frame: _UnionFrame) -> None:
    """Refuse the union as ``refit_terms`` would refuse it."""
    if frame.strong is not None:
        _handle_rejections(frame.strong.report, skip_invalid=False)
    raise_weak_refusals(frame.weak_refusals)


def _fit_in(frame: _UnionFrame, terms: Sequence[str]) -> RefitResult:
    """Fit ``terms`` (all accepted by ``frame``) on the frame's target."""
    if frame.strong is None:
        assert frame.weak is not None and frame.members is not None
        return _weak_result(terms, frame.members, frame.weak, frame.record, ())
    fit = _fit_valid(frame.strong.components, list(terms), rows=frame.strong.rows)
    return _strong_result(terms, fit, frame.record, ())


def _reduced_fit(
    frame: _UnionFrame, terms: Sequence[str]
) -> tuple[tuple[float, ...], float, float]:
    """``(coefficients, mse, nmse)`` of ``terms``, or of the zero model."""
    if terms:
        fit = _fit_in(frame, terms)
        assert fit.mse is not None
        return fit.coefficients, fit.mse, fit.nmse
    if frame.strong is None:
        assert frame.weak is not None
        return ((), *weak_zero_model(frame.weak))
    evaluator = cast("Evaluator", frame.strong.components.evaluator)
    target = evaluator.lhs_target
    selected = frame.strong.rows
    if selected is not None:
        if selected.dtype == torch.bool:
            selected = selected.flatten().nonzero().flatten()
        target = target.index_select(0, selected.to(target.device))
    mse = float((target**2).mean().item())
    return (), mse, _metrics_nmse(mse, float(target.var(correction=0).item()))


def _score_one(frame: _UnionFrame, terms: tuple[str, ...]) -> TermSetEntry:
    refused = [term for term in terms if term in frame.refused]
    if refused:
        detail = "; ".join(f"'{term}': {frame.refused[term]}" for term in refused)
        return TermSetEntry(terms, None, f"refused terms: {detail}", "refused")
    try:
        return TermSetEntry(terms, _fit_in(frame, terms), None, None)
    except (EvaluationFailedError, WeakDesignError) as error:
        return TermSetEntry(terms, None, str(error), "fit_failed")


def _refuse_duplicates(terms: Sequence[str]) -> None:
    """Refuse terms naming one column twice: "drop one" would be ambiguous."""
    seen: dict[str, str] = {}
    duplicates = []
    for term in terms:
        key = structure_term_key(lower_term(term))
        if key in seen:
            duplicates.append((seen[key], term))
        seen.setdefault(key, term)
    if duplicates:
        raise ValueError(f"duplicate terms name one column: {duplicates}")


def block_dispersion(
    dataset: PDEDataset,
    terms: Sequence[str],
    *,
    n_blocks: int = 4,
    axis: str | None = None,
    max_order: int = _DEFAULT_MAX_ORDER,
    lhs_order: int | None = None,
) -> float:
    """Refit the terms on separate grid blocks and measure coefficient spread.

    Two grid points are trimmed at both ends of every axis. The remaining
    indices along the block axis are split into ``n_blocks`` contiguous runs
    (``numpy.array_split``), each block is fitted with ``evaluate_terms``
    (``rows=`` the block), and the result is the mean over terms of
    ``|std(c, correction=0) / max(|mean(c)|, CV_DENOM_EPS)|``, where
    ``CV_DENOM_EPS = 1e-12`` and ``c`` holds one term's coefficients across
    the blocks. A law that holds on the whole grid keeps similar coefficients
    on every block.

    Args:
        dataset: A GRID ``PDEDataset`` (never mutated).
        terms: Candidate term strings (canonical funcall IR or product
            notation).
        n_blocks: Number of blocks; ``1`` gives ``0.0``.
        axis: Axis name to split along; default ``dataset.spatial_axes[0]``.
        max_order: As in ``evaluate_terms``.
        lhs_order: As in ``evaluate_terms``.

    Returns:
        The mean coefficient of variation across blocks.

    Raises:
        ValueError: If the dataset is not GRID, the resolved LHS field/axis
            is absent, or the result is non-finite.
        InvalidTermsError: Propagated from ``evaluate_terms``.
        EvaluationFailedError: Propagated from ``evaluate_terms`` when the
            fit on any block is invalid.
    """
    if dataset.topology is not DataTopology.GRID:
        raise ValueError("block_dispersion requires a GRID dataset")
    dataset = resolve_lhs_defaults(dataset)
    values = dataset.get_field(dataset.lhs_field)
    axis = dataset.spatial_axes[0] if axis is None else axis
    block_axis = cast(list[str], dataset.axis_order).index(axis)
    interior = interface_mask(
        values, axis=0, width=0, trim={d: 2 for d in range(values.ndim)}
    )
    n = values.shape[block_axis]
    shape = [1] * values.ndim
    shape[block_axis] = n
    indices = torch.arange(n, device=values.device).reshape(shape)
    coefficients: list[Tensor] = []
    for run in np.array_split(np.arange(2, n - 2), n_blocks):
        block = interior & torch.isin(
            indices, torch.as_tensor(run, device=values.device)
        )
        fit = evaluate_terms(
            dataset, terms, rows=block, max_order=max_order, lhs_order=lhs_order
        )
        assert fit.coefficients is not None
        coefficients.append(fit.coefficients)
    stacked = torch.stack(coefficients)
    denom = stacked.mean(dim=0).abs().clamp_min(CV_DENOM_EPS)
    value = float((stacked.std(dim=0, correction=0) / denom).abs().mean().item())
    if not math.isfinite(value):
        raise ValueError(f"block_dispersion: non-finite measurement {value}")
    return value


def forward_error(
    dataset: PDEDataset,
    terms: Sequence[str],
    coefficients: Sequence[float],
    *,
    method: str = "Radau",
    max_step: float | None = None,
    max_seconds: float | None = None,
) -> ForwardError:
    """Integrate the equation from the data's first time slice and compare.

    The RHS is ``kd.core.assemble_rhs`` of all terms (exactly-zero
    coefficients are omitted), integrated with ``kd.core.integrate_pde``
    over the dataset's grid. The value is ``||pred - u||_2 / ||u||_2`` over
    the full grid in float64, with ``u`` the dataset's LHS field. GRID datasets
    whose ``lhs_order`` is not 1 are refused before integration.

    ``max_seconds`` sets a cooperative deadline on ``time.monotonic()``,
    checked before and after preparing the RHS, at every RHS evaluation and
    before a measurement is returned; when it has passed the result is
    ``value=None, reason="timeout"``. A single long RHS evaluation or solver
    step is not interrupted, so a caller that needs a hard limit still runs
    this under process supervision.

    Args:
        dataset: The dataset providing the grid, initial field and target
            (never mutated).
        terms: RHS term strings of a first-order evolution law.
        coefficients: One coefficient per term.
        method: ``scipy.integrate.solve_ivp`` method.
        max_step: Maximum solver step; None lets the solver choose.
        max_seconds: Wall-clock budget in seconds; None sets no limit.

    Returns:
        A ``ForwardError`` with ``value`` set, or with ``reason`` set when the
        equation diverged, the solver failed, integration was refused (for
        example a non-GRID dataset, unsupported LHS order or a
        time-derivative term) or the deadline passed. Every result names the
        submitted terms and coefficients, the input, the interval and the
        boundary convention.

    Raises:
        ValueError: If ``terms`` and ``coefficients`` differ in length, a
            coefficient is non-finite, ``max_seconds`` is not a positive
            finite number, the error is non-finite, or the resolved LHS
            field/axis is absent from a GRID dataset.
        SearchInterrupted, torch.cuda.OutOfMemoryError, RuntimeError: An
            external stop or an allocation failure inside the integration
            (``kd.core.integrate_pde``) propagates.
    """
    from kd.core.integrator import assemble_rhs, integrate_pde

    deadline = _deadline(max_seconds)
    if len(terms) != len(coefficients):
        raise ValueError("terms and coefficients must have equal lengths")
    if dataset.topology is DataTopology.GRID:
        dataset = resolve_lhs_defaults(dataset)
    identity = _forward_identity(dataset, terms, coefficients)
    if _expired(deadline):
        return ForwardError(None, "timeout", _TIMEOUT_WARNING, **identity)
    rhs = assemble_rhs(terms, coefficients, range(len(terms)))
    if dataset.topology is DataTopology.GRID and dataset.lhs_order != 1:
        return ForwardError(
            None,
            "refused",
            "forward_error supports only first-order evolution; "
            f"got dataset lhs_order={dataset.lhs_order}.",
            **identity,
        )
    if _expired(deadline):
        return ForwardError(None, "timeout", _TIMEOUT_WARNING, **identity)
    result = integrate_pde(
        rhs, dataset, method=method, max_step=max_step, deadline=deadline
    )
    if result.timed_out:
        return ForwardError(None, "timeout", result.warning, **identity)
    if result.diverged_at_t is not None:
        return ForwardError(None, "diverged", result.warning, **identity)
    if result.warning.startswith("solve_ivp"):
        return ForwardError(None, "solve-failed", result.warning, **identity)
    if not result.success:
        return ForwardError(None, "refused", result.warning, **identity)
    assert result.predicted_field is not None
    pred = result.predicted_field.detach().cpu().numpy().astype(np.float64)
    u = dataset.get_field(dataset.lhs_field).detach().cpu().numpy().astype(np.float64)
    value = float(norm((pred - u).ravel()) / norm(u.ravel()))
    if not math.isfinite(value):
        raise ValueError(f"forward_error: non-finite measurement {value}")
    if _expired(deadline):
        return ForwardError(None, "timeout", _TIMEOUT_WARNING, **identity)
    return ForwardError(value, None, result.warning, **identity)


_TIMEOUT_WARNING = "forward_error stopped: max_seconds elapsed"


def _deadline(max_seconds: float | None) -> float | None:
    """The monotonic deadline of a ``max_seconds`` budget (None: no limit)."""
    if max_seconds is None:
        return None
    if not (math.isfinite(max_seconds) and max_seconds > 0.0):
        raise ValueError(
            f"max_seconds must be a positive finite number; got {max_seconds}"
        )
    return time.monotonic() + max_seconds


def _expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _forward_identity(
    dataset: PDEDataset, terms: Sequence[str], coefficients: Sequence[float]
) -> dict[str, Any]:
    """What a ``ForwardError`` measured: the law, the input and the interval."""
    identity: dict[str, Any] = {
        "terms": tuple(terms),
        "coefficients": tuple(float(c) for c in coefficients),
        "input_identity": compute_dataset_fingerprint(dataset),
    }
    if dataset.topology is DataTopology.GRID and dataset.axes is not None:
        times = dataset.axes[dataset.lhs_axis].values
        identity["interval"] = (float(times[0]), float(times[-1]))
        identity["boundary"] = {
            axis: "periodic" if dataset.axes[axis].is_periodic else "held_initial"
            for axis in dataset.spatial_axes
        }
    return identity


if TYPE_CHECKING:
    from kd.core.integrator import IntegrationResult
    from kd.data.trajectory import Trajectory, TrajectorySet


@dataclass(frozen=True)
class TrajectoryForwardError:
    """Outcome of ``forward_error_trajectories``.

    Attributes:
        value: ``||pred - obs||_2 / ||obs||_2`` over every trajectory, state
            and time point, or None when the laws could not be integrated or
            were not measured.
        reason: None when ``value`` is set; otherwise a reason of
            ``ForwardError``: ``"diverged"``, ``"solve-failed"`` (scipy
            ``solve_ivp`` failed), ``"refused"`` (a law is missing or extra,
            or uses a symbol other than the states, parameters and time;
            nothing was integrated) or ``"timeout"`` (the ``max_seconds``
            deadline passed: unmeasured, not a numerical outcome).
        warning: Why integration was refused, failed or stopped, naming the
            trajectory when a solve failed; empty on a clean run.
        laws: The submitted laws, target to ``(terms, coefficients)``, in
            state order; a key that names no state follows them.
        input_identity: ``compute_dataset_fingerprint`` of the trajectories.
        per_trajectory: Per trajectory id, the relative L2 error over its
            states and times; None for a trajectory that was not integrated
            to its end, or whose observed states are all zero (the relative
            error is undefined; ``value`` still includes it).
        per_target: Per target ``d<state>_dt``, the relative L2 error of that
            state over all trajectories; None unless ``value`` is set, or
            when that state is zero in every observation.
        intervals: Per trajectory id, ``(t_first, t_last)`` of its
            integration.
        solver: The ``method``, ``max_step``, ``rtol`` and ``atol`` used.
        predicted: The integrated trajectories on the observation times,
            with the ids and parameters of the input; None unless ``value``
            is set.
    """

    value: float | None
    reason: str | None
    warning: str
    laws: Mapping[str, tuple[tuple[str, ...], tuple[float, ...]]]
    input_identity: str
    per_trajectory: Mapping[str, float | None]
    per_target: Mapping[str, float | None]
    intervals: Mapping[str, tuple[float, float]]
    solver: Mapping[str, float | str | None]
    predicted: TrajectorySet | None = None


def forward_error_trajectories(
    trajectories: TrajectorySet,
    laws: Mapping[str, tuple[Sequence[str], Sequence[float]]],
    *,
    method: str = "Radau",
    max_step: float | None = None,
    max_seconds: float | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> TrajectoryForwardError:
    """Integrate one law per state along every trajectory and compare.

    ``laws`` maps each target ``derivative_target(state)`` (``dx_dt``) to
    the terms and coefficients of that state's time derivative; every state
    needs exactly one law. A law's RHS is ``kd.core.assemble_rhs`` of its
    terms (exactly-zero coefficients are omitted), and the system is
    integrated by ``kd.core.integrator.integrate_trajectory`` from each
    trajectory's first sample on that trajectory's own time points,
    trajectories in set order. A right-hand side may use the states, the
    set's parameters (constant along each trajectory), its time variable,
    ``one``, numeric literals and registry functions, under the protected
    semantics of ``FunctionRegistry.create_default`` (``refit_constants``
    fits under plain division; its laws are integrated here under
    ``safe_div``). The value is ``||pred - obs||_2 / ||obs||_2`` in float64
    over every trajectory, state and time point: the definition of
    ``forward_error``, stacked over the set.

    The first trajectory whose solve diverges, fails or is stopped ends the
    measurement: ``value``, ``per_target`` and ``predicted`` are None,
    ``per_trajectory`` keeps the trajectories finished before it, and
    ``warning`` names it. ``max_seconds`` sets one cooperative deadline on
    ``time.monotonic()`` for the whole call, checked before integrating, at
    every RHS evaluation, between trajectories and before a measurement is
    returned; as for ``forward_error``, a single long RHS evaluation or
    solver step is not interrupted.

    Args:
        trajectories: The observed trajectories (never mutated).
        laws: Target name to ``(terms, coefficients)``, one law per state.
        method: ``scipy.integrate.solve_ivp`` method.
        max_step: Maximum solver step; None lets the solver choose.
        max_seconds: Wall-clock budget in seconds; None sets no limit.
        rtol: Relative tolerance of the solver.
        atol: Absolute tolerance of the solver.

    Returns:
        A ``TrajectoryForwardError`` with ``value``, both breakdowns and the
        predicted trajectories set, or with ``reason`` set. Every result
        names the laws, the input, the intervals and the solver settings.

    Raises:
        ValueError: If a law's terms and coefficients differ in length or a
            coefficient is non-finite (the message names the law), ``rtol``,
            ``atol`` or ``max_seconds`` is not a positive finite number, the
            observed states are zero everywhere, or the error is non-finite.
        SearchInterrupted, torch.cuda.OutOfMemoryError, RuntimeError: An
            external stop or an allocation failure inside the integration
            propagates.
    """
    from kd.data.trajectory import derivative_target

    for setting, tolerance in (("rtol", rtol), ("atol", atol)):
        if not (math.isfinite(tolerance) and tolerance > 0.0):
            raise ValueError(
                f"{setting} must be a positive finite number; got {tolerance}"
            )
    deadline = _deadline(max_seconds)
    states = {derivative_target(state): state for state in trajectories.state_vars}
    order = [
        *(key for key in states if key in laws),
        *(k for k in laws if k not in states),
    ]
    rhs = _law_rhs({key: laws[key] for key in order})
    solver: dict[str, Any] = {
        "method": method,
        "max_step": max_step,
        "rtol": rtol,
        "atol": atol,
    }
    identity = _trajectory_identity(
        trajectories, {key: laws[key] for key in order}, solver
    )
    targets = list(states)
    if _expired(deadline):
        return _unmeasured_trajectories(
            "timeout", _TRAJECTORY_TIMEOUT_WARNING, identity, targets, {}
        )
    missing = [key for key in targets if key not in rhs]
    extra = [key for key in rhs if key not in states]
    if missing or extra:
        warning = (
            f"forward_error_trajectories needs one law per state, keyed {targets}; "
            f"missing {missing}, extra {extra}."
        )
        return _unmeasured_trajectories("refused", warning, identity, targets, {})
    system = {states[key]: rhs[key] for key in targets}
    return _integrate_trajectories(trajectories, system, targets, identity, deadline)


_TRAJECTORY_TIMEOUT_WARNING = "forward_error_trajectories stopped: max_seconds elapsed"


def _law_rhs(
    laws: Mapping[str, tuple[Sequence[str], Sequence[float]]],
) -> dict[str, str]:
    """Each law's integrable RHS (``assemble_rhs`` of all its terms)."""
    from kd.core.integrator import assemble_rhs

    rhs: dict[str, str] = {}
    for key, (terms, coefficients) in laws.items():
        if len(terms) != len(coefficients):
            raise ValueError(
                f"law {key!r}: terms and coefficients must have equal lengths"
            )
        try:
            rhs[key] = assemble_rhs(terms, coefficients, range(len(terms)))
        except ValueError as exc:
            raise ValueError(f"law {key!r}: {exc}") from exc
    return rhs


def _trajectory_identity(
    trajectories: TrajectorySet,
    laws: Mapping[str, tuple[Sequence[str], Sequence[float]]],
    solver: Mapping[str, Any],
) -> dict[str, Any]:
    """What a ``TrajectoryForwardError`` measured: laws, input, intervals, solver."""
    return {
        "laws": {
            key: (tuple(terms), tuple(float(c) for c in coefficients))
            for key, (terms, coefficients) in laws.items()
        },
        "input_identity": compute_dataset_fingerprint(trajectories),
        "intervals": {
            trajectory.id: (float(trajectory.t[0]), float(trajectory.t[-1]))
            for trajectory in trajectories.trajectories
        },
        "solver": dict(solver),
    }


def _unmeasured_trajectories(
    reason: str,
    warning: str,
    identity: Mapping[str, Any],
    targets: Sequence[str],
    finished: Mapping[str, float | None],
) -> TrajectoryForwardError:
    """A result without a value: only the finished trajectories keep theirs."""
    return TrajectoryForwardError(
        value=None,
        reason=reason,
        warning=warning,
        per_trajectory={key: finished.get(key) for key in identity["intervals"]},
        per_target=dict.fromkeys(targets),
        **identity,
    )


def _integration_reason(result: IntegrationResult) -> str | None:
    """The ``ForwardError`` reason of one integration; None when it succeeded."""
    if result.timed_out:
        return "timeout"
    if result.diverged_at_t is not None:
        return "diverged"
    if result.warning.startswith("solve_ivp"):
        return "solve-failed"
    if not result.success:
        return "refused"
    return None


def _integrate_trajectories(
    trajectories: TrajectorySet,
    system: Mapping[str, str],
    targets: Sequence[str],
    identity: Mapping[str, Any],
    deadline: float | None,
) -> TrajectoryForwardError:
    """Integrate ``system`` along every trajectory, in set order, and compare."""
    from kd.core.integrator import integrate_trajectory
    from kd.data.trajectory import Trajectory

    finished: dict[str, float | None] = {}
    predicted: list[Trajectory] = []
    for trajectory in trajectories.trajectories:
        if _expired(deadline):
            return _unmeasured_trajectories(
                "timeout", _TRAJECTORY_TIMEOUT_WARNING, identity, targets, finished
            )
        result = integrate_trajectory(
            system,
            trajectory,
            time_var=trajectories.time_var,
            deadline=deadline,
            **identity["solver"],
        )
        reason = _integration_reason(result)
        if reason is not None:
            warning = result.warning
            if reason != "refused":
                warning = f"trajectory {trajectory.id!r}: {warning}"
            return _unmeasured_trajectories(
                reason, warning, identity, targets, finished
            )
        assert result.predicted_field is not None
        state = result.predicted_field.numpy()
        finished[trajectory.id] = _relative_l2([state], [trajectory.state])
        predicted.append(
            Trajectory(
                id=trajectory.id, t=trajectory.t, state=state, params=trajectory.params
            )
        )
    return _compared_trajectories(
        trajectories, predicted, finished, targets, identity, deadline
    )


def _compared_trajectories(
    trajectories: TrajectorySet,
    predicted: Sequence[Trajectory],
    finished: Mapping[str, float | None],
    targets: Sequence[str],
    identity: Mapping[str, Any],
    deadline: float | None,
) -> TrajectoryForwardError:
    """The aggregate and per-target errors of a set integrated to its end."""
    from dataclasses import replace

    observed = [trajectory.state for trajectory in trajectories.trajectories]
    states = [trajectory.state for trajectory in predicted]
    value = _relative_l2(states, observed)
    if value is None:
        raise ValueError(
            "forward_error_trajectories: the observed states are zero everywhere; "
            "the relative error is undefined"
        )
    if not math.isfinite(value):
        raise ValueError(f"forward_error_trajectories: non-finite measurement {value}")
    per_target = {
        target: _relative_l2([s[row] for s in states], [o[row] for o in observed])
        for row, target in enumerate(targets)
    }
    if _expired(deadline):
        return _unmeasured_trajectories(
            "timeout", _TRAJECTORY_TIMEOUT_WARNING, identity, targets, finished
        )
    return TrajectoryForwardError(
        value=value,
        reason=None,
        warning="",
        per_trajectory=dict(finished),
        per_target=per_target,
        predicted=replace(trajectories, trajectories=tuple(predicted)),
        **identity,
    )


def _relative_l2(
    predicted: Sequence[np.ndarray], observed: Sequence[np.ndarray]
) -> float | None:
    """``||pred - obs||_2 / ||obs||_2`` over the stacked arrays, in float64.

    None when ``obs`` is zero everywhere: the relative error is undefined.
    """
    difference = np.concatenate(
        [
            (np.asarray(p, dtype=np.float64) - np.asarray(o, dtype=np.float64)).ravel()
            for p, o in zip(predicted, observed, strict=True)
        ]
    )
    reference = np.concatenate(
        [np.asarray(o, dtype=np.float64).ravel() for o in observed]
    )
    reference_norm = float(norm(reference))
    if reference_norm == 0.0:
        return None
    return float(norm(difference)) / reference_norm


def refit_constants(
    dataset: TabularDataset,
    terms: Sequence[str],
    *,
    starts: Sequence[float] = DEFAULT_STARTS,
) -> ConstantRefit:
    """Refit the numeric literals inside the terms of a tabular answer.

    The columns are ``dataset.X[:, i]`` under ``dataset.var_names`` and the
    target is ``dataset.y``, in float64. Each literal that does not only scale
    its whole term becomes a constant (the literal-position rule of
    ``kd.core.expr.constant_fit``); the literals themselves are the first
    candidate, so the terms come back unchanged when nothing improves on them.
    The terms are evaluated as written: plain division, unclamped powers and
    raw functions (``FunctionRegistry.create_plain``), not the protected
    default registry. The skeleton and the term set are not changed. A term
    whose literals were refitted comes back as funcall IR with the fitted
    values; every other term keeps the caller's spelling.

    Args:
        dataset: The tabular data the answer is fitted to.
        terms: One term per additive term, in funcall IR
            (``"exp(mul(-0.5, x1))"``) or product notation
            (``"exp(-0.5*x1)"``, lowered with each literal's sign kept).
        starts: The start values tried after the literals; each start sets
            every constant.

    Returns:
        The refitted terms and constants, the literals they replaced, the
        outer least-squares coefficients and the nmse.

    Raises:
        TypeError: If ``dataset`` is not a ``TabularDataset``; grid data are
            refitted with ``evaluate_terms``.
        KeyError: If a term names a column the dataset lacks.
        ValueError: If the table has no rows or a non-finite target, if no
            candidate gives a finite fit, if a term evaluates to a scalar
            (every term must reference a column or ``one``), if a term is
            neither funcall IR nor product notation, or if a column named
            ``c<k>`` sits beside a refitted literal.
    """
    if not isinstance(dataset, TabularDataset):
        raise TypeError(
            f"refit_constants needs a TabularDataset, got {type(dataset).__name__}"
        )
    columns = {
        name: torch.from_numpy(np.ascontiguousarray(dataset.X[:, i], dtype=np.float64))
        for i, name in enumerate(dataset.var_names)
    }
    target = torch.from_numpy(np.ascontiguousarray(dataset.y, dtype=np.float64))
    registry = FunctionRegistry.create_plain()
    irs = [lower_term(term) for term in terms]
    refit = refit_literals(irs, columns, target, registry=registry, starts=starts)
    echoed = tuple(
        written if fitted == ir else fitted
        for written, ir, fitted in zip(terms, irs, refit.terms, strict=True)
    )
    return dataclasses.replace(refit, terms=echoed)







def _handle_rejections(
    report: TermValidationReport,
    *,
    skip_invalid: bool,
) -> None:
    """Apply the fail-loud rejection policy before fitting survivors.

    Strict (``skip_invalid=False``): any rejection raises with the COMPLETE
    report. Lenient (``skip_invalid=True``): all-rejected still raises (no silent
    empty fit); otherwise survivors proceed after exactly ONE summary warning.
    """
    if not report.rejected:
        return
    if not skip_invalid:

        raise InvalidTermsError(report.rejected)
    if not report.valid:

        raise InvalidTermsError(report.rejected)

    logger.warning(
        "evaluate_terms dropped %d invalid term(s): %s",
        len(report.rejected),
        "; ".join(f"{rej.term} ({rej.reason})" for rej in report.rejected),
    )


def _prune_small_terms(
    terms: Sequence[str],
    coefficients: Sequence[float] | None,
    rtol: float,
    fixed: frozenset[int],
) -> tuple[list[str], tuple[tuple[str, float], ...]]:
    """Check the pruning inputs and split ``terms`` into kept and dropped.

    The positions in ``fixed`` are always kept and never set the reference
    magnitude: they are left out of the rule's active set.
    """
    if not 0.0 <= rtol < 1.0:
        raise ValueError(f"small_term_rtol must be in [0, 1); got {rtol}")
    if coefficients is None:
        if rtol > 0.0:
            raise ValueError("small_term_rtol > 0 requires coefficients, one per term")
        return list(terms), ()
    if len(coefficients) != len(terms):
        raise ValueError(
            f"coefficients has {len(coefficients)} entries for {len(terms)} terms"
        )
    active = [i for i in range(len(terms)) if i not in fixed]
    keep = set(small_term_indices(coefficients, active, rtol)) | fixed
    kept = [term for i, term in enumerate(terms) if i in keep]
    dropped = tuple(
        (term, float(coefficients[i])) for i, term in enumerate(terms) if i not in keep
    )
    return kept, dropped


def _require_nonempty(terms: Sequence[str]) -> list[str]:
    """Return ``terms`` as a list, failing fast on an empty input."""
    term_list = list(terms)
    if not term_list:
        raise ValueError("Empty term list: provide at least one term to evaluate.")
    return term_list


def _resolve_lhs_order(dataset: PDEDataset, lhs_order: int | None) -> int:
    """Resolve the effective LHS order: derive from dataset, or honor override.

    ``lhs_order=None`` (default) DERIVES the order from
    ``dataset.lhs_order`` — the single source of truth — so the agent entry
    stays consistent with the dataset by default. An explicit value is a
    deliberate OVERRIDE (request semantics): it may differ from
    ``dataset.lhs_order`` to probe a what-if target (e.g. the u_tt relation on
    an order-1 advection field), and is intentionally NOT reconciled against
    the dataset here. This narrow-entry override is distinct from the ``Model``
    facade, where the dataset's order is authoritative and an unsupported
    ``(algorithm, lhs_order)`` fails loud.
    """
    resolved = dataset.lhs_order if lhs_order is None else lhs_order
    if resolved == 0:
        if dataset.topology is DataTopology.TABULAR:
            raise NotImplementedError(
                "tabular datasets are not supported by evaluate_terms/validate_terms"
            )
        raise NotImplementedError(
            "homogeneous (lhs_order=0) datasets are not supported by "
            "evaluate_terms/validate_terms; the homogeneous path is "
            "plugin-private."
        )
    return resolved


def _constraint(
    dataset: PDEDataset,
    *,
    fixed: Mapping[str, float] | None,
    sketch: Sketch | SketchSpec | None,
    lhs_order: int | None,
) -> FixedConstraint | None:
    """The constraint of ``fixed`` or ``sketch`` on this fit's LHS, or None.

    Raises:
        ValueError: The constraint is refused (``resolve_constraint``).
    """
    if fixed is None and sketch is None:
        return None
    return resolve_constraint(
        dataset,
        fixed=fixed,
        sketch=sketch,
        lhs_order=_resolve_lhs_order(dataset, lhs_order),
    )


def _placed(
    constraint: FixedConstraint | None, terms: Sequence[str]
) -> FixedPlacement | None:
    """Place ``constraint`` on ``terms``; None without a constraint.

    Raises:
        InvalidTermsError: Input terms that cannot be fitted beside the fixed
            terms, every one in one report.
    """
    if constraint is None:
        return None
    placement = place_fixed(constraint, terms)
    if placement.rejected:
        raise InvalidTermsError(
            [TermRejection(term, why) for term, why in placement.rejected.items()]
        )
    return placement


def _refuse_rejected_fixed(
    report: TermValidationReport, fixed: Mapping[str, float]
) -> None:
    """A rejected fixed term raises the complete report, lenient or not.

    The equation cannot be fitted without it, so it is never dropped.
    """
    if any(rejection.term in fixed for rejection in report.rejected):
        raise InvalidTermsError(report.rejected)


def _fit_fixed(
    components: PlatformComponents,
    terms: list[str],
    fixed: Mapping[str, float],
    *,
    rows: Tensor | None,
) -> EvaluationResult:
    """Fit the free ``terms`` to the target minus the fixed ones, fail-loud.

    ``terms`` are pre-validated, in result order. With every term fixed
    nothing is solved. Either way the returned equation is measured on the
    original target, and a non-finite MSE raises.
    """
    base = cast("Evaluator", components.evaluator)
    free = [term for term in terms if term not in fixed]
    if free:
        fit = _fit_valid(lowered_components(components, fixed), free, rows=rows)
        result = merge_fixed(fit, terms, fixed, base=base, rows=rows)
    else:
        values = [fixed[term] for term in terms]
        result = measure_fixed(base, terms, values, rows=rows, complexity=0)
    if not math.isfinite(result.mse):
        raise EvaluationFailedError(
            "the equation under fixed coefficients has a non-finite MSE"
        )
    return result


def _fit_valid(
    components: PlatformComponents,
    terms: list[str],
    *,
    rows: Tensor | None,
) -> EvaluationResult:
    """Fit pre-validated terms with the components' evaluator, fail-loud.

    ``skip_invalid=False`` gives an honest fit and lets the solver-failure
    path surface as ``EvaluationFailedError``; the penalty sentinel is never
    returned. The evaluator executes the lowered IR (its column cache is
    keyed by it); the result names ``terms`` as given.
    """
    evaluator = cast("Evaluator", components.evaluator)
    irs = [lower_term(term) for term in terms]
    result = evaluator.evaluate_terms(irs, skip_invalid=False, rows=rows)
    if not result.is_valid:
        raise EvaluationFailedError(
            result.error_message or "Evaluation produced an invalid result"
        )
    return dataclasses.replace(result, terms=list(terms))
