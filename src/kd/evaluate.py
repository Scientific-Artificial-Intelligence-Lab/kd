"""Agent-facing narrow entry for evaluating candidate PDE terms.

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

Term syntax (canonical funcall IR — ENFORCED)
---------------------------------------------
Each term must be a SINGLE canonical funcall-IR term, e.g. ``"diff_x(u)"``,
``"mul(u, diff_x(u))"``, or a terminal token like ``"u_xx"``. The contract is
ENFORCED, not merely documented: every term is first passed
through the platform's own ``split_terms`` (the same canonicality gate the
LINEAR evaluator uses), so two classes of non-canonical input are rejected up
front rather than silently mis-fitted as one column:

- Infix / unary / boolean operators (``"u + u_xx"``, ``"-u"``) → ``syntax``
  rejection. The executor *can* eval a BinOp string as a single column, which
  would hand back a bogus one-coefficient fit; the syntax layer blocks it.
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
(parse/exec error, incl. the max_order hint + CUDA-OOM special case) →
``shape`` (wrong column length) → ``non-finite`` (NaN/Inf) → ``all-zero`` →
``tautology`` (duplicates the LHS target).

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

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from scipy.linalg import norm

from kd._evaluate_classify import (
    TermRejection,
    TermValidation,
    TermValidationReport,
    classify_terms,
)
from kd.core.equation import Evolution, LhsSpec, active_law, render_lhs_label
from kd.core.platform.builder import PlatformBuilder, resolve_lhs_defaults
from kd.core.platform.requirements import DerivativeReqs
from kd.data import interface_mask
from kd.data.schema import DataTopology

if TYPE_CHECKING:
    from torch import Tensor

    from kd.core.evaluator import EvaluationResult
    from kd.data.schema import PDEDataset
    from kd.search.protocol import PlatformComponents
    from kd.search.result import ExperimentResult
    from kd.viz.report import ReportResult

logger = logging.getLogger(__name__)



__all__ = [
    "ComparisonResult",
    "EvaluationFailedError",
    "ForwardError",
    "InvalidTermsError",
    "TermRejection",
    "TermValidation",
    "TermValidationReport",
    "block_dispersion",
    "evaluate_terms",
    "compare_results",
    "forward_error",
    "validate_terms",
]


@dataclass(frozen=True)
class ForwardError:
    """Outcome of ``forward_error``.

    Attributes:
        value: ``||pred - u||_2 / ||u||_2`` over the full grid, or None when
            the equation could not be integrated.
        reason: None when ``value`` is set; otherwise ``"diverged"``,
            ``"solve-failed"`` (scipy ``solve_ivp`` failed) or ``"refused"``
            (the equation or dataset is unsupported).
        warning: Why integration was refused or failed, empty on a clean run.
    """

    value: float | None
    reason: str | None
    warning: str


@dataclass
class ComparisonResult:
    """Structure refits on one dataset, separate from native search scores.

    ``evaluations`` contains successful refits; ``exclusions`` names every
    run that could not be compared and explains why. Labels are supplied by
    the caller, so a preset answer can be labelled as such in every view.
    """

    results: dict[str, ExperimentResult]
    evaluations: dict[str, EvaluationResult]
    exclusions: dict[str, str]
    lhs_label: str
    reference: str | None

    def __str__(self) -> str:
        lines = [
            "Unified platform evaluation (structure refit, same NMSE)",
            f"{'Run':<24} {'NMSE':>12} {'R^2':>10} Structure",
        ]
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
) -> ComparisonResult:
    """Refit each active evolution-law structure on the same FD features.

    The shared target is the dataset's LHS. This compares discovered
    structures, not the algorithms' native scores or original coefficients.
    Missing equations, other forms/LHSs, and refused term evaluations are
    explicitly excluded. Other errors propagate. Inputs are never mutated.
    """
    resolved = resolve_lhs_defaults(dataset)
    lhs = LhsSpec(resolved.lhs_field, resolved.lhs_axis, resolved.lhs_order)
    comparison = ComparisonResult(
        dict(results), {}, {}, render_lhs_label(lhs), dataset.ground_truth
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
                resolved, terms, max_order=max_order
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
    error, with max_order hint + CUDA-OOM handling), ``shape`` (wrong length),
    ``non-finite`` (NaN/Inf), ``all-zero``, ``tautology`` (duplicates the LHS
    target). No least-squares fit is performed and the dataset is not mutated.

    Args:
        dataset: The PDE dataset to evaluate against (never mutated).
        terms: Candidate term strings (canonical funcall IR).
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
    components = _build_components(
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
) -> EvaluationResult:
    """Evaluate candidate terms against a dataset, fail-loud.

    Pre-validates ALL terms (complete report), then fits the valid ones via the
    real platform ``Evaluator`` (OLS-on-Theta through ``LeastSquaresSolver``).

    Args:
        dataset: The PDE dataset to evaluate against (never mutated).
        terms: Candidate term strings (canonical funcall IR).
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

    Returns:
        An ``EvaluationResult`` with ``is_valid is True`` (honest metrics).
        This entry always measures ``condition_number`` (candidate-library
        collinearity). The measured matrix is the one the solver was handed:
        under ``skip_invalid=True`` that is the SURVIVING columns, not every
        string in ``terms`` — a term dropped as ``shape`` / ``all-zero`` /
        ``tautology`` is not in the number. Strict mode fits every submitted
        term or raises, so there the two coincide.

    Raises:
        ValueError: If ``terms`` is empty.
        ValueError: Propagated from ``PlatformBuilder`` when the resolved LHS
            field/axis is absent from the dataset, when the resolved
            ``lhs_order > max_order`` (the LHS is read from the same precomputed
            derivative cache that ``max_order`` bounds), or when
            ``max_order > 3`` (provider cap).
        InvalidTermsError: If terms are rejected (see ``skip_invalid``).
        EvaluationFailedError: If the final fit is invalid (the penalty sentinel
            is never returned).
    """
    term_list = _require_nonempty(terms)
    lhs_order_resolved = _resolve_lhs_order(dataset, lhs_order)
    components = _build_components(
        dataset, max_order=max_order, lhs_order=lhs_order_resolved
    )
    report = classify_terms(term_list, components, max_order=max_order)

    _handle_rejections(report, skip_invalid=skip_invalid)






    evaluator = components.evaluator
    if evaluator is None:
        raise TypeError(
            "evaluate() requires components.evaluator (term fit); got None."
        )
    result = evaluator.evaluate_terms(report.valid, skip_invalid=False, rows=rows)
    if not result.is_valid:
        raise EvaluationFailedError(
            result.error_message or "Evaluation produced an invalid result"
        )
    return result


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
        terms: Candidate term strings (canonical funcall IR).
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
) -> ForwardError:
    """Integrate the equation from the data's first time slice and compare.

    The RHS is ``kd.core.assemble_rhs`` of all terms (exactly-zero
    coefficients are omitted), integrated with ``kd.core.integrate_pde``
    over the dataset's grid. The value is ``||pred - u||_2 / ||u||_2`` over
    the full grid in float64, with ``u`` the dataset's LHS field. GRID datasets
    whose ``lhs_order`` is not 1 are refused before integration. No time limit
    is applied; a caller that needs one runs this in a subprocess.

    Args:
        dataset: The dataset providing the grid, initial field and target
            (never mutated).
        terms: RHS term strings of a first-order evolution law.
        coefficients: One coefficient per term.
        method: ``scipy.integrate.solve_ivp`` method.
        max_step: Maximum solver step; None lets the solver choose.

    Returns:
        A ``ForwardError`` with ``value`` set, or with ``reason`` set when the
        equation diverged, the solver failed, or integration was refused
        (for example a non-GRID dataset, unsupported LHS order or a
        time-derivative term).

    Raises:
        ValueError: If ``terms`` and ``coefficients`` differ in length, a
            coefficient is non-finite, the error is non-finite, or the
            resolved LHS field/axis is absent from a GRID dataset.
    """
    from kd.core.integrator import assemble_rhs, integrate_pde

    if len(terms) != len(coefficients):
        raise ValueError("terms and coefficients must have equal lengths")
    rhs = assemble_rhs(terms, coefficients, range(len(terms)))
    if dataset.topology is DataTopology.GRID:
        dataset = resolve_lhs_defaults(dataset)
        if dataset.lhs_order != 1:
            return ForwardError(
                None,
                "refused",
                "forward_error supports only first-order evolution; "
                f"got dataset lhs_order={dataset.lhs_order}.",
            )
    result = integrate_pde(rhs, dataset, method=method, max_step=max_step)
    if result.diverged_at_t is not None:
        return ForwardError(None, "diverged", result.warning)
    if result.warning.startswith("solve_ivp"):
        return ForwardError(None, "solve-failed", result.warning)
    if not result.success:
        return ForwardError(None, "refused", result.warning)
    assert result.predicted_field is not None
    pred = result.predicted_field.detach().cpu().numpy().astype(np.float64)
    u = dataset.get_field(dataset.lhs_field).detach().cpu().numpy().astype(np.float64)
    value = float(norm((pred - u).ravel()) / norm(u.ravel()))
    if not math.isfinite(value):
        raise ValueError(f"forward_error: non-finite measurement {value}")
    return ForwardError(value, None, result.warning)







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
            "plugin-private (arch041 step 3c2)."
        )
    return resolved


def _build_components(
    dataset: PDEDataset,
    *,
    max_order: int,
    lhs_order: int,
) -> PlatformComponents:
    """Assemble platform components via ``PlatformBuilder`` (no hand-wiring).

    The facade params ``max_order`` / ``lhs_order`` flow straight into
    ``DerivativeReqs`` so they cannot be silently dropped.

    This narrow entry opts IN to the candidate-library condition number: it is
    a one-shot probe whose caller is an agent choosing a term library, and one
    extra SVD is small next to the double term execution this entry already
    accepts (see the module's cost note). The search loop keeps the builder
    default (off): it runs the same solve tens of thousands of times per run
    and stores the value nowhere. ``validate_terms`` shares this helper but
    never solves, so it pays nothing for the flag.
    """
    reqs = DerivativeReqs(max_atomic_order=max_order, lhs_order=lhs_order)
    return PlatformBuilder(dataset, reqs, compute_condition_number=True).build()
