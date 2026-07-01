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

Pure constants / scalars (e.g. ``"1"``) are rejected: Theta carries no intercept
column, so a scalar that broadcasts to the wrong length would detonate the fit
unattributed. They surface as a ``shape`` rejection naming the term.

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
from collections.abc import Sequence
from typing import TYPE_CHECKING

from kd._evaluate_classify import (
    TermRejection,
    TermValidation,
    TermValidationReport,
    classify_terms,
)
from kd.core.platform.builder import PlatformBuilder
from kd.core.platform.requirements import DerivativeReqs

if TYPE_CHECKING:
    from kd.core.evaluator import EvaluationResult
    from kd.data.schema import PDEDataset
    from kd.search.protocol import PlatformComponents

logger = logging.getLogger(__name__)



__all__ = [
    "EvaluationFailedError",
    "InvalidTermsError",
    "TermRejection",
    "TermValidation",
    "TermValidationReport",
    "evaluate_terms",
    "validate_terms",
]





_DEFAULT_MAX_ORDER = 2







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

    Returns:
        An ``EvaluationResult`` with ``is_valid is True`` (honest metrics).

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



    result = components.evaluator.evaluate_terms(report.valid, skip_invalid=False)
    if not result.is_valid:
        raise EvaluationFailedError(
            result.error_message or "Evaluation produced an invalid result"
        )
    return result







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

    Decision A (DATA-0): ``lhs_order=None`` (default) DERIVES the order from
    ``dataset.lhs_order`` — the single source of truth — so the agent entry
    stays consistent with the dataset by default. An explicit value is a
    deliberate OVERRIDE (request semantics): it may differ from
    ``dataset.lhs_order`` to probe a what-if target (e.g. the u_tt relation on
    an order-1 advection field), and is intentionally NOT reconciled against
    the dataset here. This narrow-entry override is distinct from the ``Model``
    facade, where the dataset's order is authoritative and an unsupported
    ``(algorithm, lhs_order)`` fails loud.
    """
    return dataset.lhs_order if lhs_order is None else lhs_order


def _build_components(
    dataset: PDEDataset,
    *,
    max_order: int,
    lhs_order: int,
) -> PlatformComponents:
    """Assemble platform components via ``PlatformBuilder`` (no hand-wiring).

    The facade params ``max_order`` / ``lhs_order`` flow straight into
    ``DerivativeReqs`` so they cannot be silently dropped.
    """
    reqs = DerivativeReqs(max_atomic_order=max_order, lhs_order=lhs_order)
    return PlatformBuilder(dataset, reqs).build()
