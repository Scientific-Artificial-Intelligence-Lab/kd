"""Per-term classifier + report types for the ``kd.evaluate`` narrow entry.

Split out of ``kd.evaluate`` (file-size hygiene) but conceptually part of it:
this module owns the layered term-rejection classifier and the JSON-dumpable
report dataclasses. ``kd.evaluate`` re-exports the report types and calls
``classify_terms``; nothing here imports ``kd.evaluate`` (no import cycle).

Rejection layers, applied in order:
(0) syntax/composite via ``split_terms`` → (1) execution error (incl. CUDA-OOM
special case + terminal-derivative max_order hint) → (2) shape → (3) non-finite
→ (4) all-zero → (5) LHS tautology. The reason prefixes are mutually
distinguishable so callers can branch on the failure class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from kd.core.evaluator import release_cuda_memory
from kd.core.expr.naming import parse_derivative_name
from kd.core.expr.terms import split_terms

if TYPE_CHECKING:
    from kd.core.executor.context import ExecutionContext
    from kd.core.expr import PythonExecutor
    from kd.core.expr.registry import FunctionRegistry
    from kd.data.schema import PDEDataset
    from kd.search.protocol import PlatformComponents




_REASON_SYNTAX = "syntax"
_REASON_COMPOSITE = "composite"
_REASON_EXECUTION = "execution"
_REASON_SHAPE = "shape"
_REASON_NONFINITE = "non-finite"
_REASON_ALL_ZERO = "all-zero"
_REASON_TAUTOLOGY = "tautology"

_NONFINITE_REASON_TEXT = "term produced NaN/Inf values"
_ALL_ZERO_REASON_TEXT = "term produced an all-zero column"
_TAUTOLOGY_REASON_TEXT = "duplicates the LHS target (tautological fit)"



_FD_MAX_ORDER = 3







@dataclass(frozen=True)
class TermRejection:
    """A single rejected term and the reason it was rejected.

    Attributes:
        term: The offending term string.
        reason: Human-readable rejection reason. Distinguishable across the
            rejection categories; for execution errors it surfaces the original
            failure message (including the offending token/op name).
    """

    term: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe dict of this rejection."""
        return {"term": self.term, "reason": self.reason}


@dataclass(frozen=True)
class TermValidation:
    """Per-term validation outcome.

    Attributes:
        term: The term string.
        ok: Whether the term executed to a usable (finite, non-zero) column.
        reason: ``None`` when ``ok`` is True; otherwise the rejection reason.
    """

    term: str
    ok: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict of this validation."""
        return {"term": self.term, "ok": self.ok, "reason": self.reason}


@dataclass(frozen=True)
class TermValidationReport:
    """Structured result of ``validate_terms`` (JSON-dumpable, no fitting).

    Attributes:
        results: Per-term validation outcomes in input order.
        valid: Term strings that passed validation, in input order.
        rejected: ``TermRejection`` entries for the failed terms.
        ok: True iff every term passed (``rejected`` is empty).
    """

    results: list[TermValidation] = field(default_factory=list)
    valid: list[str] = field(default_factory=list)
    rejected: list[TermRejection] = field(default_factory=list)
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict of the full report (MCP-boundary contract)."""
        return {
            "ok": self.ok,
            "valid": list(self.valid),
            "rejected": [rej.to_dict() for rej in self.rejected],
            "results": [res.to_dict() for res in self.results],
        }







def _known_names(dataset: PDEDataset) -> tuple[set[str] | None, set[str] | None]:
    """Return (known_fields, known_axes) for terminal-derivative disambiguation."""
    fields = dataset.fields
    axes = dataset.axes
    known_fields = set(fields) if isinstance(fields, dict) else None
    known_axes = set(axes) if isinstance(axes, dict) else None
    return known_fields, known_axes


def classify_terms(
    terms: list[str],
    components: PlatformComponents,
    *,
    max_order: int,
) -> TermValidationReport:
    """Classify every term into valid / rejected (shared by both entries)."""
    known_fields, known_axes = _known_names(components.dataset)
    results: list[TermValidation] = []
    valid: list[str] = []
    rejected: list[TermRejection] = []

    for term in terms:
        reason = _classify_one(
            term,
            executor=components.executor,
            context=components.context,
            registry=components.registry,
            lhs_target=components.evaluator.lhs_target,
            max_order=max_order,
            known_fields=known_fields,
            known_axes=known_axes,
        )
        ok = reason is None
        results.append(TermValidation(term=term, ok=ok, reason=reason))
        if ok:
            valid.append(term)
        else:
            rejected.append(TermRejection(term=term, reason=reason or ""))

    return TermValidationReport(
        results=results, valid=valid, rejected=rejected, ok=not rejected
    )


def _classify_one(
    term: str,
    *,
    executor: PythonExecutor,
    context: ExecutionContext,
    registry: FunctionRegistry,
    lhs_target: torch.Tensor,
    max_order: int,
    known_fields: set[str] | None,
    known_axes: set[str] | None,
) -> str | None:
    """Return ``None`` if ``term`` is usable, else a rejection reason string.

    Layers, in order (see module docstring): (0) syntax/composite via
    ``split_terms``; (1) execution error — enriched with the CUDA-OOM special
    case and the terminal-derivative max_order hint; (2) shape (numel != LHS
    length); (3) non-finite; (4) all-zero; (5) tautology (bitwise-equal to the
    LHS target, relative to the caller's ``lhs_order``).

    Layers 1-4 mirror ``Evaluator._build_theta`` (additive, read-only). The
    non-finite branch is practically unreachable with safe ops on clean data but
    is kept so the contract is explicit and testable.
    """
    syntax_reason = _canonical_reason(term, registry)
    if syntax_reason is not None:
        return syntax_reason

    try:
        with torch.no_grad():
            exec_result = executor.execute(term, context)
            column = exec_result.value.detach().flatten()
    except torch.cuda.OutOfMemoryError:


        release_cuda_memory()
        return f"{_REASON_EXECUTION}: CUDA out of memory during term execution"
    except Exception as exc:
        hint = _max_order_hint(term, max_order, known_fields, known_axes)
        return f"{_REASON_EXECUTION} error for '{term}': {exc}{hint}"

    return _column_reason(column, lhs_target)


def _column_reason(column: torch.Tensor, lhs_target: torch.Tensor) -> str | None:
    """Layers 2-5: classify an executed column (shape/finite/zero/tautology)."""
    expected = lhs_target.numel()
    if column.numel() != expected:
        return (
            f"{_REASON_SHAPE} mismatch: term produces {column.numel()} elements, "
            f"expected {expected} (Theta has no intercept column; pure constants "
            f"are not supported)"
        )
    if not torch.isfinite(column).all():
        return f"{_REASON_NONFINITE}: {_NONFINITE_REASON_TEXT}"
    if bool((column == 0).all()):
        return f"{_REASON_ALL_ZERO}: {_ALL_ZERO_REASON_TEXT}"
    if torch.equal(column, lhs_target):
        return f"{_REASON_TAUTOLOGY}: {_TAUTOLOGY_REASON_TEXT}"
    return None


def _canonical_reason(term: str, registry: FunctionRegistry) -> str | None:
    """Layer 0: reject non-canonical / composite terms via ``split_terms``.

    Returns ``None`` when ``term`` is a single canonical funcall-IR term, else a
    ``syntax`` reason (infix/unary/parse error) or a ``composite`` reason listing
    the split parts so an agent can resubmit them separately.
    """
    try:
        parts = split_terms(term, registry)
    except Exception as exc:
        return f"{_REASON_SYNTAX}: non-canonical term (funcall IR required); {exc}"
    if len(parts) > 1:
        return (
            f"{_REASON_COMPOSITE}: top-level additive structure; submit as "
            f"{len(parts)} separate terms: {parts}"
        )
    return None


def _max_order_hint(
    term: str,
    max_order: int,
    known_fields: set[str] | None,
    known_axes: set[str] | None,
) -> str:
    """Return an actionable max_order hint for a terminal-derivative token.

    Uses the platform's own ``parse_derivative_name`` (not a private regex) to
    detect a same-axis terminal derivative (e.g. ``u_xxx`` -> order 3). When the
    implied order exceeds the current ``max_order`` the executor swallows the
    provider's "exceeds max_order" ValueError into a misleading "not defined"
    message; this appends the real cause. Empty string for non-derivatives.
    """
    parsed = parse_derivative_name(
        term, known_fields=known_fields, known_axes=known_axes
    )
    if parsed is None:
        return ""
    order = parsed[2]
    if order <= max_order:
        return ""
    if order <= _FD_MAX_ORDER:
        return (
            f" (looks like an order-{order} terminal derivative; current "
            f"max_order={max_order} — retry with max_order={order})"
        )
    return (
        f" (looks like an order-{order} terminal derivative; exceeds the "
        f"finite-difference cap (max order {_FD_MAX_ORDER}))"
    )
