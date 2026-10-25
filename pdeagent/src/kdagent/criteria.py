
from __future__ import annotations

import json
import math
from collections.abc import Callable, Collection, Sequence
from typing import Any, cast

from kdagent.lineage import (
    FORWARD_FRAME,
    SegmentEntry,
    SegmentLedger,
    lhs_inapplicable,
    standing_verification,
)

_FORK_SEED_STRIDE = 1000


def progress(entry: SegmentEntry) -> dict[str, Any]:
    summary = entry.summary
    if summary is None:
        return {
            "improved": False,
            "iterations_without_improvement": None,
            "segments_without_improvement": None,
        }
    return {
        "improved": (
            summary.segments_without_improvement == 0
            and summary.segment_best is not None
        ),
        "iterations_without_improvement": summary.iterations_without_improvement,
        "segments_without_improvement": summary.segments_without_improvement,
    }


def promise(
    entry: SegmentEntry,
    *,
    ledger: SegmentLedger,
    remaining_budget: int | None,
) -> dict[str, Any]:
    summary = entry.summary
    if summary is not None:
        streak = summary.segments_without_improvement
    else:
        parent = (
            None if entry.parent_run_id is None else ledger.entry(entry.parent_run_id)
        )
        inherited = (
            0
            if parent is None or parent.summary is None
            else parent.summary.segments_without_improvement
        )
        streak = inherited + 1
    projection = entry.report_projection or {}
    return {
        "plateau_streak": streak,
        "remaining_budget": remaining_budget,
        "validity": projection.get("validity"),
        "diversity": projection.get("diversity"),
        "complexity": projection.get("complexity"),
    }


def legal_moves(
    entry: SegmentEntry,
    schema_entry: dict[str, Any],
    *,
    ledger: SegmentLedger,
    available_iterations: Sequence[int],
    alternatives: Sequence[str] = (),
) -> list[str]:
    segmentation = schema_entry["segmentation"]
    pruned = ledger.is_pruned(entry.run_id)
    resumable = (
        segmentation["archive"] == "progress"
        and not pruned
        and len(available_iterations) >= 1
    )
    moves = ["fresh"]
    if alternatives:
        moves.append("switch")
    if sketch_supported(schema_entry["modes"]):
        moves.append("narrow")
    if resumable:
        moves.append("continue")
        if any(knob["resume_tier"] == "resume_safe" for knob in schema_entry["knobs"]):
            moves.append("adjust")
        if len(available_iterations) >= 2:
            moves.append("rollback")
        if segmentation["reseed"]:
            moves.append("fork")
    if not pruned and entry.status == "completed":
        moves += ["prune", "select"]
    moves.append("stop")
    return moves


def sketch_supported(modes: Sequence[dict[str, Any]]) -> bool:
    return any(
        level != "unsupported" for mode in modes for level in mode["sketch"].values()
    )


SELECTION_POLICY = "kd-agent-default-selection-v2"

FORWARD_WINDOW = 2.0


def selection_decision(
    ledger: SegmentLedger,
    *,
    dataset_id: str | None,
    exclude: Collection[str] = (),
) -> dict[str, Any]:
    domain = [
        entry
        for entry in ledger.segments()
        if entry.dataset == dataset_id
        and entry.run_id not in exclude
        and entry.status == "completed"
        and entry.law is not None
        and not ledger.is_pruned(entry.run_id)
    ]
    coverage = {
        "candidates": len(domain),
        "measured": sum(_forward_value(entry) is not None for entry in domain),
    }
    decision: dict[str, Any] = {
        "policy": SELECTION_POLICY,
        "run_id": None,
        "axis": "native_score",
        "reason": None,
        "window": None,
        "coverage": coverage,
    }
    if not domain:
        return {**decision, "reason": "empty_domain"}
    if len({_target(entry) for entry in domain}) > 1:
        return {**decision, "axis": None, "reason": "incomparable_targets"}
    if len({entry.instrument for entry in domain}) == 1:
        return {**decision, "run_id": _native_winner(domain)}
    if coverage["measured"]:
        run_id, window = _forward_winner(domain)
        return {**decision, "run_id": run_id, "axis": "forward_error", "window": window}
    if all(_forward_inapplicable(entry) for entry in domain):
        platform_run, platform_window = _platform_winner(domain)
        return {
            **decision,
            "run_id": platform_run,
            "axis": "platform_nmse",
            "reason": "forward_inapplicable",
            "window": platform_window,
        }
    return {**decision, "axis": "forward_error", "reason": "no_forward_value"}


SYSTEM_POLICY = "kd-agent-ode-system-selection-v1"


def system_decision(
    ledger: SegmentLedger, *, raw_fingerprint: str, target: str
) -> dict[str, Any] | None:
    candidates = [
        (system, standing)
        for system in ledger.systems()
        if system["raw_fingerprint"] == raw_fingerprint
        and not ledger.system_pruned(system["system_id"])
        and (standing := standing_verification(system)) is not None
    ]
    if not candidates:
        return None
    measured = [
        (value, index, system, standing)
        for index, (system, standing) in enumerate(candidates)
        if (value := standing["forward"]["value"]) is not None
    ]
    decision: dict[str, Any] = {
        "policy": SYSTEM_POLICY,
        "system_id": None,
        "verification_id": None,
        "run_id": None,
        "axis": "system_forward_error",
        "reason": None,
        "window": None,
        "coverage": {"candidates": len(candidates), "measured": len(measured)},
    }
    if not measured:
        return {**decision, "reason": "no_forward_value"}
    window = FORWARD_WINDOW * min(value for value, *_rest in measured)
    _terms, _value, _index, system, standing = min(
        (_system_terms(system), value, index, system, standing)
        for value, index, system, standing in measured
        if value <= window
    )
    return {
        **decision,
        "system_id": system["system_id"],
        "verification_id": standing["verification_id"],
        "run_id": system["members"][target],
        "window": window,
    }


def _system_terms(system: dict[str, Any]) -> int:
    return sum(len(law["support"]) for law in system["laws"].values())


def default_selection(
    ledger: SegmentLedger,
    *,
    dataset_id: str | None,
    exclude: Collection[str] = (),
) -> str | None:
    run_id: str | None = selection_decision(
        ledger, dataset_id=dataset_id, exclude=exclude
    )["run_id"]
    return run_id


def recommend(
    ledger: SegmentLedger,
    node: SegmentEntry,
    schema_entry: dict[str, Any],
    *,
    dataset_id: str | None,
    remaining_budget: int,
    perturbations: Sequence[dict[str, Any]],
    available_iterations_of: Callable[[str], Sequence[int]],
) -> dict[str, Any]:
    move: dict[str, Any] = {
        "action": "stop",
        "resume_from_run_id": None,
        "resume_iteration": None,
        "reseed": False,
        "seed": None,
        "prune": None,
        "params": None,
        "reason": None,
    }
    if remaining_budget <= 0:
        return move
    resumable = (
        schema_entry["segmentation"]["archive"] == "progress"
        and len(available_iterations_of(node.run_id)) >= 1
    )
    cell = _next_perturbation(ledger, node, perturbations)
    streak = promise(node, ledger=ledger, remaining_budget=remaining_budget)[
        "plateau_streak"
    ]
    fork = (
        _fork(
            ledger,
            node,
            dataset_id=dataset_id,
            cell=cell,
            available_iterations_of=available_iterations_of,
        )
        if streak >= 2 and schema_entry["segmentation"]["reseed"]
        else None
    )
    if fork is not None:
        return fork


    if not resumable:
        return {**move, "reason": "node_not_resumable"}
    if progress(node)["improved"]:
        return {
            **move,
            "action": "continue",
            "resume_from_run_id": node.run_id,
            "seed": node.seed,
            "params": node.params,
        }
    return {
        **move,
        "action": "adjust",
        "resume_from_run_id": node.run_id,
        "seed": node.seed,
        "params": cell,
    }


def _fork(
    ledger: SegmentLedger,
    node: SegmentEntry,
    *,
    dataset_id: str | None,
    cell: dict[str, Any] | None,
    available_iterations_of: Callable[[str], Sequence[int]],
) -> dict[str, Any] | None:
    target = default_selection(ledger, dataset_id=dataset_id)
    if target is None:
        return None
    winner = ledger.entry(target)
    iterations = tuple(available_iterations_of(target))
    if (
        winner.instrument != node.instrument
        or winner.dataset != node.dataset
        or not iterations
        or not _forkable(ledger, target, node)
    ):
        return None
    return {
        "action": "fork",
        "resume_from_run_id": target,
        "resume_iteration": max(iterations),
        "reseed": True,
        "seed": winner.seed + _FORK_SEED_STRIDE * _fork_index(ledger, target),
        "prune": node.run_id,
        "params": cell,
        "reason": None,
    }


RULE_B_NMSE_SLACK = 2.0

RULE_B_NMSE_FLOOR = 1e-9


def _target(entry: SegmentEntry) -> str:
    return json.dumps(cast(dict[str, Any], entry.law).get("lhs"), sort_keys=True)


def _term_count(entry: SegmentEntry) -> int:
    return len(cast(dict[str, Any], entry.law)["support"])


def _native_winner(domain: Sequence[SegmentEntry]) -> str:
    ranked = [
        (
            loss,
            math.inf if entry.nmse is None else entry.nmse,
            index,
            entry.run_id,
        )
        for index, entry in enumerate(domain)
        if (loss := _loss(entry)) is not None
    ]
    if not ranked:
        return domain[-1].run_id
    return min(ranked)[-1]


def _forward_status(entry: SegmentEntry) -> str:
    forward = entry.selection_forward
    return "not_measured" if forward is None else str(forward["status"])


def _forward_inapplicable(entry: SegmentEntry) -> bool:
    if _forward_status(entry) == "inapplicable":
        return True
    law = cast(dict[str, Any], entry.law)
    return "lhs" in law and lhs_inapplicable(law["lhs"]) is not None


def _forward_value(entry: SegmentEntry) -> float | None:
    forward = entry.selection_forward
    if forward is None or forward["frame"] != FORWARD_FRAME:
        return None
    value = forward["value"]
    return None if value is None else float(value)


def _forward_winner(domain: Sequence[SegmentEntry]) -> tuple[str, float]:
    measured = [
        (value, index, entry)
        for index, entry in enumerate(domain)
        if (value := _forward_value(entry)) is not None
    ]
    window = FORWARD_WINDOW * min(value for value, _index, _entry in measured)
    ranked = [
        (_term_count(entry), value, index, entry.run_id)
        for value, index, entry in measured
        if value <= window
    ]
    return min(ranked)[-1], window


def _platform_winner(domain: Sequence[SegmentEntry]) -> tuple[str | None, float | None]:
    measured = [
        (entry.platform_nmse, index, entry)
        for index, entry in enumerate(domain)
        if entry.platform_nmse is not None
    ]
    if not measured:
        return None, None
    best = min(nmse for nmse, _index, _entry in measured)
    window = max(RULE_B_NMSE_SLACK * best, RULE_B_NMSE_FLOOR)
    ranked = [
        (_term_count(entry), nmse, index, entry.run_id)
        for nmse, index, entry in measured
        if nmse <= window
    ]
    return min(ranked)[-1], window


def _loss(entry: SegmentEntry) -> float | None:
    summary = entry.summary
    if summary is None or summary.segment_best is None:
        return None
    if summary.score_direction == "max":
        return -summary.segment_best
    return summary.segment_best


def _forkable(ledger: SegmentLedger, target: str, node: SegmentEntry) -> bool:
    return node.run_id not in {entry.run_id for entry in ledger.chain(target)}


def _fork_index(ledger: SegmentLedger, run_id: str) -> int:
    return 1 + sum(1 for entry in ledger.children(run_id) if entry.action == "fork")


def _next_perturbation(
    ledger: SegmentLedger,
    node: SegmentEntry,
    perturbations: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    if not perturbations:
        return None
    spent = sum(
        1
        for entry in ledger.segments()
        if entry.dataset == node.dataset
        and entry.instrument == node.instrument
        and entry.parent_run_id is not None
        and entry.params != ledger.entry(entry.parent_run_id).params
    )
    return perturbations[min(spent, len(perturbations) - 1)]
