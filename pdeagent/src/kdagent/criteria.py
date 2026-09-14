
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from kdagent.lineage import SegmentEntry, SegmentLedger

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


def default_selection(ledger: SegmentLedger, *, dataset_id: str | None) -> str | None:
    domain = [
        entry
        for entry in ledger.segments()
        if entry.dataset == dataset_id
        and entry.status == "completed"
        and entry.law is not None
        and not ledger.is_pruned(entry.run_id)
    ]
    if not domain:
        return None
    by_nmse = len({entry.instrument for entry in domain}) > 1
    if by_nmse:
        use_platform = any(entry.platform_nmse is not None for entry in domain)
        ranked = [
            (loss, index, entry.run_id)
            for index, entry in enumerate(domain)
            if (loss := (_loss(entry, by_nmse=True) if use_platform else entry.nmse))
            is not None
        ]
        if not ranked:
            return domain[-1].run_id
        return min(ranked)[-1]
    same_instrument_ranked = [
        (
            loss,
            math.inf if entry.nmse is None else entry.nmse,
            index,
            entry.run_id,
        )
        for index, entry in enumerate(domain)
        if (loss := _loss(entry, by_nmse=False)) is not None
    ]
    if not same_instrument_ranked:
        return domain[-1].run_id
    return min(same_instrument_ranked)[-1]


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


def _loss(entry: SegmentEntry, *, by_nmse: bool) -> float | None:
    if by_nmse:
        return entry.platform_nmse
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
