
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from copy import deepcopy

from kd.core import law_agreement
from kd.core.equation import law_signature_from_evidence
from kd.search import IterationEvent, RunDirPaths, RunRecord

from kdagent.lineage import SegmentSummary

IMPROVEMENT_EPS = 1e-9


def law_from_evidence(evidence: Any) -> dict[str, Any] | None:
    if not evidence.is_valid or not evidence.support:
        return None
    catalog_fit = evidence.catalog_fit


    lhs = None if catalog_fit is None else deepcopy(catalog_fit["lhs_spec"])
    return {
        "support": evidence.support,
        "coefficients": evidence.coefficients,
        "lhs": lhs,
    }


def build_segment_report(
    run_dir: Path,
    parent: SegmentSummary | None,
    *,
    baseline_run_id: str | None = None,
) -> tuple[dict[str, Any], SegmentSummary]:
    paths = RunDirPaths(root=run_dir)
    record = RunRecord.load(paths.record)
    events = _load_events(paths.events)
    evidence = record.evidence
    direction = evidence.score_direction



    scores = [
        event.best_score for event in events if event.best_score is not None
    ]
    best = _best_of(scores, direction)
    if best is None:
        best = evidence.score
    iterations_stalled, segments_stalled, lineage_best = _plateau(
        scores, best, parent, direction
    )

    report: dict[str, Any] = {}
    score: dict[str, Any] = {"kind": evidence.score_kind, "direction": direction}
    if scores:
        score["first"] = scores[0]
        score["last"] = scores[-1]
    if best is not None:
        score["best"] = best
    report["score"] = score

    improvement: dict[str, Any] = {}
    if scores:
        improvement["within_segment"] = _improvement(scores[0], scores[-1], direction)
    if parent is not None and parent.best_score is not None and best is not None:





        if parent.segment_best is not None:
            improvement["vs_parent"] = _improvement(
                parent.segment_best, best, direction
            )
        improvement["vs_lineage_best"] = _improvement(
            parent.best_score, best, direction
        )
        improvement["baseline_run_id"] = baseline_run_id
    if improvement:
        report["improvement"] = improvement

    report["plateau"] = {
        "iterations_without_improvement": iterations_stalled,
        "segments_without_improvement": segments_stalled,
    }

    unique = [
        event.diagnostics["n_unique_candidates"]
        for event in events
        if event.diagnostics is not None
        and event.diagnostics.get("n_unique_candidates") is not None
    ]
    if unique:
        report["diversity"] = {
            "last": unique[-1],
            "mean": sum(unique) / len(unique),
        }

    complexity = [
        event.diagnostics["mean_complexity"]
        for event in events
        if event.diagnostics is not None
        and event.diagnostics.get("mean_complexity") is not None
    ]
    if complexity:



        report["complexity"] = {
            "first": complexity[0],
            "last": complexity[-1],
            "mean": sum(complexity) / len(complexity),
        }

    if events:
        n_candidates = sum(event.n_candidates for event in events)
        n_invalid = sum(event.n_invalid for event in events)
        validity: dict[str, Any] = {
            "n_candidates": n_candidates,
            "n_invalid": n_invalid,
        }



        if n_candidates > 0:
            validity["invalid_rate"] = n_invalid / n_candidates
        report["validity"] = validity

    signature = law_signature_from_evidence(evidence)
    if parent is not None and parent.signature is not None and signature is not None:
        agreement = law_agreement(parent.signature, signature)
        terms = set(signature.terms)
        parent_terms = set(parent.signature.terms)
        report["structure"] = {
            "changed": not agreement.structure,
            "support_added": sorted(terms - parent_terms),
            "support_removed": sorted(parent_terms - terms),
            "coefficient_close": agreement.coefficient,
            "max_abs_delta": agreement.max_abs_delta,
        }




    report["cost"] = {
        "search_seconds": round(record.cost.search_seconds, 3),
        "wallclock_seconds": round(record.cost.wallclock_seconds, 3),
    }

    summary = SegmentSummary(
        best_score=lineage_best,
        segment_best=best,
        score_direction=direction,
        iterations_without_improvement=iterations_stalled,
        segments_without_improvement=segments_stalled,
        signature=signature,
    )
    return report, summary


def _load_events(events_path: Path) -> tuple[IterationEvent, ...]:
    if not events_path.is_file():
        return ()
    lines = events_path.read_text(encoding="utf-8").splitlines()
    return tuple(
        IterationEvent.from_dict(json.loads(line)) for line in lines if line.strip()
    )


def _improvement(baseline: float, value: float, direction: str) -> float:
    if direction == "min":
        return baseline - value
    if direction == "max":
        return value - baseline
    raise ValueError(f"unknown score direction {direction!r}; expected 'min' or 'max'")


def _best_of(scores: Sequence[float], direction: str) -> float | None:
    if not scores:
        return None
    if direction == "min":
        return min(scores)
    if direction == "max":
        return max(scores)
    raise ValueError(f"unknown score direction {direction!r}; expected 'min' or 'max'")


def _plateau(
    scores: Sequence[float],
    best: float | None,
    parent: SegmentSummary | None,
    direction: str,
) -> tuple[int, int, float | None]:
    inherited = None if parent is None else parent.best_score
    running = inherited
    iterations = 0 if parent is None else parent.iterations_without_improvement
    for value in scores:
        if running is None or _improvement(running, value, direction) > IMPROVEMENT_EPS:
            running = value
            iterations = 0
        else:
            iterations += 1
    improved = best is not None and (
        inherited is None or _improvement(inherited, best, direction) > IMPROVEMENT_EPS
    )
    lineage_best = best if improved else inherited
    if improved or parent is None:
        return iterations, 0, lineage_best
    return iterations, parent.segments_without_improvement + 1, lineage_best
