
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kd.core.equation import to_dict as equation_to_dict
from kd.core.equation.sketch import SketchVerdict
from kd.core.equation.types import Equation
from kd.core.platform.sketch_compile import CompileReport
from kd.core.verify import VerificationReport

SKETCH_OUTCOME_ARTIFACT_TAG = "kd-sketch-outcome-v1"


@dataclass(frozen=True, kw_only=True)
class SketchOutcome:

    solution: Equation | None
    best_candidate: Equation | None
    verdict: SketchVerdict | None
    compile_report: CompileReport
    full_verify: VerificationReport | None
    failure: str | None


def sketch_outcome_payload(
    outcome: SketchOutcome,
    *,
    sketch_payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact": SKETCH_OUTCOME_ARTIFACT_TAG,
        "sketch": dict(sketch_payload),
        "verdict": asdict(outcome.verdict) if outcome.verdict is not None else None,
        "compile_report": {
            "levels": asdict(outcome.compile_report.levels),
            "notes": list(outcome.compile_report.notes),
        },
        "full_verify": (
            outcome.full_verify.to_dict() if outcome.full_verify is not None else None
        ),
        "solution": (
            equation_to_dict(outcome.solution)
            if outcome.solution is not None
            else None
        ),
        "best_candidate": (
            equation_to_dict(outcome.best_candidate)
            if outcome.best_candidate is not None
            else None
        ),
        "failure": outcome.failure,
    }


def write_sketch_artifact(
    outcome: SketchOutcome,
    *,
    sketch_payload: Mapping[str, Any],
    evidence_hash: str,
    path: str | Path,
) -> Path:
    if not evidence_hash:
        raise ValueError("evidence_hash must be a non-empty string")
    payload = sketch_outcome_payload(outcome, sketch_payload=sketch_payload)
    payload["evidence_hash"] = evidence_hash
    target = Path(path)
    tmp_path = target.with_name(f"{target.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(tmp_path, target)
    return target


__all__ = [
    "SKETCH_OUTCOME_ARTIFACT_TAG",
    "SketchOutcome",
    "sketch_outcome_payload",
    "write_sketch_artifact",
]
