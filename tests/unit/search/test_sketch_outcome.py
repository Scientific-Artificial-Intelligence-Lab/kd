
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from kd.core.platform.sketch_compile import compile_sketch
from kd.search.sketch_outcome import (
    SKETCH_OUTCOME_ARTIFACT_TAG,
    SketchOutcome,
    sketch_outcome_payload,
    write_sketch_artifact,
)

from kd.core.equation import to_dict as equation_to_dict
from kd.core.equation.sketch import Sketch, SketchVerdict, sketch_to_dict
from kd.core.equation.types import Equation
from kd.core.schema_registry import SCHEMA_REGISTRY, SchemaEntry
from kd.search.result import ExperimentResult
from tests.unit.search._sketch_fakes import burgers_sketch

_CONTENT_KEYS = frozenset(
    {
        "artifact",
        "sketch",
        "verdict",
        "compile_report",
        "full_verify",
        "solution",
        "best_candidate",
        "failure",
    }
)


def closed_law() -> tuple[Sketch, Equation, SketchOutcome]:
    sketch = burgers_sketch(holes=())
    compiled = compile_sketch(sketch)
    lifted = compiled.lift(None)
    assert lifted is not None
    verdict = sketch.matches(lifted)
    outcome = SketchOutcome(
        solution=lifted,
        best_candidate=lifted,
        verdict=verdict,
        compile_report=compiled.report,
        full_verify=None,
        failure=None,
    )
    return sketch, lifted, outcome


def test_artifact_tag_is_registered_in_the_schema_index() -> None:
    assert SKETCH_OUTCOME_ARTIFACT_TAG == "kd-sketch-outcome-v1"
    assert SCHEMA_REGISTRY[SKETCH_OUTCOME_ARTIFACT_TAG] == SchemaEntry(
        1, "kd.search.sketch_outcome"
    )


def test_outcome_carries_the_certified_and_diagnostic_faces() -> None:
    _sketch, lifted, outcome = closed_law()

    assert outcome.solution is lifted
    assert outcome.best_candidate is lifted
    assert isinstance(outcome.verdict, SketchVerdict)
    assert outcome.verdict.overall is True
    assert outcome.failure is None


def test_outcome_is_immutable() -> None:
    _sketch, _lifted, outcome = closed_law()

    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.solution = None


def test_payload_carries_the_sketch_bytes_and_every_outcome_face() -> None:
    sketch, lifted, outcome = closed_law()
    sketch_payload = sketch_to_dict(sketch)

    payload = sketch_outcome_payload(outcome, sketch_payload=sketch_payload)

    assert set(payload) >= _CONTENT_KEYS
    assert payload["artifact"] == SKETCH_OUTCOME_ARTIFACT_TAG
    assert payload["sketch"] == sketch_payload
    assert payload["solution"] == equation_to_dict(lifted)
    assert payload["best_candidate"] == equation_to_dict(lifted)
    assert payload["compile_report"]["levels"]["fixed_terms"] == "lowered"
    assert payload["verdict"]["overall"] is True
    assert payload["full_verify"] is None
    json.dumps(payload, allow_nan=False)


def test_uncertified_outcome_records_the_failure_and_no_solution() -> None:
    sketch, lifted, _certified = closed_law()
    outcome = SketchOutcome(
        solution=None,
        best_candidate=lifted,
        verdict=None,
        compile_report=compile_sketch(sketch).report,
        full_verify=None,
        failure="lift: no liftable law from final_eval",
    )

    payload = sketch_outcome_payload(outcome, sketch_payload=sketch_to_dict(sketch))

    assert payload["solution"] is None
    assert payload["verdict"] is None
    assert payload["best_candidate"] == equation_to_dict(lifted)
    assert payload["failure"] == "lift: no liftable law from final_eval"


def test_writer_links_the_sidecar_to_its_run_by_evidence_hash(
    tmp_path: Path,
) -> None:
    sketch, _lifted, outcome = closed_law()
    target = tmp_path / "sketch_outcome.json"

    written = write_sketch_artifact(
        outcome,
        sketch_payload=sketch_to_dict(sketch),
        evidence_hash="sha256:feedface",
        path=target,
    )

    assert written == target
    stored = json.loads(target.read_text(encoding="utf-8"))
    assert stored["artifact"] == SKETCH_OUTCOME_ARTIFACT_TAG
    assert stored["evidence_hash"] == "sha256:feedface"
    assert stored["sketch"] == sketch_to_dict(sketch)
    assert list(tmp_path.glob("*.tmp")) == []


def test_writer_refuses_an_unlinked_sidecar(tmp_path: Path) -> None:
    sketch, _lifted, outcome = closed_law()

    with pytest.raises(ValueError, match="evidence_hash"):
        write_sketch_artifact(
            outcome,
            sketch_payload=sketch_to_dict(sketch),
            evidence_hash="",
            path=tmp_path / "unlinked.json",
        )


def test_experiment_result_slot_defaults_to_absent() -> None:
    slot = {
        field.name: field for field in dataclasses.fields(ExperimentResult)
    }["sketch_outcome"]
    assert slot.default is None
