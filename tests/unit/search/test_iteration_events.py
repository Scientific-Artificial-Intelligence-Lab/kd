
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from kd.search.iteration_events import (
    ITEREVENT_SCHEMA_VERSION,
    ITEREVENT_SCHEME,
    IterationEvent,
    IterationEventEmitter,
    IterationEventSinkError,
)

pytestmark = pytest.mark.unit







class _StubResult:

    def __init__(self, is_valid: bool = True) -> None:
        self.is_valid = is_valid


class _StubAlgorithm:

    def __init__(
        self, best_score: float = 0.5, best_expression: str = "u_xx"
    ) -> None:
        self.best_score = best_score
        self.best_expression = best_expression


def _make_event(**overrides: Any) -> IterationEvent:
    base: dict[str, Any] = {
        "schema_version": ITEREVENT_SCHEMA_VERSION,
        "scheme": ITEREVENT_SCHEME,
        "iteration": 0,
        "n_candidates": 3,
        "n_invalid": 1,
        "best_score": 0.25,
        "best_expression": "u_xx",
        "elapsed_seconds": 0.01,
        "diagnostics": None,
    }
    base.update(overrides)
    return IterationEvent(**base)


def _emit_once(
    emitter: IterationEventEmitter,
    *,
    best_score: float = 0.5,
    n_candidates: int = 2,
    n_invalid: int = 0,
) -> None:
    algo = _StubAlgorithm(best_score=best_score)
    candidates = [f"c{i}" for i in range(n_candidates)]
    results = [_StubResult(is_valid=(i >= n_invalid)) for i in range(n_candidates)]
    emitter.on_experiment_start(algo)
    emitter.on_iteration_end(0, algo, candidates, results)







class TestSchemaDiscipline:
    def test_constants_locked(self) -> None:
        assert ITEREVENT_SCHEME == "kd-iterevent-v1"
        assert ITEREVENT_SCHEMA_VERSION == 1

    def test_frozen_key_table_equals_dataclass_fields(self) -> None:
        from kd.search.iteration_events import _ITEREVENT_V1_FIELDS

        field_names = {f.name for f in dataclasses.fields(IterationEvent)}
        assert set(_ITEREVENT_V1_FIELDS) == field_names

    def test_from_dict_rejects_unknown_key(self) -> None:
        data = _make_event().to_dict()
        data["surprise"] = 1
        with pytest.raises(ValueError, match="unknown"):
            IterationEvent.from_dict(data)

    def test_from_dict_rejects_missing_key(self) -> None:
        data = _make_event().to_dict()
        del data["n_invalid"]
        with pytest.raises(ValueError, match="missing"):
            IterationEvent.from_dict(data)

    def test_from_dict_rejects_wrong_scheme(self) -> None:
        data = _make_event().to_dict()
        data["scheme"] = "kd-iterevent-v2"



        with pytest.raises(ValueError, match="unsupported scheme"):
            IterationEvent.from_dict(data)

    def test_from_dict_rejects_wrong_version(self) -> None:
        data = _make_event().to_dict()
        data["schema_version"] = 2
        with pytest.raises(ValueError, match="unsupported schema_version"):
            IterationEvent.from_dict(data)

    def test_from_dict_rejects_non_finite_best_score(self) -> None:
        data = _make_event().to_dict()
        data["best_score"] = float("nan")
        with pytest.raises(ValueError, match="best_score"):
            IterationEvent.from_dict(data)

    def test_from_dict_rejects_wrong_typed_iteration(self) -> None:
        data = _make_event().to_dict()
        data["iteration"] = "5"
        with pytest.raises(ValueError, match="iteration"):
            IterationEvent.from_dict(data)

    def test_from_dict_rejects_bool_schema_version(self) -> None:
        data = _make_event().to_dict()
        data["schema_version"] = True
        with pytest.raises(ValueError, match="schema_version"):
            IterationEvent.from_dict(data)

    def test_from_dict_rejects_non_dict_diagnostics(self) -> None:
        data = _make_event().to_dict()
        data["diagnostics"] = [1, 2]
        with pytest.raises(ValueError, match="diagnostics"):
            IterationEvent.from_dict(data)

    def test_from_dict_rejects_int_best_expression(self) -> None:
        data = _make_event().to_dict()
        data["best_expression"] = 42
        with pytest.raises(ValueError, match="best_expression"):
            IterationEvent.from_dict(data)







class TestNonFiniteDegrade:
    @pytest.mark.parametrize(
        "bad", [float("nan"), float("inf"), float("-inf")]
    )
    def test_non_finite_best_score_degrades_to_none(self, bad: float) -> None:
        captured: list[IterationEvent] = []
        emitter = IterationEventEmitter(on_event=captured.append)
        _emit_once(emitter, best_score=bad)
        assert captured[0].best_score is None

    def test_finite_best_score_passes_through(self) -> None:
        captured: list[IterationEvent] = []
        emitter = IterationEventEmitter(on_event=captured.append)
        _emit_once(emitter, best_score=0.375)
        assert captured[0].best_score == pytest.approx(0.375)
        assert isinstance(captured[0].best_score, float)

    def test_empty_expression_normalizes_both_best_fields_to_none(self) -> None:



        captured: list[IterationEvent] = []
        emitter = IterationEventEmitter(on_event=captured.append)
        algo = _StubAlgorithm(best_score=0.0, best_expression="")
        emitter.on_experiment_start(algo)
        emitter.on_iteration_end(0, algo, ["c0"], [_StubResult(is_valid=False)])
        assert captured[0].best_expression is None
        assert captured[0].best_score is None

    def test_zero_score_with_real_expression_is_kept(self) -> None:


        captured: list[IterationEvent] = []
        emitter = IterationEventEmitter(on_event=captured.append)
        algo = _StubAlgorithm(best_score=0.0, best_expression="u_xx")
        emitter.on_experiment_start(algo)
        emitter.on_iteration_end(0, algo, ["c0"], [_StubResult()])
        assert captured[0].best_expression == "u_xx"
        assert captured[0].best_score == 0.0







class TestJsonlRoundTrip:
    def test_round_trip_equals_original(self, tmp_path: Path) -> None:
        sink = tmp_path / "events.jsonl"
        captured: list[IterationEvent] = []
        emitter = IterationEventEmitter(on_event=captured.append, jsonl_path=sink)
        algo = _StubAlgorithm(best_score=0.5)
        emitter.on_experiment_start(algo)
        for it in range(3):
            algo.best_score = 0.5 - 0.1 * it
            emitter.on_iteration_end(
                it, algo, ["a", "b"], [_StubResult(), _StubResult()]
            )
        lines = sink.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        decoded = [IterationEvent.from_dict(json.loads(line)) for line in lines]
        assert decoded == captured







class TestAllowNanDefence:
    def test_dumps_rejects_smuggled_nan(self, tmp_path: Path) -> None:
        smuggled = IterationEvent(
            schema_version=ITEREVENT_SCHEMA_VERSION,
            scheme=ITEREVENT_SCHEME,
            iteration=0,
            n_candidates=1,
            n_invalid=0,
            best_score=float("nan"),
            best_expression="u_xx",
            elapsed_seconds=0.0,
            diagnostics=None,
        )
        with pytest.raises(ValueError):
            json.dumps(smuggled.to_dict(), allow_nan=False)







class TestSinkFailLoud:
    def test_existing_file_rejected_exclusive_create(self, tmp_path: Path) -> None:
        sink = tmp_path / "events.jsonl"
        sink.write_text("pre-existing\n", encoding="utf-8")
        emitter = IterationEventEmitter(jsonl_path=sink)
        with pytest.raises(IterationEventSinkError):
            _emit_once(emitter)

    def test_existing_file_error_chains_oserror(self, tmp_path: Path) -> None:
        sink = tmp_path / "events.jsonl"
        sink.write_text("x\n", encoding="utf-8")
        emitter = IterationEventEmitter(jsonl_path=sink)
        with pytest.raises(IterationEventSinkError) as exc_info:
            _emit_once(emitter)
        assert isinstance(exc_info.value.__cause__, OSError)

    def test_unwritable_directory_path_rejected(self, tmp_path: Path) -> None:
        sink = tmp_path / "as_dir"
        sink.mkdir()
        emitter = IterationEventEmitter(jsonl_path=sink)
        with pytest.raises(IterationEventSinkError) as exc_info:
            _emit_once(emitter)
        assert isinstance(exc_info.value.__cause__, OSError)







class TestConstructorValidation:
    def test_every_n_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="every_n_iterations"):
            IterationEventEmitter(on_event=lambda e: None, every_n_iterations=0)

    def test_both_channels_none_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            IterationEventEmitter()

    def test_should_stop_always_false(self) -> None:
        emitter = IterationEventEmitter(on_event=lambda e: None)
        assert emitter.should_stop is False







class TestChannelOrdering:
    def test_sink_written_before_raising_on_event(self, tmp_path: Path) -> None:
        sink = tmp_path / "events.jsonl"

        def boom(_event: IterationEvent) -> None:
            raise RuntimeError("consumer failed")

        emitter = IterationEventEmitter(on_event=boom, jsonl_path=sink)
        with pytest.raises(RuntimeError, match="consumer failed"):
            _emit_once(emitter)
        lines = sink.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        event = IterationEvent.from_dict(json.loads(lines[0]))
        assert event.iteration == 0







class TestSinkLifetime:
    def test_sink_closed_after_experiment_end(self, tmp_path: Path) -> None:
        sink = tmp_path / "events.jsonl"
        emitter = IterationEventEmitter(jsonl_path=sink)
        algo = _StubAlgorithm()
        emitter.on_experiment_start(algo)
        emitter.on_iteration_end(0, algo, ["a"], [_StubResult()])
        assert emitter._sink is not None
        emitter.on_experiment_end(algo)
        assert emitter._sink is None

    def test_experiment_end_idempotent_without_sink(self) -> None:
        emitter = IterationEventEmitter(on_event=lambda _e: None)
        emitter.on_experiment_end(_StubAlgorithm())
        emitter.on_experiment_end(_StubAlgorithm())
        assert emitter._sink is None
