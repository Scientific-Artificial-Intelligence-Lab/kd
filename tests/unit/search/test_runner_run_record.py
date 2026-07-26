
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from kd.core.evaluator import EvaluationResult
from kd.core.platform.builder import PlatformBuilder
from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.search._torch_module_artifact import torch_module_artifact
from kd.search.dlga import DLGAConfig
from kd.search.dlga.plugin import DLGAPlugin
from kd.search.mini_table import build_mini_table
from kd.search.protocol import PlatformComponents
from kd.search.pysr import PySRConfig, PySRPlugin
from kd.search.records import RecordSchemaError, RunRecord
from kd.search.result import invalid_evaluation_result
from kd.search.run_spec import ConfigCanonicalizationError
from kd.search.runner import ExperimentRunner
from kd.search.sga import SGAConfig
from kd.search.sga.plugin import SGAPlugin
from tests.unit.search._runner_mocks import RecordingAlgorithm, StatefulAlgorithm
from tests.unit.search.pysr.conftest import FakePySRBackend, make_backend_factory


class _DeclaredRecordingAlgorithm(RecordingAlgorithm):
    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "declared", "seed": 23}


class _ConfiglessRecordingAlgorithm(RecordingAlgorithm):
    @property
    def config(self) -> dict[str, Any]:
        return {}


class _PartlyInvalidAlgorithm(RecordingAlgorithm):
    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return [
            EvaluationResult(
                mse=float(index + 1),
                nmse=float(index + 1),
                r2=0.0,
                is_valid=index % 2 == 0,
            )
            for index, _candidate in enumerate(candidates)
        ]


class _ExactQuadraticModel(nn.Module):

    def forward(self, *, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"u": 1.0 + x * x + t * t}


def _make_dataset(name: str = "run-record-test") -> PDEDataset:
    x = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = 1.0 + xg * xg + tg * tg
    return PDEDataset(
        name=name,
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo("x", x), "t": AxisInfo("t", t)},
        axis_order=["x", "t"],
        fields={"u": FieldData("u", u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _recording_components(
    *,
    name: str = "run-record-test",
    context: object | None = None,
) -> PlatformComponents:
    return PlatformComponents(
        dataset=_make_dataset(name),
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=context if context is not None else SimpleNamespace(),
        registry=MagicMock(),
    )


def _run_record(
    algorithm: RecordingAlgorithm,
    *,
    max_iterations: int = 1,
    batch_size: int = 2,
    preprocessing_seconds: float | None = None,
    context: object | None = None,
) -> RunRecord:
    result = ExperimentRunner(
        algorithm=algorithm,
        max_iterations=max_iterations,
        batch_size=batch_size,
    ).run(
        _recording_components(context=context),
        preprocessing_seconds=preprocessing_seconds,
    )
    assert result.run_record is not None
    return result.run_record


@pytest.mark.integration
def test_recording_run_populates_observed_cost_and_none_slots() -> None:
    iterations = 3
    batch_size = 4
    record = _run_record(
        RecordingAlgorithm(),
        max_iterations=iterations,
        batch_size=batch_size,
    )
    cost = record.cost

    assert cost.search_seconds > 0.0
    assert cost.wallclock_seconds == cost.search_seconds
    assert cost.preprocessing_seconds is None
    assert cost.surrogate_train_seconds is None
    assert cost.tokens_in is None
    assert cost.tokens_out is None
    assert cost.tokens_cached is None
    assert cost.api_cost_usd is None


    assert cost.boundary_results == iterations * batch_size
    assert cost.boundary_invalid_results == 0


@pytest.mark.integration
def test_boundary_invalid_results_count_only_invalid() -> None:
    record = _run_record(
        _PartlyInvalidAlgorithm(),
        max_iterations=2,
        batch_size=4,
    )

    assert record.cost.boundary_results == 8
    assert record.cost.boundary_invalid_results == 4


@pytest.mark.integration
def test_preprocessing_time_propagates_with_owned_exact_arithmetic() -> None:
    record = _run_record(RecordingAlgorithm(), preprocessing_seconds=0.25)

    assert record.cost.preprocessing_seconds == 0.25
    assert record.cost.wallclock_seconds == record.cost.search_seconds + 0.25


@pytest.mark.integration
def test_evidence_identity_uses_declared_algorithm_and_manifest_facts() -> None:
    result = ExperimentRunner(
        algorithm=_DeclaredRecordingAlgorithm(),
        max_iterations=1,
    ).run(_recording_components())

    assert result.run_record is not None
    assert result.manifest is not None
    evidence = result.run_record.evidence
    assert evidence.instrument == "declared"
    assert evidence.seed == result.manifest.seed == 23
    assert (
        evidence.dataset_cache_fingerprint
        == result.manifest.dataset_cache_fingerprint
    )


@pytest.mark.integration
def test_evidence_identity_falls_back_to_algorithm_class_name() -> None:
    record = _run_record(_ConfiglessRecordingAlgorithm())

    assert record.evidence.instrument == "_ConfiglessRecordingAlgorithm"


def _production_record(instrument: str) -> RunRecord:
    dataset = _make_dataset(name=f"format-{instrument}")
    plugin: SGAPlugin | DLGAPlugin | PySRPlugin
    if instrument == "sga":
        plugin = SGAPlugin(
            SGAConfig(
                num=2,
                depth=2,
                width=2,
                maxit=1,
                str_iters=1,
                seed=0,
            )
        )
    elif instrument == "dlga":
        plugin = DLGAPlugin(
            DLGAConfig(
                pop_size=2,
                max_modules=1,
                max_module_length=1,
                lhs_auto_select=False,
                epsilon=0.0,
                seed=0,
            ),
            surrogate_model=_ExactQuadraticModel(),
        )
    elif instrument == "pysr":
        backend = FakePySRBackend()
        plugin = PySRPlugin(
            PySRConfig(seed=0),
            backend_factory=make_backend_factory(backend),
        )
    else:
        raise ValueError(f"unknown instrument: {instrument}")

    components = PlatformBuilder(dataset, plugin.derivative_requirements).build()
    runner = ExperimentRunner(
        algorithm=plugin,
        max_iterations=1,
        batch_size=plugin.runner_batch_size,
    )
    result = runner.run(components)
    assert result.run_record is not None
    return result.run_record


def _record_key_tree(record: RunRecord) -> dict[str, frozenset[str]]:
    payload = record.to_dict()
    return {
        "envelope": frozenset(payload),
        "cost": frozenset(payload["cost"]),
        "evidence": frozenset(payload["evidence"]),
    }


@pytest.mark.integration
def test_three_instruments_emit_identical_record_format_and_table_rows() -> None:
    records = [_production_record(name) for name in ("sga", "dlga", "pysr")]

    key_trees = [_record_key_tree(record) for record in records]
    assert key_trees[0] == key_trees[1] == key_trees[2]
    assert {record.evidence.instrument for record in records} == {
        "sga",
        "dlga",
        "pysr",
    }
    data_rows = [
        line
        for line in build_mini_table(records).splitlines()[2:]
        if line.startswith("|")
    ]
    assert len(data_rows) == 3


@pytest.mark.integration
def test_identical_mock_runs_have_deterministic_evidence_and_hash() -> None:
    first = _run_record(_DeclaredRecordingAlgorithm())
    second = _run_record(_DeclaredRecordingAlgorithm())

    assert first.evidence.to_dict() == second.evidence.to_dict()
    assert first.evidence_hash == second.evidence_hash


@pytest.mark.integration
@pytest.mark.parametrize(
    ("context", "expected"),
    [
        pytest.param(SimpleNamespace(), None, id="absent"),
        pytest.param(
            SimpleNamespace(
                training_result=SimpleNamespace(elapsed_seconds=1.5)
            ),
            1.5,
            id="present",
        ),
    ],
)
def test_surrogate_training_time_reads_through_context(
    context: object,
    expected: float | None,
) -> None:
    record = _run_record(RecordingAlgorithm(), context=context)

    assert record.cost.surrogate_train_seconds == expected


class _TokenReportingAlgorithm(RecordingAlgorithm):

    def __init__(self, totals: object) -> None:
        super().__init__()
        self._totals = totals

    @property
    def llm_token_totals(self) -> object:
        return self._totals


@pytest.mark.integration
@pytest.mark.parametrize(
    ("totals", "expected_in", "expected_out"),
    [
        pytest.param({"tokens_in": 12, "tokens_out": 5}, 12, 5, id="valid-dict"),
        pytest.param({"tokens_in": 7}, 7, None, id="partial-dict"),
        pytest.param({"tokens_in": 0, "tokens_out": 0}, 0, 0, id="zero-is-valid"),
        pytest.param(None, None, None, id="attr-value-none"),
        pytest.param(MagicMock(), None, None, id="non-dict-mock"),
        pytest.param(
            {"tokens_in": True, "tokens_out": 5}, None, 5, id="bool-rejected"
        ),
        pytest.param(
            {"tokens_in": -1, "tokens_out": 5}, None, 5, id="negative-rejected"
        ),
    ],
)
def test_llm_token_totals_probe_populates_or_coerces(
    totals: object,
    expected_in: int | None,
    expected_out: int | None,
) -> None:
    record = _run_record(_TokenReportingAlgorithm(totals))

    assert record.cost.tokens_in == expected_in
    assert record.cost.tokens_out == expected_out


    assert record.cost.tokens_cached is None
    assert record.cost.api_cost_usd is None


@pytest.mark.integration
def test_llm_token_totals_absent_attribute_leaves_both_none() -> None:


    record = _run_record(RecordingAlgorithm())

    assert record.cost.tokens_in is None
    assert record.cost.tokens_out is None


class _RaisingTokenAlgorithm(RecordingAlgorithm):

    @property
    def llm_token_totals(self) -> dict[str, int]:
        raise RuntimeError("token telemetry backend unavailable")


class _RaisingSurrogateAlgorithm(RecordingAlgorithm):

    @property
    def surrogate_train_seconds(self) -> float:
        raise ValueError("surrogate timer exploded")


@pytest.mark.integration
def test_raising_token_property_does_not_abort_run() -> None:


    record = _run_record(_RaisingTokenAlgorithm())

    assert record.cost.tokens_in is None
    assert record.cost.tokens_out is None


@pytest.mark.integration
def test_raising_surrogate_property_falls_back_to_none() -> None:


    record = _run_record(_RaisingSurrogateAlgorithm())

    assert record.cost.surrogate_train_seconds is None


@pytest.mark.integration
def test_completed_invalid_final_result_still_emits_evidence() -> None:
    algorithm = RecordingAlgorithm()
    algorithm.final_eval_result = EvaluationResult(
        mse=float("inf"),
        nmse=float("nan"),
        r2=-float("inf"),
        is_valid=False,
        error_message="no valid candidate",
    )

    record = _run_record(algorithm)

    assert record.evidence.is_valid is False
    assert record.evidence.catalog_fit is None
    assert record.evidence.mse is None
    assert record.evidence.nmse is None
    assert record.evidence.r2 is None
    assert math.isfinite(record.cost.wallclock_seconds)







def _mock_components() -> PlatformComponents:
    return PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=MagicMock(),
        registry=MagicMock(),
    )


@pytest.mark.integration
def test_d1_invalid_reason_flows_from_plugin_result_to_record() -> None:


    algorithm = RecordingAlgorithm()
    algorithm.final_eval_result = invalid_evaluation_result(
        "evaluation blew up", score=None, reason="evaluation_error"
    )
    result = ExperimentRunner(algorithm=algorithm, max_iterations=1).run(
        _mock_components()
    )
    assert result.run_record is not None
    assert result.run_record.evidence.is_valid is False
    assert result.run_record.evidence.invalid_reason == "evaluation_error"


@pytest.mark.integration
def test_d4_restore_path_marks_manifest_resumed(tmp_path: Path) -> None:


    source_runner = ExperimentRunner(algorithm=StatefulAlgorithm(), max_iterations=1)
    checkpoint = tmp_path / "ckpt.pt"
    source_runner.save_checkpoint(checkpoint)

    resumed_runner = ExperimentRunner(algorithm=StatefulAlgorithm(), max_iterations=1)
    resumed_runner.load_checkpoint(checkpoint)
    result = resumed_runner.run(_mock_components())

    assert result.manifest is not None
    assert result.manifest.resumed is True


@pytest.mark.integration
def test_d4_fresh_run_manifest_not_resumed() -> None:

    result = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1).run(
        _mock_components()
    )
    assert result.manifest is not None
    assert result.manifest.resumed is False







class _NumpyIntConfigAlgorithm(RecordingAlgorithm):

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "numpy_int", "seed": np.int64(7)}


class _BadSurrogate(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.hook_fn = lambda value: value


class _InjectedSurrogateAlgorithm(RecordingAlgorithm):

    def __init__(self) -> None:
        super().__init__()
        self._bad_model = _BadSurrogate()

    @property
    def artifacts(self) -> dict[str, Any]:
        return {"surrogate_model": torch_module_artifact(self._bad_model)}


def test_item2_non_serializable_config_raises_at_entry_before_search() -> None:


    algorithm = _NumpyIntConfigAlgorithm()
    runner = ExperimentRunner(algorithm=algorithm, max_iterations=3)
    with pytest.raises(ConfigCanonicalizationError, match="seed"):
        runner.run(_mock_components())
    assert "propose" not in algorithm.call_log


def test_item2_injected_non_serializable_surrogate_raises_at_entry() -> None:


    algorithm = _InjectedSurrogateAlgorithm()
    runner = ExperimentRunner(algorithm=algorithm, max_iterations=3)
    with pytest.raises(RecordSchemaError) as exc_info:
        runner.run(_mock_components())
    assert isinstance(exc_info.value.__cause__, TypeError)
    assert "propose" not in algorithm.call_log







def test_item7_empty_checkpoint_state_marks_manifest_not_resumed(
    tmp_path: Path,
) -> None:



    source_runner = ExperimentRunner(algorithm=RecordingAlgorithm(), max_iterations=1)
    checkpoint = tmp_path / "empty_ckpt.pt"
    source_runner.save_checkpoint(checkpoint)

    resumed_runner = ExperimentRunner(
        algorithm=RecordingAlgorithm(), max_iterations=1
    )
    resumed_runner.load_checkpoint(checkpoint)
    assert resumed_runner._pending_restore_state == {}

    result = resumed_runner.run(_mock_components())
    assert result.manifest is not None
    assert result.manifest.resumed is False
