
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult

pytestmark = pytest.mark.unit

_N_SAMPLES = 8


def _base_eval_result() -> EvaluationResult:
    return EvaluationResult(
        mse=0.01,
        nmse=0.02,
        r2=0.98,
        complexity=2,
        coefficients=torch.tensor([1.0, 0.5]),
        is_valid=True,
        error_message="",
        selected_indices=[0, 1],
        residuals=torch.zeros(_N_SAMPLES),
        terms=["u", "u_x"],
        expression="add(u, u_x)",
    )


def _wrap_experiment_result(final_eval: EvaluationResult) -> ExperimentResult:
    return ExperimentResult(
        best_expression="add(u, u_x)",
        best_score=0.02,
        iterations=10,
        early_stopped=False,
        final_eval=final_eval,
        actual=torch.zeros(_N_SAMPLES),
        predicted=torch.zeros(_N_SAMPLES),
        dataset_name="test",
        algorithm_name="sga",
        config={},
        recorder=VizRecorder(),
    )


def _save_base_result(tmp_path: Path, name: str = "base.json") -> Path:
    path = tmp_path / name
    _wrap_experiment_result(_base_eval_result()).save(path)
    return path


def _rewrite_final_eval_score_keys(
    src: Path, dst: Path, score_keys: dict[str, Any]
) -> None:
    with src.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    final_eval = data["final_eval"]
    final_eval.pop("aic", None)
    final_eval.pop("score", None)
    final_eval.update(score_keys)
    with dst.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)







def test_new_format_save_load_round_trips_score(tmp_path: Path) -> None:
    eval_result = EvaluationResult(
        mse=0.01,
        nmse=0.02,
        r2=0.98,
        score=-42.5,
        complexity=1,
        coefficients=torch.tensor([1.5]),
        is_valid=True,
        residuals=torch.zeros(_N_SAMPLES),
        terms=["u_xx"],
        expression="u_xx",
    )
    path = tmp_path / "roundtrip.json"
    _wrap_experiment_result(eval_result).save(path)

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    assert "score" in raw["final_eval"], "to_dict must emit the migrated key"
    assert "aic" not in raw["final_eval"], "legacy key must not be written"

    loaded = ExperimentResult.load(path)
    assert loaded.final_eval.score == pytest.approx(-42.5)







def test_loader_reads_score_key(tmp_path: Path) -> None:
    base = _save_base_result(tmp_path)
    target = tmp_path / "score_only.json"
    _rewrite_final_eval_score_keys(base, target, {"score": 3.14})

    loaded = ExperimentResult.load(target)
    assert loaded.final_eval.score == pytest.approx(3.14)







def test_loader_falls_back_to_legacy_aic_key(tmp_path: Path) -> None:
    base = _save_base_result(tmp_path)
    target = tmp_path / "legacy_aic.json"
    _rewrite_final_eval_score_keys(base, target, {"aic": -7.5})

    loaded = ExperimentResult.load(target)
    assert loaded.final_eval.score == pytest.approx(-7.5)


def test_loader_legacy_aic_none_sentinel_preserved(tmp_path: Path) -> None:
    base = _save_base_result(tmp_path)
    target = tmp_path / "legacy_aic_none.json"
    _rewrite_final_eval_score_keys(base, target, {"aic": None})

    loaded = ExperimentResult.load(target)
    assert loaded.final_eval.score is None







def test_loader_score_key_wins_over_legacy_aic(tmp_path: Path) -> None:
    base = _save_base_result(tmp_path)
    target = tmp_path / "both_keys.json"
    _rewrite_final_eval_score_keys(base, target, {"score": 1.0, "aic": 99.0})

    loaded = ExperimentResult.load(target)
    assert loaded.final_eval.score == pytest.approx(1.0)







def test_loader_missing_both_keys_raises_keyerror(tmp_path: Path) -> None:
    base = _save_base_result(tmp_path)
    target = tmp_path / "neither_key.json"
    _rewrite_final_eval_score_keys(base, target, {})

    with pytest.raises(KeyError):
        ExperimentResult.load(target)
