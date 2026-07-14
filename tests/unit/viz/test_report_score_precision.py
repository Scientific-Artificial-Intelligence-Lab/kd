
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch

from kd.core.evaluator import EvaluationResult
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.viz.report import generate_report

_N_SAMPLES = 8



_SCORE_CELL_RE = re.compile(r"<th>Best[^<]*</th>\s*<td>([^<]+)</td>")


def _make_result(best_score: float) -> ExperimentResult:
    recorder = VizRecorder()
    recorder.log("_best_score", best_score)
    recorder.log("_best_expr", "expr")
    eval_result = EvaluationResult(
        mse=0.01,
        nmse=0.005,
        r2=0.99,
        score=best_score,
        complexity=2,
        coefficients=torch.tensor([1.0, 0.5]),
        is_valid=True,
        error_message="",
        selected_indices=[0, 1],
        residuals=torch.zeros(_N_SAMPLES),
        terms=["u", "u_x"],
        expression="add(u, u_x)",
    )
    return ExperimentResult(
        best_expression="add(u, u_x)",
        best_score=best_score,
        iterations=1,
        early_stopped=False,
        final_eval=eval_result,
        actual=torch.linspace(0.0, 1.0, _N_SAMPLES),
        predicted=torch.linspace(0.0, 1.0, _N_SAMPLES),
        dataset_name="precision_probe",
        algorithm_name="TestPlugin",
        config={"algorithm": "sga"},
        recorder=recorder,
        score_kind="reward",
        score_direction="max",
    )


def _rendered_score_cell(best_score: float, tmp_path: Path) -> str:
    output = tmp_path / "report.html"
    generate_report(_make_result(best_score), [], output)
    content = output.read_text(encoding="utf-8")
    match = _SCORE_CELL_RE.search(content)
    assert match is not None, (
        "could not locate the 'Best {kind}' score row in the report HTML; "
        "rows found: "
        f"{[ln.strip() for ln in content.splitlines() if 'Best' in ln]!r}"
    )
    return match.group(1)


def test_small_reward_keeps_significant_digits(tmp_path: Path) -> None:
    cell = _rendered_score_cell(0.94864, tmp_path)
    assert cell == "0.94864", (
        f"expected significant-digit formatting '0.94864', got {cell!r} "
        "(the current .2f formatting truncates to '0.95')."
    )
    assert cell != "0.95"


def test_large_magnitude_score_preserves_precision(tmp_path: Path) -> None:
    value = -28.776750720652373
    cell = _rendered_score_cell(value, tmp_path)
    rendered = float(cell)
    rel_err = abs(rendered - value) / abs(value)
    assert rel_err < 1e-4, (
        f"rendered score {cell!r} loses precision (rel err {rel_err:.2e}); "
        "significant-figure formatting must keep >= 5 sig figs."
    )
    assert cell != "-28.78"
