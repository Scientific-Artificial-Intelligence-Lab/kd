
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest
import torch

from kd.core.evaluator import EvaluationResult
from kd.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd.search.recorder import VizRecorder
from kd.search.result import ExperimentResult
from kd.viz import VizEngine





_TWO_PI = 2.0 * 3.141592653589793


def _make_pde_dataset(nx: int = 20, nt: int = 10) -> PDEDataset:
    x = torch.linspace(0.0, _TWO_PI, nx)
    t = torch.linspace(0.0, 1.0, nt)
    u_field = torch.sin(x).unsqueeze(1) * torch.exp(-t).unsqueeze(0)
    return PDEDataset(
        name="autograd_warning_dataset",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u_field)},
        lhs_field="u",
        lhs_axis="t",
    )


def _make_diverging_sga_autograd_result(
    dataset: PDEDataset,
) -> ExperimentResult:



    rng = torch.Generator()
    rng.manual_seed(0)
    n_samples = dataset.get_shape()[0] * dataset.get_shape()[1]
    actual = torch.randn(n_samples, generator=rng)
    predicted = actual + torch.randn(n_samples, generator=rng) * 0.01
    residuals = predicted - actual

    recorder = VizRecorder()
    recorder.log("best_aic", 1.0)
    recorder.log("_best_score", 1.0)
    recorder.log("_best_expr", "u**3")
    recorder.log("_n_candidates", 1)

    return ExperimentResult(
        best_expression="u**3",
        best_score=1.0,
        iterations=1,
        early_stopped=False,
        final_eval=EvaluationResult(
            mse=0.01,
            nmse=0.01,
            r2=0.5,
            score=-50.0,
            complexity=1,
            coefficients=torch.tensor([0.0, 100.0]),
            is_valid=True,
            error_message="",
            selected_indices=[0, 1],
            residuals=residuals,
            terms=["u", "u**3"],
            expression="u**3",
        ),
        actual=actual,
        predicted=predicted,
        dataset_name=dataset.name,
        algorithm_name="SGA",
        config={"algorithm": "sga", "use_autograd": True},
        recorder=recorder,
    )







@pytest.mark.integration
class TestM1AutogradWarningDeduplication:

    def test_autograd_domain_note_appears_once_in_html(self, tmp_path: Path) -> None:
        dataset = _make_pde_dataset()
        result = _make_diverging_sga_autograd_result(dataset)
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(result, dataset=dataset)

        html = report.report.read_text() if report.report is not None else ""

        sentinel = "Domain note: this run fitted derivatives in an autograd"
        occurrences = html.count(sentinel)
        assert occurrences == 1, (
            f"Expected the autograd domain note to appear exactly once in "
            f"the rendered HTML, got {occurrences} occurrences. The Tier 2 "
            f"plot warning-forwarding paths bypass the engine-level dedup. "
            f"Warnings collected: {report.warnings}"
        )

    def test_autograd_domain_note_appears_once_in_report_warnings(
        self, tmp_path: Path
    ) -> None:
        dataset = _make_pde_dataset()
        result = _make_diverging_sga_autograd_result(dataset)
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(result, dataset=dataset)

        sentinel = "Domain note: this run fitted derivatives in an autograd"
        matching = [w for w in report.warnings if sentinel in w]
        assert len(matching) == 1, (
            f"Expected exactly one warning containing the autograd domain "
            f"note in report.warnings; got {len(matching)}: {matching}"
        )





        assert any(
            "integrate" in w.lower() or "step size" in w.lower()
            for w in report.warnings
        ), (
            f"integration-failure path was not exercised; expected an "
            f"'integrate' or 'step size' warning, got: {report.warnings}"
        )







@pytest.mark.integration
class TestL2VizFixtureExplicitConfig:

    def test_viz_pipeline_helper_sets_algorithm_key(self) -> None:
        from tests.integration.test_viz_pipeline import (
            _make_experiment_result,
        )

        result = _make_experiment_result()
        assert "algorithm" in result.config, (
            "Viz pipeline test fixture is missing config['algorithm']. "
            "Without it, the autograd-warning branch in VizEngine cannot "
            "be exercised by these tests."
        )

        assert result.config["algorithm"] in {"sga", "discover", "dlga"}, (
            f"config['algorithm'] should be a known algorithm name, got "
            f"{result.config['algorithm']!r}"
        )

    def test_viz_pipeline_helper_sets_use_autograd_key(self) -> None:
        from tests.integration.test_viz_pipeline import (
            _make_experiment_result,
        )

        result = _make_experiment_result()
        assert "use_autograd" in result.config, (
            "Viz pipeline test fixture is missing config['use_autograd']. "
            "Without it, the autograd-warning branch silently no-ops."
        )
        assert isinstance(result.config["use_autograd"], bool), (
            f"config['use_autograd'] must be a bool, got "
            f"{type(result.config['use_autograd']).__name__}"
        )
